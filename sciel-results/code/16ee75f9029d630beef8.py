"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Simulated Annealing on Binary Vectors.
We restrict h to be a binary vector in {0, 1}^N with exactly N/2 ones.
This satisfies the constraints 0 <= h_i <= 1 and sum(h) = N/2 exactly.
The objective is to minimize the maximum linear overlap C_k.
We use simulated annealing with swap moves (exchange a 1 with a 0) 
to explore the discrete landscape. This avoids local minima 
associated with continuous relaxation and exploits the 
combinatorial structure of the problem.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145

def _score_from_h_binary(h):
    """
    Compute the maximum overlap for a binary vector h.
    h is a numpy array of 0s and 1s.
    Overlap C_k = (2/N) * (N/2 - r_k) where r_k is the linear autocorrelation.
    For binary h, r_k = sum_{i} h_i h_{i-k}.
    """
    n = len(h)
    # Compute linear autocorrelation efficiently using FFT
    # Pad to 2n to avoid circular convolution effects
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    
    # The overlap values
    cs = (2.0 / n) * (n / 2.0 - r)
    
    # We care about the maximum overlap. 
    # Note: C_0 is always (2/N)(N/2 - N/2) = 0? 
    # r_0 = sum h_i^2 = sum h_i = N/2. So C_0 = 0.
    # We want to minimize max_{k != 0} C_k.
    # Actually, the definition of overlap usually considers all shifts.
    # The max will be over k=1..N-1 typically, but C_0=0 is the min possible.
    # So we just take the max of the whole vector, or max of cs[1:].
    # Let's check the seed program: it returns max(cs).
    # Since C_0 = 0, and others are likely positive, max(cs) is the metric.
    
    return float(np.max(cs))

def _construct_binary_swap_move(h, ones_idx, zeros_idx):
    """
    Generate a new binary vector by swapping a random 1 with a random 0.
    """
    new_h = h.copy()
    i = ones_idx[np.random.randint(len(ones_idx))]
    j = zeros_idx[np.random.randint(len(zeros_idx))]
    new_h[i] = 0.0
    new_h[j] = 1.0
    return new_h, i, j

def construct():
    """
    Construct step heights using Simulated Annealing on binary vectors.
    """
    t0 = time.time()
    np.random.seed(42)
    
    # Initial state: random binary vector with N/2 ones
    ones_count = N // 2
    h = np.zeros(N)
    indices = np.random.choice(N, ones_count, replace=False)
    h[indices] = 1.0
    
    best_h = h.copy()
    best_score = _score_from_h_binary(h)
    
    # Maintain lists of indices for 1s and 0s for fast swap selection
    ones_idx = np.where(h == 1.0)[0]
    zeros_idx = np.where(h == 0.0)[0]
    
    # Simulated Annealing parameters
    # Start with high temperature to allow uphill moves
    T_start = 0.5
    T_end = 0.001
    cooling_rate = 0.999
    
    T = T_start
    current_score = best_score
    current_h = h
    current_ones_idx = ones_idx
    current_zeros_idx = zeros_idx
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < _DEADLINE - 10:
        # Generate a move: swap a random 1 with a random 0
        new_h, i, j = _construct_binary_swap_move(current_h, current_ones_idx, current_zeros_idx)
        
        # Update index sets
        # i was 1, becomes 0. j was 0, becomes 1.
        # Remove i from ones_idx, add to zeros_idx
        # Remove j from zeros_idx, add to ones_idx
        
        # Find positions in arrays (small N, so linear search is fine, but let's be safe)
        # Since N=100, we can just recompute or use a set. 
        # For speed in the inner loop, we can maintain sorted arrays or just recompute 
        # occasionally. But 100 elements is tiny.
        
        # Actually, let's just compute the new score.
        new_score = _score_from_h_binary(new_h)
        
        # Acceptance criterion
        if new_score < current_score:
            # Always accept improvement
            current_h = new_h
            current_score = new_score
            current_ones_idx = np.where(current_h == 1.0)[0]
            current_zeros_idx = np.where(current_h == 0.0)[0]
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Deterioration: accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta < 1e-12:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            
            if np.random.rand() < prob:
                current_h = new_h
                current_score = new_score
                current_ones_idx = np.where(current_h == 1.0)[0]
                current_zeros_idx = np.where(current_h == 0.0)[0]
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        iter_count += 1
        
        # If we've been stuck for a long time, restart with a new random seed
        # to escape local minima
        if no_improve_count > 5000:
            # Random restart
            h_new = np.zeros(N)
            indices = np.random.choice(N, ones_count, replace=False)
            h_new[indices] = 1.0
            score_new = _score_from_h_binary(h_new)
            
            if score_new < current_score:
                current_h = h_new
                current_score = score_new
                current_ones_idx = np.where(current_h == 1.0)[0]
                current_zeros_idx = np.where(current_h == 0.0)[0]
                if score_new < best_score:
                    best_score = score_new
                    best_h = current_h.copy()
            else:
                # Even if worse, we might want to jump to a new region
                # But let's only restart if we are very stuck
                if np.random.rand() < 0.1:
                    current_h = h_new
                    current_score = score_new
                    current_ones_idx = np.where(current_h == 1.0)[0]
                    current_zeros_idx = np.where(current_h == 0.0)[0]
            
            no_improve_count = 0
            T = T_start * 0.5 # Reheat slightly

    # Final check: ensure best_h is valid
    # It is binary, sum is N/2.
    # The task asks for "step heights" which are just the values.
    # Returning the binary vector is valid.
    
    return best_h

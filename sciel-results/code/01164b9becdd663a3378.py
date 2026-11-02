import numpy as np
import time

N = 100
DEADLINE = 140.0  # Safety margin from 150s

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initialize binary sequence
    # Start with a random permutation of 50 ones and 50 zeros
    h = np.zeros(N, dtype=int)
    ones_indices = rng.choice(N, N // 2, replace=False)
    h[ones_indices] = 1
    
    # Compute initial autocorrelation c11
    # c11[k] = sum h[i] h[i+k]
    # Using FFT is fast for initialization
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] corresponds to lag k (with some shift convention, but symmetric)
    # For real h, autocorr is even.
    # c11[k] = autocorr[k] for k=0..N-1?
    # np.fft.ifft(np.fft.fft(h)*np.fft.fft(h).conj()) gives circular autocorrelation R[k] = sum h[n] h[n-k]
    # Due to symmetry R[k] = R[N-k] = sum h[n] h[n+k].
    # So we can just use autocorr directly, ensuring we look at k=1..N-1.
    c11 = autocorr
    
    # Get current min
    # Exclude k=0
    min_val = np.min(c11[1:])
    
    best_min_val = min_val
    best_h = h.copy()
    best_c11 = c11.copy()
    
    # Local Search / Simulated Annealing
    # We want to maximize min(c11[1:])
    
    # Precompute indices for quick access?
    # We need to evaluate delta for all k for a given swap (i, j).
    
    # To speed up, we can identify "bottleneck" lags where c11[k] is minimal.
    # We only care about improving the minimum.
    
    # Simple hill climbing with random restarts or perturbations
    
    current_min = min_val
    stagnation_count = 0
    
    # To avoid O(N) update for every swap if we do many, 
    # O(N) is 100 ops. 10,000 swaps = 10^6 ops. Trivial.
    
    while time.time() - t0 < DEADLINE:
        # Identify lags with minimum value
        # c11[0] is max, ignore.
        min_lags = np.where(c11[1:] == current_min)[0] + 1
        
        # Pick a random swap
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        # Calculate delta c11 for all k
        # delta[k] = -h[i+k] - h[i-k] + h[j+k] + h[j-k]
        # Indices mod N
        
        # Vectorized update
        # Note: i and j are distinct.
        # h is 0 or 1.
        
        # We can just compute the new c11 for the affected parts?
        # No, delta is sparse? No, delta depends on h at i+k and j+k for all k.
        # So delta is a full vector.
        
        # Construct indices
        idx_i_plus = (i + np.arange(N)) % N
        idx_i_minus = (i - np.arange(N)) % N
        idx_j_plus = (j + np.arange(N)) % N
        idx_j_minus = (j - np.arange(N)) % N
        
        delta = -h[idx_i_plus] - h[idx_i_minus] + h[idx_j_plus] + h[idx_j_minus]
        
        # Apply delta to c11 to get new_c11
        new_c11 = c11 + delta
        
        # Check new min
        new_min = np.min(new_c11[1:])
        
        # Acceptance criterion
        # We want to maximize min.
        # If new_min > current_min, accept.
        # If new_min == current_min, accept with probability 0.5 (to explore).
        # If new_min < current_min, reject (or accept with very low prob, but let's stick to hill climbing first).
        
        accept = False
        if new_min > current_min:
            accept = True
        elif new_min == current_min:
            # Random walk on plateau
            if rng.random() < 0.5:
                accept = True
        
        if accept:
            # Update state
            h[i] = 0
            h[j] = 1
            c11 = new_c11
            current_min = new_min
            
            if current_min > best_min_val:
                best_min_val = current_min
                best_h = h.copy()
                best_c11 = c11.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # If stagnation is high, maybe perturb?
            # For now, let it run.
            
            # Optional: If we hit a theoretical max, break?
            # Max possible min is floor((N^2/4 - N/2)/(N-1)) ?
            # For N=100, approx 24.
            if best_min_val >= 24:
                break
        else:
            # Reject, no change to h or c11
            pass

    # Calculate final score based on best_h
    # Score is raw value of max overlap
    # Max Overlap = N/4 - min(c11[1:])
    # Using best_c11
    
    min_c11_best = np.min(best_c11[1:])
    max_overlap = N / 4.0 - min_c11_best
    
    # The problem asks to return the sequence h.
    # The evaluator calculates the score.
    # We return best_h.
    
    # Ensure best_h is float?
    # "sequence of step heights in [0, 1]"
    # Binary is in [0,1].
    
    return best_h.astype(float)

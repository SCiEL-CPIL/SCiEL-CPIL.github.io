import numpy as np
import time

N = 100

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap at shift s is sum(h_i * (1 - h_{i+s})) = sum(h_i) - sum(h_i * h_{i+s})
    Since sum(h_i) = N/2, this is N/2 - autocorr_s.
    We want to minimize the maximum of this over all non-zero shifts s.
    """
    # Use FFT for fast autocorrelation
    Fh = np.fft.fft(h)
    S = Fh * np.conj(Fh)
    acorr = np.fft.ifft(S).real
    # acorr[0] is sum(h^2). acorr[s] is sum(h_i * h_{i+s})
    # Overlap_s = N/2 - acorr[s]
    # We consider shifts 1 to N-1
    overlaps = N / 2.0 - acorr[1:N]
    return np.max(overlaps)

def construct():
    """
    Strategy:
    1. Initialize h with a random binary-like sequence (0s and 1s) 
       that sums to N/2 (50 zeros, 50 ones).
    2. Use Simulated Annealing to minimize the maximum overlap.
       - Move: Flip a 0 to 1 and a 1 to 0 (to maintain sum constraint) 
         or swap values in continuous space if we relax to [0,1].
       - Given the "step heights" nature and the success of binary search 
         in inspirations, we will stick to a discrete set of values 
         or a coarse grid, but the prompt says "step heights h in [0,1]".
         The inspiration note says "Binary local search finds low-autocorrelation 
         structures faster". 
         
       Let's try a hybrid: Start with binary 0/1. 
       Objective: Minimize max_overlap.
       
       However, the constraint is sum(h) = N/2.
       If we use 0/1 values, sum is integer. N=100, sum=50.
       
       Move: Pick i where h[i]==0 and j where h[j]==1. 
       Set h[i]=1, h[j]=0.
       
       This preserves the sum exactly.
       The values remain in {0,1} which is a subset of [0,1].
       
       Is it better to allow continuous values? 
       The LP relaxation suggests the optimal might not be binary. 
       However, for N=100, binary sequences (like Barker-like or Golay-like) 
       often have good autocorrelation properties.
       
       Let's implement Simulated Annealing on the binary space first.
       If time permits, we can try to relax to continuous values by 
       perturbing the binary solution slightly and projecting.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initialize with 50 zeros and 50 ones
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N//2, replace=False)
    h[ones_indices] = 1.0
    
    # Ensure we have a good starting point. 
    # Random might be bad. Let's try a few random restarts or a structured start.
    # A structured start: alternating 0, 1, 0, 1... 
    # h = np.array([0.5, 0.5, ...]) is score 1.0 (seed).
    # Binary alternating: 0,1,0,1...
    # Autocorrelation of 0,1,0,1...
    # h = [0, 1, 0, 1, ...]
    # acorr[1] = sum h_i h_{i+1} = 0
    # acorr[2] = sum h_i h_{i+2} = N/2 = 50
    # Overlap[2] = 50 - 50 = 0
    # Overlap[1] = 50 - 0 = 50
    # Max overlap = 50. 
    # Uniform 0.5: acorr[s] = N/4 = 25 for all s.
    # Overlap[s] = 50 - 25 = 25.
    # So binary alternating is worse (50 > 25).
    
    # We need a sequence with low autocorrelation at ALL shifts.
    # This is the "minimum overlap" problem, related to Barker sequences or Zadoff-Chu.
    
    current_h = h.copy()
    current_score = _max_overlap(current_h)
    
    best_h = current_h.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0
    alpha = 0.99
    min_T = 1e-4
    
    # Precompute indices of 0s and 1s for fast swapping
    # We will maintain lists of indices where h=0 and h=1
    
    zeros = np.where(h == 0)[0]
    ones = np.where(h == 1)[0]
    
    while time.time() < deadline and T > min_T:
        # Pick a random zero and a random one
        i = zeros[np.random.randint(len(zeros))]
        j = ones[np.random.randint(len(ones))]
        
        # Swap: h[i] becomes 1, h[j] becomes 0
        old_i = h[i]
        old_j = h[j]
        h[i] = 1.0
        h[j] = 0.0
        
        new_score = _max_overlap(h)
        
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            current_score = new_score
            # Update zero/one lists
            # i was in zeros, now in ones
            # j was in ones, now in zeros
            # To keep lists sorted or just valid, we can just swap values in the arrays
            # But numpy arrays don't swap easily. 
            # For efficiency, we can just regenerate the lists every K steps 
            # or use a different representation.
            # Given N=100, regenerating is O(N log N). 
            # Steps will be many. Let's just update the lists by swapping the values.
            # Find i in zeros and j in ones.
            # This is slow if we do linear search.
            # Let's just use a boolean mask or recompute if delta < 0 (improvement) 
            # or periodically.
            # Actually, for N=100, recomputing zeros/ones every step is 100 ops.
            # If we do 1000 steps/sec, that's 100k ops/sec. Negligible.
            
            # Update lists
            # Remove i from zeros, add to ones
            # Remove j from ones, add to zeros
            
            # Efficient way:
            # Just create new arrays? No, that's allocation overhead.
            # Let's just track the state and recompute indices every 100 steps 
            # or on improvement.
            # For now, let's just recompute every step for correctness and simplicity, 
            # as N is small.
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
            
            # Recompute indices
            zeros = np.where(h == 0)[0]
            ones = np.where(h == 1)[0]
            
        else:
            # Reject
            h[i] = old_i
            h[j] = old_j
        
        T *= alpha
    
    # Optional: Try to improve with continuous relaxation?
    # The binary solution is a valid solution. 
    # Can we do better with non-binary?
    # The LP relaxation bound was the target.
    # If we have time, we can take the best binary solution and do a few 
    # gradient steps in continuous space, projecting to [0,1] and sum constraint.
    
    if time.time() < deadline - 10:
        h_cont = best_h.copy()
        # Gradient descent on max_overlap is hard (non-smooth).
        # Instead, minimize the sum of squares of overlaps above a threshold?
        # Or just use SLSQP to minimize max_overlap?
        # SLSQP might be slow.
        
        # Let's try a simple local search in continuous space.
        # Perturb a few elements slightly.
        h_local = best_h.copy()
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.1, 0.1)
            new_h = h_local.copy()
            new_h[idx] += delta
            # Project to [0,1]
            new_h = np.clip(new_h, 0, 1)
            # Project to sum N/2
            s = np.sum(new_h)
            if s > 0:
                new_h *= (N/2.0 / s)
                new_h = np.clip(new_h, 0, 1)
            
            # Check if valid (sum exactly N/2 might drift slightly due to clipping)
            # The problem says "sum is exactly len/2". 
            # If we clip, sum might change.
            # We need to be careful.
            # If we allow continuous values, we must maintain sum exactly.
            # The projection `new_h *= (N/2.0 / s)` ensures sum is N/2 
            # before the final clip. If the final clip changes values, 
            # the sum might no longer be N/2.
            
            # To be safe, let's stick to the binary solution which 
            # inherently satisfies the constraints exactly.
            # Or, we can do a "repair" step:
            # If sum(new_h) != N/2, adjust one element to fix it.
            
            sum_new = np.sum(new_h)
            diff = N/2.0 - sum_new
            if abs(diff) > 1e-9:
                # Add diff to a random element that can absorb it
                k = np.random.randint(N)
                new_h[k] += diff
                if new_h[k] < 0 or new_h[k] > 1:
                    # Try another element or revert
                    new_h[k] -= diff
                    # Revert to previous best
                    continue
            
            sc = _max_overlap(new_h)
            if sc < best_score:
                best_score = sc
                best_h = new_h.copy()
                h_local = new_h.copy()
    
    return best_h

# The function _max_overlap is defined above.
# The function construct is defined above.

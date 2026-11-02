import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O_max(h):
    """Compute max overlap of h with shifted 1-h. Returns scalar."""
    # Overlap at shift k: sum_j h[j] * (1 - h[(j-k) mod N]) for positive shift
    # = sum_j h[j] - sum_j h[j]*h[(j-k) mod N]
    # sum_j h[j] = N/2 always (by constraint)
    # So overlap = SCALE * (N/2 - autocorr[k])
    # where autocorr[k] = sum_j h[j]*h[(j-k) mod N]
    
    # For binary h, autocorr[k] = number of positions j where h[j]=1 and h[(j-k)%N]=1
    # We can compute all autocorrelations via FFT or direct.
    
    # Direct computation for N=100: O(N^2) = 10000 ops, very fast
    # But let's use a smarter approach:
    # autocorr[k] = sum_{j=0}^{N-1} h[j] * h[(j-k) % N]
    
    h_int = (h >= 0.5).astype(np.int32)
    
    autocorr = np.zeros(N, dtype=np.float64)
    for k in range(N):
        # h[(j-k) % N] for j=0..N-1
        # This is np.roll(h, k)
        shifted = np.roll(h_int, k)
        autocorr[k] = np.dot(h_int, shifted)
    
    # Overlap at shift k = SCALE * (N/2 - autocorr[k])
    overlaps = SCALE * (N / 2.0 - autocorr)
    return float(overlaps.max())

def _compute_O_array(h):
    """Compute all overlaps as array."""
    h_int = (h >= 0.5).astype(np.int32)
    autocorr = np.zeros(N, dtype=np.float64)
    for k in range(N):
        shifted = np.roll(h_int, k)
        autocorr[k] = np.dot(h_int, shifted)
    overlaps = SCALE * (N / 2.0 - autocorr)
    return overlaps

def _sa_optimize(h_init, deadline, rng, max_iter=200000):
    """Simulated annealing on binary space with fixed sum constraint."""
    h = h_init.copy()
    # Ensure h is binary and has exactly N/2 ones
    h = (h >= 0.5).astype(np.float64)
    num_ones = int(h.sum())
    target_ones = N // 2
    
    # Adjust to have exactly target_ones
    if num_ones > target_ones:
        ones_idx = np.where(h == 1.0)[0]
        to_remove = num_ones - target_ones
        rng.shuffle(ones_idx)
        h[ones_idx[:to_remove]] = 0.0
    elif num_ones < target_ones:
        zeros_idx = np.where(h == 0.0)[0]
        to_add = target_ones - num_ones
        rng.shuffle(zeros_idx)
        h[zeros_idx[:to_add]] = 1.0
    
    best_h = h.copy()
    best_val = _compute_O_max(h)
    current_val = best_val
    
    # SA parameters
    T_start = 0.5
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    T = T_start
    
    ones_idx = np.where(h == 1.0)[0]
    zeros_idx = np.where(h == 0.0)[0]
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        T *= cooling
        
        # Pick random 1 and random 0 to swap
        if len(ones_idx) == 0 or len(zeros_idx) == 0:
            break
        
        i = ones_idx[rng.integers(len(ones_idx))]
        j = zeros_idx[rng.integers(len(zeros_idx))]
        
        # Try swap
        h[i] = 0.0
        h[j] = 1.0
        
        new_val = _compute_O_max(h)
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            current_val = new_val
            # Update indices
            ones_idx = np.where(h == 1.0)[0]
            zeros_idx = np.where(h == 0.0)[0]
            
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Reject
            h[i] = 1.0
            h[j] = 0.0
    
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = float('inf')
    
    # Generate diverse initial solutions
    inits = []
    
    # Random binary with exactly N/2 ones
    for _ in range(10):
        h = np.zeros(N)
        ones_pos = rng.choice(N, size=N//2, replace=False)
        h[ones_pos] = 1.0
        inits.append(h)
    
    # Structured inits
    idx = np.arange(N)
    for f in [1, 2, 3, 5, 7, 11, 13]:
        h = (0.5 + 0.5 * np.cos(2.0 * np.pi * f * idx / N) >= 0.5).astype(np.float64)
        # Adjust to exactly N/2 ones
        num_ones = int(h.sum())
        if num_ones > N//2:
            ones_idx = np.where(h == 1.0)[0]
            rng.shuffle(ones_idx)
            h[ones_idx[:num_ones - N//2]] = 0.0
        elif num_ones < N//2:
            zeros_idx = np.where(h == 0.0)[0]
            rng.shuffle(zeros_idx)
            h[zeros_idx[:N//2 - num_ones]] = 1.0
        inits.append(h)
    
    # Run SA on each init with time budget
    time_per_init = (deadline - start - 5.0) / len(inits)
    
    for init_h in inits:
        if time.time() > deadline - 3.0:
            break
        
        # Compute max iterations for this init based on time
        elapsed = time.time() - start
        remaining = deadline - start - elapsed - 2.0
        if remaining < 1.0:
            break
        
        # Estimate: each SA iteration takes ~50us for N=100
        max_iter = int(remaining / 0.00005)
        max_iter = min(max_iter, 500000)
        
        try:
            h_result, val_result = _sa_optimize(init_h, deadline, rng, max_iter=max_iter)
            if val_result < best_val:
                best_val = val_result
                best_h = h_result.copy()
        except Exception:
            continue
    
    if best_h is None:
        # Fallback: random binary
        h = np.zeros(N)
        ones_pos = rng.choice(N, size=N//2, replace=False)
        h[ones_pos] = 1.0
        best_h = h
    
    return best_h

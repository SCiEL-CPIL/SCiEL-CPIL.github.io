import numpy as np
import time

N = 400
DEADLINE = 145  # Safety margin

def _compute_c3(h):
    """Compute max absolute value of autoconvolution sidelobes."""
    n = len(h)
    nfft = 2 * n
    H = np.fft.rfft(h, n=nfft)
    conv = np.fft.irfft(H * H, n=nfft)
    # Sidelobes are indices 1 to n-1 and n+1 to 2n-2 (circularly shifted in linear conv context)
    # Linear autoconvolution of length 2n-1.
    # The peak is at index n-1 (lag 0).
    # We want max |conv[k]| for k != n-1.
    # Note: irfft returns real array of length nfft.
    # The linear convolution result is in conv[0:2n-1].
    # The peak (lag 0) is at index n-1.
    # So we take max(abs(conv[0:n-1]), abs(conv[n:2n-1]))
    if n == 1:
        return 0.0
    peak_idx = n - 1
    left = conv[0:peak_idx]
    right = conv[peak_idx+1:2*n-1]
    return np.max(np.abs(np.concatenate([left, right])))

def _local_search_best_improvement(h, deadline):
    """
    Local search using Best Improvement strategy.
    Tries flipping each bit, keeps the one that reduces C3 the most.
    """
    best_h = h.copy()
    best_c3 = _compute_c3(best_h)
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        best_delta = 1e-9  # Threshold for improvement
        best_idx = -1
        best_new_c3 = best_c3
        
        # Try flipping each bit
        for i in range(N):
            h_new = best_h.copy()
            h_new[i] *= -1
            c3_new = _compute_c3(h_new)
            
            if c3_new < best_new_c3 - best_delta:
                best_new_c3 = c3_new
                best_idx = i
                improved = True
                
        if improved:
            best_h[best_idx] *= -1
            best_c3 = best_new_c3
            no_improve_count = 0
        else:
            break
            
    return best_h, best_c3

def _perturb(h, p_flip=0.1):
    """Randomly flip a fraction of bits."""
    h_new = h.copy()
    num_flips = int(N * p_flip)
    if num_flips == 0:
        num_flips = 1
    idxs = np.random.choice(N, num_flips, replace=False)
    h_new[idxs] *= -1
    return h_new

def _initialize_structured():
    """Generate diverse initial sequences."""
    seqs = []
    np.random.seed(42)
    
    # 1. Random
    seqs.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 2. M-sequence like (LFSR)
    h = np.zeros(N)
    state = 1
    for i in range(N):
        h[i] = 1.0 if (state & 1) else -1.0
        state = (state << 1) | ((state >> 9) ^ (state >> 7) ^ (state >> 4) ^ 1)
        state &= 0xFFFF
    seqs.append(h)
    
    # 3. Sinusoidal
    t = np.linspace(0, 4*np.pi, N)
    h_sin = np.sign(np.sin(t))
    h_sin[h_sin == 0] = 1.0
    seqs.append(h_sin)
    
    # 4. Block structure
    h_block = np.ones(N)
    h_block[N//4:N//2] = -1
    h_block[3*N//4:] = -1
    seqs.append(h_block)
    
    # 5. Golay-like (if possible, but N=400 is not a power of 2, so approximate)
    # Just add another random
    seqs.append(np.random.choice([-1.0, 1.0], size=N))
    
    return seqs

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_h = None
    best_c3 = np.inf
    
    initializations = _initialize_structured()
    
    # Phase 1: Local Search from each initialization
    for h0 in initializations:
        if time.time() > deadline - 10:
            break
        
        # Run local search with a sub-deadline to allow multiple restarts
        sub_deadline = min(deadline - 5, start_time + 20) # Max 20s per LS run initially
        
        h_ls, c3_ls = _local_search_best_improvement(h0, sub_deadline)
        
        if c3_ls < best_c3:
            best_c3 = c3_ls
            best_h = h_ls.copy()
            
    # Phase 2: Iterated Local Search (ILS)
    # Continue from the best solution found so far
    if best_h is not None:
        current_h = best_h.copy()
        current_c3 = best_c3
        
        while time.time() < deadline - 5:
            # Perturb
            h_perturbed = _perturb(current_h, p_flip=0.05 + 0.1 * np.random.random())
            
            # Local Search from perturbed state
            # Allocate remaining time minus safety margin
            sub_deadline = deadline - 5
            h_ls, c3_ls = _local_search_best_improvement(h_perturbed, sub_deadline)
            
            # Accept if better
            if c3_ls < current_c3 - 1e-12:
                current_h = h_ls
                current_c3 = c3_ls
                if c3_ls < best_c3:
                    best_c3 = c3_ls
                    best_h = h_ls.copy()
            else:
                # If not better, we might still accept with some probability (SA-like)
                # But for ILS, usually we just restart from the global best if stuck?
                # Or keep the perturbed state?
                # Let's keep the global best as the anchor for next perturbation
                # to avoid drifting too far.
                pass
                
    return best_h


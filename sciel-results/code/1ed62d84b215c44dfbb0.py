"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 130.0
    
    # Precompute FFT parameters
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    # Precompute twiddles for fast sign-flip updates
    k_indices = np.arange(fsize // 2 + 1)
    twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
    
    # Strategy: Direct discrete search using multi-start local search with 
    # diverse initial patterns. This avoids the pitfalls of continuous relaxation
    # and directly optimizes the discrete sign variables.
    
    rng = random.Random(42)
    
    def _local_search(h, max_iter=5000):
        """
        Local search: flip signs to reduce the objective.
        Uses efficient FFT updates.
        """
        h = h.copy()
        H = np.fft.rfft(h, fsize)
        current_val = _score(h)
        
        # Greedy hill climb with random restarts within the local search
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 5:
            improved = False
            # Random order of indices to avoid bias
            order = rng.sample(range(N), N)
            for i in order:
                if time.time() > deadline - 5:
                    return h, current_val
                old_val = h[i]
                new_val = -old_val
                h[i] = new_val
                delta = -2 * old_val
                H += delta * twiddles[i]
                
                ac = np.fft.irfft(H * H, fsize)
                new_ac_max = np.max(np.abs(ac[:size]))
                s = np.sum(h)
                if abs(s) < 1e-9:
                    new_score = 1e10
                else:
                    new_score = new_ac_max / (s * s)
                
                if new_score < current_val:
                    current_val = new_score
                    improved = True
                else:
                    h[i] = old_val
                    H -= delta * twiddles[i]
            iters += 1
            
        return h, current_val
    
    def _simulated_annealing(h, T_init=1.0, T_final=0.01, max_iter=5000):
        """
        Simulated annealing for escaping local minima.
        """
        h = h.copy()
        H = np.fft.rfft(h, fsize)
        current_val = _score(h)
        best_h = h.copy()
        best_val = current_val
        T = T_init
        cooling = (T_final / T_init) ** (1.0 / max_iter)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            i = rng.randint(0, N - 1)
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac[:size]))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            delta_score = new_score - current_val
            if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = old_val
                H -= delta * twiddles[i]
            T *= cooling
            
        return best_h, best_val
    
    # Generate diverse initial candidates
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Random
    for _ in range(3):
        inits.append(np.array([1.0 if rng.random() > 0.5 else -1.0 for _ in range(N)]))
    
    # 3. Chirp-like patterns
    x = np.arange(N)
    inits.append(np.sign(np.cos(np.pi * x**2 / N)))
    inits.append(np.sign(np.sin(np.pi * x / N)))
    
    # 4. Block patterns
    for block_size in [10, 20, 50]:
        h = np.ones(N)
        for start in range(0, N, block_size * 2):
            h[start:start+block_size] = -1
        inits.append(h)
        
    # 5. Mersenne-like patterns (if N allows)
    # Just use a few more random ones with different seeds
    for seed in [1, 2, 3]:
        rng2 = random.Random(seed)
        inits.append(np.array([1.0 if rng2.random() > 0.5 else -1.0 for _ in range(N)]))
    
    best_h = inits[0].copy()
    best_val = _score(best_h)
    
    # Run local search on each initial candidate
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        h_ls, val_ls = _local_search(h_init, max_iter=2000)
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls.copy()
            
    # Run simulated annealing on the best solution found so far
    if time.time() < deadline - 10:
        h_sa, val_sa = _simulated_annealing(best_h, T_init=0.5, T_final=0.01, max_iter=10000)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
            
    # Final local search polish
    if time.time() < deadline - 5:
        h_final, val_final = _local_search(best_h, max_iter=5000)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
            
    return best_h

# EVOLVE-BLOCK-END

import numpy as np
import time
import itertools

N = 50
DEADLINE = 145  # Safety margin for 150s limit

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def construct():
    start_time = time.time()
    
    # The objective is scale invariant: f(c*h) = f(h).
    # We can fix the L2 norm to 1, or simply search over a grid of amplitudes.
    # A key insight from similar inequalities (e.g., Boas, Hardy-Littlewood) is that
    # extremizers often have a "binary" or "few-level" structure.
    
    # Strategy:
    # 1. Try sparse supports with varying sizes.
    # 2. For each support, try a small set of amplitude patterns (e.g., all 1s, or weighted).
    # 3. Use a simple local search (hill climbing) on the discrete amplitudes 
    #    restricted to a grid of values {0, 0.5, 1.0, 1.5, ...} or normalized variants.
    
    best_score = 0.0
    best_h = np.zeros(N)
    
    # Pre-defined candidate supports
    # 1. Full support
    # 2. Sparse supports (random subsets)
    # 3. Structured supports (e.g., every k-th)
    
    supports = []
    supports.append(set(range(N))) # Full
    
    # Random sparse supports
    np.random.seed(42)
    for k in [5, 10, 15, 20, 25, 30, 40]:
        for _ in range(20):
            supports.append(set(np.random.choice(N, k, replace=False)))
            
    # Structured supports
    for step in [2, 3, 4, 5]:
        supports.append(set(range(0, N, step)))
        
    # Amplitude grids
    # Since scale is invariant, we can fix the max amplitude to 1.0 and vary others relatively.
    # Or just try a grid of absolute values and let the scale invariance handle the rest.
    # Actually, let's fix L2 norm = 1 for the candidate h.
    
    # We will iterate through supports. For each support, we try to optimize the amplitudes
    # using a simple coordinate ascent or random perturbation on a discrete grid.
    
    # Discrete amplitude levels: 
    # Let's try levels: 0.0, 0.25, 0.5, 0.75, 1.0
    # And then normalize to L2=1.
    
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # For efficiency, we won't do full exhaustive search on 50 dims.
    # We will do a "random restart + local search" on the discrete amplitude vector.
    
    max_evals = 50000
    
    for support in supports:
        if time.time() > DEADLINE:
            break
        
        if len(support) == 0:
            continue
            
        # Initialize with all 1s on support
        h = np.zeros(N)
        h[list(support)] = 1.0
        
        # Local search on discrete levels
        # We represent the state as indices into 'levels' for each position in support.
        # But simpler: just perturb the actual values and snap to nearest level?
        # Or: treat it as a combinatorial problem on the subset of active elements.
        
        # Let's do a simple hill climbing where we change one element's level at a time.
        # To speed up, we'll precompute the convolution effect? No, just recompute.
        # N=50, conv is fast.
        
        current_score = _score(h)
        if current_score > best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Local search
        # We'll try to improve by changing one amplitude at a time to another level.
        # To avoid O(N * L) per step being too slow if we do many steps, 
        # we'll limit the number of local search iterations per support.
        
        for _ in range(50): # Max 50 improvement steps per support
            if time.time() > DEADLINE:
                break
                
            improved = False
            # Shuffle order of elements to avoid bias
            elems = list(support)
            np.random.shuffle(elems)
            
            for idx in elems:
                # Try changing h[idx] to each other level
                original_val = h[idx]
                for new_level in levels:
                    if new_level == original_val:
                        continue
                    
                    h[idx] = new_level
                    # Quick check: if all zeros, skip
                    if np.all(h == 0):
                        h[idx] = original_val
                        continue
                        
                    new_score = _score(h)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_h = h.copy()
                        break # Move to next element after improvement
                    else:
                        h[idx] = original_val
                
                if improved:
                    break # Restart scan if any improvement found
    
    # If we have time, try a "continuous" local search on the best discrete solution
    # using a simple gradient-free method (Nelder-Mead) on the continuous amplitudes.
    if best_score > 0 and time.time() < DEADLINE - 10:
        # Use the best_h found so far
        h_init = best_h.copy()
        
        # Normalize to L2=1 for stability in optimization
        norm = np.linalg.norm(h_init)
        if norm > 0:
            h_init = h_init / norm
            
        # Objective to minimize: -score(h)
        def objective(x):
            if np.any(x < 0):
                return 1e10
            return -_score(x)
            
        try:
            from scipy.optimize import minimize
            res = minimize(objective, h_init, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success or res.fun < -best_score:
                h_opt = np.maximum(res.x, 0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass

    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print("Best score:", _score(np.array(result)))

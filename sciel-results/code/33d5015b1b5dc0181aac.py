import numpy as np
import time

def _precompute_matrix(n_samples):
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    k = np.arange(24)
    Z = np.exp(1j * np.outer(theta, k))
    return Z

# Pre-compute a high-resolution matrix for evaluation
Z_mat = _precompute_matrix(20000)

def _fast_supnorm(c):
    """
    Fast sup norm evaluation.
    c: array of 24 coefficients (+1/-1)
    """
    vals = Z_mat @ c
    return np.max(np.abs(vals))

def _local_search_2opt(c, deadline):
    """
    Local search flipping 1 or 2 bits.
    Returns improved coefficients and score.
    """
    c = c.copy().astype(float)
    current_score = _fast_supnorm(c)
    n = 24
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        best_delta = 0
        best_flip = None
        best_type = None # 1 for single, 2 for double
        
        # Single flips
        for i in range(n):
            c[i] = -c[i]
            new_score = _fast_supnorm(c)
            c[i] = -c[i] # revert
            
            delta = current_score - new_score
            if delta > best_delta:
                best_delta = delta
                best_flip = (i,)
                best_type = 1
                
        # If no single flip improved, try double flips
        if best_delta <= 0:
            for i in range(n):
                for j in range(i+1, n):
                    c[i] = -c[i]
                    c[j] = -c[j]
                    new_score = _fast_supnorm(c)
                    c[i] = -c[i] # revert
                    c[j] = -c[j] # revert
                    
                    delta = current_score - new_score
                    if delta > best_delta:
                        best_delta = delta
                        best_flip = (i, j)
                        best_type = 2
                        
        if best_delta > 1e-9:
            # Apply best flip
            if best_type == 1:
                c[best_flip[0]] = -c[best_flip[0]]
            else:
                c[best_flip[0]] = -c[best_flip[0]]
                c[best_flip[1]] = -c[best_flip[1]]
                
            current_score = _fast_supnorm(c)
            improved = True
            
    return c, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # Safety margin
    
    # Initial Seeds
    seeds = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    g = np.array(p[:24], dtype=float)
    seeds.append(g)
    seeds.append(-g)
    seeds.append(g[::-1])
    seeds.append(-g[::-1])
    
    # 2. Random seeds
    rng = np.random.default_rng(42)
    for _ in range(10):
        seeds.append(rng.choice([-1.0, 1.0], size=24))
        
    best_coeffs = None
    best_score = np.inf
    
    # Evaluate seeds
    for s in seeds:
        score = _fast_supnorm(s)
        if score < best_score:
            best_score = score
            best_coeffs = s.copy()
            
    # Iterated Local Search
    while time.time() < deadline - 10.0:
        # Perturb the best found so far
        current = best_coeffs.copy()
        
        # Random perturbation: flip 3-6 random bits
        n_flips = rng.integers(3, 7)
        flip_idx = rng.choice(24, size=n_flips, replace=False)
        for idx in flip_idx:
            current[idx] = -current[idx]
            
        # Local Search
        current, current_score = _local_search_2opt(current, deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current.copy()
            
    # Final refinement: run local search on the very best found
    best_coeffs, best_score = _local_search_2opt(best_coeffs, deadline)
    
    # Convert to int list
    result = np.sign(best_coeffs).astype(int)
    result[result == 0] = 1
    return result.tolist()

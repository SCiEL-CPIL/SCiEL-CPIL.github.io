import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    N = 24
    sqrt_N = np.sqrt(N)
    
    # Precompute evaluation matrix for search
    M_search = 2048
    angles_search = np.linspace(0, 2*np.pi, M_search, endpoint=False)
    k_idx = np.arange(N)
    exponents_search = np.outer(angles_search, k_idx)
    V_search = np.exp(1j * exponents_search)
    
    def score_search(coeffs):
        vals = V_search @ coeffs
        return np.max(np.abs(vals)) / sqrt_N

    def local_search(initial_coeffs):
        coeffs = initial_coeffs.copy()
        current_score = score_search(coeffs)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                old_val = coeffs[i]
                coeffs[i] = -old_val
                new_score = score_search(coeffs)
                if new_score < current_score - 1e-15:
                    current_score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
        return coeffs, current_score

    # Store top candidates
    candidates = [] # list of (score, coeffs)
    
    def add_candidate(coeffs, sc):
        # Simple insertion sort for small list
        candidates.append((sc, coeffs.copy()))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 20:
            candidates.pop()

    # Golay-Rudin-Shapiro start
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs = np.array(p[:N], dtype=float)
    
    # First run
    coeffs, sc = local_search(grs)
    add_candidate(coeffs, sc)
    
    # Iterated Local Search
    iteration = 0
    while time.time() < deadline:
        if iteration > 0 and candidates:
            # Pick a random candidate from top 5 to perturb
            top_k = min(5, len(candidates))
            idx = np.random.randint(0, top_k)
            base_coeffs, _ = candidates[idx]
            init = base_coeffs.copy()
            # Flip 2 to 5 random bits
            num_flips = np.random.randint(2, 6)
            if num_flips > N:
                num_flips = N
            idx_flip = np.random.choice(N, num_flips, replace=False)
            init[idx_flip] *= -1
        else:
            init = np.random.choice([-1.0, 1.0], size=N)
            
        coeffs, sc = local_search(init)
        add_candidate(coeffs, sc)
        
        iteration += 1

    if not candidates:
        return [1] * N

    # Re-evaluate top candidates with higher precision
    M_final = 10000
    angles_final = np.linspace(0, 2*np.pi, M_final, endpoint=False)
    exponents_final = np.outer(angles_final, k_idx)
    V_final = np.exp(1j * exponents_final)
    
    def score_final(coeffs):
        vals = V_final @ coeffs
        return np.max(np.abs(vals)) / sqrt_N

    best_final_score = np.inf
    best_final_coeffs = None
    
    for sc, coeffs in candidates:
        # Only re-evaluate if it's potentially good
        # sc is based on M_search. 
        # We can just re-evaluate all top 20.
        final_sc = score_final(coeffs)
        if final_sc < best_final_score:
            best_final_score = final_sc
            best_final_coeffs = coeffs

    if best_final_coeffs is None:
        return [1] * N

    return [int(c) for c in best_final_coeffs]

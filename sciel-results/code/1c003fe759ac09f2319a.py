import numpy as np
import time

def construct():
    """
    Finds 24 coefficients in {+1, -1} minimizing the sup norm of P(z) on |z|=1.
    Uses local search with random restarts and a dense grid for evaluation.
    """
    N_COEFFS = 24
    # Time limit management
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin
    
    # Grid for evaluation
    # Dense enough to capture peaks, but fast enough for search
    # The evaluator uses golden section, but for search, a fixed grid is fine.
    # Let's use 2048 points.
    N_GRID = 2048
    thetas = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Precompute powers of z: A[i, k] = z[i]^k
    # A is (N_GRID, N_COEFFS)
    A = np.zeros((N_GRID, N_COEFFS), dtype=np.complex128)
    A[:, 0] = 1.0
    for k in range(1, N_COEFFS):
        A[:, k] = A[:, k-1] * z
        
    def get_max_norm(c):
        # c is a numpy array of shape (N_COEFFS,) with values +1/-1
        P = A.dot(c)
        return np.max(np.abs(P))
        
    def local_search(c, max_time=None):
        """
        Performs local search starting from c.
        Returns the best c found.
        """
        current_c = c.copy()
        current_val = get_max_norm(current_c)
        
        # We want to minimize current_val
        improved = True
        while improved and (max_time is None or time.time() - start_time < max_time):
            improved = False
            best_val = current_val
            best_c = current_c.copy()
            best_flip_idx = -1
            
            # Try flipping each bit
            for k in range(N_COEFFS):
                # Flip bit k
                # P_new = P_old - 2 * c[k] * A[:, k]
                # We don't have P_old stored, so we recompute or store it.
                # Recomputing is simpler and fast enough for N=24, K=2048.
                # But we can be smarter.
                
                # Let's just recompute for simplicity first, optimize if needed.
                trial_c = current_c.copy()
                trial_c[k] = -trial_c[k]
                trial_val = get_max_norm(trial_c)
                
                if trial_val < best_val:
                    best_val = trial_val
                    best_c = trial_c
                    best_flip_idx = k
            
            if best_flip_idx != -1:
                current_c = best_c
                current_val = best_val
                improved = True
            else:
                # No single bit flip improves. 
                # Try a random perturbation to escape local minimum?
                # Or just stop and let the outer loop restart.
                # Since we do many restarts, stopping is fine.
                break
                
        return current_c, current_val

    # Seed with Golay-Rudin-Shapiro as one of the initial candidates
    # The seed program logic:
    def get_gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n])
        
    initial_candidates = []
    
    # 1. GR Sequence
    initial_candidates.append(get_gr_sequence(N_COEFFS))
    
    # 2. Random candidates
    # Generate a few random ones
    for _ in range(20):
        initial_candidates.append(np.random.choice([-1, 1], size=N_COEFFS))
        
    # 3. Maybe some structured ones?
    # Alternating
    initial_candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N_COEFFS)]))
    # All ones
    initial_candidates.append(np.ones(N_COEFFS, dtype=int))
    
    best_c_overall = None
    best_val_overall = np.inf
    
    # Run local search from each initial candidate
    # If time runs out, we might not finish all, but each is fast.
    for c_init in initial_candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        c_res, val_res = local_search(c_init)
        
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_c_overall = c_res
            
    # If we have time, do more random restarts
    # Each local search is very fast (ms). We can do hundreds.
    while time.time() - start_time < TIME_LIMIT:
        c_init = np.random.choice([-1, 1], size=N_COEFFS)
        c_res, val_res = local_search(c_init)
        
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_c_overall = c_res
            
    # Final check: ensure best_c_overall is not None
    if best_c_overall is None:
        # Fallback to GR
        return get_gr_sequence(N_COEFFS).tolist()
        
    return best_c_overall.tolist()

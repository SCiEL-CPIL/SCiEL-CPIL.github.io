import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    def compute_radius(c):
        """Compute the maximum radius for given centers."""
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        min_b = np.min(np.minimum(bx, by))
        return min(min_d / 2, min_b)
    
    def energy(c, r):
        """
        Energy function: sum of squared violations.
        Returns (energy, min_violation) where min_violation < 0 means feasible.
        """
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Pairwise violations: want d_ij >= 2r
        viol_d = np.maximum(0, 2 * r - d)
        E_d = np.sum(viol_d ** 2)
        
        # Boundary violations: want dist_to_wall >= r
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        b = np.minimum(bx, by)
        viol_b = np.maximum(0, r - b)
        E_b = np.sum(viol_b ** 2)
        
        E = E_d + E_b
        min_viol = np.max(np.concatenate([viol_d.ravel(), viol_b.ravel()]))
        
        return E, min_viol
    
    def anneal(r, n_iter=50000, T0=0.05, Tend=0.001, seed=42):
        """Simulated annealing to find valid packing for radius r."""
        rng = np.random.RandomState(seed)
        
        # Initialize with jittered grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N].copy()
        c += rng.normal(0, 0.005, c.shape)
        c = np.clip(c, -0.5 + 0.01, 0.5 - 0.01)
        
        best_c = c.copy()
        best_r_found = compute_radius(c)
        best_E, _ = energy(c, r)
        
        # Annealing schedule
        for i in range(n_iter):
            T = T0 * (Tend / T0) ** (i / n_iter)
            
            # Generate move: pick random circle, perturb
            idx = rng.randint(N)
            step = 0.01 * T / T0 + 0.001
            delta = rng.normal(0, step, 2)
            c_new = c.copy()
            c_new[idx] += delta
            c_new = np.clip(c_new, -0.5, 0.5)
            
            E_new, _ = energy(c_new, r)
            
            # Accept if better, or with probability exp(-dE/T)
            if E_new < best_E:
                accept = True
            else:
                dE = E_new - best_E
                accept = rng.random() < np.exp(-dE / (T + 1e-10))
            
            if accept:
                c = c_new
                if E_new < best_E:
                    best_E = E_new
                    best_c = c.copy()
                    r_found = compute_radius(c)
                    if r_found > best_r_found:
                        best_r_found = r_found
            
            # Time check
            if time.time() - start_time > timeout - 5:
                break
        
        return best_E, best_c, best_r_found
    
    # Binary search for maximum feasible radius
    r_low = 0.06
    r_high = 0.075
    
    best_c = None
    best_r = 0.0
    
    # First, find a good packing at moderate radius
    for r_try in [0.065, 0.068, 0.070, 0.071, 0.0713]:
        if time.time() - start_time > timeout - 10:
            break
            
        E, c_found, r_found = anneal(r_try)
        
        if E < 1e-6 and r_found > best_r:
            best_r = r_found
            best_c = c_found
            
            # Try to improve further
            if r_found >= r_try - 1e-4:
                # Try slightly larger radius
                r_try2 = r_try + 0.0005
                E2, c2, r2 = anneal(r_try2)
                if E2 < 1e-6 and r2 > best_r:
                    best_r = r2
                    best_c = c2
    
    # Final refinement: try to push radius up from best found
    if best_c is not None:
        for _ in range(5):
            if time.time() - start_time > timeout - 5:
                break
            
            r_new = best_r + 0.0002
            E, c_new, r_found = anneal(r_new)
            if E < 1e-6 and r_found > best_r:
                best_r = r_found
                best_c = c_new
            else:
                break
    
    # Fallback: use grid if nothing found
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_r = compute_radius(best_c)
    
    return best_c

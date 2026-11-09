import numpy as np
import time
import sys
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P0 = np.ones((n, n)) / (n * n)
    
    start_time = time.time()
    deadline = start_time + 580
    
    # Pre-compute projection indices for each slope
    xs = np.arange(n)
    ys = np.arange(n)
    X_grid, Y_grid = np.meshgrid(xs, ys, indexing='xy')
    
    proj_info = {}
    for r in [-1, 0, 1, 2]:
        vals = X_grid + r * Y_grid
        min_val = int(np.min(vals))
        max_val = int(np.max(vals))
        n_unique = max_val - min_val + 1
        indices = (vals - min_val).ravel()
        proj_info[r] = (indices, n_unique)
    
    def get_projected_entropies(P):
        P_flat = P.ravel()
        entropies = {}
        for r in [-1, 0, 1, 2]:
            indices, n_unique = proj_info[r]
            counts = np.bincount(indices, weights=P_flat, minlength=n_unique)
            probs = counts[counts > 1e-15]
            if len(probs) == 0:
                entropies[r] = 0.0
            else:
                probs = probs / probs.sum()
                entropies[r] = -np.sum(probs * np.log2(probs))
        return entropies

    def calculate_score(P):
        ents = get_projected_entropies(P)
        D = ents[-1]
        M = max(ents[0], ents[1], ents[2])
        if M < 1e-9:
            return -1.0
        return D / M

    best_score = calculate_score(P0)
    best_P = P0.copy()
    best_Z = np.log(P0 + 1e-15)
    
    def softmax(Z_flat):
        Z = Z_flat.reshape(n, n)
        Z_max = np.max(Z)
        exp_Z = np.exp(Z - Z_max)
        P = exp_Z / np.sum(exp_Z)
        return P

    def smooth_max(H0, H1, H2, tau):
        arr = np.array([H0, H1, H2])
        m = np.max(arr)
        if m == 0:
            return 0.0
        scaled = arr / tau
        sm = np.max(scaled)
        return tau * (sm + np.log(np.sum(np.exp(scaled - sm))))

    def objective_smooth(Z_flat, tau):
        P = softmax(Z_flat)
        ents = get_projected_entropies(P)
        D = ents[-1]
        M_smooth = smooth_max(ents[0], ents[1], ents[2], tau)
        if M_smooth < 1e-9:
            return 0.0
        return -D / M_smooth

    # Phase 1: Random restarts with moderate tau
    tau_phase1 = 0.3
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        Z_init = np.random.normal(0, 1, (n, n)).flatten()
        
        try:
            res = minimize(lambda z: objective_smooth(z, tau_phase1), Z_init, 
                          method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-9})
            
            P_opt = softmax(res.x)
            score_opt = calculate_score(P_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
                best_Z = res.x.copy()
                
        except Exception:
            continue
    
    # Phase 2: Polish with tighter tau from best starting point
    tau_phase2 = 0.1
    if time.time() < deadline - 30:
        try:
            res = minimize(lambda z: objective_smooth(z, tau_phase2), best_Z, 
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            P_opt = softmax(res.x)
            score_opt = calculate_score(P_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
                best_Z = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Very tight polish
    tau_phase3 = 0.05
    if time.time() < deadline - 15:
        try:
            res = minimize(lambda z: objective_smooth(z, tau_phase3), best_Z, 
                          method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-11})
            P_opt = softmax(res.x)
            score_opt = calculate_score(P_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
                best_Z = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Discrete support refinement via greedy submodular heuristic
    # Try to improve by adjusting which cells have significant mass
    if time.time() < deadline - 10:
        # Start from current best
        P_work = best_P.copy()
        
        # Identify cells with very small mass and try zeroing them out,
        # then redistribute mass to improve D/M ratio
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Randomly perturb the support: zero out some small cells
            P_trial = P_work.copy()
            flat = P_trial.ravel()
            
            # Find cells with mass below threshold
            threshold = np.percentile(flat[flat > 0], 20) if np.any(flat > 0) else 0
            small_cells = np.where((flat < threshold) & (flat > 0))[0]
            
            if len(small_cells) == 0:
                continue
            
            # Randomly zero out some of the small cells
            num_zero = max(1, len(small_cells) // 2)
            to_zero = np.random.choice(small_cells, size=min(num_zero, len(small_cells)), replace=False)
            flat[to_zero] = 0
            
            # Normalize
            s = flat.sum()
            if s > 1e-9:
                flat /= s
                P_trial = flat.reshape(n, n)
                score_trial = calculate_score(P_trial)
                
                if score_trial > best_score:
                    best_score = score_trial
                    best_P = P_trial.copy()
                    best_Z = np.log(P_trial + 1e-15)
            
            # Also try adding mass back to zeroed cells
            if np.any(flat == 0):
                zero_cells = np.where(flat == 0)[0]
                if len(zero_cells) > 0:
                    # Add small mass to some zero cells
                    num_add = min(5, len(zero_cells))
                    to_add = np.random.choice(zero_cells, size=num_add, replace=False)
                    flat2 = flat.copy()
                    flat2[to_add] = 1e-4
                    s2 = flat2.sum()
                    if s2 > 1e-9:
                        flat2 /= s2
                        P_trial2 = flat2.reshape(n, n)
                        score_trial2 = calculate_score(P_trial2)
                        
                        if score_trial2 > best_score:
                            best_score = score_trial2
                            best_P = P_trial2.copy()
                            best_Z = np.log(P_trial2 + 1e-15)
    
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        return P0
    best_P /= s
    
    return best_P

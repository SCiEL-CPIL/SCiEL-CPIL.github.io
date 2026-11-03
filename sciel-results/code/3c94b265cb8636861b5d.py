import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))
    
    def negative_score(params, n):
        """Objective function for optimization: minimize -score"""
        h = np.zeros(n)
        n_atoms = len(params) // 3
        for i in range(n_atoms):
            center = params[i*3]
            width = params[i*3+1]
            amp = params[i*3+2]
            if width <= 0 or amp <= 0:
                continue
            # Create rectangular pulse
            start = int(np.floor(center - width/2))
            end = int(np.ceil(center + width/2))
            start = max(0, start)
            end = min(n, end)
            if start < end:
                h[start:end] += amp
        # Ensure non-negative
        h = np.maximum(h, 0)
        return -calc_score(h)
    
    # Try multiple random initializations of atom configurations
    np.random.seed(42)
    for trial in range(200):
        # Random number of atoms (2 to 8)
        n_atoms = np.random.randint(2, 9)
        
        # Random initial parameters
        params = []
        for i in range(n_atoms):
            center = np.random.uniform(0, n)
            width = np.random.uniform(1, 5)
            amp = np.random.uniform(0.1, 1.0)
            params.extend([center, width, amp])
        
        # Define bounds: center in [0,n], width in [0.5, n], amp in [0, 10]
        bounds = []
        for i in range(n_atoms):
            bounds.append((0, n))      # center
            bounds.append((0.5, n))    # width
            bounds.append((0, 10))     # amp
        
        # Optimize
        try:
            result = minimize(negative_score, params, args=(n,), 
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-6})
            
            # Extract optimized h
            h_opt = np.zeros(n)
            for i in range(n_atoms):
                center = result.x[i*3]
                width = result.x[i*3+1]
                amp = result.x[i*3+2]
                if width <= 0 or amp <= 0:
                    continue
                start = int(np.floor(center - width/2))
                end = int(np.ceil(center + width/2))
                start = max(0, start)
                end = min(n, end)
                if start < end:
                    h_opt[start:end] += amp
            h_opt = np.maximum(h_opt, 0)
            
            score = calc_score(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Local refinement: coordinate descent on the best solution
    if best_score > 0:
        h = best_h.copy()
        for _iter in range(50):
            improved = False
            for i in range(n):
                if h[i] == 0:
                    continue
                # Try small perturbations
                for delta in [-0.01, 0.01, -0.005, 0.005]:
                    h_new = h.copy()
                    h_new[i] += delta
                    if h_new[i] < 0:
                        continue
                    # Renormalize to keep L1 = 1
                    if np.sum(h_new) > 0:
                        h_new = h_new / np.sum(h_new)
                    score_new = calc_score(h_new)
                    if score_new > best_score + 1e-8:
                        best_score = score_new
                        best_h = h_new.copy()
                        h = h_new.copy()
                        improved = True
                        break
            if not improved:
                break
    
    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)

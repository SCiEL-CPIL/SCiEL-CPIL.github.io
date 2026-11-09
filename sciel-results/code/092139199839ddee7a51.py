"""Improved broadband AR coating design using simulated annealing + local polish."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating using simulated annealing
    to escape local minima, followed by local polishing."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 10.0  # 10s safety margin
    
    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    n_L = materials["L"]
    n_H = materials["H"]
    band = problem.get("band_nm", [400, 800])
    lam_min, lam_max = band[0], band[1]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)
    
    def compute_reflectance(labels, thicknesses_nm):
        """Exact characteristic-matrix method for normal incidence, lossless non-dispersive.
        Returns mean reflectance in percent over the band."""
        n_layers = len(labels)
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d < 1e-9:
                continue
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            i_c = 1j
            M00 = c
            M01 = i_c / (n_j * s)
            M10 = i_c * n_j * s
            M11 = c
            new00 = M[0,0]*M00 + M[0,1]*M10
            new01 = M[0,0]*M01 + M[0,1]*M11
            new10 = M[1,0]*M00 + M[1,1]*M10
            new11 = M[1,0]*M01 + M[1,1]*M11
            M = np.array([[new00, new01],[new10, new11]])
        Y_sub = substrate
        Y = (M[0,0] + M[0,1]*Y_sub) / (M[1,0] + M[1,1]*Y_sub)
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)
    
    def make_symmetric_labels(n_layers):
        """Build a symmetric label sequence starting with L on air side."""
        if n_layers % 2 != 0:
            n_layers -= 1
        if n_layers < 2:
            return ["L"]
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full
    
    def project(x, labels):
        """Project x onto feasible region: 0 <= x_i <= max_layer, sum <= max_total."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return x
    
    def objective(x, labels):
        x = project(x, labels)
        return compute_reflectance(labels, list(x))
    
    def simulated_annealing(labels, x0, n_iter=5000, T0=5.0, T1=0.01, alpha=0.9995):
        """Simulated annealing to find good thickness configuration."""
        n_var = len(labels)
        best_x = x0.copy()
        best_val = objective(best_x, labels)
        curr_x = x0.copy()
        curr_val = best_val
        T = T0
        
        for it in range(n_iter):
            if time.time() > DEADLINE - 5:
                break
            # Generate perturbation
            idx = random.randint(0, n_var - 1)
            step_size = max_layer * 0.1 * (0.5 + 0.5 * np.exp(-it / (n_iter * 0.3)))
            perturbation = random.gauss(0, step_size)
            new_x = curr_x.copy()
            new_x[idx] += perturbation
            new_x = project(new_x, labels)
            
            new_val = objective(new_x, labels)
            delta = new_val - curr_val
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                curr_x = new_x
                curr_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
            
            T *= alpha
        
        return best_x, best_val
    
    def local_polish(x, labels, max_iter=200):
        """Local polish using Nelder-Mead."""
        from scipy.optimize import minimize
        best_x = x.copy()
        best_val = objective(best_x, labels)
        
        bounds = [(0.0, max_layer)] * len(labels)
        cons = [{'type': 'ineq', 'fun': lambda x: max_total - np.sum(np.clip(x, 0, max_layer))}]
        
        try:
            res = minimize(lambda x: objective(x, labels), best_x, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_x = project(res.x, labels)
                best_val = objective(best_x, labels)
        except Exception:
            pass
        
        if time.time() < DEADLINE - 20:
            try:
                res2 = minimize(lambda x: objective(x, labels), best_x, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12})
                if res2.fun < best_val:
                    best_x = project(res2.x, labels)
                    best_val = objective(best_x, labels)
            except Exception:
                pass
        
        return best_x, best_val
    
    # Try several layer counts with SA + polish
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    center = 0.5 * (lam_min + lam_max)
    
    for n_layers in [12, 10, 14, 8, 16, 6]:
        if time.time() >= DEADLINE:
            break
        
        labels = make_symmetric_labels(n_layers)
        if len(labels) > max_layers:
            continue
        
        n_var = len(labels)
        x0 = np.zeros(n_var)
        for i, lab in enumerate(labels):
            n_i = materials[lab]
            x0[i] = center / (4.0 * n_i)
        x0 = project(x0, labels)
        
        # Phase 1: Simulated annealing
        sa_x, sa_val = simulated_annealing(labels, x0, n_iter=3000)
        
        # Phase 2: Local polish
        pol_x, pol_val = local_polish(sa_x, labels)
        
        # Validate
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in pol_x):
            continue
        if sum(pol_x) > max_total + 1e-6:
            pol_x = pol_x * (max_total / sum(pol_x))
        
        val = compute_reflectance(labels, list(pol_x))
        if val < best_score:
            best_score = val
            best_labels = labels
            best_thick = list(pol_x)
    
    # Fallback
    if best_labels is None:
        center = 0.5 * (lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END

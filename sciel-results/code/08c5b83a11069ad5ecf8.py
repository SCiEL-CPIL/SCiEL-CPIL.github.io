# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    directly against the exact characteristic-matrix reflectance objective."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 5.0  # leave 5s safety margin
    
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
        if n_layers == 0:
            # Bare substrate reflectance
            r0 = (ambient - substrate) / (ambient + substrate)
            return float(np.abs(r0)**2 * 100.0)
            
        # Pre-compute delta for each layer and wavelength
        # delta[i, j] = 2*pi*n_i*d_i/lambda_j
        # We can compute this on the fly to save memory if n_layers is large, but 20*401 is small.
        
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d < 1e-9:
                continue
            # delta = 2*pi*n*d/lambda
            delta = 2.0 * np.pi * n_j * d / wavelengths  # shape (n_wl,)
            c = np.cos(delta)          # real-ish
            s = np.sin(delta)
            i_c = 1j
            # per-wavelength 2x2 matrices:
            # M_j[0,0] = c
            # M_j[0,1] = i_c / (n_j * s)
            # M_j[1,0] = i_c * n_j * s
            # M_j[1,1] = c
            M00 = c
            M01 = i_c / (n_j * s)
            M10 = i_c * n_j * s
            M11 = c
            # Multiply M = M * M_j  (vector as column, matrices act on left)
            new00 = M[0,0]*M00 + M[0,1]*M10
            new01 = M[0,0]*M01 + M[0,1]*M11
            new10 = M[1,0]*M00 + M[1,1]*M10
            new11 = M[1,0]*M01 + M[1,1]*M11
            M = np.array([[new00, new01],[new10, new11]])
        
        # Admittance: Y = (m11 + m12*Y_sub)/(m21 + m22*Y_sub) with Y_sub = n_sub (normal incidence, lossless)
        Y_sub = substrate
        Y = (M[0,0] + M[0,1]*Y_sub) / (M[1,0] + M[1,1]*Y_sub)
        # Reflectance: R = |(Y - n_amb)/(Y + n_amb)|^2
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)  # percent
    
    def make_symmetric_labels(n_layers):
        """Build a symmetric label sequence: e.g. L H H L L H H L ... Actually standard: 
        Use a half-stack and mirror. For even n, e.g. n=8: half = 4 labels, full = half + reverse(half)
        We'll use half pattern starting with L on air side."""
        if n_layers % 2 != 0:
            n_layers -= 1
        half = n_layers // 2
        # Simple half pattern: alternating L,H starting with L
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full

    def optimize_stack(n_layers, n_restarts=5):
        labels = make_symmetric_labels(n_layers)
        if len(labels) == 0:
            return None
        n_var = len(labels)
        
        # Initial guess: quarter-wave at band center
        center = 0.5*(lam_min + lam_max)
        
        def objective(x):
            x = np.clip(x, 0.0, max_layer)
            # Enforce total via projection if needed (soft)
            s = np.sum(x)
            if s > max_total:
                x = x * (max_total / s)
                x = np.clip(x, 0.0, max_layer)
            return compute_reflectance(labels, list(x))
        
        best_x = None
        best_val = float('inf')
        
        # Use Simulated Annealing for global search
        # Then polish with L-BFGS-B
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            # Random initialization
            if restart == 0:
                x0 = np.zeros(n_var)
                for i, lab in enumerate(labels):
                    n_i = materials[lab]
                    x0[i] = center / (4.0 * n_i)
            else:
                x0 = np.random.uniform(0, max_layer, n_var)
            
            x0 = np.clip(x0, 0.0, max_layer)
            s0 = np.sum(x0)
            if s0 > max_total:
                x0 = x0 * (max_total / s0)
                x0 = np.clip(x0, 0.0, max_layer)
            
            # Simulated Annealing
            T = 50.0  # Initial temperature (in percent reflectance units)
            alpha = 0.995  # Cooling rate
            x_curr = x0.copy()
            curr_val = objective(x_curr)
            
            # If we start with a very bad solution, T might need to be higher
            # But 50% is huge, so it's fine.
            
            for step in range(500):  # Fixed number of steps, or time-based
                if time.time() > DEADLINE - 5:
                    break
                
                # Generate neighbor
                x_new = x_curr.copy()
                # Perturb a few random layers
                n_perturb = np.random.randint(1, 4)
                idx = np.random.choice(n_var, n_perturb, replace=False)
                # Perturb amount proportional to T? Or fixed?
                # Let's use a fixed step size that shrinks slightly
                step_size = 20.0 * (T / 50.0) + 1.0
                x_new[idx] += np.random.uniform(-step_size, step_size, n_perturb)
                
                x_new = np.clip(x_new, 0.0, max_layer)
                s_new = np.sum(x_new)
                if s_new > max_total:
                    x_new = x_new * (max_total / s_new)
                    x_new = np.clip(x_new, 0.0, max_layer)
                
                new_val = objective(x_new)
                delta_val = new_val - curr_val
                
                if delta_val < 0 or np.random.random() < np.exp(-delta_val / max(T, 1e-9)):
                    x_curr = x_new
                    curr_val = new_val
                    
                T *= alpha
                
            # Polish with L-BFGS-B
            from scipy.optimize import minimize
            try:
                bounds = [(0.0, max_layer)] * n_var
                cons = [{'type': 'ineq', 'fun': lambda x: max_total - np.sum(np.clip(x,0,max_layer))}]
                res = minimize(objective, x_curr, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.success and res.fun < curr_val:
                    x_final = np.clip(res.x, 0.0, max_layer)
                    s_final = np.sum(x_final)
                    if s_final > max_total:
                        x_final = x_final * (max_total / s_final)
                        x_final = np.clip(x_final, 0.0, max_layer)
                    final_val = objective(x_final)
                    if final_val < best_val:
                        best_val = final_val
                        best_x = x_final.copy()
                else:
                    if curr_val < best_val:
                        best_val = curr_val
                        best_x = x_curr.copy()
            except Exception:
                if curr_val < best_val:
                    best_val = curr_val
                    best_x = x_curr.copy()
            
            # Update global best
            if best_x is not None and best_val < curr_val:
                 pass # already handled in loop? No, best_val is local to restart loop?
                 # Wait, best_val is defined outside restart loop.
                 # The logic above updates best_val inside the restart loop.
        
        if best_x is None:
            return None
        
        return labels, list(best_x), best_val
    
    # Try several layer counts
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Prioritize higher layer counts for better broadband performance
    for n_layers in [14, 16, 12, 18, 10, 8]:
        if time.time() >= DEADLINE - 5:
            break
        res = optimize_stack(n_layers, n_restarts=3)
        if res is None:
            continue
        labels, th, val = res
        # Validate
        if len(labels) > max_layers:
            continue
        if any(t < 0 or t > max_layer for t in th):
            continue
        if sum(th) > max_total:
            # Scale down
            scale = max_total / sum(th)
            th = [t*scale for t in th]
            if any(t > max_layer for t in th):
                continue
        # Recompute to be safe
        val2 = compute_reflectance(labels, th)
        if val2 < best_score:
            best_score = val2
            best_labels = labels
            best_thick = th
    
    # Fallback
    if best_labels is None:
        center = 0.5*(lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center/(4.0*n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END

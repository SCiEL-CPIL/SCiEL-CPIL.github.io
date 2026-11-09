"""Antireflection coating design using differential evolution with structural priors."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating by optimizing layer thicknesses
    directly against the exact characteristic-matrix reflectance objective."""
    
    t_start = time.time()
    DEADLINE = 150.0 - 10.0  # leave 10s safety margin
    
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
            # Bare substrate
            r = (substrate - ambient) / (substrate + ambient)
            return float(np.abs(r)**2 * 100.0)
        
        M = np.eye(2, dtype=complex)
        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses_nm[i]
            if d < 1e-10:
                continue
            delta = 2.0 * np.pi * n_j * d / wavelengths
            c = np.cos(delta)
            s = np.sin(delta)
            # Handle near-zero sin to avoid division by zero
            mask = np.abs(s) > 1e-12
            M00 = c
            M01 = np.zeros_like(c, dtype=complex)
            M10 = np.zeros_like(c, dtype=complex)
            M11 = c
            M01[mask] = 1j / (n_j * s[mask])
            M10[mask] = 1j * n_j * s[mask]
            # Where sin is near zero, the layer acts as identity in some sense
            # Actually if d=0 we skip, but if sin=0 with d>0, the matrix is [[1,0],[0,1]] effectively
            # Let's just set M01 and M10 to 0 where sin is tiny (which is correct limit)
            
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
    
    def make_labels_pattern(n_layers):
        """Create a label pattern: start with L, alternate, ensure symmetry if possible.
        For broadband AR, a common pattern is symmetric: L H H L ... L H H L
        Or simply alternating. Let's use a symmetric alternating pattern."""
        if n_layers <= 0:
            return []
        if n_layers % 2 == 0:
            half = n_layers // 2
            half_labels = []
            for i in range(half):
                half_labels.append("L" if i % 2 == 0 else "H")
            return half_labels + list(reversed(half_labels))
        else:
            # Odd: center is L, then alternate
            labels = []
            # Build from center out
            center_idx = n_layers // 2
            labels = [""] * n_layers
            labels[center_idx] = "L"
            for i in range(1, center_idx + 1):
                # Alternate: if distance is odd, H; if even, L? 
                # Let's just do: center L, next H, next L, etc.
                if i % 2 == 1:
                    labels[center_idx - i] = "H"
                    labels[center_idx + i] = "H"
                else:
                    labels[center_idx - i] = "L"
                    labels[center_idx + i] = "L"
            return labels
    
    def objective(x, labels):
        """Objective function: mean reflectance. x is thicknesses."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        # If still over due to clipping, scale again
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
        return compute_reflectance(labels, list(x))
    
    # We will try several layer counts and label patterns
    # Use Differential Evolution for global search
    from scipy.optimize import differential_evolution, minimize
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    # Layer counts to try: even counts are symmetric
    n_layers_list = [8, 10, 12, 14, 16, 18, 20]
    
    for n_layers in n_layers_list:
        if time.time() >= DEADLINE:
            break
        
        labels = make_labels_pattern(n_layers)
        n_var = len(labels)
        
        # Initial bounds
        bounds = [(0.0, max_layer)] * n_var
        
        # Strategy: Parameterize with a base thickness and gradient
        # But DE works on the raw vector. Let's just use DE on the raw vector.
        # To help convergence, we can use a seed with quarter-wave values.
        center_wl = 0.5 * (lam_min + lam_max)
        x0 = []
        for i, lab in enumerate(labels):
            n_i = materials[lab]
            x0.append(center_wl / (4.0 * n_i))
        x0 = np.clip(x0, 0.0, max_layer)
        s0 = np.sum(x0)
        if s0 > max_total:
            x0 = x0 * (max_total / s0)
            x0 = np.clip(x0, 0.0, max_layer)
        
        def obj_func(x):
            return objective(x, labels)
        
        # Run Differential Evolution
        # maxiter and popsize control the search
        # We have ~150s total. Let's allocate time per n_layers.
        # n_layers_list has 7 items. ~20s each?
        # DE can be slow. Let's use a smaller population and fewer iterations.
        # popsize=10, maxiter=100 might be reasonable.
        
        try:
            res = differential_evolution(
                obj_func,
                bounds,
                seed=42,
                maxiter=100,
                popsize=10,
                tol=1e-8,
                polish=True,  # Use L-BFGS-B at the end
                init='sobol',
                mutation=(0.3, 0.9),
                recombination=0.8
            )
            
            if res.success or res.fun < best_score:
                best_x = np.clip(res.x, 0.0, max_layer)
                s = np.sum(best_x)
                if s > max_total:
                    best_x = best_x * (max_total / s)
                    best_x = np.clip(best_x, 0.0, max_layer)
                val = objective(best_x, labels)
                if val < best_score:
                    best_score = val
                    best_labels = labels
                    best_thick = list(best_x)
        except Exception:
            pass
        
        # Also try a local optimization from the quarter-wave guess
        if time.time() < DEADLINE - 5:
            try:
                res2 = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 100, 'ftol': 1e-12})
                if res2.fun < best_score:
                    best_x = np.clip(res2.x, 0.0, max_layer)
                    s = np.sum(best_x)
                    if s > max_total:
                        best_x = best_x * (max_total / s)
                        best_x = np.clip(best_x, 0.0, max_layer)
                    val = objective(best_x, labels)
                    if val < best_score:
                        best_score = val
                        best_labels = labels
                        best_thick = list(best_x)
            except Exception:
                pass
    
    # Fallback
    if best_labels is None:
        center = 0.5*(lam_min + lam_max)
        best_labels = ["L"]
        best_thick = [center/(4.0*n_L)]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END

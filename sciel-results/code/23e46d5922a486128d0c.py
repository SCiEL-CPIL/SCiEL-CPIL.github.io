import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy:
    1. Exploit the fact that the union area at height c is Lx(c) * Ly(c).
    2. Minimize the exact volume integral int_0^1 Lx(c) Ly(c) dc.
    3. Optimize x and y independently (14 variables, fixing x0=0, y0=0 for translation invariance).
    4. Use Differential Evolution for global search, followed by Nelder-Mead for local refinement.
    """
    N = 8
    deadline = time.time() + 580.0
    
    # Precompute shear factors
    i_vals = np.arange(N, dtype=np.float64)
    shear = i_vals / N
    width = 1.0 / N

    def calc_union_length(x, c):
        """
        Calculate the length of the union of intervals [x_i + c*shear_i, x_i + c*shear_i + width].
        x: array of shape (N,)
        c: scalar
        """
        L = x + c * shear
        R = L + width
        order = np.argsort(L)
        L_s = L[order]
        R_s = R[order]
        
        total_len = 0.0
        current_max_r = -np.inf
        for i in range(N):
            if L_s[i] > current_max_r:
                total_len += R_s[i] - L_s[i]
                current_max_r = R_s[i]
            else:
                if R_s[i] > current_max_r:
                    current_max_r = R_s[i]
        return total_len

    def get_breakpoints(x):
        """
        Find all c in [0, 1] where the ordering of interval endpoints changes.
        """
        # Endpoints:
        # Left: a = shear_i, b = x_i
        # Right: a = shear_i, b = x_i + width
        a_coeffs = np.concatenate([shear, shear])
        b_coeffs = np.concatenate([x, x + width])
        
        bps = set()
        bps.add(0.0)
        bps.add(1.0)
        
        for p in range(2 * N):
            for q in range(p + 1, 2 * N):
                da = a_coeffs[p] - a_coeffs[q]
                db = b_coeffs[p] - b_coeffs[q]
                if abs(da) < 1e-15:
                    continue
                c_val = -db / da
                if 0.0 <= c_val <= 1.0:
                    bps.add(c_val)
                    
        return np.sort(np.array(list(bps), dtype=np.float64))

    def exact_volume(params):
        """
        params: array of shape (14,)
        params[0:7] are x_1..x_7
        params[7:14] are y_1..y_7
        x_0 and y_0 are fixed to 0.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        x[1:] = params[0:7]
        y[1:] = params[7:14]
        
        bps_x = get_breakpoints(x)
        bps_y = get_breakpoints(y)
        
        # Merge breakpoints
        all_bps = np.unique(np.concatenate([bps_x, bps_y]))
        
        if len(all_bps) < 2:
            return 0.0
            
        total_vol = 0.0
        for i in range(len(all_bps) - 1):
            c0 = all_bps[i]
            c1 = all_bps[i+1]
            if c1 - c0 < 1e-15:
                continue
                
            # Evaluate Lx and Ly at endpoints
            Lx0 = calc_union_length(x, c0)
            Lx1 = calc_union_length(x, c1)
            Ly0 = calc_union_length(y, c0)
            Ly1 = calc_union_length(y, c1)
            
            # Slopes
            dx = (Lx1 - Lx0) / (c1 - c0)
            dy = (Ly1 - Ly0) / (c1 - c0)
            
            # Intercepts
            bx = Lx0 - dx * c0
            by = Ly0 - dy * c0
            
            # Integrate (dx*c + bx)(dy*c + by)
            term1 = dx * dy * (c1**3 - c0**3) / 3.0
            term2 = (dx * by + bx * dy) * (c1**2 - c0**2) / 2.0
            term3 = bx * by * (c1 - c0)
            
            total_vol += term1 + term2 + term3
            
        return total_vol

    # Initial guess: Symmetric configuration
    # x_i = -i/16, y_i = -i/16
    # Since x_0=0, x_i = i/16 - 0? No, relative positions.
    # Let's use x_i = i/16 - 0.5 * (N-1)/N? 
    # Actually, just use small random or linear.
    # Symmetric: x_i = -i / (2*N)
    # If x_0=0, then x_i = -i/(2*N) + 0? No, x_0 is 0.
    # x_i = -i / (2*N) implies x_0=0.
    init_x = -np.arange(1, N) / (2.0 * N)
    init_y = -np.arange(1, N) / (2.0 * N)
    init_params = np.concatenate([init_x, init_y])
    
    # Bounds for optimization
    # The relative positions don't need to be huge. [-1, 1] is safe.
    bounds = [(-1.0, 1.0)] * 14

    best_params = init_params
    best_val = exact_volume(init_params)
    
    # Phase 1: Differential Evolution
    if time.time() < deadline - 100:
        try:
            res_de = differential_evolution(
                exact_volume, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-5, 
                seed=42,
                polish=False # We will polish later
            )
            if np.all(np.isfinite(res_de.x)) and res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x
        except Exception:
            pass

    # Phase 2: Local Refinement (Nelder-Mead)
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                exact_volume, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.all(np.isfinite(res_nm.x)) and res_nm.fun < best_val:
                best_val = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
            
    # Phase 3: Further Refinement (Powell)
    if time.time() < deadline - 5:
        try:
            res_pw = minimize(
                exact_volume, 
                best_params, 
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if np.all(np.isfinite(res_pw.x)) and res_pw.fun < best_val:
                best_val = res_pw.fun
                best_params = res_pw.x
        except Exception:
            pass

    # Construct the 64x2 array
    x_final = np.zeros(N)
    y_final = np.zeros(N)
    x_final[1:] = best_params[0:7]
    y_final[1:] = best_params[7:14]
    
    positions = np.zeros((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = x_final[i]
        positions[k, 1] = y_final[j]

    return positions

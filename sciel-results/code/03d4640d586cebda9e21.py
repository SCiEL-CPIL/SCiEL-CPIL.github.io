# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as so

def kakeya_needle_3d8() -> np.ndarray:
    """
    Solves the discrete 3D Kakeya needle problem for N=8 using exact analytical volume
    calculation and L-BFGS optimization.
    
    Returns a (64, 2) array of base positions (x_k, y_k).
    """
    N = 8
    delta = 1.0 / N
    num_tubes = N * N
    
    # Precompute indices for tubes (i, j)
    i_idx = np.array([k // N for k in range(num_tubes)], dtype=np.float64)
    j_idx = np.array([k % N for k in range(num_tubes)], dtype=np.float64)

    # Helper to compute the exact volume of the union of sheared tubes
    def compute_volume_exact(positions):
        """
        positions: (64, 2) array of (x_k, y_k)
        """
        # Extract base positions for unique i and j
        # Tube k has i = k//N, j = k%N.
        # x_k depends only on i (for the X-dimension shear and base).
        # y_k depends only on j (for the Y-dimension shear and base).
        # However, the problem defines x_k and y_k independently for each tube k.
        # But the shear is c*i/N for x and c*j/N for y.
        # The union in X depends on the set of intervals { [x_k + c*i_k/N, x_k + delta + c*i_k/N] }
        # The union in Y depends on the set of intervals { [y_k + c*j_k/N, y_k + delta + c*j_k/N] }
        
        # Note: Multiple tubes can share the same i. If they have different x_k, 
        # they contribute multiple intervals to the X union.
        # Similarly for Y.
        
        # Construct endpoints for X dimension
        # For each tube k, we have an interval [Lx_k(c), Rx_k(c)]
        # Lx_k(c) = x_k + c * i_k / N
        # Rx_k(c) = x_k + delta + c * i_k / N
        # We represent each endpoint as (A, B) where val = A + B*c.
        # For X: A = x_k, B = i_k / N
        # For Y: A = y_k, B = j_k / N
        
        eps_x = []
        eps_y = []
        
        for k in range(num_tubes):
            xk = positions[k, 0]
            yk = positions[k, 1]
            ik = i_idx[k]
            jk = j_idx[k]
            
            # X endpoints: Left (type 0), Right (type 1)
            eps_x.append((xk, ik / N, 0))
            eps_x.append((xk + delta, ik / N, 1))
            
            # Y endpoints: Left (type 0), Right (type 1)
            eps_y.append((yk, jk / N, 0))
            eps_y.append((yk + delta, jk / N, 1))

        # Find critical points where the relative order of any two endpoints changes
        # val1(c) = A1 + B1*c, val2(c) = A2 + B2*c
        # val1(c) = val2(c) => c = (A2 - A1) / (B1 - B2)
        
        crit_x = {0.0, 1.0}
        for i in range(len(eps_x)):
            A1, B1, _ = eps_x[i]
            for j in range(i + 1, len(eps_x)):
                A2, B2, _ = eps_x[j]
                if abs(B1 - B2) > 1e-12:
                    c = (A2 - A1) / (B1 - B2)
                    if 0.0 < c < 1.0:
                        crit_x.add(c)
        
        crit_y = {0.0, 1.0}
        for i in range(len(eps_y)):
            A1, B1, _ = eps_y[i]
            for j in range(i + 1, len(eps_y)):
                A2, B2, _ = eps_y[j]
                if abs(B1 - B2) > 1e-12:
                    c = (A2 - A1) / (B1 - B2)
                    if 0.0 < c < 1.0:
                        crit_y.add(c)
        
        all_crit = sorted(crit_x.union(crit_y))
        
        # Helper to calculate union length at a specific c
        def calc_union_length(eps, c):
            # Calculate values
            vals = []
            for A, B, typ in eps:
                vals.append((A + B * c, typ))
            vals.sort()
            
            curr_len = 0.0
            count = 0
            start_pos = 0.0
            total = 0.0
            for val, typ in vals:
                if typ == 0: # Left
                    if count == 0:
                        start_pos = val
                    count += 1
                else: # Right
                    count -= 1
                    if count == 0:
                        total += val - start_pos
            return total

        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_start = all_crit[k]
            c_end = all_crit[k + 1]
            h = c_end - c_start
            if h < 1e-12:
                continue
            
            Lx_s = calc_union_length(eps_x, c_start)
            Lx_e = calc_union_length(eps_x, c_end)
            Ly_s = calc_union_length(eps_y, c_start)
            Ly_e = calc_union_length(eps_y, c_end)
            
            # Lx(c) and Ly(c) are linear in c on this interval.
            # Lx(c) = Lx_s + (c - c_start) * (Lx_e - Lx_s) / h
            # Let u = (c - c_start) / h, so c = c_start + u*h
            # Lx = Lx_s + u * dx, where dx = Lx_e - Lx_s
            # Ly = Ly_s + u * dy, where dy = Ly_e - Ly_s
            # Product P(u) = (Lx_s + u*dx)(Ly_s + u*dy)
            # Integral of P(u) from 0 to 1 is:
            # Lx_s*Ly_s + 0.5*(Lx_s*dy + Ly_s*dx) + (1/3)*dx*dy
            # Multiply by h to get integral over c.
            
            dx = Lx_e - Lx_s
            dy = Ly_e - Ly_s
            
            term = Lx_s * Ly_s
            term += 0.5 * (Lx_s * dy + Ly_s * dx)
            term += (1.0 / 3.0) * dx * dy
            
            total_vol += h * term
            
        return total_vol

    # Initial guess from Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Construct initial positions
    # The Keich construction typically assigns x based on i and y based on j.
    # Here we have 64 tubes.
    # We can use the same base offset for all tubes with the same i, and same for j.
    # Or we can optimize all 128 variables.
    # Optimizing all 128 variables allows for more flexibility but is harder.
    # Let's try optimizing the 16 unique base positions (8 for X, 8 for Y) first.
    # If tube k has i, x_k = X_base[i]. If tube k has j, y_k = Y_base[j].
    # This reduces the dimension from 128 to 16.
    
    def objective(params):
        # params: (16,) array. First 8 are X bases, next 8 are Y bases.
        X_base = params[:N]
        Y_base = params[N:]
        
        # Expand to 64 tubes
        positions = np.empty((num_tubes, 2))
        for k in range(num_tubes):
            i = int(i_idx[k])
            j = int(j_idx[k])
            positions[k, 0] = X_base[i]
            positions[k, 1] = Y_base[j]
            
        return compute_volume_exact(positions)
    
    # Initial params
    initial_params = np.concatenate([keich, keich])
    
    start_time = time.time()
    deadline = start_time + 550.0
    
    best_vol = np.inf
    best_params = initial_params
    
    # Try multiple restarts to escape local minima
    rng = np.random.default_rng(42)
    
    # First run from Keich
    res = so.minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12})
    if res.fun < best_vol:
        best_vol = res.fun
        best_params = res.x
    if time.time() > deadline:
        break

    # Random restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        # Perturb the best solution or start random
        if restarts % 2 == 0:
            x0 = best_params + rng.normal(0, 0.1, size=16)
        else:
            # Random start in a reasonable range
            x0 = rng.uniform(-0.5, 1.5, size=16)
        
        res = so.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x
        
        restarts += 1
        
    # Final check: expand best_params to full 64x2 positions
    X_base = best_params[:N]
    Y_base = best_params[N:]
    
    final_positions = np.empty((num_tubes, 2))
    for k in range(num_tubes):
        i = int(i_idx[k])
        j = int(j_idx[k])
        final_positions[k, 0] = X_base[i]
        final_positions[k, 1] = Y_base[j]
        
    return final_positions

# EVOLVE-BLOCK-END

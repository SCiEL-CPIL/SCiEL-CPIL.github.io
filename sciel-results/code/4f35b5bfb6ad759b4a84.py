import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes to minimize the union volume.
    Uses multiple restarts with L-BFGS-B on the piecewise-linear objective.
    """
    N = 8
    w = 1.0 / N
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    def union_length(U, c):
        """
        Computes the length of the union of N intervals [U_i + c*i/N, U_i + c*i/N + w].
        """
        starts = U + c * np.arange(N) / N
        ends = starts + w
        
        # Sort intervals by start point
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Merge overlapping intervals
        length = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for i in range(1, N):
            if s[i] > cur_e:
                length += cur_e - cur_s
                cur_s = s[i]
                cur_e = e[i]
            else:
                if e[i] > cur_e:
                    cur_e = e[i]
        length += cur_e - cur_s
        return length

    def objective(U):
        """
        Minimizes the integral of L(c)^2 over c in [0, 1].
        This corresponds to minimizing the volume when U_x = U_y = U.
        """
        # Use a fine grid for integration
        c_vals = np.linspace(0, 1, 200)
        total = 0.0
        for c in c_vals:
            l = union_length(U, c)
            total += l * l
        return total / len(c_vals)

    bounds = [(-1.0, 1.0)] * N
    
    # Multiple restarts
    best_U = keich.copy()
    best_obj = objective(keich)
    
    rng = np.random.default_rng(42)
    for restart in range(20):
        if restart == 0:
            U_init = keich.copy()
        else:
            U_init = keich + rng.normal(0, 0.1, N)
        
        res = minimize(objective, U_init, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_U = res.x
    
    U_opt = best_U

    # Construct the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = U_opt[i_flat]
    base_y = U_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check: ensure finite
    if not np.all(np.isfinite(positions)):
        # Fallback to seed if optimization failed
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

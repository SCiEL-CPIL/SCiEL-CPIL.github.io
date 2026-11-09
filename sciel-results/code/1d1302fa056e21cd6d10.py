# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Minimizes union volume using exact 2D area computation with interval merging
    and Gauss-Legendre quadrature, optimized via Nelder-Mead.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill

    # --- Initial Keich construction ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x_base = keich.copy()
    y_base = keich.copy()

    # --- Exact 2D area of union of sheared rectangles in (coord, c) plane ---
    def area_2d(base, N):
        """
        Compute the area of the union of N sheared rectangles in the (x, c) plane.
        Rectangle i: at height c, x-range is [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
        This is a parallelogram with vertices:
          (base[i], 0), (base[i]+(i+1)/N, 0), (base[i]+(i+1)/N+i/N, 1), (base[i]+i/N, 1)
        
        We use Gauss-Legendre quadrature in c, and at each c compute the 1D union length.
        """
        # Use 64-point Gauss-Legendre quadrature for accuracy
        c_vals, w_vals = np.polynomial.legendre.leggauss(64)
        c_vals = 0.5 * (c_vals + 1.0)  # map from [-1,1] to [0,1]
        w_vals = 0.5 * w_vals  # scale weights
        
        total_area = 0.0
        for c, w in zip(c_vals, w_vals):
            # At height c, intervals are [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N]
            lefts = base + c * np.arange(N) / N
            rights = lefts + (np.arange(N) + 1) / N
            
            # Merge intervals to compute union length
            order = np.argsort(lefts)
            lefts_sorted = lefts[order]
            rights_sorted = rights[order]
            
            total_len = 0.0
            cur_l = lefts_sorted[0]
            cur_r = rights_sorted[0]
            for k in range(1, N):
                if lefts_sorted[k] <= cur_r:
                    cur_r = max(cur_r, rights_sorted[k])
                else:
                    total_len += cur_r - cur_l
                    cur_l = lefts_sorted[k]
                    cur_r = rights_sorted[k]
            total_len += cur_r - cur_l
            
            total_area += w * total_len
        
        return total_area

    # --- Objective: minimize Vx * Vy (equivalently log(Vx) + log(Vy) for numerical stability) ---
    def objective(params):
        x_base_opt = params[:N]
        y_base_opt = params[N:]
        vx = area_2d(x_base_opt, N)
        vy = area_2d(y_base_opt, N)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Optimize using Nelder-Mead ---
    initial_params = np.concatenate([x_base, y_base])
    best_params = initial_params.copy()
    best_obj = objective(best_params)

    # Run Nelder-Mead with time guard
    result = None
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': 20000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
                'initial_simplex': 0.1 * np.ones(2 * N)
            }
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass

    # --- Local refinement with coordinate descent ---
    if time.time() < deadline - 10.0:
        step = 1e-4
        for iteration in range(500):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params[idx]
                for delta in [step, -step, step/2, -step/2, step/4, -step/4]:
                    best_params[idx] = old_val + delta
                    obj = objective(best_params)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                        break
                    best_params[idx] = old_val
                if not improved:
                    best_params[idx] = old_val
            if not improved:
                step *= 0.7
                if step < 1e-7:
                    break

    # --- Extract final base positions ---
    x_final = best_params[:N]
    y_final = best_params[N:]

    # --- Build final positions array ---
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # Precompute constants
    idx_arr = np.arange(N, dtype=np.float64)
    widths = (idx_arr + 1.0) / N  # width of each tube: (i+1)/8

    def compute_area(base):
        """
        Exact area of union of N parallelograms in (x, c) plane.
        Parallelogram i: x from base[i] + c*i/N to base[i] + (i+1)/N + c*i/N, c in [0,1].
        
        Uses sweep-line: find critical c values where interval endpoints cross,
        then integrate union length piecewise linearly.
        """
        # Critical c values: where left_i(c) = left_j(c) or left_i(c) = right_j(c) or right_i(c) = right_j(c)
        # left_i(c) = base[i] + c*i/N, right_i(c) = left_i(c) + (i+1)/N
        
        # Collect all critical c values in [0,1]
        critical_c = {0.0, 1.0}
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # left_i = left_j: base[i] + c*i/N = base[j] + c*j/N
                # c*(i-j)/N = base[j] - base[i]
                denom = (i - j) / N
                if abs(denom) > 1e-15:
                    c_val = (base[j] - base[i]) / denom
                    if 0.0 <= c_val <= 1.0:
                        critical_c.add(c_val)
                # left_i = right_j: base[i] + c*i/N = base[j] + (j+1)/N + c*j/N
                # c*(i-j)/N = base[j] + (j+1)/N - base[i]
                if abs(denom) > 1e-15:
                    c_val = (base[j] + (j + 1) / N - base[i]) / denom
                    if 0.0 <= c_val <= 1.0:
                        critical_c.add(c_val)
                # right_i = left_j: base[i] + (i+1)/N + c*i/N = base[j] + c*j/N
                # c*(i-j)/N = base[j] - base[i] - (i+1)/N
                if abs(denom) > 1e-15:
                    c_val = (base[j] - base[i] - (i + 1) / N) / denom
                    if 0.0 <= c_val <= 1.0:
                        critical_c.add(c_val)
                # right_i = right_j: base[i] + (i+1)/N + c*i/N = base[j] + (j+1)/N + c*j/N
                # c*(i-j)/N = base[j] + (j+1)/N - base[i] - (i+1)/N
                if abs(denom) > 1e-15:
                    c_val = (base[j] + (j + 1) / N - base[i] - (i + 1) / N) / denom
                    if 0.0 <= c_val <= 1.0:
                        critical_c.add(c_val)
        
        c_sorted = np.array(sorted(critical_c))
        
        if len(c_sorted) < 2:
            # No crossings, just compute directly
            c_mid = 0.5
            lefts = base + c_mid * idx_arr / N
            rights = lefts + widths
            order = np.argsort(lefts)
            lefts_s = lefts[order]
            rights_s = rights[order]
            total_len = 0.0
            cur_l = lefts_s[0]
            cur_r = rights_s[0]
            for k in range(1, N):
                if lefts_s[k] <= cur_r:
                    cur_r = max(cur_r, rights_s[k])
                else:
                    total_len += cur_r - cur_l
                    cur_l = lefts_s[k]
                    cur_r = rights_s[k]
            total_len += cur_r - cur_l
            return total_len
        
        # Integrate piecewise
        area = 0.0
        for m in range(len(c_sorted) - 1):
            c1 = c_sorted[m]
            c2 = c_sorted[m + 1]
            if c2 - c1 < 1e-15:
                continue
            
            # Union length is piecewise linear in c within [c1, c2]
            # Evaluate at c1 and c2 (and midpoint for safety)
            c_mid = 0.5 * (c1 + c2)
            
            def union_len(c):
                lefts = base + c * idx_arr / N
                rights = lefts + widths
                order = np.argsort(lefts)
                lefts_s = lefts[order]
                rights_s = rights[order]
                total_len = 0.0
                cur_l = lefts_s[0]
                cur_r = rights_s[0]
                for k in range(1, N):
                    if lefts_s[k] <= cur_r:
                        cur_r = max(cur_r, rights_s[k])
                    else:
                        total_len += cur_r - cur_l
                        cur_l = lefts_s[k]
                        cur_r = rights_s[k]
                total_len += cur_r - cur_l
                return total_len
            
            l1 = union_len(c1)
            l2 = union_len(c2)
            lm = union_len(c_mid)
            
            # Trapezoidal rule (should be exact if linear, but use 3-point for safety)
            seg_area = (c2 - c1) / 6.0 * (l1 + 4 * lm + l2)
            area += seg_area
        
        return area

    def objective(params):
        x_base = params[:N]
        y_base = params[N:]
        ax = compute_area(x_base)
        ay = compute_area(y_base)
        if ax <= 0 or ay <= 0:
            return 1e10
        return ax * ay

    # --- Generate diverse initial points ---
    def keich_init():
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    def star_init(rng):
        """All tubes converge toward a common center region."""
        center = rng.uniform(-0.5, 0.5, size=2)
        x_base = np.zeros(N)
        y_base = np.zeros(N)
        for i in range(N):
            # Tube i at c=1 should be near center
            # x at c=1: x_base[i] + i/N
            # So x_base[i] = center_x - i/N + small jitter
            x_base[i] = center[0] - i / N + rng.normal(0, 0.05)
            y_base[i] = center[1] - i / N + rng.normal(0, 0.05)
        return x_base, y_base

    def random_init(rng):
        x_base = rng.uniform(-1, 1, size=N)
        y_base = rng.uniform(-1, 1, size=N)
        return x_base, y_base

    def keich_jittered_init(rng):
        keich = keich_init()
        x_base = keich + rng.normal(0, 0.1, size=N)
        y_base = keich + rng.normal(0, 0.1, size=N)
        return x_base, y_base

    rng = np.random.default_rng(42)

    # Collect initial points
    init_points = []
    
    # Keich
    kb = keich_init()
    init_points.append(np.concatenate([kb, kb]))
    
    # Star constructions
    for _ in range(3):
        xb, yb = star_init(rng)
        init_points.append(np.concatenate([xb, yb]))
    
    # Jittered Keich
    for _ in range(3):
        xb, yb = keich_jittered_init(rng)
        init_points.append(np.concatenate([xb, yb]))
    
    # Random
    for _ in range(2):
        xb, yb = random_init(rng)
        init_points.append(np.concatenate([xb, yb]))
    
    # --- Optimize each initial point ---
    best_params = init_points[0].copy()
    best_obj = objective(best_params)

    for init_idx, init_pt in enumerate(init_points):
        if time.time() > deadline - 30.0:
            break
        
        current_obj = objective(init_pt)
        if current_obj < best_obj:
            best_obj = current_obj
            best_params = init_pt.copy()
        
        # Try L-BFGS-B
        try:
            result = minimize(
                objective,
                init_pt,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead from best found so far
        if time.time() > deadline - 20.0:
            break
        try:
            nm_start = best_params if best_obj < current_obj else init_pt
            result = minimize(
                objective,
                nm_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-9,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.05 * np.ones(2 * N)
                }
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # --- Final local refinement with coordinate descent ---
    if time.time() < deadline - 15.0:
        step = 1e-4
        for iteration in range(300):
            if time.time() > deadline - 15.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params[idx]
                found_better = False
                for delta in [step, -step, step/2, -step/2, step/4, -step/4, step/8, -step/8]:
                    best_params[idx] = old_val + delta
                    obj = objective(best_params)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                        found_better = True
                        break
                if not found_better:
                    best_params[idx] = old_val
            if not improved:
                step *= 0.6
                if step < 1e-8:
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

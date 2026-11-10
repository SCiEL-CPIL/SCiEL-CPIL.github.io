import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    X0 = keich.copy()
    Y0 = keich.copy()

    # Precompute coefficients for endpoints
    # X endpoints: x_i + c * i/N and x_i + delta + c * i/N
    # A1_i = x_i, B1_i = i/N
    # A2_i = x_i + delta, B2_i = i/N
    
    def compute_volume(X, Y):
        # X, Y are arrays of length N
        # Build arrays of A and B for X and Y
        # X: 2N points
        # A_x = [x_0, x_0+d, x_1, x_1+d, ...]
        # B_x = [0, 0, 1/N, 1/N, ..., 7/N, 7/N]
        # is_right = [0, 1, 0, 1, ...]
        
        A_x = np.empty(2 * N)
        B_x = np.empty(2 * N)
        is_r_x = np.empty(2 * N, dtype=int)
        for i in range(N):
            A_x[2*i] = X[i]
            B_x[2*i] = i / N
            is_r_x[2*i] = 0
            A_x[2*i+1] = X[i] + delta
            B_x[2*i+1] = i / N
            is_r_x[2*i+1] = 1
            
        A_y = np.empty(2 * N)
        B_y = np.empty(2 * N)
        is_r_y = np.empty(2 * N, dtype=int)
        for j in range(N):
            A_y[2*j] = Y[j]
            B_y[2*j] = j / N
            is_r_y[2*j] = 0
            A_y[2*j+1] = Y[j] + delta
            B_y[2*j+1] = j / N
            is_r_y[2*j+1] = 1

        # Find critical points for X
        crit_x = {0.0, 1.0}
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_x[i] != B_x[j]:
                    c = (A_x[j] - A_x[i]) / (B_x[i] - B_x[j])
                    if 0.0 < c < 1.0:
                        crit_x.add(c)
                        
        # Find critical points for Y
        crit_y = {0.0, 1.0}
        for i in range(2 * N):
            for j in range(i + 1, 2 * N):
                if B_y[i] != B_y[j]:
                    c = (A_y[j] - A_y[i]) / (B_y[i] - B_y[j])
                    if 0.0 < c < 1.0:
                        crit_y.add(c)
                        
        all_crit = sorted(crit_x.union(crit_y))
        
        def calc_union_length(A, B, is_r, c):
            vals = A + B * c
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_is_r = is_r[order]
            
            count = 0
            start_pos = 0.0
            total = 0.0
            for k in range(len(sorted_vals)):
                if sorted_is_r[k] == 0:
                    if count == 0:
                        start_pos = sorted_vals[k]
                    count += 1
                else:
                    count -= 1
                    if count == 0:
                        total += sorted_vals[k] - start_pos
            return total

        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_start = all_crit[k]
            c_end = all_crit[k + 1]
            h = c_end - c_start
            if h < 1e-12:
                continue

            Lx_s = calc_union_length(A_x, B_x, is_r_x, c_start)
            Lx_e = calc_union_length(A_x, B_x, is_r_x, c_end)
            Ly_s = calc_union_length(A_y, B_y, is_r_y, c_start)
            Ly_e = calc_union_length(A_y, B_y, is_r_y, c_end)

            dx = Lx_e - Lx_s
            dy = Ly_e - Ly_s

            term = Lx_s * Ly_s
            term += 0.5 * (Lx_s * dy + Ly_s * dx)
            term += (1.0 / 3.0) * dx * dy

            total_vol += h * term

        return total_vol

    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)

    p0 = np.concatenate([X0, Y0])

    start_time = time.time()
    deadline = start_time + 550.0 # Safety margin

    best_p = p0.copy()
    best_v = objective(p0)

    # Use Differential Evolution for global search
    # Bounds: X and Y should be in [-1, 1] roughly
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res_de = so.differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42, polish=True, tol=1e-8)
        if time.time() < deadline:
            if res_de.fun < best_v:
                best_v = res_de.fun
                best_p = res_de.x.copy()
    except Exception:
        pass

    # Polish with Powell
    if time.time() < deadline - 60:
        try:
            res_pw = so.minimize(objective, best_p, method='Powell', options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res_pw.fun < best_v:
                best_v = res_pw.fun
                best_p = res_pw.x.copy()
        except Exception:
            pass

    X_opt = best_p[:N]
    Y_opt = best_p[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    return positions

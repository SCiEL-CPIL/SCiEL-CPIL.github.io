import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    deadline = time.time() + 580.0
    
    # Exact volume via critical-point integration of 1-D union lengths
    def compute_volume(X, Y):
        eps_x = [(X[i], i / N, 0) for i in range(N)] + [(X[i] + delta, i / N, 1) for i in range(N)]
        eps_y = [(Y[j], j / N, 0) for j in range(N)] + [(Y[j] + delta, j / N, 1) for j in range(N)]
        
        def find_crit(eps):
            crit = set()
            crit.add(0.0)
            crit.add(1.0)
            for i in range(len(eps)):
                A1, B1, _ = eps[i]
                for j in range(i + 1, len(eps)):
                    A2, B2, _ = eps[j]
                    if abs(B1 - B2) > 1e-15:
                        c_val = (A2 - A1) / (B1 - B2)
                        if 0.0 < c_val < 1.0:
                            crit.add(c_val)
            return crit
        
        crit = sorted(find_crit(eps_x).union(find_crit(eps_y)))
        
        def union_length(eps, c):
            pts = [(A + B * c, is_r) for A, B, is_r in eps]
            pts.sort(key=lambda x: x[0])
            total = 0.0
            count = 0
            start_pos = 0.0
            for val, is_r in pts:
                if is_r == 0:
                    if count == 0:
                        start_pos = val
                    count += 1
                else:
                    count -= 1
                    if count == 0:
                        total += val - start_pos
            return total
        
        total_vol = 0.0
        for k in range(len(crit) - 1):
            c0 = crit[k]
            c1 = crit[k + 1]
            h = c1 - c0
            if h < 1e-12:
                continue
            Lx0 = union_length(eps_x, c0)
            Lx1 = union_length(eps_x, c1)
            Ly0 = union_length(eps_y, c0)
            Ly1 = union_length(eps_y, c1)
            dx = Lx1 - Lx0
            dy = Ly1 - Ly0
            term = Lx0 * Ly0 + 0.5 * (Lx0 * dy + Ly0 * dx) + (1.0 / 3.0) * dx * dy
            total_vol += h * term
        return total_vol
    
    def objective(params):
        return compute_volume(params[:N], params[N:])
    
    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    p0 = np.concatenate([keich, keich])
    best_p = p0.copy()
    best_v = objective(p0)
    
    rng = np.random.default_rng(42)
    starts = [p0.copy()]
    
    # Random perturbations
    for attempt in range(40):
        if time.time() > deadline - 100:
            break
        sigma = 0.15 * (1 + 0.5 * np.sin(attempt * 0.3))
        perturbation = rng.normal(0, sigma, size=2 * N)
        starts.append(p0 + perturbation)
    
    # Structured starts
    sym_x = np.linspace(-delta / 2, -(N - 1) * delta / 2, N)
    sym_y = np.linspace(-delta / 2, -(N - 1) * delta / 2, N)
    starts.append(np.concatenate([sym_x, sym_y]))
    
    anti_x = np.linspace(0, (N - 1) * delta, N)
    anti_y = np.linspace(0, (N - 1) * delta, N)
    starts.append(np.concatenate([anti_x, anti_y]))
    
    # Diagonal offset
    diag_x = np.linspace(0, 0.5, N)
    diag_y = np.linspace(0, 0.5, N) + 0.2
    starts.append(np.concatenate([diag_x, diag_y]))
    
    for p_start in starts:
        if time.time() > deadline - 80:
            break
        try:
            res = np.optimize.minimize(objective, p_start, method='Nelder-Mead',
                                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 20:
        try:
            res = np.optimize.minimize(objective, best_p, method='Nelder-Mead',
                                       options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
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

<<<<<<< SEARCH
    def make_grid_init(w, h, rng, noise_scale=0.01):
        best_init = None
        best_sum = -1
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs[i] = (col + 0.5) * (w / cols)
                ys[i] = (row + 0.5) * (h / rows)
                # Initial radius estimate: min distance to boundary / 2
                r_est = 0.5 * min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = r_est
            
            xs += rng.normal(0, noise_scale, N)
            ys += rng.normal(0, noise_scale, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            # Simple heuristic score
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_hex_init(w, h, rng):
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
            [7, 7, 7],
            [4, 4, 4, 4, 5],
        ]
        best_init = None
        best_sum = -1
        
        for pattern in patterns:
            if sum(pattern) != N: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            idx = 0
            n_rows = len(pattern)
            row_h = h / n_rows
            
            for row_idx, n_cols in enumerate(pattern):
                col_w = w / n_cols
                offset = 0.0 if row_idx % 2 == 0 else col_w / 2
                for col_idx in range(n_cols):
                    if idx >= N: break
                    x = (col_idx + 0.5) * col_w + offset
                    y = (row_idx + 0.5) * row_h
                    xs[idx] = x
                    ys[idx] = y
                    r_est = 0.5 * min(x, w - x, y, h - y)
                    rs[idx] = r_est
                    idx += 1
            
            if idx != N: continue
            
            xs += rng.normal(0, 0.01, N)
            ys += rng.normal(0, 0.01, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_random_init(w, h, rng):
        xs = rng.uniform(0.1, w - 0.1, N)
        ys = rng.uniform(0.1, h - 0.1, N)
        rs = rng.uniform(0.05, 0.2, N)
        return np.concatenate([xs, ys, rs])

    def is_valid(params, w, h, tol=1e-5):
        xs = params[0:N]
        ys = params[N:2*N]
        rs = params[2*N:3*N]
        
        if np.any(rs < -tol): return False
        if np.any(xs < -tol) or np.any(w - xs < -tol): return False
        if np.any(ys < -tol) or np.any(h - ys < -tol): return False
        if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
        if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
=======
    def make_grid_init(w, h, rng, noise_scale=0.01):
        best_init = None
        best_sum = -1
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs[i] = (col + 0.5) * (w / cols)
                ys[i] = (row + 0.5) * (h / rows)
                r_est = 0.5 * min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = r_est
            
            xs += rng.normal(0, noise_scale, N)
            ys += rng.normal(0, noise_scale, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_hex_init(w, h, rng):
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
            [7, 7, 7],
            [4, 4, 4, 4, 5],
        ]
        best_init = None
        best_sum = -1
        
        for pattern in patterns:
            if sum(pattern) != N: continue
            
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            idx = 0
            n_rows = len(pattern)
            row_h = h / n_rows
            
            for row_idx, n_cols in enumerate(pattern):
                col_w = w / n_cols
                offset = 0.0 if row_idx % 2 == 0 else col_w / 2
                for col_idx in range(n_cols):
                    if idx >= N: break
                    x = (col_idx + 0.5) * col_w + offset
                    y = (row_idx + 0.5) * row_h
                    xs[idx] = x
                    ys[idx] = y
                    r_est = 0.5 * min(x, w - x, y, h - y)
                    rs[idx] = r_est
                    idx += 1
            
            if idx != N: continue
            
            xs += rng.normal(0, 0.01, N)
            ys += rng.normal(0, 0.01, N)
            rs = np.clip(rs, 0.01, 0.5)
            
            params = np.concatenate([xs, ys, rs])
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_init = params
        return best_init

    def make_random_init(w, h, rng):
        xs = rng.uniform(0.1, w - 0.1, N)
        ys = rng.uniform(0.1, h - 0.1, N)
        rs = rng.uniform(0.05, 0.2, N)
        return np.concatenate([xs, ys, rs])

    def is_valid(params, w, h, tol=1e-5):
        xs = params[0:N]
        ys = params[N:2*N]
        rs = params[2*N:3*N]
        
        if np.any(rs < -tol): return False
        if np.any(xs < -tol) or np.any(w - xs < -tol): return False
        if np.any(ys < -tol) or np.any(h - ys < -tol): return False
        if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
        if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
        
        # Use Delaunay for fast validation
        points = np.column_stack([xs, ys])
        edges = _delaunay_edges(points)
        # Also check all pairs to be safe (N=21 is small)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def optimize_phase(params0, w, h, beta, lambda_pen, lambda_eq, max_iter, edges=None, time_budget=None):
        """Run one optimization phase with given parameters."""
        max_r = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        for i in range(N):
            bounds.append((0.0, max_r))
        
        try:
            res = minimize(objective_and_grad, params0, 
                           args=(w, h, beta, lambda_pen, edges, lambda_eq),
                           jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            if res.success and is_valid(res.x, w, h):
                return res.x
        except Exception:
            pass
        return None
>>>>>>> REPLACE


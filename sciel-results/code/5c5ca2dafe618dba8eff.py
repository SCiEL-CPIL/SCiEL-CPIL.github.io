"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_score = -np.inf
    best_result = None

    def _elapsed():
        return time.time() - start_time

    def _optimize_radii_lp(x, y):
        """
        Solve for maximum radii given fixed centers using Linear Programming.
        Maximize sum(r) subject to r_i + r_j <= d_ij and r_i <= dist_to_bound.
        Returns radii array and success flag.
        """
        # Vectorized distance calculation
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Upper triangular indices for pairwise constraints
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Build A_ub matrix for pairwise constraints
        # Each constraint: r_i + r_j <= d_ij
        A_pairs = np.zeros((len(pair_dists), N))
        A_pairs[np.arange(len(pair_dists)), i_idx] = 1.0
        A_pairs[np.arange(len(pair_dists)), j_idx] = 1.0
        b_pairs = pair_dists
        
        # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        bounds_r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        # r_i <= bounds_r[i]  =>  r_i <= b_i
        A_bounds = np.eye(N)
        b_bounds = bounds_r
        
        # Combine constraints
        A_ub = np.vstack([A_pairs, A_bounds])
        b_ub = np.concatenate([b_pairs, b_bounds])
        
        c = -np.ones(N) # Minimize -sum(r)
        
        try:
            # Use HiGHS solver if available, it's very fast for small LPs
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*N, method='highs')
            if res.success:
                return res.x, True
            else:
                # Fallback to iterative shrink if LP fails (rare)
                return _iterative_shrink(x, y), True
        except Exception:
            return _iterative_shrink(x, y), True

    def _iterative_shrink(x, y):
        """Fallback: Iteratively shrink radii to satisfy constraints."""
        r = np.array([min(x[i], 1.0-x[i], y[i], 1.0-y[i]) for i in range(N)])
        for _ in range(20):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    if r[i] + r[j] > d:
                        # Shrink proportionally
                        current_sum = r[i] + r[j]
                        if current_sum > 0:
                            scale = d / current_sum * 0.999
                            r[i] *= scale
                            r[j] *= scale
                        changed = True
            if not changed:
                break
        return r

    def _evaluate(x, y, r, eps=1e-9):
        """Check validity and return sum of radii."""
        if np.any(x < -eps) or np.any(x > 1.0 + eps): return -np.inf
        if np.any(y < -eps) or np.any(y > 1.0 + eps): return -np.inf
        if np.any(r < -eps): return -np.inf
        if np.any(x < r - eps) or np.any(x > 1.0 - r + eps): return -np.inf
        if np.any(y < r - eps) or np.any(y > 1.0 - r + eps): return -np.inf
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        min_dists = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any(dists[mask] < min_dists[mask] - eps):
            return -np.inf
        return np.sum(r)

    def _make_hex_init():
        """Hexagonal lattice initialization."""
        # 6 rows of approx 5-6 circles
        rows = [6, 5, 6, 5, 6, 4] # Total 32
        x = []
        y = []
        row_h = 1.0 / len(rows)
        for row_idx, count in enumerate(rows):
            ry = (row_idx + 0.5) * row_h
            offset = 0.0
            if row_idx % 2 == 1:
                offset = 0.5 / count
            for c in range(count):
                cx = (c + 0.5 + offset) / count
                x.append(cx)
                y.append(ry)
        return np.array(x[:N]), np.array(y[:N])

    def _make_grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = []
        y = []
        for i in range(side):
            for j in range(side):
                if len(x) < N:
                    x.append((i + 0.5) / side)
                    y.append((j + 0.5) / side)
        return np.array(x[:N]), np.array(y[:N])

    def _make_random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, N), rng.uniform(0.1, 0.9, N)

    def _smooth_objective(params):
        """
        Smooth surrogate objective for L-BFGS-B.
        Uses soft-max penalty for overlaps to provide gradients.
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Estimate radii roughly to guide the search
        # We don't want to solve LP here (too slow for gradient steps), 
        # so we use a heuristic radius or fixed small radius for penalty calculation?
        # Better: use the LP result from the last call if cached, or just a simple heuristic.
        # For L-BFGS-B, we can use a penalty method on positions assuming radii are determined by local constraints.
        
        # Let's use a simple heuristic radius: min distance to neighbor / 2
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        
        # Approximate max radius for each circle based on nearest neighbor
        min_dists = np.min(dists, axis=1)
        min_bnds = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        approx_r = np.minimum(min_dists / 2.0, min_bnds)
        
        # Penalty for overlaps with approx_r
        r = approx_r
        min_dists_req = r[:, None] + r[None, :]
        diff = min_dists_req - dists
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Soft max penalty
        eps = 1e-4
        smooth_max = 0.5 * (diff + np.sqrt(diff*diff + eps*eps))
        pen = np.sum(smooth_max[mask])
        
        # Boundary penalty
        bound_pen = 0.0
        # x - r < 0 => r - x > 0
        pen_left = r - x
        pen_right = x - (1.0 - r)
        pen_bottom = r - y
        pen_top = y - (1.0 - r)
        
        b_pen = np.sum(0.5 * (pen_left + np.sqrt(pen_left**2 + eps*eps))) + \
                np.sum(0.5 * (pen_right + np.sqrt(pen_right**2 + eps*eps))) + \
                np.sum(0.5 * (pen_bottom + np.sqrt(pen_bottom**2 + eps*eps))) + \
                np.sum(0.5 * (pen_top + np.sqrt(pen_top**2 + eps*eps)))
        
        # Objective: Maximize sum(r) -> Minimize -sum(r) + Penalty
        # Note: approx_r is a function of x,y, so this is a valid smooth function.
        return -np.sum(r) + 100.0 * (pen + b_pen)

    def _nelder_mead_objective(params):
        """
        Objective for Nelder-Mead.
        Solves LP exactly for radii.
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Clamp positions to [0,1] to help optimizer
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        r, success = _optimize_radii_lp(x, y)
        if not success:
            return 1e6
            
        # Verify validity (LP should guarantee this, but numerical errors can occur)
        score = _evaluate(x, y, r)
        if score == -np.inf:
            return 1e6
        return -score

    # Initializations
    inits = []
    x_hex, y_hex = _make_hex_init()
    inits.append((x_hex, y_hex))
    
    x_grid, y_grid = _make_grid_init()
    inits.append((x_grid, y_grid))
    
    for seed in range(5):
        x_r, y_r = _make_random_init(seed + 100)
        inits.append((x_r, y_r))

    bounds_pos = [(0.0, 1.0)] * (2 * N)

    for idx, (x0, y0) in enumerate(inits):
        if _elapsed() > deadline - 20:
            break

        # 1. Coarse Optimization with L-BFGS-B on positions
        # We use the smooth objective to find a good basin.
        p0 = np.concatenate([x0, y0])
        
        try:
            res = minimize(_smooth_objective, p0, method='L-BFGS-B', 
                           bounds=bounds_pos,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            p_opt = res.x
        except Exception:
            p_opt = p0

        x_curr = p_opt[:N]
        y_curr = p_opt[N:2*N]
        
        # 2. Refinement with Nelder-Mead on positions
        # This uses the exact LP solver, so it's more expensive but precise.
        if _elapsed() < deadline - 10:
            try:
                res2 = minimize(_nelder_mead_objective, p_opt, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                p_final = res2.x
            except Exception:
                p_final = p_opt
        else:
            p_final = p_opt
            
        x_final = np.clip(p_final[:N], 0, 1)
        y_final = np.clip(p_final[N:2*N], 0, 1)
        
        # 3. Final LP solve for exact radii
        r_final, success = _optimize_radii_lp(x_final, y_final)
        if not success:
            continue
            
        score = _evaluate(x_final, y_final, r_final)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([x_final, y_final, r_final])

    # If no valid result, return a safe grid
    if best_result is None:
        x_grid, y_grid = _make_grid_init()
        r_grid, _ = _optimize_radii_lp(x_grid, y_grid)
        best_result = np.column_stack([x_grid, y_grid, r_grid])
        best_score = _evaluate(x_grid, y_grid, r_grid)

    return best_result

if __name__ == "__main__":
    result = construct()
    print(result)

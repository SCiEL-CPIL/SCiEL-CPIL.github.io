"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21
DEADLINE = 110  # seconds, safety margin before 120s limit


def _validate_and_score(circles):
    """Validate circles and compute score (sum of radii)."""
    if len(circles) != N:
        return -1.0
    if np.any(circles[:, 2] < 0):
        return -1.0
    if np.any(np.isnan(circles)) or np.any(np.isinf(circles)):
        return -1.0
    return float(np.sum(circles[:, 2]))


def _check_validity(circles, w, h):
    """Check if circles are valid (non-overlapping, within bounds)."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < -1e-9 or x + r > w + 1e-9:
            return False
        if y - r < -1e-9 or y + r > h + 1e-9:
            return False
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True


def construct():
    start_time = time.time()
    best_score = -1.0
    best_result = None
    
    # Initial configuration: try different rectangle aspect ratios
    # w + h = 2, so h = 2 - w
    # Try several initial widths
    
    def run_optimization(w_init, x0, y0, r0):
        """Run optimization for given initial conditions."""
        h = 2.0 - w_init
        if h <= 0 or w_init <= 0:
            return None, -1.0
        
        # Variables: [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1, w]
        # Total: 3*N + 1 = 64 variables
        n_var = 3 * N + 1
        
        def objective(v):
            """Negative sum of radii (we minimize)."""
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            h = 2.0 - w
            if h <= 0.01 or w <= 0.01:
                return 1e10
            # Penalty for out-of-bounds
            penalty = 0.0
            for i in range(N):
                if xs[i] - rs[i] < -1e-6:
                    penalty += 100 * (xs[i] - rs[i] + 1e-6)**2
                if xs[i] + rs[i] > w + 1e-6:
                    penalty += 100 * (xs[i] + rs[i] - w - 1e-6)**2
                if ys[i] - rs[i] < -1e-6:
                    penalty += 100 * (ys[i] - rs[i] + 1e-6)**2
                if ys[i] + rs[i] > h + 1e-6:
                    penalty += 100 * (ys[i] + rs[i] - h - 1e-6)**2
            return -np.sum(rs) + penalty
        
        # Constraints: non-overlap
        def make_nonoverlap_constraint(i, j):
            def constraint(v):
                xi = v[i*3]
                yi = v[i*3+1]
                ri = v[i*3+2]
                xj = v[j*3]
                yj = v[j*3+1]
                rj = v[j*3+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                # dist >= min_dist  =>  dist_sq >= min_dist^2
                # This is non-convex, so use penalty in objective instead
                return 0  # placeholder
            return constraint
        
        # Use SLSQP with boundary constraints handled via penalty
        # Simpler: just use SLSQP with bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((0, 1.0))  # r
        bounds.append((0.5, 1.5))  # w
        
        # Initial guess
        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]
        v0[3*N] = w_init
        
        # Run SLSQP
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success or res.fun < objective(v0):
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                w = res.x[3*N]
                h = 2.0 - w
                circles = np.column_stack([xs, ys, rs])
                if _check_validity(circles, w, h):
                    score = _validate_and_score(circles)
                    return circles, score
        except Exception:
            pass
        
        return None, -1.0
    
    # Generate several initial configurations
    initial_configs = []
    
    # Grid-like initializations with different aspect ratios
    for w_frac in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        w = w_frac
        h = 2.0 - w
        if h <= 0.1:
            continue
        
        # Arrange circles in a grid pattern
        cols = max(1, int(np.round(N * w / (w + h))))
        rows = max(1, int(np.ceil(N / cols)))
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            x0[i] = (col + 0.5) * w / cols
            y0[i] = (row + 0.5) * h / rows
            r0[i] = min(w / (2 * cols), h / (2 * rows)) * 0.95
        
        initial_configs.append((w, x0, y0, r0))
    
    # Also try random initializations
    np.random.seed(42)
    for trial in range(5):
        w = np.random.uniform(0.7, 1.3)
        h = 2.0 - w
        if h <= 0.1:
            continue
        x0 = np.random.uniform(0.1, w - 0.1, N)
        y0 = np.random.uniform(0.1, h - 0.1, N)
        r0 = np.full(N, 0.08)
        initial_configs.append((w, x0, y0, r0))
    
    # Run optimizations
    for w_init, x0, y0, r0 in initial_configs:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        circles, score = run_optimization(w_init, x0, y0, r0)
        if circles is not None and score > best_score:
            best_score = score
            best_result = circles
    
    # If we have a good result, try a final polish with tighter tolerances
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        # Extract current best parameters
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        
        # Estimate w from the bounding box
        w_est = max(np.max(xs + rs), 0.5)
        h_est = max(np.max(ys + rs), 0.5)
        # Adjust to satisfy w + h = 2
        scale = 2.0 / (w_est + h_est)
        w_polish = w_est * scale
        h_polish = h_est * scale
        
        if 0.3 < w_polish < 1.7:
            # Polish with SLSQP
            n_var = 3 * N + 1
            
            def objective_polish(v):
                xs_p = v[0:3*N:3]
                ys_p = v[1:3*N:3]
                rs_p = v[2:3*N:3]
                w_p = v[3*N]
                h_p = 2.0 - w_p
                if h_p <= 0.01 or w_p <= 0.01:
                    return 1e10
                penalty = 0.0
                for i in range(N):
                    if xs_p[i] - rs_p[i] < -1e-8:
                        penalty += 1000 * (xs_p[i] - rs_p[i] + 1e-8)**2
                    if xs_p[i] + rs_p[i] > w_p + 1e-8:
                        penalty += 1000 * (xs_p[i] + rs_p[i] - w_p - 1e-8)**2
                    if ys_p[i] - rs_p[i] < -1e-8:
                        penalty += 1000 * (ys_p[i] - rs_p[i] + 1e-8)**2
                    if ys_p[i] + rs_p[i] > h_p + 1e-8:
                        penalty += 1000 * (ys_p[i] + rs_p[i] - h_p - 1e-8)**2
                # Non-overlap penalty
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs_p[i]-xs_p[j])**2 + (ys_p[i]-ys_p[j])**2)
                        min_d = rs_p[i] + rs_p[j]
                        if dist < min_d:
                            penalty += 1000 * (min_d - dist)**2
                return -np.sum(rs_p) + penalty
            
            bounds_polish = []
            for i in range(N):
                bounds_polish.append((0, 2.0))
                bounds_polish.append((0, 2.0))
                bounds_polish.append((0, 1.0))
            bounds_polish.append((0.3, 1.7))
            
            v0_polish = np.zeros(n_var)
            for i in range(N):
                v0_polish[i*3] = xs[i]
                v0_polish[i*3+1] = ys[i]
                v0_polish[i*3+2] = rs[i]
            v0_polish[3*N] = w_polish
            
            try:
                res = minimize(objective_polish, v0_polish, method='SLSQP',
                              bounds=bounds_polish, options={'maxiter': 300, 'ftol': 1e-12})
                if res.fun < objective_polish(v0_polish):
                    xs_p = res.x[0:3*N:3]
                    ys_p = res.x[1:3*N:3]
                    rs_p = res.x[2:3*N:3]
                    w_p = res.x[3*N]
                    h_p = 2.0 - w_p
                    circles_p = np.column_stack([xs_p, ys_p, rs_p])
                    if _check_validity(circles_p, w_p, h_p):
                        score_p = _validate_and_score(circles_p)
                        if score_p > best_score:
                            best_score = score_p
                            best_result = circles_p
            except Exception:
                pass
    
    # Fallback: return a valid grid configuration if nothing better found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_score = _validate_and_score(best_result)
    
    return best_result

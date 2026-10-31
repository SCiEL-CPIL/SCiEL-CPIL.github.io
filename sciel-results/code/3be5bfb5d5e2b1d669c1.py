"""Rewritten program: model-and-optimize approach for circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    deadline = time.time() + 110.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h, tol=1e-7):
        if np.any(r < -tol):
            return False
        if np.any(x < -tol) or np.any(x > w + tol):
            return False
        if np.any(y < -tol) or np.any(y > h + tol):
            return False
        # Check circle-wall distance
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def penalty_function(vars_, w, h, penalty_factor=100.0):
        x, y, r = unpack(vars_)
        penalty = 0.0
        
        # Wall constraints (circle must be inside)
        penalty += penalty_factor * np.sum(np.maximum(0, r - x) ** 2)
        penalty += penalty_factor * np.sum(np.maximum(0, r - (w - x)) ** 2)
        penalty += penalty_factor * np.sum(np.maximum(0, r - y) ** 2)
        penalty += penalty_factor * np.sum(np.maximum(0, r - (h - y)) ** 2)
        
        # Negative radius
        penalty += penalty_factor * np.sum(np.maximum(0, -r) ** 2)
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = np.maximum(0, r[i] + r[j] - dist)
                penalty += penalty_factor * overlap ** 2
        
        return penalty - np.sum(r)

    def objective_constrained(vars_, w, h):
        x, y, r = unpack(vars_)
        return -np.sum(r)

    def make_constraints(w, h):
        def cons_inside(vars_):
            x, y, r = unpack(vars_)
            return np.concatenate([
                x - r,
                w - x - r,
                y - r,
                h - y - r
            ])

        def cons_non_overlap(vars_):
            x, y, r = unpack(vars_)
            cons = []
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    cons.append(dist - r[i] - r[j])
            return np.array(cons)

        def cons_positive_r(vars_):
            x, y, r = unpack(vars_)
            return r

        return [
            {'type': 'ineq', 'fun': cons_inside},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]

    def make_bounds(w, h):
        bounds = []
        for i in range(N):
            # x: [0, w]
            bounds.append((0.0, w))
            # y: [0, h]
            bounds.append((0.0, h))
            # r: [0, min(w,h)/2]
            r_max = min(w, h) / 2.0
            bounds.append((0.0, r_max))
        return bounds

    def hexagonal_initial(w, h, n_cols=None):
        """Generate a hexagonal packing initial guess."""
        if n_cols is None:
            n_cols = int(np.sqrt(N * w / h)) + 1
        
        n_rows = int(np.ceil(N / n_cols))
        
        # Adjust if we have space issues
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        # Estimate radius
        est_r = min(w / n_cols, h / n_rows) * 0.45

        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x_center = (col + 0.5) * w / n_cols
            y_center = (row + 0.5) * h / n_rows
            
            # Stagger odd rows for hexagonal packing
            if row % 2 == 1:
                x_center += 0.5 * w / n_cols
                # Clamp to stay inside
                if x_center + est_r > w:
                    x_center = w - est_r
                elif x_center - est_r < 0:
                    x_center = est_r
            
            xs[i] = x_center
            ys[i] = y_center

        return xs, ys

    def random_initial(w, h):
        """Generate a random initial guess."""
        xs = np.random.uniform(0.05, w - 0.05, N)
        ys = np.random.uniform(0.05, h - 0.05, N)
        
        # Estimate radii based on minimum distances
        min_dists = np.zeros(N)
        for i in range(N):
            min_d = min(xs[i], w - xs[i], ys[i], h - ys[i])
            for j in range(N):
                if i != j:
                    d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_d = min(min_d, d)
            min_dists[i] = min_d
        
        radii = 0.4 * min_dists
        return xs, ys, radii

    # Strategy 1: Optimized aspect ratio search with SLSQP
    # Include diverse aspect ratios from very elongated to square
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0, 4.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    # Sort by likelihood of success (near-square first, but keep diversity)
    aspect_ratios.sort(key=lambda ar: abs(ar - 1.25))
    
    max_restarts_per_ar = 3
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        constraints = make_constraints(w, h)
        bounds = make_bounds(w, h)
        
        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break

            if restart == 0:
                # Use hexagonal initial guess
                xs, ys = hexagonal_initial(w, h)
                # Calculate radii using vectorized min distance
                min_dists = np.zeros(N)
                for i in range(N):
                    min_d = min(xs[i], w - xs[i], ys[i], h - ys[i])
                    d = np.sqrt((xs[i] - xs)**2 + (ys[i] - ys)**2)
                    d[i] = np.inf
                    min_d = min(min_d, np.min(d))
                    min_dists[i] = min_d
                radii = 0.42 * min_dists
            elif restart == 1:
                # Grid-based initialization with slight perturbation
                n_cols = int(np.sqrt(N * w / h)) + 1
                n_rows = int(np.ceil(N / n_cols))
                while n_cols * n_rows < N:
                    if n_cols <= n_rows:
                        n_cols += 1
                    else:
                        n_rows += 1
                xs = np.zeros(N)
                ys = np.zeros(N)
                for i in range(N):
                    col = i % n_cols
                    row = i // n_cols
                    xs[i] = (col + 0.5) * w / n_cols
                    ys[i] = (row + 0.5) * h / n_rows
                # Add small random perturbation
                jitter = 0.03 * min(w, h)
                xs = np.clip(xs + np.random.normal(0, jitter, N), 0.01, w - 0.01)
                ys = np.clip(ys + np.random.normal(0, jitter, N), 0.01, h - 0.01)
                # Calculate radii
                min_dists = np.zeros(N)
                for i in range(N):
                    min_d = min(xs[i], w - xs[i], ys[i], h - ys[i])
                    d = np.sqrt((xs[i] - xs)**2 + (ys[i] - ys)**2)
                    d[i] = np.inf
                    min_d = min(min_d, np.min(d))
                    min_dists[i] = min_d
                radii = 0.4 * min_dists
            else:
                # Random restart with jitter
                xs, ys, radii = random_initial(w, h)
                # Add small jitter
                jitter = 0.02 * min(w, h)
                xs = np.clip(xs + np.random.normal(0, jitter, N), 0.01, w - 0.01)
                ys = np.clip(ys + np.random.normal(0, jitter, N), 0.01, h - 0.01)

            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3 * i] = xs[i]
                x0[3 * i + 1] = ys[i]
                x0[3 * i + 2] = radii[i]

            try:
                # Use SLSQP for local optimization
                result = minimize(
                    objective_constrained, x0, method='SLSQP',
                    constraints=constraints, bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
                )
                
                x, y, r = unpack(result.x)
                
                # Check validity
                if validate(x, y, r, w, h):
                    # Clip small negative values
                    x = np.clip(x, 0, w)
                    y = np.clip(y, 0, h)
                    r = np.clip(r, 0, None)
                    
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
                        
            except Exception:
                continue

    # Strategy 2: Try penalty-based approach with different initializations
    # This can sometimes escape local minima
    if time.time() < deadline - 25:
        for ar in [1.0, 1.2, 1.4, 1.5, 2.0]:
            if time.time() > deadline - 15:
                break

            w = 2.0 / (1.0 + ar)
            h = 2.0 - w
            bounds = make_bounds(w, h)
            
            # Try a few random restarts with penalty method
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                    
                xs, ys, radii = random_initial(w, h)
                x0 = np.zeros(3 * N)
                for i in range(N):
                    x0[3 * i] = xs[i]
                    x0[3 * i + 1] = ys[i]
                    x0[3 * i + 2] = radii[i]

                try:
                    result = minimize(
                        lambda v: penalty_function(v, w, h),
                        x0, method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
                    )
                    
                    x, y, r = unpack(result.x)
                    
                    # Check validity
                    if validate(x, y, r, w, h):
                        x = np.clip(x, 0, w)
                        y = np.clip(y, 0, h)
                        r = np.clip(r, 0, None)
                        
                        score = np.sum(r)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([x, y, r])
                            
                except Exception:
                    continue

    # Strategy 3: Final high-precision polish pass on the best result found
    if best_result is not None and time.time() < deadline - 10:
        # Determine the aspect ratio from the best result
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]
        
        # Estimate w and h from the bounding box
        w_est = max(x_best + r_best, 2.0 / (1.0 + 1.25))
        h_est = max(y_best + r_best, 2.0 - w_est)
        
        # Try several candidate aspect ratios near the best one
        for ar in [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5]:
            if time.time() > deadline - 5:
                break
                
            w = 2.0 / (1.0 + ar)
            h = 2.0 - w
            
            # Check if current best fits in this rectangle
            if np.max(x_best + r_best) > w + 1e-6 or np.max(y_best + r_best) > h + 1e-6:
                continue
                
            constraints = make_constraints(w, h)
            bounds = make_bounds(w, h)
            
            x0 = best_result.flatten()
            
            try:
                # High-precision polish with SLSQP
                result = minimize(
                    objective_constrained, x0, method='SLSQP',
                    constraints=constraints, bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
                )
                
                x, y, r = unpack(result.x)
                
                # Check validity
                if validate(x, y, r, w, h, tol=1e-8):
                    x = np.clip(x, 0, w)
                    y = np.clip(y, 0, h)
                    r = np.clip(r, 0, None)
                    
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
                        
            except Exception:
                continue
        
        # Second polish pass with even tighter tolerance
        if time.time() < deadline - 5:
            x_best = best_result[:, 0]
            y_best = best_result[:, 1]
            r_best = best_result[:, 2]
            
            for ar in [1.2, 1.25, 1.3]:
                if time.time() > deadline - 3:
                    break
                    
                w = 2.0 / (1.0 + ar)
                h = 2.0 - w
                
                if np.max(x_best + r_best) > w + 1e-6 or np.max(y_best + r_best) > h + 1e-6:
                    continue
                    
                constraints = make_constraints(w, h)
                bounds = make_bounds(w, h)
                
                x0 = best_result.flatten()
                
                try:
                    result = minimize(
                        objective_constrained, x0, method='SLSQP',
                        constraints=constraints, bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
                    )
                    
                    x, y, r = unpack(result.x)
                    
                    if validate(x, y, r, w, h, tol=1e-9):
                        x = np.clip(x, 0, w)
                        y = np.clip(y, 0, h)
                        r = np.clip(r, 0, None)
                        
                        score = np.sum(r)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([x, y, r])
                            
                except Exception:
                    continue

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END

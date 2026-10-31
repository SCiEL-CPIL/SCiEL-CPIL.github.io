"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    # Focused aspect ratio search around square and slight deviations
    aspect_ratios = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25])
    
    # Number of restarts per aspect ratio
    max_restarts_per_ar = 6

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def smooth_penalty(x, y, r, w, h, beta=100.0):
        """Smooth penalty for constraint violations using soft-min."""
        # Wall constraints: x - r >= 0, w - x - r >= 0, y - r >= 0, h - y - r >= 0
        wall_viol = np.concatenate([
            np.minimum(x - r, 0),
            np.minimum(w - x - r, 0),
            np.minimum(y - r, 0),
            np.minimum(h - y - r, 0)
        ])
        
        # Overlap constraints: dist - ri - rj >= 0 (vectorized)
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        rsum = r[idx_i] + r[idx_j]
        overlap_viol = np.minimum(dist - rsum, 0)
        
        all_viol = np.concatenate([wall_viol, overlap_viol])
        # Soft penalty: sum of soft-min
        neg_viol = -all_viol  # positive when violated
        # log-sum-exp for smooth max
        max_val = np.max(neg_viol)
        if max_val > 1e-12:
            penalty = np.log(np.sum(np.exp(beta * (neg_viol - max_val)))) / beta + max_val
        else:
            penalty = 0.0
        return penalty * 200.0

    def validate(x, y, r, w, h):
        # Check if circles are inside rectangle and non-overlapping
        if np.any(r < 1e-10):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def objective(vars_):
        x, y, r = unpack(vars_)
        w, h = objective.w, objective.h
        return -np.sum(r) + smooth_penalty(x, y, r, w, h)

    def make_constraints(w, h):
        """Returns constraints for SLSQP - keep them for hard enforcement."""
        
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
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            return dist - r[idx_i] - r[idx_j]

        def cons_positive_r(vars_):
            x, y, r = unpack(vars_)
            return r - 1e-10

        return [
            {'type': 'ineq', 'fun': cons_inside},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]

    def hex_init(w, h, restart):
        """Generate hexagonal grid initialization with optional jitter."""
        ar_wh = w / h
        # Choose grid dimensions based on aspect ratio
        if ar_wh >= 1.0:
            n_cols = int(np.round(np.sqrt(N * ar_wh)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / ar_wh)))
            n_cols = int(np.ceil(N / n_rows))
        
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        # Hexagonal offset: vertical spacing is sqrt(3)/2 * cell_w for optimal packing
        hex_h = cell_w * np.sqrt(3) / 2 if cell_w < cell_h else cell_h
        est_r = 0.45 * min(cell_w, hex_h)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x_center = (col + 0.5) * cell_w
            y_center = (row + 0.5) * cell_h
            
            # Stagger odd rows
            if row % 2 == 1:
                x_center += 0.5 * cell_w
                x_center = np.clip(x_center, est_r, w - est_r)

            xs[i] = x_center
            ys[i] = y_center

        if restart > 0:
            jitter_scale = 0.1 * min(cell_w, cell_h)
            np.random.seed(42 + restart)
            xs += np.random.normal(0, jitter_scale, N)
            ys += np.random.normal(0, jitter_scale, N)
            xs = np.clip(xs, 0.01, w - 0.01)
            ys = np.clip(ys, 0.01, h - 0.01)

        # Compute initial radii from minimum distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
        
        r0 = 0.40 * min_dist
        radii = np.full(N, r0)
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = radii[i]
        
        return x0

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)
        objective.w = w
        objective.h = h

        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break

            x0 = hex_init(w, h, restart)

            try:
                # Phase 1: SLSQP with penalty + constraints
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 1200, 'ftol': 1e-12}
                )

                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
                        
                        # Phase 2: Polish with tighter tolerance
                        if time.time() < deadline - 8:
                            try:
                                result2 = minimize(
                                    objective, result.x, method='SLSQP',
                                    constraints=constraints,
                                    options={'maxiter': 5000, 'ftol': 1e-15}
                                )
                                x2, y2, r2 = unpack(result2.x)
                                if validate(x2, y2, r2, w, h):
                                    score2 = np.sum(r2)
                                    if score2 > best_score:
                                        best_score = score2
                                        best_result = np.column_stack([x2, y2, r2])
                            except Exception:
                                pass

            except Exception:
                continue

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        w = h = 1.0
        n_cols = 5
        n_rows = 5
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(cells) < N:
                    cx = (j + 0.5) * w / n_cols
                    cy = (i + 0.5) * h / n_rows
                    r = 0.45 * min(w / n_cols, h / n_rows)
                    cells.append((cx, cy, r))
        return np.array(cells)

    # Final global polish on the best solution found
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        
        # Determine which aspect ratio gave the best result by checking bounds
        # Try to polish with the same w,h that produced it
        # We need to recover w,h - try to infer from the solution
        # For now, try a few candidate rectangles
        for ar_test in [0.9, 0.95, 1.0, 1.05, 1.1]:
            if time.time() > deadline - 3:
                break
            w_test = 2.0 / (1.0 + ar_test)
            h_test = 2.0 - w_test
            
            # Check if current best fits in this rectangle
            if (np.all(x_best >= r_best - 1e-8) and np.all(x_best <= w_test - r_best + 1e-8) and
                np.all(y_best >= r_best - 1e-8) and np.all(y_best <= h_test - r_best + 1e-8)):
                try:
                    objective.w = w_test
                    objective.h = h_test
                    constraints_final = make_constraints(w_test, h_test)
                    x_final = np.column_stack([x_best, y_best, r_best]).flatten()
                    result_final = minimize(
                        objective, x_final, method='SLSQP',
                        constraints=constraints_final,
                        options={'maxiter': 8000, 'ftol': 1e-15}
                    )
                    xf, yf, rf = unpack(result_final.x)
                    if validate(xf, yf, rf, w_test, h_test):
                        score_final = np.sum(rf)
                        if score_final > best_score:
                            best_score = score_final
                            best_result = np.column_stack([xf, yf, rf])
                except Exception:
                    pass
                break  # Only try the first fitting rectangle
    
    return best_result
# EVOLVE-BLOCK-END

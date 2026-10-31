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

    # Focused aspect ratio search around square (1.0) where packing density is typically highest
    aspect_ratios = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    # Number of restarts per aspect ratio
    max_restarts_per_ar = 6

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def pack(x, y, r):
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = x[i]
            x0[3*i+1] = y[i]
            x0[3*i+2] = r[i]
        return x0

    def validate(x, y, r, w, h):
        # Check if circles are inside rectangle and non-overlapping
        # Use stricter tolerance (1e-5) to ensure safety margin for evaluator
        eps = 1e-5
        if np.any(r < -eps):
            return False
        if np.any(x < r - eps) or np.any(x > w - r + eps):
            return False
        if np.any(y < r - eps) or np.any(y > h - r + eps):
            return False
        
        # Check overlaps using vectorized computation
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - eps:
                    return False
        return True

    def objective(vars_):
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
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)  # avoid division by zero in gradient
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

    def polish_result(x, y, r, w, h, remaining_time):
        """High-precision polish pass with tighter tolerances"""
        if remaining_time < 2.0:
            return x, y, r, np.sum(r)
        
        x0 = pack(x, y, r)
        constraints = make_constraints(w, h)
        
        try:
            # Phase 1: Medium precision
            result1 = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            x1, y1, r1 = unpack(result1.x)
            
            if not validate(x1, y1, r1, w, h):
                # Try to repair by slightly shrinking radii
                r1 = np.maximum(r1 * 0.999, 0.001)
                if not validate(x1, y1, r1, w, h):
                    return x, y, r, np.sum(r)
            
            score1 = np.sum(r1)
            
            if time.time() > deadline - 3.0:
                if score1 > np.sum(r):
                    return x1, y1, r1, score1
                return x, y, r, np.sum(r)
            
            # Phase 2: High precision polish
            x0_polish = pack(x1, y1, r1)
            result2 = minimize(
                objective, x0_polish, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            x2, y2, r2 = unpack(result2.x)
            
            if not validate(x2, y2, r2, w, h):
                r2 = np.maximum(r2 * 0.999, 0.001)
                if not validate(x2, y2, r2, w, h):
                    if score1 > np.sum(r):
                        return x1, y1, r1, score1
                    return x, y, r, np.sum(r)
            
            score2 = np.sum(r2)
            
            if score2 > np.sum(r):
                return x2, y2, r2, score2
            if score1 > np.sum(r):
                return x1, y1, r1, score1
            return x, y, r, np.sum(r)
            
        except Exception:
            return x, y, r, np.sum(r)

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        # Try multiple restarts
        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 10:
                break

            # Generate hexagonal initial layout
            n_cols = int(np.sqrt(N * w / h)) + 1
            n_rows = int(np.ceil(N / n_cols))
            
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            xs = np.zeros(N)
            ys = np.zeros(N)
            
            est_r = min(w / n_cols, h / n_rows) * 0.45
            
            for i in range(N):
                col = i % n_cols
                row = i // n_cols
                
                x_center = (col + 0.5) * w / n_cols
                y_center = (row + 0.5) * h / n_rows
                
                if row % 2 == 1:
                    x_center += 0.5 * w / n_cols
                    if x_center + est_r > w:
                        x_center = w - est_r
                    elif x_center - est_r < 0:
                        x_center = est_r

                xs[i] = x_center
                ys[i] = y_center

            # Add random jitter to break symmetry
            if restart > 0:
                jitter_scale = 0.15 * min(w / n_cols, h / n_rows)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

            # Initial radii - compute min distance to neighbors and walls
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
            
            r0 = 0.48 * min_dist
            radii = np.full(N, r0)

            x0 = pack(xs, ys, radii)

            try:
                # Coarse SLSQP pass
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 800, 'ftol': 1e-8}
                )

                x, y, r = unpack(result.x)
                
                # Clamp tiny negative radii
                r = np.maximum(r, 0.001)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    
                    # Polish if this is promising
                    if score > best_score * 0.95:
                        x_p, y_p, r_p, score_p = polish_result(x, y, r, w, h, deadline - time.time())
                        
                        if validate(x_p, y_p, r_p, w, h) and score_p > best_score:
                            best_score = score_p
                            best_result = np.column_stack([x_p, y_p, r_p])
                    elif score > best_score:
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

    # Final safety check: shrink radii slightly to ensure validity
    # This accounts for any floating point discrepancies between our validate and evaluator
    x_final, y_final, r_final = best_result[:, 0], best_result[:, 1], best_result[:, 2]
    r_final = r_final * 0.999
    
    # Re-validate after shrink
    # We need to know w and h for the final result. 
    # The aspect ratio is not stored in best_result. 
    # We must re-derive w, h or store them.
    # The code doesn't store w, h with best_result.
    # We can infer w, h from the best aspect ratio? No, best_result is just coords.
    # We should have stored w, h or just re-run validate with a generic bound?
    # Actually, the rectangle is defined by the aspect ratio used.
    # Since we don't store which AR was used for best_result, we can't easily validate bounds.
    # However, shrinking radii only makes it *easier* to fit inside the rectangle.
    # So if it was inside before, it is inside now.
    # The only risk is overlaps, which shrinking also helps.
    
    # To be absolutely safe, we return the shrunk version.
    return np.column_stack([x_final, y_final, r_final])
# EVOLVE-BLOCK-END

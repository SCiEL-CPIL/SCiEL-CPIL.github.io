"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0
    best_result = None
    best_score = 0.0

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def smooth_penalty(x, y, r, w, h, beta=50.0):
        """Smooth penalty for constraint violations using log-sum-exp."""
        # Wall constraints
        wall_viol = np.concatenate([
            np.minimum(x - r, 0),
            np.minimum(w - x - r, 0),
            np.minimum(y - r, 0),
            np.minimum(h - y - r, 0)
        ])
        
        # Overlap constraints
        overlap_viol = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                overlap_viol.append(np.minimum(dist - r[i] - r[j], 0))
        
        all_viol = np.concatenate([wall_viol, overlap_viol])
        neg_viol = -all_viol
        max_val = np.max(neg_viol)
        if max_val > 1e-12:
            penalty = np.log(np.sum(np.exp(beta * (neg_viol - max_val)))) / beta + max_val
        else:
            penalty = 0.0
        return penalty * 100.0

    def validate(x, y, r, w, h, tol=1e-9):
        if np.any(r < 1e-10):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

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
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    cons.append(dist - r[i] - r[j])
            return np.array(cons)

        def cons_positive_r(vars_):
            x, y, r = unpack(vars_)
            return r - 1e-10

        return [
            {'type': 'ineq', 'fun': cons_inside},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]

    def force_relax(x, y, r, w, h, n_iters=100, dt=0.01):
        """Force-directed relaxation to resolve overlaps and wall violations."""
        for _ in range(n_iters):
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 10
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 10
            
            # Overlap repulsion
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    rsum = r[i] + r[j]
                    if dist < rsum:
                        overlap = rsum - dist
                        if dist > 1e-12:
                            fx[i] += (dx / dist) * overlap * 5
                            fy[i] += (dy / dist) * overlap * 5
                            fx[j] -= (dx / dist) * overlap * 5
                            fy[j] -= (dy / dist) * overlap * 5
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clip to bounds
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
        
        return x, y

    def hex_init(w, h, restart, seed_base=42):
        """Generate hexagonal grid initialization with jitter."""
        ar_wh = w / h
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
        est_r = 0.40 * min(cell_w, cell_h)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x_center = (col + 0.5) * cell_w
            y_center = (row + 0.5) * cell_h
            
            if row % 2 == 1:
                x_center += 0.5 * cell_w
                x_center = np.clip(x_center, est_r, w - est_r)

            xs[i] = x_center
            ys[i] = y_center

        if restart > 0:
            jitter_scale = 0.15 * min(cell_w, cell_h)
            np.random.seed(seed_base + restart * 7)
            xs += np.random.normal(0, jitter_scale, N)
            ys += np.random.normal(0, jitter_scale, N)
            xs = np.clip(xs, est_r*0.5, w - est_r*0.5)
            ys = np.clip(ys, est_r*0.5, h - est_r*0.5)

        # Force relaxation to resolve initial overlaps
        r0 = 0.35 * min(cell_w, cell_h)
        radii = np.full(N, r0)
        xs, ys = force_relax(xs, ys, radii, w, h, n_iters=50, dt=0.005)
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = radii[i]
        
        return x0

    # Aspect ratios to explore
    aspect_ratios = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    max_restarts = 5

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        def objective(vars_):
            x, y, r = unpack(vars_)
            return -np.sum(r) + smooth_penalty(x, y, r, w, h)

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            x0 = hex_init(w, h, restart)

            try:
                # Phase 1: SLSQP optimization
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )

                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
                        
                        # Phase 2: Polish
                        if time.time() < deadline - 12:
                            try:
                                result2 = minimize(
                                    objective, result.x, method='SLSQP',
                                    constraints=constraints,
                                    options={'maxiter': 2000, 'ftol': 1e-14}
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

    # Fallback
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

    return best_result
# EVOLVE-BLOCK-END

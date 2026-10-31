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

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
        if np.any(r < 0):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
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
                    dist = np.sqrt(dx*dx + dy*dy)
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

    def force_directed_relax(x, y, r, w, h, n_steps=500, alpha=0.1):
        """
        Force-directed relaxation using contact graph model.
        Treats circles as nodes with springs for contact constraints.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # Build contact graph: edges where circles are close
        # Use a dynamic threshold based on current radii
        for step in range(n_steps):
            # Decay alpha for convergence
            alpha_step = alpha * (1.0 - step / n_steps)
            
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            
            # Pairwise repulsion for overlapping circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
            
            # Wall constraints: push circles away from walls
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    forces_x[i] += (r[i] - x[i]) * 2.0
                # Right wall
                if x[i] > w - r[i]:
                    forces_x[i] -= (x[i] - (w - r[i])) * 2.0
                # Bottom wall
                if y[i] < r[i]:
                    forces_y[i] += (r[i] - y[i]) * 2.0
                # Top wall
                if y[i] > h - r[i]:
                    forces_y[i] -= (y[i] - (h - r[i])) * 2.0
            
            # Update positions
            x += alpha_step * forces_x
            y += alpha_step * forces_y
            
            # Project back to valid region
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
        
        return x, y, r

    def grid_initialization(n_cols, n_rows, w, h, stagger=True):
        """Generate grid-based initial positions."""
        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x_center = (col + 0.5) * w / n_cols
            y_center = (row + 0.5) * h / n_rows
            
            if stagger and row % 2 == 1:
                x_center += 0.5 * w / n_cols
                if x_center > w - 0.1:
                    x_center = w - 0.1
                elif x_center < 0.1:
                    x_center = 0.1
            
            xs[i] = x_center
            ys[i] = y_center
        
        return xs, ys

    def compute_initial_radii(x, y, w, h):
        """Compute initial radii based on minimum distances."""
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, x[i], w - x[i], y[i], h - y[i])
        
        return 0.45 * min_dist

    # Search over aspect ratios
    aspect_ratios = np.linspace(0.6, 1.8, 12)
    max_restarts_per_ar = 4

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 8:
                break

            # Generate initial layout
            n_cols = int(np.sqrt(N * w / h)) + 1
            n_rows = int(np.ceil(N / n_cols))
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            xs, ys = grid_initialization(n_cols, n_rows, w, h, stagger=True)

            # Add random jitter for diversity
            if restart > 0:
                jitter_scale = 0.08 * min(w / n_cols, h / n_rows)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                xs = np.clip(xs, 0.02, w - 0.02)
                ys = np.clip(ys, 0.02, h - 0.02)

            # Compute initial radii
            r0 = compute_initial_radii(xs, ys, w, h)
            radii = np.full(N, r0)

            # Force-directed relaxation to improve initial packing
            xs_fd, ys_fd, _ = force_directed_relax(xs, ys, radii, w, h, n_steps=300, alpha=0.15)

            # Build initial vector
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = xs_fd[i]
                x0[3*i+1] = ys_fd[i]
                x0[3*i+2] = radii[i]

            try:
                # SLSQP optimization
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 800, 'ftol': 1e-10}
                )

                x, y, r = unpack(result.x)
                
                # Final force-directed polish
                x_pol, y_pol, r_pol = force_directed_relax(x, y, r, w, h, n_steps=200, alpha=0.05)
                
                # Validate and score
                if validate(x_pol, y_pol, r_pol, w, h):
                    score = np.sum(r_pol)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_pol, y_pol, r_pol])
                
                # Also check SLSQP result directly
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])

            except Exception:
                continue

    # Fallback
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

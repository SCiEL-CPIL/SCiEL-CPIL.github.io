"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def remaining():
        return max(0.0, deadline - time.time())

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h, tol=1e-9):
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False

        # Pairwise overlap check
        for i in range(N):
            dx = x[i] - x[i+1:]
            dy = y[i] - y[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            if np.any(dist < r[i] + r[i+1:] - tol):
                return False
        return True

    def hexagonal_init(w, h, seed):
        rng = np.random.default_rng(seed)
        # Estimate grid size
        # For N=21, a 5x4 or 4x5 grid is common.
        # Let's try to fit N circles in a hexagonal grid.
        # Approximate area per circle ~ pi*r^2. Total area w*h.
        # r ~ sqrt(w*h / (N*pi))
        est_r = np.sqrt(w * h / (N * np.pi))
        
        # Hexagonal spacing
        dx = 2 * est_r
        dy = np.sqrt(3) * est_r
        
        # Determine grid dimensions
        n_cols = int(np.ceil(w / dx))
        n_rows = int(np.ceil(h / dy))
        
        # Adjust to ensure we have at least N positions
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
            
            # Center of cell
            x_center = (col + 0.5) * w / n_cols
            y_center = (row + 0.5) * h / n_rows
            
            # Stagger odd rows for hexagonal packing
            if row % 2 == 1:
                x_center += 0.5 * w / n_cols
                # Clamp to stay inside if it goes out of bounds
                if x_center > w - est_r:
                    x_center = w - est_r
                elif x_center < est_r:
                    x_center = est_r

            xs[i] = x_center
            ys[i] = y_center

        # Add random jitter to break symmetry
        jitter_scale = 0.05 * est_r
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        # Clip to valid range (roughly)
        xs = np.clip(xs, est_r*0.5, w - est_r*0.5)
        ys = np.clip(ys, est_r*0.5, h - est_r*0.5)

        # Calculate initial radii based on min distance to neighbors and walls
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
            
        r0 = 0.45 * min_dist
        radii = np.full(N, r0)
        
        return xs, ys, radii

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

    # Search over aspect ratios
    # w + h = 2
    # ar = h/w -> h = ar*w -> w(1+ar)=2 -> w = 2/(1+ar)
    aspect_ratios = np.linspace(0.7, 1.3, 7)
    
    for ar in aspect_ratios:
        if remaining() < 5.0:
            break
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        constraints = make_constraints(w, h)
        
        # Try multiple restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if remaining() < 2.0:
                break
            
            seed = int(ar * 1000) + restart * 7
            xs, ys, radii = hexagonal_init(w, h, seed)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            try:
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
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

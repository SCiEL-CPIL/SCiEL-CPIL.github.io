"""Circle packing in unit square: 26 circles, maximize sum of radii.
Strategy: Global optimization via dual_annealing (simulated annealing + L-BFGS-B)
with smooth quadratic penalty, plus Delaunay-contact-graph constrained polish.
This is a fundamentally different search family from grow-and-relax/SLSQP:
- Explores the full 78-D (26*3) space globally via annealing
- Uses smooth penalties (not hard constraints) for the annealing phase
- Refines with Delaunay-reduced constraint L-BFGS-B
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 135.0  # Safety margin from 150s limit


def _validity(circles, eps=1e-9):
    """Check validity. Returns (is_valid, sum_radii)."""
    n = len(circles)
    s = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False, 0.0
        if x < r + eps or x > 1 - r - eps or y < r + eps or y > 1 - r - eps:
            return False, 0.0
        s += r
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            d2 = (x - x2) ** 2 + (y - y2) ** 2
            if d2 < (r + r2) ** 2 - 1e-8:
                return False, 0.0
    return True, s


def _delaunay_edges(pos):
    """Get Delaunay edges (pairs of nearby centers)."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        # Also add near-neighbor pairs not in Delaunay
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                if d < 0.3:  # heuristic: close enough to matter
                    edges.add((i, j))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _penalty_objective(vars, edges, wall_margin=0.0, pair_margin=0.0):
    """Smooth quadratic penalty objective: minimize -sum(r) + penalties.
    vars = [x0, y0, x1, y1, ..., r0, r1, ...]  (78 dims)
    """
    n = N
    x = vars[:2 * n].reshape(n, 2)
    rad = vars[2 * n:]
    
    # Negative sum of radii (we minimize)
    obj = -np.sum(rad)
    
    # Wall penalties (quadratic)
    for i in range(n):
        # x[i,0] >= rad[i]
        viol = max(0, rad[i] - x[i, 0])
        obj += 100.0 * viol ** 2
        # x[i,0] <= 1 - rad[i]
        viol = max(0, x[i, 0] + rad[i] - 1.0)
        obj += 100.0 * viol ** 2
        # x[i,1] >= rad[i]
        viol = max(0, rad[i] - x[i, 1])
        obj += 100.0 * viol ** 2
        # x[i,1] <= 1 - rad[i]
        viol = max(0, x[i, 1] + rad[i] - 1.0)
        obj += 100.0 * viol ** 2
    
    # Pair penalties (quadratic, only for Delaunay edges)
    for (i, j) in edges:
        d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
        viol = max(0, rad[i] + rad[j] - d)
        obj += 100.0 * viol ** 2
    
    return obj


def _penalty_grad(vars, edges):
    """Gradient of penalty objective."""
    n = N
    x = vars[:2 * n].reshape(n, 2)
    rad = vars[2 * n:]
    grad = np.zeros(3 * n)
    
    # d(-sum(rad))/d(rad) = -1
    grad[2 * n:] = -1.0
    
    # Wall gradients
    for i in range(n):
        # x[i,0] >= rad[i]: penalty = 100*(rad[i]-x[i,0])^2 if active
        viol = rad[i] - x[i, 0]
        if viol > 0:
            grad[2 * i] += 200.0 * (-viol)   # d/dx
            grad[2 * n + i] += 200.0 * viol   # d/dr
        viol = x[i, 0] + rad[i] - 1.0
        if viol > 0:
            grad[2 * i] += 200.0 * viol
            grad[2 * n + i] += 200.0 * viol
        viol = rad[i] - x[i, 1]
        if viol > 0:
            grad[2 * i + 1] += 200.0 * (-viol)
            grad[2 * n + i] += 200.0 * viol
        viol = x[i, 1] + rad[i] - 1.0
        if viol > 0:
            grad[2 * i + 1] += 200.0 * viol
            grad[2 * n + i] += 200.0 * viol
    
    # Pair gradients
    for (i, j) in edges:
        dx = x[i, 0] - x[j, 0]
        dy = x[i, 1] - x[j, 1]
        d = np.hypot(dx, dy)
        if d < 1e-12:
            d = 1e-12
        viol = rad[i] + rad[j] - d
        if viol > 0:
            # penalty = 100 * viol^2
            # d/dx_i = 100 * 2 * viol * d/dx_i (rad[i]+rad[j]-d)
            # d/dx_i (-d) = -dx/d
            factor = 200.0 * viol
            grad[2 * i] += factor * (-dx / d)
            grad[2 * i + 1] += factor * (-dy / d)
            grad[2 * j] += factor * (dx / d)
            grad[2 * j + 1] += factor * (dy / d)
            grad[2 * n + i] += factor
            grad[2 * n + j] += factor
    
    return grad


def _dual_annealing_run(pos0, r0, edges, time_budget=10.0):
    """Run dual_annealing from initial point. Returns best (pos, r, score)."""
    t_start = time.time()
    
    vars0 = np.zeros(3 * N)
    vars0[:2 * N] = pos0.flatten()
    vars0[2 * N:] = r0
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.0, 0.5))  # r
    
    def objective(v):
        return _penalty_objective(v, edges)
    
    def gradient(v):
        return _penalty_grad(v, edges)
    
    try:
        from scipy.optimize import dual_annealing
        res = dual_annealing(
            objective,
            bounds=bounds,
            x0=vars0,
            maxiter=30,
            maxfun=5000,
            seed=42,
            initial_temp=1.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            progress=False,
        )
        
        if res.x is not None:
            new_pos = res.x[:2 * N].reshape(N, 2)
            new_r = res.x[2 * N:]
            # Clip to valid range
            new_r = np.clip(new_r, 0.0, 0.5)
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, 1.0)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, 1.0)
            
            circles = list(zip(new_pos[:, 0], new_pos[:, 1], new_r))
            valid, score = _validity(circles)
            if valid:
                return new_pos, new_r, score
    except Exception:
        pass
    
    return pos0, r0, -np.sum(r0)


def _delaunay_polish(pos, r, time_budget=8.0):
    """Polish using L-BFGS-B with Delaunay-reduced constraints via penalty."""
    t_start = time.time()
    edges = _delaunay_edges(pos)
    
    vars0 = np.zeros(3 * N)
    vars0[:2 * N] = pos.flatten()
    vars0[2 * N:] = r
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    def objective(v):
        return _penalty_objective(v, edges)
    
    def gradient(v):
        return _penalty_grad(v, edges)
    
    try:
        res = minimize(
            objective,
            vars0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        if res.x is not None:
            new_pos = res.x[:2 * N].reshape(N, 2)
            new_r = res.x[2 * N:]
            new_r = np.clip(new_r, 0.0, 0.5)
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, 1.0)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, 1.0)
            
            circles = list(zip(new_pos[:, 0], new_pos[:, 1], new_r))
            valid, score = _validity(circles)
            if valid:
                return new_pos, new_r, score
    except Exception:
        pass
    
    return pos, r, -np.sum(r)


def _max_radius_fix(pos, iters=30):
    """Given fixed positions, maximize radii via Jacobi."""
    n = N
    r = np.zeros(n)
    for _ in range(iters):
        for i in range(n):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)
    return r


def _get_initializations():
    """Generate diverse initial configurations."""
    inits = []
    rng = np.random.default_rng(12345)
    
    # 1. Grid-like (5x5 + 1 extra)
    side = 5
    pts = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            pts.append(((i + 0.5) / side, (j + 0.5) / side))
            idx += 1
    if len(pts) < N:
        pts.append((0.5, 0.5))
    inits.append(np.array(pts[:N]))
    
    # 2. Hexagonal packing
    pts = []
    spacing = 1.0 / 5
    row = 0
    y = spacing / 2
    while len(pts) < N and y < 1.0:
        x_offset = (row % 2) * spacing * 0.866 / 2
        x = spacing / 2 + x_offset
        while x < 1.0 and len(pts) < N:
            pts.append((x, y))
            x += spacing
        y += spacing * 0.866
        row += 1
    if len(pts) >= N:
        inits.append(np.array(pts[:N]))
    
    # 3. Random configurations
    for seed in [1, 2, 3, 7, 13, 21, 34, 55, 89, 144]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        inits.append(pts)
    
    # 4. Corner-weighted random
    for seed in [100, 200, 300]:
        rng = np.random.default_rng(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            # Bias toward corners and edges
            if i < 4:
                pts[i] = rng.random(2) * 0.3  # near corner
            elif i < 8:
                pts[i, 0] = rng.random()
                pts[i, 1] = rng.random() * 0.2  # near bottom
            else:
                pts[i] = rng.random(2)
        inits.append(pts)
    
    return inits


def construct():
    t_start = time.time()
    best_score = -1.0
    best_circles = None
    
    def update_best(pos, r):
        nonlocal best_score, best_circles
        circles = list(zip(pos[:, 0], pos[:, 1], r))
        valid, s = _validity(circles)
        if valid and s > best_score:
            best_score = s
            best_circles = [tuple(c) for c in circles]
    
    inits = _get_initializations()
    
    for init_idx, pos0 in enumerate(inits):
        if time.time() > DEADLINE:
            break
        
        # Set initial radii
        r0 = _max_radius_fix(pos0, iters=20)
        circles = list(zip(pos0[:, 0], pos0[:, 1], r0))
        valid, s = _validity(circles)
        if valid:
            update_best(pos0, r0)
        
        # Phase 1: Dual annealing (global exploration)
        if time.time() < DEADLINE - 15:
            edges = _delaunay_edges(pos0)
            new_pos, new_r, score = _dual_annealing_run(pos0, r0, edges, time_budget=10.0)
            if score > 0:
                update_best(new_pos, new_r)
                pos0, r0 = new_pos, new_r
        
        # Phase 2: Delaunay-constrained L-BFGS-B polish
        if time.time() < DEADLINE - 5:
            new_pos, new_r, score = _delaunay_polish(pos0, r0, time_budget=8.0)
            if score > 0:
                update_best(new_pos, new_r)
                pos0, r0 = new_pos, new_r
        
        # Phase 3: Another polish pass (refined edges)
        if time.time() < DEADLINE - 2:
            new_pos, new_r, score = _delaunay_polish(pos0, r0, time_budget=5.0)
            if score > 0:
                update_best(new_pos, new_r)
    
    # Final: try one more global pass with best found as seed
    if best_circles is not None and time.time() < DEADLINE - 10:
        best_pos = np.array([[c[0], c[1]] for c in best_circles])
        best_r = np.array([c[2] for c in best_circles])
        edges = _delaunay_edges(best_pos)
        
        # Diversify: add small random perturbation
        rng = np.random.default_rng(999)
        perturbed_pos = best_pos + rng.normal(0, 0.02, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0, 1)
        perturbed_r = np.clip(best_r + rng.normal(0, 0.01, N), 0, 0.5)
        
        new_pos, new_r, score = _dual_annealing_run(perturbed_pos, perturbed_r, edges, time_budget=8.0)
        if score > 0:
            update_best(new_pos, new_r)
        
        if time.time() < DEADLINE - 3:
            new_pos, new_r, score = _delaunay_polish(new_pos, new_r, time_budget=5.0)
            if score > 0:
                update_best(new_pos, new_r)
    
    if best_circles is None:
        # Fallback: 5x5 grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.95)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.array(best_circles)


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)

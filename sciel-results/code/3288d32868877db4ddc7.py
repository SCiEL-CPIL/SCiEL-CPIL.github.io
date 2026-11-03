"""
Pack 50 equal circles in a unit square using a Delaunay-based contact graph
with hard-constraint SLSQP polishing, preceded by a fast hard-disk relaxation
to get into the basin of attraction of the optimum.

Strategy (different from the seed):
1. Initialisation: 7x7 grid + jitter, hexagonal, 8x8 grid, random.
2. Fast hard-disk relaxation: iteratively push overlapping circles apart
   (like a simple molecular dynamics / relaxation) to get a feasible
   configuration with a decent radius.
3. Delaunay triangulation to identify the contact graph (edges where
   distance < 2*r + epsilon).
4. SLSQP with hard constraints:
   - Objective: maximize r
   - Constraints: pairwise distances >= 2*r, wall distances >= r
   - Variables: (x_1,...,x_N, y_1,...,y_N, r)
5. Iterate: re-compute contact graph, polish with SLSQP on the active
   constraints only (or all constraints for safety).
6. Multi-start with diverse initializations.

This is structurally different from the seed's soft-min + L-BFGS-B approach:
- Uses exact hard constraints instead of smooth approximation
- Uses Delaunay to identify which constraints are active
- Uses a physics-inspired relaxation phase before optimization
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill


def _radius(centers):
    """Compute the max common radius for given centers."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)


def _delaunay_edges(centers, r, tol=1e-6):
    """
    Identify contact edges via Delaunay triangulation.
    Return list of (i, j) pairs where dist(i,j) < 2*r + tol,
    plus wall contacts.
    """
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                i = simplex[k]
                j = simplex[(k + 1) % 3]
                edges.add((min(i, j), max(i, j)))
        # Also add all pairs that are close (Delaunay might miss some)
        # For 50 points, check all pairs near 2r
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2 * r + tol * 10:
                    edges.add((i, j))
        return list(edges)
    except Exception:
        # Fallback: all pairs within 2*r + margin
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2 * r + 0.01:
                    edges.append((i, j))
        return edges


def _slsqp_polish(centers, r0, time_limit=8.0):
    """
    SLSQP with hard constraints to maximize radius.
    Variables: [x_1,...,x_N, y_1,...,y_N, r]
    """
    x0 = np.zeros(2 * N + 1)
    x0[:N] = centers[:, 0]
    x0[N:2*N] = centers[:, 1]
    x0[2*N] = r0
    
    start_time = time.time()
    killed = [False]
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            killed[0] = True
            raise RuntimeError("Time limit")
        return False
    
    def objective(x):
        return -x[2*N]  # maximize r
    
    def obj_grad(x):
        g = np.zeros_like(x)
        g[2*N] = -1.0
        return g
    
    constraints = []
    
    # Pairwise distance constraints: ||c_i - c_j|| >= 2*r
    # In SLSQP form: ||c_i - c_j|| - 2*r >= 0
    # Use a sparse set: all pairs (for safety with 50 points, ~1225 constraints)
    # This is manageable for SLSQP.
    for i in range(N):
        for j in range(i + 1, N):
            def con(x, i=i, j=j):
                xi = x[i]
                yi = x[N + i]
                xj = x[j]
                yj = x[N + j]
                r = x[2*N]
                d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return d - 2 * r
            def con_grad(x, i=i, j=j):
                g = np.zeros_like(x)
                xi = x[i]
                yi = x[N + i]
                xj = x[j]
                yj = x[N + j]
                r = x[2*N]
                dx = xi - xj
                dy = yi - yj
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-12:
                    g[i] = dx / d
                    g[N + i] = dy / d
                    g[j] = -dx / d
                    g[N + j] = -dy / d
                    g[2*N] = -2.0
                return g
            constraints.append({'type': 'ineq', 'fun': con, 'jac': con_grad})
    
    # Wall constraints: 0.5 - |x_i| >= r and 0.5 - |y_i| >= r
    for i in range(N):
        # x_i <= 0.5 - r  =>  0.5 - r - x_i >= 0
        def con_x_right(x, i=i):
            return 0.5 - x[2*N] - x[i]
        def con_x_right_grad(x, i=i):
            g = np.zeros_like(x)
            g[i] = -1.0
            g[2*N] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': con_x_right, 'jac': con_x_right_grad})
        
        # -x_i <= 0.5 - r  =>  0.5 - r + x_i >= 0
        def con_x_left(x, i=i):
            return 0.5 - x[2*N] + x[i]
        def con_x_left_grad(x, i=i):
            g = np.zeros_like(x)
            g[i] = 1.0
            g[2*N] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': con_x_left, 'jac': con_x_left_grad})
        
        # y_i <= 0.5 - r
        def con_y_top(x, i=i):
            return 0.5 - x[2*N] - x[N + i]
        def con_y_top_grad(x, i=i):
            g = np.zeros_like(x)
            g[N + i] = -1.0
            g[2*N] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': con_y_top, 'jac': con_y_top_grad})
        
        # -y_i <= 0.5 - r
        def con_y_bottom(x, i=i):
            return 0.5 - x[2*N] + x[N + i]
        def con_y_bottom_grad(x, i=i):
            g = np.zeros_like(x)
            g[N + i] = 1.0
            g[2*N] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': con_y_bottom, 'jac': con_y_bottom_grad})
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.001, 0.5)]
    
    try:
        res = minimize(
            objective, x0, jac=obj_grad,
            method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-10, 'disp': False},
            callback=callback
        )
        if res.x is not None:
            cx = res.x[:N]
            cy = res.x[N:2*N]
            r = res.x[2*N]
            return np.column_stack([cx, cy]), r
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers, r0


def _hard_disk_relax(centers, max_iters=500, dt=0.005):
    """
    Simple hard-disk relaxation: push overlapping circles apart.
    This helps escape from infeasible configurations.
    """
    c = centers.copy()
    for it in range(max_iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                # Current radius estimate
                wall_r = np.min(0.5 - np.abs(c))
                min_pair = np.min(np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2)) if False else 0)
                # Just use a fixed target based on current best estimate
                r_est = _radius(c)
                target = 2 * r_est + 0.001
                if dist < target and dist > 1e-10:
                    overlap = target - dist
                    push = d / dist * overlap * 0.5
                    c[i] += push
                    c[j] -= push
                    moved = True
            # Wall push
            for k in range(2):
                if c[i, k] > 0.5 - r_est:
                    c[i, k] = 0.5 - r_est
                    moved = True
                elif c[i, k] < -0.5 + r_est:
                    c[i, k] = -0.5 + r_est
                    moved = True
        
        if not moved:
            break
    
    # Clip to bounds
    c = np.clip(c, -0.5, 0.5)
    return c


def _init_grid_7x7():
    """7x7 grid (49 points) plus one extra."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add 50th point at center offset
    grid = np.vstack([grid, [0.0, 0.0]])
    # Jitter slightly
    np.random.seed(42)
    grid += np.random.uniform(-0.01, 0.01, size=(N, 2))
    return grid[:N]


def _init_hexagonal():
    """Hexagonal lattice."""
    points = []
    spacing = 0.14
    dy = spacing * np.sqrt(3) / 2
    for j in range(8):
        for i in range(8):
            x = i * spacing - 0.5 + spacing / 2
            y = j * dy - 0.5 + dy / 2
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    # Center and scale
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points *= 0.45 / max_ext
    # Jitter
    np.random.seed(123)
    points += np.random.uniform(-0.005, 0.005, size=(N, 2))
    return points[:N]


def _init_grid_8x8():
    """8x8 grid (64 points), take first 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    np.random.seed(77)
    grid += np.random.uniform(-0.008, 0.008, size=(N, 2))
    return grid[:N]


def _init_random(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def construct():
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_hexagonal(),
        _init_grid_8x8(),
        _init_random(1),
        _init_random(2),
        _init_random(3),
    ]
    
    # Phase 1: Hard-disk relaxation + SLSQP polish on each init
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Relax
        relaxed = _hard_disk_relax(init, max_iters=200)
        r_relax = _radius(relaxed)
        
        # SLSQP polish
        time_left = DEADLINE - 10 - (time.time() - start_time)
        alloc = min(10.0, time_left / max(1, len(inits) - i))
        polished, r_polish = _slsqp_polish(relaxed, r_relax, time_limit=alloc)
        r_final = _radius(polished)
        
        if r_final > best_r:
            best_r = r_final
            best_centers = polished.copy()
    
    # Phase 2: Additional random restarts with SLSQP
    for seed in range(10, 25):
        if time.time() - start_time > DEADLINE - 8:
            break
        
        init = _init_random(seed)
        relaxed = _hard_disk_relax(init, max_iters=100)
        r_relax = _radius(relaxed)
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        alloc = min(6.0, time_left / max(1, 15))
        polished, r_polish = _slsqp_polish(relaxed, r_relax, time_limit=alloc)
        r_final = _radius(polished)
        
        if r_final > best_r:
            best_r = r_final
            best_centers = polished.copy()
    
    # Phase 3: Final polish on the best with more iterations
    if best_centers is not None and time.time() - start_time < DEADLINE - 3:
        time_left = DEADLINE - 3 - (time.time() - start_time)
        final_polished, r_final = _slsqp_polish(best_centers, best_r, time_limit=min(time_left, 10.0))
        r_check = _radius(final_polished)
        if r_check > best_r:
            best_r = r_check
            best_centers = final_polished.copy()
    
    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid_7x7()
    
    # Final clip to ensure inside square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

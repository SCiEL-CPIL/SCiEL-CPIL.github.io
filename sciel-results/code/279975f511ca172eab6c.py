"""Pack 50 equal circles in a unit square using Delaunay-based Contact Graph Optimization."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _get_delaunay_edges(centers):
    """Get edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
        return list(edges)
    except Exception:
        # Fallback to k-nearest neighbors if Delaunay fails (e.g., collinear points)
        edges = set()
        for i in range(N):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            nn = np.argsort(dists)[:5]
            for j in nn:
                edges.add(tuple(sorted((i, j))))
        return list(edges)

def _expand_radius(centers, r_start, r_target, time_limit=5.0):
    """
    Incrementally increase radius from r_start to r_target.
    At each step, resolve collisions using local gradient descent on the contact graph.
    """
    centers = centers.copy()
    start_time = time.time()
    
    # Step size for radius increase
    dr = 0.001
    current_r = r_start
    
    while current_r < r_target and time.time() - start_time < time_limit:
        # Get contact graph
        edges = _get_delaunay_edges(centers)
        
        # Check for violations
        violations = []
        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2 * current_r:
                violations.append((i, j, d))
        
        # Check wall violations
        for i in range(N):
            if centers[i, 0] < -0.5 + current_r or centers[i, 0] > 0.5 - current_r or \
               centers[i, 1] < -0.5 + current_r or centers[i, 1] > 0.5 - current_r:
                # Move towards center
                centers[i, 0] = np.clip(centers[i, 0], -0.5 + current_r, 0.5 - current_r)
                centers[i, 1] = np.clip(centers[i, 1], -0.5 + current_r, 0.5 - current_r)
        
        if not violations:
            current_r += dr
            continue
        
        # Resolve violations
        # Use a simple iterative solver
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            moved = False
            for i, j, d in violations:
                if d < 1e-8:
                    # Random nudge
                    centers[i] += np.random.normal(0, 0.01, 2)
                    centers[j] -= np.random.normal(0, 0.01, 2)
                    moved = True
                    continue
                
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * current_r:
                    # Push apart
                    push = (2 * current_r - dist) / 2.0
                    unit = diff / dist
                    centers[i] += push * unit
                    centers[j] -= push * unit
                    moved = True
            
            # Re-check walls
            for i in range(N):
                centers[i, 0] = np.clip(centers[i, 0], -0.5 + current_r, 0.5 - current_r)
                centers[i, 1] = np.clip(centers[i, 1], -0.5 + current_r, 0.5 - current_r)
            
            if not moved:
                break
        
        # If we couldn't resolve, try smaller step
        if len(violations) > 0:
            current_r = max(r_start, current_r - 0.0005)
        else:
            current_r += dr
    
    return centers

def _slsqp_polish(centers, time_limit=10.0):
    """
    Polish the configuration using SLSQP to maximize the minimum distance.
    We maximize r subject to:
    1. |c_i - c_j| >= 2r for all i < j
    2. -0.5 + r <= c_ik <= 0.5 - r for all i, k
    """
    x0 = centers.ravel()
    start_time = time.time()
    
    def objective(x):
        return -x[-1]  # Maximize r (last variable)
    
    constraints = []
    
    # Pairwise distance constraints
    # |c_i - c_j|^2 >= (2r)^2
    # (x[2i] - x[2j])^2 + (x[2i+1] - x[2j+1])^2 - 4*x[-1]^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def cons(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                r = x[-1]
                return dx*dx + dy*dy - 4*r*r
            constraints.append({'type': 'ineq', 'fun': cons})
    
    # Wall constraints
    # x[2i] >= -0.5 + r  => x[2i] - x[-1] >= -0.5
    # x[2i] <= 0.5 - r   => -x[2i] - x[-1] >= -0.5
    # x[2i+1] >= -0.5 + r => x[2i+1] - x[-1] >= -0.5
    # x[2i+1] <= 0.5 - r  => -x[2i+1] - x[-1] >= -0.5
    for i in range(N):
        def cons_x_left(x, i=i):
            return x[2*i] - x[-1] + 0.5
        def cons_x_right(x, i=i):
            return -x[2*i] - x[-1] + 0.5
        def cons_y_left(x, i=i):
            return x[2*i+1] - x[-1] + 0.5
        def cons_y_right(x, i=i):
            return -x[2*i+1] - x[-1] + 0.5
        constraints.append({'type': 'ineq', 'fun': cons_x_left})
        constraints.append({'type': 'ineq', 'fun': cons_x_right})
        constraints.append({'type': 'ineq', 'fun': cons_y_left})
        constraints.append({'type': 'ineq', 'fun': cons_y_right})
    
    # Initial r
    r0 = _radius(centers)
    x0[-1] = r0
    
    # Bounds for centers
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False},
            callback=callback
        )
        if res.success or res.x is not None:
            new_centers = res.x[:2*N].reshape(N, 2)
            return new_centers
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _make_hexagonal_grid(n):
    """Create a hexagonal grid of n points centered in the square."""
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    for i in range(8):
        y = -0.5 + (i + 0.5) * dy
        for j in range(7):
            x = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter to square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            p = np.random.uniform(-0.4, 0.4, 2)
            points = np.vstack([points, p])
            
    return points

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal grid
    hex_centers = _make_hexagonal_grid(N)
    if hex_centers is not None and len(hex_centers) == N:
        candidates.append(hex_centers)
        
    # 2. Square grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    sq_centers = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    candidates.append(sq_centers)
    
    # 3. Random starts
    for i in range(3):
        c_rand = np.random.uniform(-0.4, 0.4, size=(N, 2))
        candidates.append(c_rand)
        
    # Process each candidate
    for idx, centers in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        current_centers = centers.copy()
        
        # Step 1: Expand radius using Hard-Disk method
        r_init = _radius(current_centers)
        r_target = 0.0714  # Slightly above best known
        
        time_left_step = min(5.0, DEADLINE - 15 - (time.time() - start_time))
        if time_left_step <= 0: 
            break
            
        new_centers = _expand_radius(current_centers, r_init, r_target, time_limit=time_left_step)
        r_new = _radius(new_centers)
        
        if r_new > _radius(current_centers):
            current_centers = new_centers
            
        # Step 2: Polish with SLSQP
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 1.0:
            polished = _slsqp_polish(current_centers, time_limit=min(10.0, time_left))
            r_pol = _radius(polished)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished.copy()
                
    if best_centers is None:
        # Fallback
        best_centers = _make_hexagonal_grid(N)
        
    return best_centers

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = _radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r}")

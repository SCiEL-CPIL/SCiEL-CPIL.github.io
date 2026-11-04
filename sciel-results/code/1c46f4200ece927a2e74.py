"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3.0)
TRI_AREA = S3 / 4.0  # area of equilateral triangle with side 1

def _min_area(pts):
    """Compute the minimum triangle area among all C(N,3) triples."""
    n = len(pts)
    min_a = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_a:
                    min_a = area
    return min_a

def _objective(x):
    """Negative of min area (to maximize via minimization)."""
    pts = x.reshape(-1, 2)
    return -_min_area(pts)

def _penalty(x):
    """Penalty for points outside the triangle."""
    pts = x.reshape(-1, 2)
    penalty = 0.0
    for p in pts:
        x_c, y_c = p
        # Check if point is inside the equilateral triangle
        # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
        # Half-planes:
        # 1. y >= 0
        # 2. y <= sqrt(3)*x
        # 3. y <= -sqrt(3)*(x-1)
        if y_c < -1e-9:
            penalty += (-y_c) ** 2
        if y_c > S3 * x_c + 1e-9:
            penalty += (y_c - S3 * x_c) ** 2
        if y_c > -S3 * (x_c - 1) + 1e-9:
            penalty += (y_c + S3 * (x_c - 1)) ** 2
    return penalty

def _constrained_objective(x):
    """Objective with penalty for being outside the triangle."""
    base = _objective(x)
    pen = _penalty(x)
    return base + 100.0 * pen

def _project_to_triangle(pts):
    """Project points onto the triangle if they are outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x_c, y_c = pts[i]
        # Simple projection: clamp to nearest edge if outside
        # This is a rough projection; better to use proper constrained optimization
        # For now, just clip to the bounding box and hope for the best
        # Actually, let's do a proper projection using the half-plane constraints
        # This is complex, so let's just use penalty method in optimization
        pass
    return pts

def _random_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    while True:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            x = 0.5 * u + 0.5 * v
            y = (S3 / 2.0) * v
            return np.array([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Initial points: use a grid-like pattern inside the triangle
    # Place points at vertices and centroid-like positions
    # For 11 points, we can use a hexagonal packing approximation
    points = []
    
    # Place some points near vertices and edges
    # Vertex 1: (0,0), Vertex 2: (1,0), Vertex 3: (0.5, sqrt(3)/2)
    # Centroid: (0.5, sqrt(3)/6)
    
    # Use a structured initial guess
    # 3 vertices (slightly inset), 3 edge midpoints (slightly inset), 5 interior
    eps = 0.02
    points.append([eps, eps])
    points.append([1 - eps, eps])
    points.append([0.5, S3/2 - eps * S3])
    # Edge midpoints (inset)
    points.append([0.5, eps])
    points.append([0.25 + eps*0.5, S3/4 + eps*0.5])  # midpoint of left edge
    points.append([0.75 - eps*0.5, S3/4 + eps*0.5])  # midpoint of right edge
    # Interior points - hexagonal pattern
    cx, cy = 0.5, S3/6
    r = 0.15
    for angle in [0, 60, 120, 180, 240, 300]:
        rad = np.radians(angle)
        px = cx + r * np.cos(rad)
        py = cy + r * np.sin(rad)
        if len(points) < N:
            points.append([px, py])
    
    points = np.array(points[:N])
    
    # Ensure all points are inside
    best_pts = points.copy()
    best_val = _min_area(best_pts)
    
    # Differential Evolution for global search
    bounds = []
    for i in range(N):
        bounds.append([(0, 0), (1, 1)])  # Rough bounds, will use penalty
    
    # Use a tighter bounding box
    bounds = [(0, 1), (0, S3/2)] * N
    
    try:
        # First pass: differential evolution
        de_result = differential_evolution(
            _constrained_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-4,
            popsize=15,
            polish=False,
            init='latinhypercube',
            update_interval=1,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            de_pts = de_result.x.reshape(-1, 2)
            de_val = _min_area(de_pts)
            if de_val > best_val:
                best_pts = de_pts.copy()
                best_val = de_val
            
            # Nelder-Mead polish from DE result
            nm_result = minimize(
                _constrained_objective,
                de_result.x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            elapsed = time.time() - start_time
            if elapsed < deadline - 10:
                nm_pts = nm_result.x.reshape(-1, 2)
                nm_val = _min_area(nm_pts)
                if nm_val > best_val:
                    best_pts = nm_pts.copy()
                    best_val = nm_val
    except Exception:
        pass
    
    # Multiple restarts with Nelder-Mead from random starts
    elapsed = time.time() - start_time
    restarts_done = 0
    while elapsed < deadline - 15 and restarts_done < 20:
        # Generate random initial points
        pts = np.array([_random_in_triangle(np.random.RandomState(rng.randint(0, 1000000))) for _ in range(N)])
        
        nm_result = minimize(
            _constrained_objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        
        elapsed = time.time() - start_time
        if elapsed >= deadline - 5:
            break
        
        nm_pts = nm_result.x.reshape(-1, 2)
        nm_val = _min_area(nm_pts)
        if nm_val > best_val:
            best_pts = nm_pts.copy()
            best_val = nm_val
        
        restarts_done += 1
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        nm_result = minimize(
            _constrained_objective,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        final_pts = nm_result.x.reshape(-1, 2)
        final_val = _min_area(final_pts)
        if final_val > best_val:
            best_pts = final_pts.copy()
            best_val = final_val
    
    # Ensure points are inside the triangle (final check)
    for i in range(N):
        x_c, y_c = best_pts[i]
        # If slightly outside due to numerical issues, project inward
        if y_c < 0:
            best_pts[i, 1] = 0.001
        if y_c > S3 * x_c:
            # Project onto left edge
            t = y_c / S3
            if t > 1:
                t = 1
            best_pts[i, 0] = t
            best_pts[i, 1] = S3 * t
        if y_c > -S3 * (x_c - 1):
            # Project onto right edge
            t = (1 - x_c)
            if t > 1:
                t = 1
            best_pts[i, 0] = 1 - t
            best_pts[i, 1] = S3 * t
    
    return best_pts

import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # Distance from origin
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return -1.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    return min(min_boundary, min_pair)


def _initial_hexagonal():
    """Generate an initial packing based on a hexagonal grid clipped to the unit circle."""
    points = []
    # Hexagonal spacing. If r is the circle radius, spacing is ~2r.
    # We want to fit ~50 circles. Area of unit circle is pi. Area of 50 circles is 50*pi*r^2.
    # 50*r^2 < 1 => r < 1/sqrt(50) ~ 0.141.
    # So spacing ~ 2*0.12 = 0.24.
    spacing = 0.24
    # Generate a grid of hexagonal points
    # Hexagonal lattice vectors:
    # v1 = (spacing, 0)
    # v2 = (spacing/2, spacing*sqrt(3)/2)
    
    # We will iterate over a bounding box
    # Max index: 1 / spacing ~ 4. So -5 to 5 should cover it.
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (0.5 * spacing if j % 2 != 0 else 0)
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x*x + y*y) <= 1.0 - spacing/2.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, we need to select N.
    # If we have fewer, we need to add random points or adjust spacing.
    # For N=50, spacing 0.24 usually gives around 50-60 points.
    if len(points) > N:
        # Select the N points closest to the center? Or random?
        # Closest to center is usually better for packing density near boundary.
        dists = np.sum(points**2, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:N]]
    elif len(points) < N:
        # Add random points inside the circle
        while len(points) < N:
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            points = np.vstack([points, [r * np.cos(theta), r * np.sin(theta)]])
            
    return points

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # Initial guess: Hexagonal grid
    init_points = _initial_hexagonal()
    update_best(init_points)

    def objective(vec):
        if time.time() - start_time > deadline:
            return 1e10
        pts = vec.reshape(N, 2)
        s = _score(pts)
        return -s

    # Multi-start Nelder-Mead
    # We will run several iterations, each time perturbing the best solution found so far
    current_best = best_points.copy()
    
    # Number of restarts depends on time. Let's aim for ~5-10 restarts if fast, or 1-2 if slow.
    # Nelder-Mead on 100D is slow. Let's limit maxiter.
    
    while time.time() - start_time < deadline - 10:
        # Add noise to current best to escape local minima
        # Noise scale should be small, maybe 0.05
        noise = np.random.normal(0, 0.05, (N, 2))
        x0 = current_best + noise
        
        # Ensure points are inside the circle (roughly) to help optimizer
        # Clip distances to 0.95
        dists = np.sqrt(np.sum(x0**2, axis=1))
        mask = dists > 0.95
        if np.any(mask):
            x0[mask] = x0[mask] * (0.95 / dists[mask])
            
        try:
            # Use Nelder-Mead with adaptive step
            # maxiter needs to be small enough to allow restarts
            # 100D problem, 2000 iterations might take ~1-2 seconds depending on machine.
            # Let's try 5000 iterations with adaptive.
            result = minimize(objective, x0.flatten(), method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result.fun < 0: # Valid score
                pts_opt = result.x.reshape(N, 2)
                s = update_best(pts_opt)
                if s > best_score - 1e-6: # If we improved or stayed same, update current_best
                    current_best = pts_opt.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = _initial_hexagonal()

    return best_points

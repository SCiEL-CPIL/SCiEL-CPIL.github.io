"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_score(pts):
    """Compute the packing radius for a set of points."""
    if len(pts) != N:
        return 0.0
    # Check all points are inside unit circle
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        return 0.0
    # Compute minimum pairwise distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    # Compute minimum distance to boundary
    min_boundary = 1.0 - np.max(norms)
    r = min(min_dist / 2.0, min_boundary)
    return r


def _initial_positions_hexagonal():
    """Generate initial positions using hexagonal lattice."""
    # Use a hexagonal packing pattern
    points = []
    # Start with a central point and rings
    # Ring 0: center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        points.append([0.3 * np.cos(angle), 0.3 * np.sin(angle)])
    
    # Ring 2: 12 points
    for angle in np.linspace(0, 2*np.pi, 13)[:-1]:
        points.append([0.6 * np.cos(angle), 0.6 * np.sin(angle)])
    
    # Ring 3: 18 points
    for angle in np.linspace(0, 2*np.pi, 19)[:-1]:
        points.append([0.9 * np.cos(angle), 0.9 * np.sin(angle)])
    
    # We need exactly 50 points
    points = np.array(points[:N])
    return points


def _gradient_descent(pts, max_iter=10000, lr=0.01, beta=100):
    """Gradient descent to maximize the minimum radius."""
    pts = pts.copy()
    
    for iteration in range(max_iter):
        # Compute current radius
        norms = np.linalg.norm(pts, axis=1)
        min_boundary = 1.0 - np.max(norms)
        
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        
        r = min(min_dist / 2.0, min_boundary)
        
        # Compute gradient of the objective
        # We want to maximize r = min(min_dist/2, min_boundary)
        # Use soft-min approximation
        grad = np.zeros_like(pts)
        
        # Gradient contribution from boundary constraint
        if min_boundary < min_dist / 2.0:
            # Push points inward
            for i in range(N):
                norm_i = np.linalg.norm(pts[i])
                if norm_i > 1e-10:
                    grad[i] -= (pts[i] / norm_i) * lr
                else:
                    grad[i] -= np.array([lr, 0.0])
        
        # Gradient contribution from pairwise distance
        if min_dist / 2.0 < min_boundary:
            i, j = min_pair
            diff = pts[i] - pts[j]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-10:
                unit_diff = diff / norm_diff
                grad[i] += unit_diff * lr
                grad[j] -= unit_diff * lr
        
        # Update positions
        pts += grad
        
        # Project onto feasible region (inside unit circle)
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts[i] = pts[i] / norms[i]
    
    return pts


def _simulated_annealing(pts, max_iter=50000, t_start=0.01, t_end=0.0001, cool_rate=0.9999):
    """Simulated annealing to find better packing."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    temperature = t_start
    
    for iteration in range(max_iter):
        # Generate perturbation
        idx = np.random.randint(N)
        perturbation = np.random.randn(2) * temperature
        new_pts = pts.copy()
        new_pts[idx] += perturbation
        
        # Check if new point is inside unit circle
        if np.linalg.norm(new_pts[idx]) > 1.0:
            continue
        
        new_score = _compute_score(new_pts)
        
        if new_score >= best_score:
            pts = new_pts
            best_pts = new_pts.copy()
            best_score = new_score
        elif np.random.rand() < np.exp((new_score - best_score) / temperature):
            pts = new_pts
        
        temperature = max(t_end, temperature * cool_rate)
    
    return best_pts


def _polish(pts, max_iter=5000, tol=1e-12):
    """High-precision polish phase."""
    pts = pts.copy()
    
    for iteration in range(max_iter):
        # Compute all pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    dists[i, j] = np.linalg.norm(pts[i] - pts[j])
        
        # Find minimum distance
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] < min_dist:
                    min_dist = dists[i, j]
                    min_pair = (i, j)
        
        # Compute boundary distances
        norms = np.linalg.norm(pts, axis=1)
        min_boundary = 1.0 - np.max(norms)
        
        r = min(min_dist / 2.0, min_boundary)
        
        # If we're very close to optimal, stop
        if min_dist > 2 * min_boundary + tol:
            # Boundary is the limiting factor, push all points slightly inward
            for i in range(N):
                norm_i = np.linalg.norm(pts[i])
                if norm_i > 1e-10:
                    pts[i] *= (1.0 - tol / 100)
            break
        
        if min_boundary > min_dist / 2.0 + tol:
            # Distance is the limiting factor, try to increase it
            i, j = min_pair
            diff = pts[i] - pts[j]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-10:
                unit_diff = diff / norm_diff
                step = tol / 100
                pts[i] += unit_diff * step
                pts[j] -= unit_diff * step
            break
        
        # Otherwise, we're balanced, do a small random perturbation
        idx = np.random.randint(N)
        pts[idx] += np.random.randn(2) * tol / 1000
        
        # Project onto feasible region
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts[i] = pts[i] / norms[i]
    
    return pts


def construct():
    """Main function to construct the circle packing."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Generate initial positions
    pts = _initial_positions_hexagonal()
    
    # Phase 1: Gradient descent
    pts = _gradient_descent(pts, max_iter=5000, lr=0.005)
    
    # Phase 2: Simulated annealing
    pts = _simulated_annealing(pts, max_iter=30000, t_start=0.005, t_end=0.00005)
    
    # Phase 3: High-precision polish
    if time.time() < deadline:
        pts = _polish(pts, max_iter=3000, tol=1e-12)
    
    # Final projection to ensure validity
    norms = np.linalg.norm(pts, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            pts[i] = pts[i] / norms[i]
    
    return pts
# EVOLVE-BLOCK-END

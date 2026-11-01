"""Pack 50 equal circles in a unit circle using simulated annealing with multi-restart."""
import numpy as np
import time
from math import pi, sqrt, cos, sin

N = 50
TARGET = 50  # number of circles

def _score(pts):
    """Compute the maximum radius r such that circles of radius r centered at pts
    are all inside the unit circle and non-overlapping.
    Returns (r, valid) where valid is False if any centre is outside or non-finite."""
    if len(pts) != TARGET:
        return 0.0, False
    if not np.all(np.isfinite(pts)):
        return 0.0, False
    # Distance from origin
    r_orig = np.sqrt(np.sum(pts**2, axis=1))
    if np.any(r_orig > 1.0):
        return 0.0, False
    # Min distance to boundary
    min_boundary = np.min(1.0 - r_orig)
    # Pairwise distances
    n = len(pts)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((pts[i] - pts[j])**2))
            if d < min_pair:
                min_pair = d
    r = min(min_boundary, min_pair / 2.0)
    return r, True


def _sa_optimize(pts, time_budget=30.0, T0=0.05, T_end=1e-6, alpha=0.999):
    """Simulated annealing to maximize the minimum radius."""
    pts = pts.copy()
    current_r, _ = _score(pts)
    best_pts = pts.copy()
    best_r = current_r
    T = T0
    step = 0.02
    n = len(pts)
    deadline = time.time() + time_budget
    
    while time.time() < deadline:
        # Pick a random point and perturb it
        idx = np.random.randint(n)
        old_pos = pts[idx].copy()
        # Random perturbation
        angle = np.random.uniform(0, 2 * pi)
        mag = np.random.uniform(0, step)
        pts[idx, 0] += mag * cos(angle)
        pts[idx, 1] += mag * sin(angle)
        
        new_r, valid = _score(pts)
        if not valid:
            pts[idx] = old_pos
            continue
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = pts.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            if np.random.random() < np.exp(delta / T):
                current_r = new_r
            else:
                pts[idx] = old_pos
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            # Reduce step size
            step *= 0.95
            if step < 0.001:
                break
    
    return best_pts, best_r


def _hexagonal_init(n_circles):
    """Generate initial positions on a hexagonal lattice pattern."""
    pts = []
    # Start with a dense hex lattice and pick points
    spacing = 0.25
    k = 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt(3) / 2
            if x*x + y*y <= 0.85:  # Keep inside with margin
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) < n_circles:
        # Add more points
        spacing = 0.2
        k = 4
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt(3) / 2
                if x*x + y*y <= 0.85:
                    pts = np.vstack([pts, [[x, y]]])
    
    # If still not enough, add random points
    while len(pts) < n_circles:
        angle = np.random.uniform(0, 2*pi)
        r = np.random.uniform(0, 0.8)
        pts = np.vstack([pts, [[r*cos(angle), r*sin(angle)]]])
    
    # Select n_circles points (try to maximize spacing)
    if len(pts) >= n_circles:
        # Simple selection: take first n_circles
        pts = pts[:n_circles]
    return pts


def _random_init(n_circles):
    """Generate random initial positions inside the unit circle."""
    angles = np.random.uniform(0, 2*pi, n_circles)
    radii = np.sqrt(np.random.uniform(0, 1, n_circles)) * 0.8
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _ring_init(n_circles):
    """Generate initial positions on concentric rings."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        angle = 2*pi*i/6
        pts.append([0.25*cos(angle), 0.25*sin(angle)])
    # Ring 2: 12 points
    for i in range(12):
        angle = 2*pi*i/12
        pts.append([0.5*cos(angle), 0.5*sin(angle)])
    # Ring 3: remaining
    remaining = n_circles - len(pts)
    if remaining > 0:
        for i in range(remaining):
            angle = 2*pi*i/remaining
            pts.append([0.75*cos(angle), 0.75*sin(angle)])
    return np.array(pts[:n_circles])


def construct():
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple initializations and SA runs
    initializations = [
        _hexagonal_init,
        _ring_init,
        _random_init
    ]
    
    run_count = 0
    for init_func in initializations:
        for _ in range(3):  # 3 runs per initialization
            if time.time() > deadline - 35.0:  # Need at least 35s for final SA
                break
            
            pts = init_func(N)
            # Quick SA optimization
            pts, r = _sa_optimize(pts, time_budget=20.0, T0=0.05, T_end=1e-6, alpha=0.9995)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
            run_count += 1
    
    # Final polish with longer SA on the best solution
    if best_pts is not None and time.time() < deadline - 5.0:
        remaining_time = min(25.0, deadline - time.time() - 2.0)
        if remaining_time > 5.0:
            best_pts, best_r = _sa_optimize(best_pts, time_budget=remaining_time, T0=0.02, T_end=1e-7, alpha=0.9998)
    
    # Ensure validity
    if best_pts is None:
        # Fallback: return a valid configuration
        best_pts = _hexagonal_init(N)
    
    r, valid = _score(best_pts)
    if not valid:
        # Last resort: return points well inside
        angles = np.linspace(0, 2*pi, N, endpoint=False)
        radii = np.full(N, 0.5)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Adjust to ensure no overlaps
        # For now, just return as is; the score will be computed by evaluator
    
    return best_pts

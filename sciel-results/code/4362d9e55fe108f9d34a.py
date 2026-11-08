"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Triangle vertices for barycentric coordinate conversion
_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _H]])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Constraints:
      y >= 0
      y <= sqrt(3) * x        (left edge from origin to top)
      y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)  (right edge)
    """
    pts = pts.copy()
    # Clip y >= 0
    pts[pts[:, 1] < 0, 1] = 0.0
    # Clip to left edge: y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    min_x_left = pts[:, 1] / np.sqrt(3)
    mask_left = pts[:, 0] < min_x_left
    pts[mask_left, 0] = min_x_left[mask_left]
    # Clip to right edge: y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    max_x_right = 1.0 - pts[:, 1] / np.sqrt(3)
    mask_right = pts[:, 0] > max_x_right
    pts[mask_right, 0] = max_x_right[mask_right]
    # Final clip for numerical safety
    pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1)
    return pts


def _objective(x):
    """Negative min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Create a symmetric initial configuration for 11 points.
    Uses a pattern inspired by known optimal/near-optimal Heilbronn configurations.
    Place points with 3-fold rotational symmetry where possible.
    """
    # Use barycentric coordinates for better symmetry
    # 11 points: 3 at vertices, 3 at edge midpoints, 4 interior
    # Actually, let's use a more balanced arrangement
    
    # Center point
    cx, cy = 0.5, _H / 3  # centroid of the triangle
    
    # 6 points around center at radius r (hexagonal)
    r = 0.22
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [[cx, cy]]
    for a in angles:
        px = cx + r * np.cos(a)
        py = cy + r * np.sin(a)
        pts.append([px, py])
    # 4 more points near the corners but not too close
    pts.append([0.12, 0.08])
    pts.append([0.88, 0.08])
    pts.append([0.5, 0.65])
    pts.append([0.32, 0.32])
    
    arr = np.array(pts[:N], dtype=float)
    return _project_to_triangle(arr).flatten()


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u,v,w) with u+v+w=1 to Cartesian."""
    x = bary[:, 0] * 0.0 + bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 0] * 0.0 + bary[:, 1] * 0.0 + bary[:, 2] * _H
    return np.column_stack([x, y])


def _generate_structured_start(seed):
    """Generate structured initial configurations using barycentric coordinates."""
    rng = np.random.RandomState(seed)
    # Generate 11 points in barycentric space (u,v,w >= 0, u+v+w=1)
    # Use Dirichlet distribution for diversity
    alpha = rng.uniform(0.5, 2.0, size=3)
    bary = rng.dirichlet(alpha, size=N)
    pts = _barycentric_to_cartesian(bary)
    return _project_to_triangle(pts).flatten()


def _random_initial_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        while u + v > 1.0:
            u, v = rng.random(), rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    arr = np.array(pts, dtype=float)
    return _project_to_triangle(arr).flatten()


def _simulated_annealing(x0, max_time=5.0, T0=0.01, cooling=0.95, rng=None):
    """Simulated annealing to escape local optima."""
    if rng is None:
        rng = np.random.RandomState()
    
    start = time.time()
    current_x = x0.copy()
    current_val = _objective(current_x)
    best_x = current_x.copy()
    best_val = current_val
    T = T0
    
    while time.time() < start + max_time and T > 1e-10:
        # Perturb: move a random point by a small random amount
        idx = rng.randint(N)
        delta = rng.normal(0, T, size=2)
        new_x = current_x.copy()
        new_x[2*idx] += delta[0]
        new_x[2*idx+1] += delta[1]
        
        new_pts = _project_to_triangle(new_x.reshape(N, 2))
        new_val = _objective(new_pts.flatten())
        
        # Accept if better or with probability exp(-delta/T)
        if new_val < current_val:
            current_x = new_x
            current_val = new_val
            if new_val < best_val:
                best_x = new_x
                best_val = new_val
        else:
            prob = np.exp((current_val - new_val) / T)
            if rng.random() < prob:
                current_x = new_x
                current_val = new_val
        
        T *= cooling
    
    return best_x, best_val


def _local_optimize(x0, max_time=10.0, use_sa=True):
    """Run SLSQP local optimization from x0, optionally followed by simulated annealing.
    Returns (best_x, best_val)."""
    start = time.time()
    
    # Use SLSQP with bounds to keep points in triangle
    bounds = []
    for i in range(N):
        bounds.append((0, 1))   # x
        bounds.append((0, _H))  # y
    
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    try:
        res = minimize(
            _objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Simulated annealing to escape local optima
    if use_sa and time.time() < start + max_time - 1.0:
        sa_time = min(3.0, max_time - (time.time() - start) - 1.0)
        if sa_time > 0.5:
            sa_x, sa_val = _simulated_annealing(
                best_x, max_time=sa_time, T0=0.008, cooling=0.97,
                rng=np.random.RandomState(int(time.time() * 1000) % 100000)
            )
            if sa_val < best_val:
                best_x = sa_x
                best_val = sa_val
    
    elapsed = time.time() - start
    return best_x, best_val, elapsed


def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Multi-start optimization
    best_x = None
    best_val = np.inf
    
    # Start 1: symmetric configuration
    x0 = _symmetric_initial_config()
    x, val, _ = _local_optimize(x0, max_time=12.0, use_sa=True)
    best_x, best_val = x, val
    
    if time.time() > deadline - 30:
        return _project_to_triangle(best_x.reshape(N, 2))
    
    # Additional structured starts using barycentric coordinates
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        x0 = _generate_structured_start(seed + 500)
        x, val, _ = _local_optimize(x0, max_time=8.0, use_sa=True)
        if val < best_val:
            best_x, best_val = x, val
    
    # Additional random starts
    for seed in range(30):
        if time.time() > deadline - 15:
            break
        x0 = _random_initial_config(seed + 100)
        x, val, _ = _local_optimize(x0, max_time=6.0, use_sa=True)
        if val < best_val:
            best_x, best_val = x, val
    
    # Final high-precision polish with tighter tolerance
    if time.time() < deadline - 8:
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, _H))
        try:
            # First pass with moderate precision
            res = minimize(
                _objective,
                best_x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-13, 'disp': False}
            )
            if res.success or res.fun < best_val:
                best_x = res.x
                best_val = res.fun
            
            # Second pass with very tight tolerance
            if time.time() < deadline - 4:
                res2 = minimize(
                    _objective,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
                )
                if res2.success or res2.fun < best_val:
                    best_x = res2.x
                    best_val = res2.fun
        except Exception:
            pass
    
    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts

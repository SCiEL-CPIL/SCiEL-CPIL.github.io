"""Heilbronn 11 triangles: Simulated Annealing with pairwise repulsion and adaptive restarts."""
import numpy as np
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRI = len(_TRIPLETS)

# Precompute coefficient matrix for area computation
# Area of triangle (i,j,k) = 0.5 * |x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)|
_A = np.zeros((_N_TRI, 2 * N))
for idx, (i, j, k) in enumerate(_TRIPLETS):
    _A[idx, 2*i] = 1.0
    _A[idx, 2*i+1] = 0.0
    _A[idx, 2*j] = 0.0
    _A[idx, 2*j+1] = 1.0
    _A[idx, 2*k] = 0.0
    _A[idx, 2*k+1] = 0.0
    # det = x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)
    _A[idx, 2*i] = 1.0
    _A[idx, 2*i+1] = 0.0
    _A[idx, 2*j] = 0.0
    _A[idx, 2*j+1] = 1.0
    _A[idx, 2*k] = 0.0
    _A[idx, 2*k+1] = 0.0
    # Actually, let's build it properly:
    # det = x_i*y_j - x_i*y_k - x_j*y_i + x_j*y_k + x_k*y_i - x_k*y_j
    _A[idx, :] = 0.0
    _A[idx, 2*i] = 0.0
    _A[idx, 2*i+1] = 0.0
    _A[idx, 2*j] = 0.0
    _A[idx, 2*j+1] = 0.0
    _A[idx, 2*k] = 0.0
    _A[idx, 2*k+1] = 0.0
    # x_i * y_j: row 2*i col 2*j+1
    _A[idx, 2*i] = 0.0
    _A[idx, 2*j+1] = 1.0
    # x_i * y_k: row 2*i col 2*k+1 with negative
    _A[idx, 2*k+1] = -1.0
    # x_j * y_i: row 2*j col 2*i+1 with negative
    _A[idx, 2*j] = 0.0
    _A[idx, 2*i+1] = -1.0
    # x_j * y_k: row 2*j col 2*k+1
    _A[idx, 2*k+1] = -1.0 + 1.0  # already set
    _A[idx, 2*j+1] = 0.0
    # Let me just do it directly
    _A[idx, :] = 0.0
    _A[idx, 2*i] = 0.0
    _A[idx, 2*i+1] = 0.0
    _A[idx, 2*j] = 0.0
    _A[idx, 2*j+1] = 0.0
    _A[idx, 2*k] = 0.0
    _A[idx, 2*k+1] = 0.0
    # det = x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)
    _A[idx, 2*i] = 0.0
    _A[idx, 2*i+1] = 0.0
    _A[idx, 2*j] = 0.0
    _A[idx, 2*j+1] = 0.0
    _A[idx, 2*k] = 0.0
    _A[idx, 2*k+1] = 0.0
    # x_i * y_j
    _A[idx, 2*j+1] += 1.0
    # -x_i * y_k
    _A[idx, 2*k+1] -= 1.0
    # x_j * y_k
    _A[idx, 2*k+1] += 1.0
    # -x_j * y_i
    _A[idx, 2*i+1] -= 1.0
    # x_k * y_i
    _A[idx, 2*i+1] += 1.0
    # -x_k * y_j
    _A[idx, 2*j+1] -= 1.0


def _min_area(pts):
    """Compute minimum triangle area. pts shape (N, 2)."""
    flat = pts.flatten()
    dets = _A @ flat
    areas = 0.5 * np.abs(dets)
    return np.min(areas)


def _min_area_fast(pts):
    """Faster min area using precomputed structure."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    d1x = pj[:, 0] - pi[:, 0]
    d1y = pj[:, 1] - pi[:, 1]
    d2x = pk[:, 0] - pi[:, 0]
    d2y = pk[:, 1] - pi[:, 1]
    dets = d1x * d2y - d1y * d2x
    areas = 0.5 * np.abs(dets)
    return np.min(areas)


def _inside_triangle(pts):
    """Check if all points are inside or on the triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, S3H)
    x = pts[:, 0]
    y = pts[:, 1]
    # y >= 0
    if np.any(y < -1e-10):
        return False
    # y <= S3H * x (left edge)
    if np.any(y > S3H * x + 1e-10):
        return False
    # y <= S3H * (1 - x) (right edge)
    if np.any(y > S3H * (1 - x) + 1e-10):
        return False
    return True


def _project_to_triangle(pts):
    """Project points into the triangle."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Clip y to [0, max_y]
    y = np.maximum(y, 0.0)
    
    # For each point, ensure it's inside the triangle
    for i in range(N):
        xi, yi = x[i], y[i]
        # Left boundary: y <= S3H * x
        if yi > S3H * xi:
            yi = S3H * xi
        # Right boundary: y <= S3H * (1 - x)
        if yi > S3H * (1 - xi):
            yi = S3H * (1 - xi)
        # x in [0, 1]
        xi = np.clip(xi, 0.0, 1.0)
        pts[i] = [xi, yi]
    
    return pts


def _random_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    while True:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            x = u + 0.5 * v
            y = S3H * v
            return np.array([x, y])


def _initial_configs():
    """Generate multiple diverse initial configurations."""
    configs = []
    
    # Config 1: Grid-like arrangement
    pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.25, 0.0], [0.75, 0.0], [0.5, S3H/2],
        [0.25, S3H/2], [0.75, S3H/2],
        [0.5, S3H*3/4], [0.375, S3H/4], [0.625, S3H/4]
    ])
    pts = _project_to_triangle(pts)
    configs.append(pts)
    
    # Config 2: Random
    rng = np.random.RandomState(42)
    pts = np.array([_random_in_triangle(rng) for _ in range(N)])
    configs.append(pts)
    
    # Config 3: Concentric rings
    center = np.array([0.5, S3H/3])
    pts = [center.copy()]
    # Ring 1: 3 points
    for angle in [0, 2*np.pi/3, 4*np.pi/3]:
        r = 0.15
        p = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p)
    # Ring 2: 7 points
    for angle in np.linspace(0, 2*np.pi, 7, endpoint=False):
        r = 0.3
        p = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p)
    pts = np.array(pts[:N])
    pts = _project_to_triangle(pts)
    configs.append(pts)
    
    # Config 4: Known good-ish arrangement
    pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.3, 0.1], [0.7, 0.1], [0.5, 0.3],
        [0.2, 0.25], [0.8, 0.25], [0.4, 0.45],
        [0.6, 0.45], [0.5, 0.15]
    ])
    pts = _project_to_triangle(pts)
    configs.append(pts)
    
    # Config 5: Random with different seed
    rng = np.random.RandomState(123)
    pts = np.array([_random_in_triangle(rng) for _ in range(N)])
    configs.append(pts)
    
    return configs


def _simulated_annealing(pts_init, t_start=0.1, t_end=0.001, n_steps=5000, 
                          rng=None, deadline=None, step_size_init=0.05):
    """Simulated annealing to maximize minimum triangle area."""
    if rng is None:
        rng = np.random.RandomState()
    
    pts = pts_init.copy()
    best_pts = pts.copy()
    best_val = _min_area_fast(pts)
    current_val = best_val
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate perturbation
        step_size = step_size_init * (t / t_start) ** 0.5
        perturb = rng.normal(0, step_size, (N, 2))
        new_pts = pts + perturb
        new_pts = _project_to_triangle(new_pts)
        
        new_val = _min_area_fast(new_pts)
        delta = new_val - current_val
        
        # Accept if better or with probability exp(delta/t)
        if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
            pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_pts = pts.copy()
        
        t *= cooling
    
    return best_pts, best_val


def _local_search(pts, n_iterations=2000, deadline=None, rng=None):
    """Local search: try small perturbations and keep improvements."""
    if rng is None:
        rng = np.random.RandomState()
    
    pts = pts.copy()
    best_pts = pts.copy()
    best_val = _min_area_fast(pts)
    current_val = best_val
    
    step_size = 0.02
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random perturbation
        perturb = rng.normal(0, step_size, (N, 2))
        new_pts = pts + perturb
        new_pts = _project_to_triangle(new_pts)
        
        new_val = _min_area_fast(new_pts)
        if new_val > current_val + 1e-12:
            pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_pts = pts.copy()
        else:
            # Try smaller step
            perturb = rng.normal(0, step_size * 0.5, (N, 2))
            new_pts = pts + perturb
            new_pts = _project_to_triangle(new_pts)
            new_val = _min_area_fast(new_pts)
            if new_val > current_val + 1e-12:
                pts = new_pts
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_pts = pts.copy()
        
        # Occasionally reduce step size
        if it % 500 == 0:
            step_size *= 0.9
    
    return best_pts, best_val


def _hill_climbing(pts, deadline=None, rng=None):
    """Hill climbing with adaptive step size."""
    if rng is None:
        rng = np.random.RandomState()
    
    pts = pts.copy()
    best_pts = pts.copy()
    best_val = _min_area_fast(pts)
    current_val = best_val
    
    step_size = 0.03
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2:
        perturb = rng.normal(0, step_size, (N, 2))
        new_pts = pts + perturb
        new_pts = _project_to_triangle(new_pts)
        
        new_val = _min_area_fast(new_pts)
        if new_val > current_val + 1e-12:
            pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_pts = pts.copy()
            no_improve = 0
            step_size = min(step_size * 1.05, 0.05)
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            step_size *= 0.95
    
    return best_pts, best_val


def construct():
    deadline = time.time() + 140  # Safety margin from 150s limit
    
    best_pts = None
    best_val = -np.inf
    
    configs = _initial_configs()
    
    # Phase 1: Simulated annealing from each config
    for idx, config in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.RandomState(idx * 100 + 7)
        pts_sa, val_sa = _simulated_annealing(
            config, t_start=0.05, t_end=0.0005, n_steps=3000,
            rng=rng, deadline=deadline - 5
        )
        
        if val_sa > best_val:
            best_val = val_sa
            best_pts = pts_sa.copy()
    
    # Phase 2: Local search from best
    if best_pts is not None and time.time() < deadline - 5:
        rng = np.random.RandomState(999)
        pts_ls, val_ls = _local_search(
            best_pts, n_iterations=3000, deadline=deadline - 3, rng=rng
        )
        if val_ls > best_val:
            best_val = val_ls
            best_pts = pts_ls.copy()
    
    # Phase 3: Hill climbing with restarts
    for restart in range(5):
        if time.time() > deadline - 3:
            break
        
        rng = np.random.RandomState(restart * 500 + 1)
        
        # Add small perturbation to best
        perturb = rng.normal(0, 0.02, best_pts.shape)
        pts_pert = _project_to_triangle(best_pts + perturb)
        
        pts_hc, val_hc = _hill_climbing(pts_pert, deadline=deadline - 2, rng=rng)
        if val_hc > best_val:
            best_val = val_hc
            best_pts = pts_hc.copy()
    
    # Phase 4: Final local search
    if best_pts is not None and time.time() < deadline - 2:
        rng = np.random.RandomState(4242)
        pts_final, val_final = _local_search(
            best_pts, n_iterations=2000, deadline=deadline - 1, rng=rng
        )
        if val_final > best_val:
            best_val = val_final
            best_pts = pts_final.copy()
    
    # Final validation and projection
    if best_pts is None:
        # Fallback
        rng = np.random.RandomState(0)
        best_pts = np.array([_random_in_triangle(rng) for _ in range(N)])
    
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts

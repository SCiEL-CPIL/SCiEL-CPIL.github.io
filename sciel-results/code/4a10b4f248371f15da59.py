"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 150s limit
DEADLINE_MARGIN = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _structured_initial():
    """Generate a structured initial configuration: vertices, edge midpoints, centroid."""
    # Vertices
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0.5, S3H])
    # Edge midpoints
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m02 = (v0 + v2) / 2
    # Centroid
    c = (v0 + v1 + v2) / 3
    # Additional points: midpoints of medians or other symmetric positions
    # Point at 1/3 along each median from vertex to opposite midpoint
    p_a = (2/3) * v0 + (1/3) * m01  # Actually let's use better symmetric points
    # Use points that divide edges in 1:2 ratio
    e1 = (2*v0 + v1) / 3
    e2 = (v0 + 2*v1) / 3
    e3 = (2*v1 + v2) / 3
    e4 = (v1 + 2*v2) / 3
    e5 = (2*v2 + v0) / 3
    e6 = (v2 + 2*v0) / 3
    
    # Select 11 points: 3 vertices + 3 edge midpoints + centroid + 4 more
    # For better symmetry, use: 3 vertices, 3 edge midpoints, centroid, and 4 points at 1/4 and 3/4 along edges
    q1 = (3*v0 + v1) / 4
    q2 = (v0 + 3*v1) / 4
    q3 = (3*v1 + v2) / 4
    q4 = (v1 + 3*v2) / 4
    
    pts = np.array([v0, v1, v2, m01, m12, m02, c, q1, q2, q3, q4])
    return pts

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _sa_step(current, current_score, temp, rng):
    """One step of simulated annealing: perturb and accept/reject."""
    # Perturb: move 1-3 random points by small random amount
    n_perturb = rng.randint(1, 4)
    idx = rng.choice(N, size=n_perturb, replace=False)
    noise = rng.normal(0, 0.05 * temp + 0.001, size=(n_perturb, 2))
    candidate = current.copy()
    candidate[idx] += noise
    candidate = _project_cartesian(candidate)
    
    new_score = _vectorized_min_area_cart(candidate)
    
    # Accept if better, or with probability exp((new - current)/temp)
    if new_score >= current_score:
        return candidate, new_score, True
    else:
        prob = np.exp((new_score - current_score) / (temp + 1e-10))
        if rng.random() < prob:
            return candidate, new_score, True
        else:
            return current, current_score, False

def _local_polish(points, max_iter=500):
    """Local polish using Nelder-Mead on the hard minimum."""
    def objective(flat):
        cart = flat.reshape(N, 2)
        cart = _project_cartesian(cart)
        return -_vectorized_min_area_cart(cart)
    
    try:
        res = minimize(objective, points.flatten(), method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        final = res.x.reshape(N, 2)
        final = _project_cartesian(final)
        return final, _vectorized_min_area_cart(final)
    except:
        return points, _vectorized_min_area_cart(points)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Start with structured initial configuration
    best_cart = _structured_initial()
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Simulated annealing phase
    T_start = 0.1
    T_end = 0.0001
    n_sa_steps = 5000
    
    current = best_cart.copy()
    current_score = best_score
    
    for i in range(n_sa_steps):
        if time.time() > deadline - 30:  # Leave 30s for polish
            break
        
        # Geometric cooling
        temp = T_start * (T_end / T_start) ** (i / n_sa_steps)
        
        candidate, cand_score, accepted = _sa_step(current, current_score, temp, rng)
        if accepted:
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best_score = current_score
                best_cart = current.copy()
    
    # Local polish from best configuration
    if time.time() < deadline - 10:
        polished, polish_score = _local_polish(best_cart, max_iter=1000)
        if polish_score > best_score:
            best_score = polish_score
            best_cart = polished
    
    # Try a few more polish attempts from slightly perturbed versions
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        perturbed = best_cart + rng.normal(0, 0.01, size=(N, 2))
        perturbed = _project_cartesian(perturbed)
        polished, polish_score = _local_polish(perturbed, max_iter=500)
        if polish_score > best_score:
            best_score = polish_score
            best_cart = polished
    
    return best_cart
# EVOLVE-BLOCK-END

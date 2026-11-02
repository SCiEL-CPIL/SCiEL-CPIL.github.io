"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Symmetry-constrained rubber-band model with Nelder-Mead on hard min-area.
- Fixed 3-row grid topology: 4-3-4 points
- Global affine parameters control row positions
- Small local perturbations break symmetry
- Nelder-Mead (derivative-free) directly on the non-smooth min objective
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

# Pre-compute triplet indices
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]


def _min_triangle_area(pts):
    """Compute the hard minimum triangle area."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _check_inside(pts):
    """Check all points are inside or on the boundary of the equilateral triangle."""
    eps = 1e-9
    for x, y in pts:
        if y < -eps or x < -eps or x > 1 + eps:
            return False
        if y > S3H * x + eps and y > S3H * (1 - x) + eps:
            # Check: y <= S3H*x (below left edge) and y <= S3H*(1-x) (below right edge)
            # Actually for triangle (0,0),(1,0),(0.5,S3H):
            # Left edge from (0,0) to (0.5,S3H): y = 2*S3H*x = sqrt(3)*x
            # Right edge from (1,0) to (0.5,S3H): y = sqrt(3)*(1-x)
            # Point is inside if y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
            if y > S3H * x + eps:
                return False
            if y > S3H * (1 - x) + eps:
                return False
    return True


def _rubber_band_positions(params):
    """
    Generate 11 points from a compact parameter vector.
    
    Topology: 3 rows
    Row 0 (bottom): 4 points
    Row 1 (middle): 3 points  
    Row 2 (top): 4 points
    
    Parameters (10 total):
    p[0]: vertical offset of row 0 (0 to 0.1)
    p[1]: vertical offset of row 1 (0.2 to 0.5)
    p[2]: vertical offset of row 2 (0.5 to 0.9)
    p[3]: horizontal spread factor (0.7 to 1.0)
    p[4]: shear factor (-0.1 to 0.1)
    p[5]: row 0 x-center offset (-0.05 to 0.05)
    p[6]: row 1 x-center offset (-0.05 to 0.05)
    p[7]: row 2 x-center offset (-0.05 to 0.05)
    p[8]: symmetry break amplitude (0 to 0.03)
    p[9]: top row compression (0.8 to 1.0)
    
    Returns: (11, 2) array of Cartesian points
    """
    p = np.clip(params, -0.5, 1.5)  # loose clip for safety
    
    y0 = p[0] + 0.05       # row 0 y
    y1 = p[1] + 0.35       # row 1 y  
    y2 = p[2] + 0.65       # row 2 y
    
    spread = p[3] + 0.85   # horizontal spread
    shear = p[4]           # shear
    cx0 = p[5]             # row 0 center x
    cx1 = p[6]             # row 1 center x
    cx2 = p[7]             # row 2 center x
    sym_break = p[8]       # symmetry breaking
    top_comp = p[9] + 0.9  # top row compression
    
    pts = np.zeros((N, 2))
    
    # Row 0: 4 points at y=y0, x = cx0 + spread * [-0.3, -0.1, 0.1, 0.3]
    row0_offsets = np.array([-0.3, -0.1, 0.1, 0.3]) * spread
    for i in range(4):
        x = cx0 + row0_offsets[i] + shear * (y0 - 0.3)
        y = y0
        # Small perturbation
        x += sym_break * 0.1 * (i - 1.5)
        pts[i] = [x, y]
    
    # Row 1: 3 points at y=y1, x = cx1 + spread * [-0.2, 0, 0.2]
    row1_offsets = np.array([-0.2, 0.0, 0.2]) * spread
    for i in range(3):
        x = cx1 + row1_offsets[i] + shear * (y1 - 0.3)
        y = y1
        x += sym_break * 0.05 * (i - 1)
        pts[4 + i] = [x, y]
    
    # Row 2: 4 points at y=y2, x = cx2 + top_comp * spread * [-0.3, -0.1, 0.1, 0.3]
    row2_offsets = np.array([-0.3, -0.1, 0.1, 0.3]) * spread * top_comp
    for i in range(4):
        x = cx2 + row2_offsets[i] + shear * (y2 - 0.3)
        y = y2
        x += sym_break * 0.1 * (i - 1.5)
        pts[8 + i] = [x, y]
    
    return pts


def _project_to_triangle(pts):
    """Project points to be inside the equilateral triangle."""
    pts = pts.copy()
    for i in range(N):
        x, y = pts[i]
        # Clamp to triangle
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        # Check left edge: y <= sqrt(3)*x
        if y > S3H * x:
            # Project onto left edge
            if x > 1e-10:
                y = S3H * x
            else:
                x = 0
                y = 0
        # Check right edge: y <= sqrt(3)*(1-x)
        if y > S3H * (1 - x):
            if x < 1 - 1e-10:
                y = S3H * (1 - x)
            else:
                x = 1
                y = 0
        pts[i] = [x, y]
    return pts


def _objective(params):
    """Negative of min triangle area (for minimization)."""
    pts = _rubber_band_positions(params)
    pts = _project_to_triangle(pts)
    
    # Penalty if points go outside (shouldn't happen after projection)
    if not _check_inside(pts):
        return 1.0
    
    area = _min_triangle_area(pts)
    return -area


def _structured_init():
    """Create structured initial parameter vector."""
    # Good starting point: roughly evenly spaced rows
    return np.array([
        0.05,    # row 0 y offset
        0.15,    # row 1 y offset  
        0.15,    # row 2 y offset
        0.0,     # spread
        0.0,     # shear
        0.0,     # row 0 cx
        0.0,     # row 1 cx
        0.0,     # row 2 cx
        0.0,     # sym break
        0.0,     # top comp
    ])


def _random_init(rng):
    """Random initial parameter vector."""
    return np.array([
        rng.uniform(0, 0.15),
        rng.uniform(0.1, 0.3),
        rng.uniform(0.1, 0.3),
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(0, 0.03),
        rng.uniform(-0.1, 0.1),
    ])


def _local_nm_optimize(params, time_budget=5.0):
    """Run Nelder-Mead optimization with time budget."""
    start = time.time()
    
    def time_check(params):
        if time.time() - start > time_budget:
            raise TimeoutError()
    
    try:
        result = minimize(
            _objective,
            params,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
                'callback': time_check
            }
        )
        return result.x
    except (TimeoutError, Exception):
        return params


def _greedy_init():
    """Greedy initialization: place points to maximize min distance."""
    rng = np.random.RandomState(42)
    pts = np.zeros((N, 2))
    
    # Place first point at center
    pts[0] = [0.5, S3H / 3]
    
    for i in range(1, N):
        best_pos = None
        best_min_dist = -1
        for _ in range(500):
            # Random point in triangle
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                continue
            x = u + 0.5 * v
            y = S3H * v
            cand = np.array([x, y])
            
            # Min distance to existing points
            dists = np.linalg.norm(pts[:i] - cand, axis=1)
            min_dist = np.min(dists)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = cand
        
        if best_pos is not None:
            pts[i] = best_pos
    
    return pts


def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses symmetry-constrained rubber-band model with Nelder-Mead optimization.
    """
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_params = None
    best_score = -1.0
    best_pts = None
    
    # Generate diverse initial parameter sets
    inits = []
    
    # Structured
    inits.append(_structured_init())
    
    # Random
    for _ in range(8):
        inits.append(_random_init(rng))
    
    # Greedy-based initialization (convert to param space approximately)
    greedy_pts = _greedy_init()
    # Extract approximate parameters from greedy points
    # Row 0: points 0-3, Row 1: points 4-6, Row 2: points 7-10
    if len(greedy_pts) >= N:
        y0 = np.mean(greedy_pts[:4, 1])
        y1 = np.mean(greedy_pts[4:7, 1])
        y2 = np.mean(greedy_pts[7:11, 1])
        cx0 = np.mean(greedy_pts[:4, 0])
        cx1 = np.mean(greedy_pts[4:7, 0])
        cx2 = np.mean(greedy_pts[7:11, 0])
        
        # Estimate spread from row 0
        x_spread = np.std(greedy_pts[:4, 0])
        spread_est = x_spread / 0.2  # rough estimate
        
        greedy_params = np.array([
            y0 - 0.05,
            y1 - 0.35,
            y2 - 0.65,
            spread_est - 0.85,
            0.0,
            cx0 - 0.5,
            cx1 - 0.5,
            cx2 - 0.5,
            0.01,
            0.0
        ])
        inits.append(greedy_params)
    
    # Multi-start Nelder-Mead
    for idx, init_params in enumerate(inits):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Time budget per optimization
        remaining = DEADLINE - 15 - (time.time() - start_time)
        time_budget = min(8.0, remaining / max(1, len(inits) - idx))
        time_budget = max(2.0, time_budget)
        
        # Run Nelder-Mead
        opt_params = _local_nm_optimize(init_params, time_budget)
        
        # Evaluate
        pts = _rubber_band_positions(opt_params)
        pts = _project_to_triangle(pts)
        
        if not _check_inside(pts):
            continue
        
        score = _min_triangle_area(pts)
        
        if score > best_score:
            best_score = score
            best_params = opt_params.copy()
            best_pts = pts.copy()
    
    # If we have time, do a few more random restarts from the best
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Perturb best params
            perturbed = best_params + rng.normal(0, 0.02, size=10)
            opt_params = _local_nm_optimize(perturbed, time_budget=5.0)
            
            pts = _rubber_band_positions(opt_params)
            pts = _project_to_triangle(pts)
            
            if not _check_inside(pts):
                continue
            
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_params = opt_params.copy()
                best_pts = pts.copy()
    
    # Fallback: if rubber-band model failed, use greedy
    if best_pts is None:
        best_pts = _greedy_init()
        best_score = _min_triangle_area(best_pts)
    
    # Final validation and correction
    best_pts = _project_to_triangle(best_pts)
    
    # Ensure points are strictly inside (add tiny epsilon if on boundary)
    for i in range(N):
        x, y = best_pts[i]
        if y < 1e-6:
            y = 1e-6
        if x < 1e-6:
            x = 1e-6
        if x > 1 - 1e-6:
            x = 1 - 1e-6
        # Check against edges
        if y > S3H * x - 1e-6:
            y = max(0, S3H * x - 1e-6)
        if y > S3H * (1 - x) - 1e-6:
            y = max(0, S3H * (1 - x) - 1e-6)
        best_pts[i] = [x, y]
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")

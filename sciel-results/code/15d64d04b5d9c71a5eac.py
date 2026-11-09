"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
CENTER = np.array([0.5, H / 3])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)


def _min_area_and_indices(points):
    """Compute minimum triangle area and the indices of the smallest triangle.
    Returns (min_area, (i, j, k))."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    min_idx = np.argmin(areas)
    return areas[min_idx], (_idx_i[min_idx], _idx_j[min_idx], _idx_k[min_idx])


def _soft_min_area(points, beta=100.0):
    """Smooth approximation of min triangle area using log-sum-exp.
    This is a smooth lower bound on the true min area. Maximizing it pushes all areas up."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    m = areas.min()
    shifted = -beta * (areas - m)
    max_val = np.max(shifted)
    log_sum = max_val + np.log(np.sum(np.exp(shifted - max_val)))
    soft_min = m - log_sum / beta
    return soft_min


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) with u+v<=1 to Cartesian.
    Point = u*A + v*B + (1-u-v)*C where A=(0,0), B=(1,0), C=(0.5,H)."""
    u = bary[0]
    v = bary[1]
    w = 1.0 - u - v
    x = u * 0.0 + v * 1.0 + w * 0.5
    y = u * 0.0 + v * 0.0 + w * H
    return np.array([x, y])


def _params_to_points(params):
    """Convert 2*N params (barycentric u,v pairs) to N Cartesian points.
    Each pair (u,v) must satisfy u>=0, v>=0, u+v<=1."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        if u < 0: u = 0.0
        if v < 0: v = 0.0
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        points[i] = np.array([v + (1 - u - v) * 0.5, (1 - u - v) * H])
    return points


def _lattice_initial_points():
    """Generate initial points on a triangular lattice inside the equilateral triangle.
    Uses a 3x3 grid in barycentric coordinates for 11 points."""
    points = []
    # 3x3 grid in barycentric: (u,v) where u,v in {0, 1/3, 2/3} and u+v <= 1
    for row in range(4):
        for col in range(4 - row):
            u = row / 3.0
            v = col / 3.0
            if u + v <= 1.0 + 1e-9:
                pt = _bary_to_cart(np.array([u, v]))
                points.append(pt)
    # We have 10 points from the lattice, add center as 11th
    points.append(CENTER.copy())
    return np.array(points[:N])


def _lattice_params(points):
    """Convert Cartesian points to barycentric params (u,v) for each point."""
    params = np.zeros(2 * N)
    for i in range(N):
        x, y = points[i]
        s = 1.0 - y / H
        d = 2.0 * (x - 0.5)
        u = (s - d) / 2.0
        v = (s + d) / 2.0
        params[2 * i] = max(0.0, u)
        params[2 * i + 1] = max(0.0, v)
    return params


def _coordinate_descent_polish(points, deadline, rng, max_iter=5000):
    """Coordinate descent that moves points involved in the smallest triangles.
    For the smallest triangle (i,j,k), try moving each of the 3 points in small
    random directions to increase the area. Also try moving other points away
    from the smallest triangle to avoid creating new small triangles."""
    current_min, (ti, tj, tk) = _min_area_and_indices(points)
    best_points = points.copy()
    best_min = current_min
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        # Get the current smallest triangle
        current_min, (ti, tj, tk) = _min_area_and_indices(points)
        
        # Strategy 1: Move one of the three points in the smallest triangle
        # to increase its area
        moved = False
        for pt_idx in [ti, tj, tk]:
            # Try several random directions
            for _ in range(8):
                direction = rng.randn(2)
                direction /= (np.linalg.norm(direction) + 1e-10)
                for step in [0.01, 0.005, 0.002, 0.001]:
                    new_point = points[pt_idx] + direction * step
                    if not _is_inside(new_point):
                        continue
                    old_pt = points[pt_idx].copy()
                    points[pt_idx] = new_point
                    new_min, _ = _min_area_and_indices(points)
                    if new_min > best_min:
                        best_min = new_min
                        best_points = points.copy()
                        moved = True
                    else:
                        points[pt_idx] = old_pt
        
        # Strategy 2: Move a point NOT in the smallest triangle to a random
        # position that might help
        if not moved and rng.random() < 0.3:
            other_indices = [i for i in range(N) if i not in [ti, tj, tk]]
            if other_indices:
                pt_idx = rng.choice(other_indices)
                direction = rng.randn(2)
                direction /= (np.linalg.norm(direction) + 1e-10)
                for step in [0.01, 0.005, 0.002]:
                    new_point = points[pt_idx] + direction * step
                    if not _is_inside(new_point):
                        continue
                    old_pt = points[pt_idx].copy()
                    points[pt_idx] = new_point
                    new_min, _ = _min_area_and_indices(points)
                    if new_min > best_min:
                        best_min = new_min
                        best_points = points.copy()
                    else:
                        points[pt_idx] = old_pt
        
        # Check for stagnation and add small noise
        if it % 100 == 99:
            for i in range(N):
                noise = rng.randn(2) * 0.005
                new_pt = points[i] + noise
                if _is_inside(new_pt):
                    old_pt = points[i].copy()
                    points[i] = new_pt
                    new_min, _ = _min_area_and_indices(points)
                    if new_min < best_min - 0.001:
                        points[i] = old_pt
    
    return best_points, best_min


def _simulated_annealing(start_points, deadline, rng, max_steps=20000, beta_start=50.0, beta_end=200.0):
    """Simulated annealing directly on Cartesian coordinates.
    Uses soft-min area as the smooth objective. Adaptive beta."""
    points = start_points.copy()
    best_points = points.copy()
    best_score = _min_area_and_indices(points)[0]
    
    T0 = 0.02
    T_end = 0.0005
    cooling = (T_end / T0) ** (1.0 / max_steps)
    
    current_obj = -_soft_min_area(points, beta_start)
    
    for step in range(max_steps):
        if time.time() > deadline - 3.0:
            break
        
        T = T0 * (cooling ** step)
        # Adaptive beta: increase over time to make soft-min sharper
        progress = step / max_steps
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Step size decreases with temperature
        step_size = 0.01 * (1.0 + 5.0 * T / T0)
        
        # Pick a random point to move
        pt_idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-10)
        
        new_point = points[pt_idx] + direction * step_size
        if not _is_inside(new_point):
            continue
        
        old_pt = points[pt_idx].copy()
        points[pt_idx] = new_point
        
        new_obj = -_soft_min_area(points, beta)
        delta = new_obj - current_obj
        
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_obj = new_obj
            score = _min_area_and_indices(points)[0]
            if score > best_score:
                best_score = score
                best_points = points.copy()
        else:
            points[pt_idx] = old_pt
    
    return best_points, best_score


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill

    # Phase 1: Start with lattice configuration
    lat_pts = _lattice_initial_points()
    best_points = lat_pts.copy()
    best_score = _min_area_and_indices(lat_pts)[0]
    
    # Phase 2: Long simulated annealing with adaptive beta
    rng = np.random.RandomState(42)
    sa_points, sa_score = _simulated_annealing(
        best_points, deadline, rng, max_steps=30000, beta_start=50.0, beta_end=300.0
    )
    if sa_score > best_score:
        best_points = sa_points
        best_score = sa_score
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 15.0:
        cd_points, cd_score = _coordinate_descent_polish(
            best_points, deadline, rng, max_iter=8000
        )
        if cd_score > best_score:
            best_points = cd_points
            best_score = cd_score
    
    # Phase 4: Another SA round from the polished solution
    if time.time() < deadline - 10.0:
        rng2 = np.random.RandomState(12345)
        sa2_points, sa2_score = _simulated_annealing(
            best_points, deadline, rng2, max_steps=15000, beta_start=100.0, beta_end=400.0
        )
        if sa2_score > best_score:
            best_points = sa2_points
            best_score = sa2_score
    
    # Phase 5: Final coordinate descent polish
    if time.time() < deadline - 5.0:
        cd2_points, cd2_score = _coordinate_descent_polish(
            best_points, deadline, rng, max_iter=5000
        )
        if cd2_score > best_score:
            best_points = cd2_points
            best_score = cd2_score
    
    # Ensure all points are inside
    for i in range(N):
        if not _is_inside(best_points[i]):
            best_points[i] = 0.95 * best_points[i] + 0.05 * CENTER
    
    return best_points

# EVOLVE-BLOCK-END

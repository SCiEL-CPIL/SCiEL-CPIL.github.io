"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Direct active-set local search: identify the bottleneck (minimum-area) triangle.
2. Use Nelder-Mead simplex method to locally optimize the 3 points of the bottleneck
   triangle to increase its area, subject to staying inside the parent triangle.
3. Accept moves only if the global minimum area improves.
4. Combine with simulated annealing-style perturbations for escaping local optima.
5. Multi-start with diverse initial configurations.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, NelderMead

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
def _get_triplets(n):
    triplets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triplets.append((i, j, k))
    return np.array(triplets)

TRIPLETS = _get_triplets(N)
I_IDX = TRIPLETS[:, 0]
J_IDX = TRIPLETS[:, 1]
K_IDX = TRIPLETS[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points.
    Returns (min_area, (i, j, k)) where (i,j,k) is the index of the min triangle."""
    n = len(pts)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _vectorized_min_area_and_idx(pts):
    """Vectorized computation of minimum triangle area and its indices."""
    p_i = pts[I_IDX]
    p_j = pts[J_IDX]
    p_k = pts[K_IDX]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    min_triplet = TRIPLETS[min_idx]
    
    return min_area, (min_triplet[0], min_triplet[1], min_triplet[2])

def _check_inside_triangle(pts):
    """Check if all points are inside or on the boundary of the unit triangle."""
    # Triangle vertices: A(0,0), B(1,0), C(0.5, sqrt(3)/2)
    x, y = pts[:, 0], pts[:, 1]
    # y >= 0
    if np.any(y < -1e-10):
        return False
    # y <= S3H * (1 - |x - 0.5|) for the triangle
    # Upper boundary: from (0,0) to (0.5, S3H): y = S3H * x / 0.5 = 2*S3H*x
    #                 from (0.5, S3H) to (1,0): y = S3H * (1-x) / 0.5 = 2*S3H*(1-x)
    # So y <= 2*S3H*min(x, 1-x)
    upper = 2 * S3H * np.minimum(x, 1 - x)
    if np.any(y > upper + 1e-10):
        return False
    # x between 0 and 1
    if np.any(x < -1e-10) or np.any(x > 1 + 1e-10):
        return False
    return True

def _project_point_to_triangle(x, y):
    """Project a single point (x,y) to the closest point inside the triangle."""
    # If already inside, return as-is
    if y >= 0 and y <= 2 * S3H * min(x, 1 - x) and 0 <= x <= 1:
        return x, y
    
    # Project to closest point in the triangle
    # The triangle has 3 edges. Find closest point on each edge, then take min distance.
    # Edge 1: (0,0) to (1,0)
    # Edge 2: (1,0) to (0.5, S3H)
    # Edge 3: (0.5, S3H) to (0,0)
    
    def _closest_on_segment(px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            return ax, ay
        t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / len_sq))
        return ax + t*dx, ay + t*dy
    
    # Closest on each edge
    c1x, c1y = _closest_on_segment(x, y, 0, 0, 1, 0)
    c2x, c2y = _closest_on_segment(x, y, 1, 0, 0.5, S3H)
    c3x, c3y = _closest_on_segment(x, y, 0.5, S3H, 0, 0)
    
    d1 = (x-c1x)**2 + (y-c1y)**2
    d2 = (x-c2x)**2 + (y-c2y)**2
    d3 = (x-c3x)**2 + (y-c3y)**2
    
    min_d = min(d1, d2, d3)
    if min_d == d1:
        return c1x, c1y
    elif min_d == d2:
        return c2x, c2y
    else:
        return c3x, c3y

def _project_to_triangle(pts):
    """Project all points to be inside the triangle."""
    result = np.zeros_like(pts)
    for i in range(len(pts)):
        result[i, 0], result[i, 1] = _project_point_to_triangle(pts[i, 0], pts[i, 1])
    return result

def _random_point_in_triangle(rng):
    """Generate a random point uniformly in the triangle."""
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    # Barycentric: (1-r1-r2, r1, r2)
    x = r1 + 0.5 * r2
    y = S3H * r2
    return x, y

def _greedy_init(rng):
    """Greedy initialization: place points one by one to maximize the minimum distance."""
    pts = np.zeros((N, 2))
    # Start with centroid
    pts[0] = [0.5, S3H/3]
    
    for i in range(1, N):
        best_pos = None
        best_min_dist = 0
        # Try many random positions
        for _ in range(2000):
            x, y = _random_point_in_triangle(rng)
            pos = np.array([x, y])
            # Compute minimum distance to existing points
            dists = np.sqrt(np.sum((pts[:i] - pos)**2, axis=1))
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        pts[i] = best_pos
    
    return pts

def _structured_init():
    """Create a structured initial configuration based on known good patterns."""
    # Use a grid-like pattern in barycentric coordinates
    pts = np.zeros((N, 2))
    
    # Place points at various barycentric positions
    bary_coords = [
        (1/3, 1/3, 1/3),      # centroid
        (0, 0.5, 0.5),        # midpoint of BC
        (0.5, 0, 0.5),        # midpoint of AC
        (0.5, 0.5, 0),        # midpoint of AB
        (0.1, 0.45, 0.45),    # near A
        (0.45, 0.1, 0.45),    # near B
        (0.45, 0.45, 0.1),    # near C
        (0.2, 0.4, 0.4),      # interior
        (0.4, 0.2, 0.4),      # interior
        (0.4, 0.4, 0.2),      # interior
        (0.3, 0.35, 0.35),    # interior
    ]
    
    for i, (u, v, w) in enumerate(bary_coords):
        x = v + 0.5 * w
        y = S3H * w
        pts[i] = [x, y]
    
    return pts

def _active_set_local_search(pts, rng, max_iters=500, time_limit=10.0):
    """
    Active-set local search:
    1. Find the minimum-area triangle.
    2. Use Nelder-Mead to optimize those 3 points to increase the global minimum area.
    3. Accept if global minimum improves.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_min, current_min_idx = _vectorized_min_area_and_idx(current_pts)
    best_pts = current_pts.copy()
    best_min = current_min
    
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if no_improve_count > max_no_improve:
            break
        
        min_area, min_idx = _vectorized_min_area_and_idx(current_pts)
        i, j, k = min_idx
        
        # Extract the 3 points of the bottleneck triangle
        p_i = current_pts[i].copy()
        p_j = current_pts[j].copy()
        p_k = current_pts[k].copy()
        
        # Define objective: maximize the minimum area when these 3 points move
        # We'll use Nelder-Mead to optimize the 6 coordinates of these 3 points
        
        def objective(x_flat):
            new_pts = current_pts.copy()
            new_pts[i] = x_flat[0:2]
            new_pts[j] = x_flat[2:4]
            new_pts[k] = x_flat[4:6]
            
            # Project to triangle
            new_pts = _project_to_triangle(new_pts)
            
            # Compute global minimum area
            min_a, _ = _vectorized_min_area_and_idx(new_pts)
            return -min_a  # Minimize negative area = maximize area
        
        x0 = np.concatenate([p_i, p_j, p_k])
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            new_pts = current_pts.copy()
            new_pts[i] = result.x[0:2]
            new_pts[j] = result.x[2:4]
            new_pts[k] = result.x[4:6]
            new_pts = _project_to_triangle(new_pts)
            
            new_min, _ = _vectorized_min_area_and_idx(new_pts)
            
            if new_min > current_min + 1e-10:
                current_pts = new_pts
                current_min = new_min
                no_improve_count = 0
                
                if current_min > best_min:
                    best_min = current_min
                    best_pts = current_pts.copy()
            else:
                no_improve_count += 1
                
        except Exception:
            no_improve_count += 1
    
    return best_pts, best_min

def _simulated_annealing(pts, rng, n_steps=5000, T_start=0.01, T_end=0.001, time_limit=20.0):
    """Simulated annealing with direct evaluation of the non-smooth objective."""
    start_time = time.time()
    current_pts = pts.copy()
    current_min, _ = _vectorized_min_area_and_idx(current_pts)
    best_pts = current_pts.copy()
    best_min = current_min
    
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Update temperature (geometric cooling)
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Random perturbation
        new_pts = current_pts.copy()
        # Perturb 1-3 random points
        n_perturb = rng.randint(1, 3)
        indices = rng.sample(range(N), n_perturb)
        for idx in indices:
            noise = np.random.normal(0, 0.01, 2)
            new_pts[idx] += noise
        
        # Project to triangle
        new_pts = _project_to_triangle(new_pts)
        
        new_min, _ = _vectorized_min_area_and_idx(new_pts)
        
        # Accept if better, or with probability exp((new-old)/T) if worse
        delta = new_min - current_min
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pts = new_pts
            current_min = new_min
            
            if current_min > best_min:
                best_min = current_min
                best_pts = current_pts.copy()
    
    return best_pts, best_min

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_score = 0.0
    best_pts = None
    
    # Multiple initialization strategies
    inits = []
    
    # Structured init
    inits.append(_structured_init())
    
    # Greedy init
    inits.append(_greedy_init(rng))
    
    # Random inits
    for _ in range(5):
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        inits.append(pts)
    
    # For each init, run active-set local search + simulated annealing
    for init_idx, init_pts in enumerate(inits):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Phase 1: Active-set local search
        pts, score = _active_set_local_search(init_pts, rng, max_iters=300, time_limit=15.0)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Phase 2: Simulated annealing to escape local optima
        if time.time() - start_time > DEADLINE - 10:
            break
        pts2, score2 = _simulated_annealing(pts, rng, n_steps=3000, T_start=0.005, T_end=0.0005, time_limit=15.0)
        
        if score2 > best_score:
            best_score = score2
            best_pts = pts2.copy()
        
        # Phase 3: Another round of active-set on the SA result
        if time.time() - start_time > DEADLINE - 5:
            break
        pts3, score3 = _active_set_local_search(pts2, rng, max_iters=200, time_limit=10.0)
        
        if score3 > best_score:
            best_score = score3
            best_pts = pts3.copy()
    
    # Final polish: one more active-set round on the best found
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        final_pts, final_score = _active_set_local_search(best_pts, rng, max_iters=200, time_limit=8.0)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _structured_init()
    
    # Ensure all points are inside the triangle
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts

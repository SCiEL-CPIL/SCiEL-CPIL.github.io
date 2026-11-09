import numpy as np
import scipy.spatial
import time

N = 14

def _vectorized_score(pts):
    """Compute min triangle area / hull area using vectorized ops."""
    n = len(pts)
    # Compute all pairwise vectors
    # For each triple (i,j,k): area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # Use broadcasting: shape (n, n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    # For triple (i,j,k): v1 = pts[j]-pts[i], v2 = pts[k]-pts[i]
    # cross = v1[0]*v2[1] - v1[1]*v2[0]
    # v1 = diff[i,j], v2 = diff[i,k]
    # cross[i,j,k] = diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]
    cross = diff[:, :, 0][:, :, np.newaxis] * diff[:, :, 1][:, np.newaxis, :] - \
            diff[:, :, 1][:, :, np.newaxis] * diff[:, :, 0][:, np.newaxis, :]
    # cross has shape (n, n, n), where cross[i,j,k] is the cross product
    # We only need i<j<k
    areas = 0.5 * np.abs(cross)
    # Get upper triangle indices
    iu, ju, ku = np.triu_indices(n, k=2)
    min_area = np.min(areas[iu, ju, ku])
    
    # Hull area
    try:
        hull = scipy.spatial.ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_score = -np.inf
    
    # Multiple diverse initializations
    initializations = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initializations.append(np.vstack([np.cos(angles), np.sin(angles)]).T)
    
    # 2. Two concentric circles
    half = N // 2
    angles1 = np.linspace(0, 2*np.pi, half, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, N-half, endpoint=False) + np.pi/(N-half)
    pts1 = np.vstack([np.cos(angles1), np.sin(angles1)]).T * 0.7
    pts2 = np.vstack([np.cos(angles2), np.sin(angles2)]).T * 1.0
    initializations.append(np.vstack([pts1, pts2]))
    
    # 3. Grid-like (approximate)
    # Place points in a 4x4 grid pattern, then select 14
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts[:N])
    initializations.append(grid_pts)
    
    # 4. Random uniform in unit circle
    np.random.seed(42)
    r = np.sqrt(np.random.uniform(0, 1, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    initializations.append(np.vstack([r*np.cos(theta), r*np.sin(theta)]).T)
    
    # 5. Hexagonal packing approximation
    # Just another random start with different seed
    np.random.seed(123)
    r = np.sqrt(np.random.uniform(0, 1, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    initializations.append(np.vstack([r*np.cos(theta), r*np.sin(theta)]).T)
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit:
            break
        
        # Simulated annealing for each initialization
        pts = init_pts.copy()
        current_score = _vectorized_score(pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = pts.copy()
        
        if current_score <= 0:
            continue
            
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995
        step_size = 0.05
        min_step = 0.001
        
        while time.time() - start_time < time_limit:
            # Perturb a random point
            idx = np.random.randint(N)
            perturbation = np.random.normal(0, step_size, 2)
            new_pts = pts.copy()
            new_pts[idx] += perturbation
            
            new_score = _vectorized_score(new_pts)
            
            if new_score <= 0:
                T *= cooling_rate
                continue
                
            # Accept if better, or with probability exp(-(delta)/T)
            delta = current_score - new_score
            if new_score > current_score or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = pts.copy()
            
            T *= cooling_rate
            # Adapt step size
            if T < T_min * 10:
                step_size = max(min_step, step_size * 0.95)
            else:
                step_size = max(min_step, step_size * 0.99)
        
        if time.time() - start_time > time_limit:
            break
    
    # Final polish: simple local search on best_points
    if best_points is not None and time.time() - start_time < time_limit - 5:
        pts = best_points.copy()
        current_score = _vectorized_score(pts)
        step_size = 0.01
        for _ in range(500):
            if time.time() - start_time > time_limit - 2:
                break
            idx = np.random.randint(N)
            perturbation = np.random.normal(0, step_size, 2)
            new_pts = pts.copy()
            new_pts[idx] += perturbation
            new_score = _vectorized_score(new_pts)
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = pts.copy()
            else:
                step_size *= 0.98
                if step_size < 0.0001:
                    break
    
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.vstack([np.cos(angles), np.sin(angles)]).T
    
    return best_points

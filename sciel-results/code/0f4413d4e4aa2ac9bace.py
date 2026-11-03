"""Pack 50 equal circles in a unit square using Simulated Annealing + High-Beta L-BFGS-B Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    """
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    y = -beta * all_constraints
    max_y = np.max(y)
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    x0 = centers.ravel()
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _simulated_annealing(centers, time_limit=10.0):
    """
    Simulated annealing to explore the configuration space.
    Moves: random displacement of one or a few circles.
    Energy: -radius (we want to minimize energy, i.e., maximize radius).
    But radius is non-smooth. We use the soft-min surrogate for the energy function
    to guide the annealing, but accept moves based on the exact radius if possible?
    Actually, using the exact radius for acceptance is better for final quality,
    but the soft-min is smoother.
    
    Let's use the exact radius for the energy function in SA.
    E = -radius(centers)
    
    Moves:
    1. Pick 1-3 random circles.
    2. Displace them by a random vector with magnitude ~ current_temperature.
    3. Check if they are still inside the square. If not, reject or reflect.
       Better: just reject if out of bounds.
    4. Compute new radius.
    5. Accept if new_radius > old_radius (energy decreases).
    6. Accept with probability exp((new_radius - old_radius) / T) if new_radius < old_radius.
       Note: delta_E = -new_radius - (-old_radius) = old_radius - new_radius.
       So prob = exp((old_radius - new_radius) / T).
    
    Temperature schedule: T = T0 * (alpha)^iter or T = T0 * (1 - t/t_max).
    """
    start_time = time.time()
    centers = centers.copy()
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T0 = 0.05
    T_end = 0.001
    alpha = 0.9995  # Cooling rate
    
    T = T0
    
    # Estimate number of iterations possible
    # Each iteration involves radius calculation which is O(N^2).
    # N=50 -> 2500 ops. ~1us? No, numpy overhead. ~10-50us.
    # So ~20k-100k iterations per second.
    # With 10s, we can do ~1M iterations.
    
    while time.time() - start_time < time_limit:
        # Check time more frequently
        if time.time() - start_time > time_limit - 0.1:
            break
            
        # Pick random circles to move
        n_move = np.random.randint(1, 4)  # 1, 2, or 3
        indices = np.random.choice(N, size=n_move, replace=False)
        
        # Random displacement
        # Scale displacement by T
        disp = np.random.normal(0, T, size=(n_move, 2))
        
        new_centers = centers.copy()
        new_centers[indices] += disp
        
        # Check bounds
        # If any point is out of bounds, reject
        if np.any(new_centers < -0.5) or np.any(new_centers > 0.5):
            T *= alpha
            continue
            
        new_r = _radius(new_centers)
        
        # Acceptance
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # delta_r = current_r - new_r > 0
            # prob = exp(-delta_r / T)
            if T > 1e-6:
                prob = np.exp(-(current_r - new_r) / T)
                if np.random.random() < prob:
                    centers = new_centers
                    current_r = new_r
        
        T *= alpha
        
        # Periodic cooling adjustment if T drops too fast?
        # Standard geometric cooling is fine.
        
    return best_centers, best_r

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # Initial candidate: Hexagonal grid
    # Generate a grid slightly larger than needed and center it
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    points = []
    for i in range(8):
        y = -0.5 + (i + 0.5) * dy
        for j in range(7):
            x = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
    hex_centers = np.array(points)
    # Filter to square
    mask = (hex_centers[:, 0] >= -0.5) & (hex_centers[:, 0] <= 0.5) & (hex_centers[:, 1] >= -0.5) & (hex_centers[:, 1] <= 0.5)
    hex_centers = hex_centers[mask]
    
    if len(hex_centers) > N:
        # Keep first N? Or center them?
        # For SA, any valid start is fine.
        hex_centers = hex_centers[:N]
    elif len(hex_centers) < N:
        # Pad with random points?
        # This shouldn't happen.
        pass
        
    initial_centers = hex_centers
    
    # Phase 1: Simulated Annealing
    # Use a good chunk of the time for SA to explore.
    sa_time = min(60.0, DEADLINE - 30) # Leave 30s for polish and other candidates
    
    sa_centers, sa_r = _simulated_annealing(initial_centers, time_limit=sa_time)
    
    if sa_r > best_r:
        best_r = sa_r
        best_centers = sa_centers.copy()
        
    # Phase 2: Polish with L-BFGS-B
    # Use the best center from SA
    if best_centers is not None:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 5.0:
            # High beta polish
            polished = _local_refine_smooth(best_centers, time_limit=min(15.0, time_left), beta=500.0)
            r_pol = _radius(polished)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished.copy()
                
            # Second polish with even higher beta if time remains
            time_left2 = DEADLINE - 2 - (time.time() - start_time)
            if time_left2 > 5.0 and best_r > 0.071:
                polished2 = _local_refine_smooth(best_centers, time_limit=min(10.0, time_left2), beta=1000.0)
                r_pol2 = _radius(polished2)
                
                if r_pol2 > best_r:
                    best_r = r_pol2
                    best_centers = polished2.copy()
    
    # If SA didn't improve much, try a different start?
    # Or just return best.
    
    if best_centers is None:
        # Fallback
        best_centers = initial_centers
        
    return best_centers

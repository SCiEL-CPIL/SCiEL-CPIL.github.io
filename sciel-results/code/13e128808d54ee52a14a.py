"""Pack 50 equal circles in a unit square using Voronoi-based local search and SA."""
import numpy as np
import time
from scipy.spatial import Delaunay

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _init_hexagonal(n_x, n_y, spacing):
    """Create a hexagonal lattice of points, centered and scaled."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing
            y = j * dy
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    points = points[:N]
    # Center
    points -= np.mean(points, axis=0)
    # Scale to fit in [-0.45, 0.45]
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points *= (0.45 / max_ext)
    return points

def _init_grid(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _local_search(centers, time_limit=5.0):
    """
    Local search using Delaunay triangulation to find nearest neighbors.
    We try to increase the minimum radius by moving points that are in tight contact.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_r = _radius(best_centers)
    step_size = 0.01
    
    # Precompute Delaunay
    try:
        tri = Delaunay(best_centers)
        # Get edges
        edges = tri.simplices
        # Unique pairs
        pairs = set()
        for simplex in edges:
            for i in range(3):
                for j in range(i+1, 3):
                    pairs.add((simplex[i], simplex[j]))
        pairs = list(pairs)
    except Exception:
        return best_centers

    while time.time() - start_time < time_limit:
        # Find the current bottleneck
        r = best_r
        # Identify points that are "tight" against walls or each other
        # We will do a random walk of small perturbations, accepting if r improves
        # or with a small probability if it doesn't (to escape local minima)
        
        # Pick a random point
        idx = np.random.randint(N)
        
        # Propose a move
        move = np.random.normal(0, step_size, size=2)
        new_centers = best_centers.copy()
        new_centers[idx] += move
        
        # Check bounds strictly
        if np.any(np.abs(new_centers) > 0.5):
            continue
            
        new_r = _radius(new_centers)
        
        if new_r > best_r:
            best_centers = new_centers
            best_r = new_r
            # Update Delaunay occasionally? It's expensive. 
            # For local search, we can just use the old neighbors or recompute if improvement is significant.
            # Recomputing every step is too slow for 145s budget if we do many steps.
            # Let's recompute if we haven't in a while or if we moved a lot.
            # Actually, let's just recompute every 100 successful moves or so? 
            # Or just keep the old graph? The graph might become invalid.
            # Let's recompute if the move was large or every 50 iterations.
            # For simplicity and correctness, let's recompute if we improve.
            try:
                tri = Delaunay(best_centers)
                pairs = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            pairs.add((simplex[i], simplex[j]))
                pairs = list(pairs)
            except Exception:
                pass
        else:
            # Simulated annealing style acceptance
            # Probability of accepting worse move
            # We don't have a temperature, so let's just use a fixed small prob
            if np.random.rand() < 0.01:
                best_centers = new_centers
                # best_r stays the same (we track best_r separately)
                # But we are moving away from the best known center? 
                # This is risky. Let's stick to hill climbing for stability, 
                # and rely on multiple restarts for global search.
                pass

    return best_centers

def _anneal(centers, time_limit=10.0):
    """
    Simulated Annealing on the positions.
    Objective: Maximize r.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _radius(current)
    best = current.copy()
    best_r = current_r
    
    T = 0.01
    T_min = 0.0001
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Perturb a random point
        idx = np.random.randint(N)
        step = np.random.normal(0, 0.05, size=2)
        new_pos = current[idx] + step
        
        # Check bounds
        if np.any(np.abs(new_pos) > 0.5):
            continue
            
        new_centers = current.copy()
        new_centers[idx] = new_pos
        new_r = _radius(new_centers)
        
        delta_r = new_r - current_r
        
        if delta_r > 0:
            current = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            # Accept with probability exp(delta_r / T)
            # delta_r is negative
            prob = np.exp(delta_r / T)
            if np.random.rand() < prob:
                current = new_centers
                current_r = new_r
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Maybe reheat slightly?
            # T = 0.005

    return best

def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    inits = []
    
    # 1. Rectangular Grids
    inits.append(_init_grid(7, 8))
    inits.append(_init_grid(8, 7))
    inits.append(_init_grid(5, 10))
    inits.append(_init_grid(10, 5))
    
    # 2. Hexagonal Grids
    for sp in [0.13, 0.14, 0.15, 0.16]:
        inits.append(_init_hexagonal(10, 10, sp))
        
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    np.random.seed(123)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))

    # Phase 1: Local Search (Hill Climbing) on each init
    # This is fast and finds local optima quickly.
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_left = DEADLINE - 20 - (time.time() - start_time)
        alloc = min(2.0, time_left / max(1, len(inits) - i))
        
        refined = _local_search(init, time_limit=alloc)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()

    # Phase 2: Simulated Annealing on the best result and a few others
    # SA is slower per step but can escape local minima.
    candidates = [best_centers.copy()]
    # Add some other good ones if we have time
    # Let's just run SA on the best one found so far, and maybe 2 others if time permits.
    
    # We'll run SA on the best result
    if time.time() - start_time < DEADLINE - 5:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        sa_result = _anneal(best_centers, time_limit=time_left * 0.5)
        r_sa = _radius(sa_result)
        if r_sa > best_r:
            best_r = r_sa
            best_centers = sa_result.copy()
            
        # If time remains, run SA on a random restart
        if time.time() - start_time < DEADLINE - 5:
            time_left = DEADLINE - 5 - (time.time() - start_time)
            np.random.seed(int(time.time()))
            rand_init = np.random.uniform(-0.45, 0.45, size=(N, 2))
            # Quick local search first
            ls_result = _local_search(rand_init, time_limit=min(1.0, time_left/4))
            sa_result2 = _anneal(ls_result, time_limit=time_left * 0.5)
            r_sa2 = _radius(sa_result2)
            if r_sa2 > best_r:
                best_r = r_sa2
                best_centers = sa_result2.copy()

    return best_centers

if __name__ == "__main__":
    c = construct()
    print(_radius(c))

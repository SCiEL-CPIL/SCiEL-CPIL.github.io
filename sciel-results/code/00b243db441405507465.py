"""Physics-based circle packing optimizer for 200 equal circles in a unit square."""
N = 200
import time
import numpy as np
from scipy.spatial import cKDTree

# EVOLVE-BLOCK-START

def _radius_of_placement(centers):
    """Compute the largest common radius for a given set of centers."""
    # Distance from each center to the nearest wall
    dist_to_wall = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_wall)
    
    # Minimum pairwise distance / 2 using KDTree for efficiency
    if len(centers) <= 1:
        min_pair = np.inf
    else:
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_wall, min_pair)


def _fibonacci_lattice_init():
    """Create Fibonacci lattice (sunflower pattern) initialization."""
    # Fibonacci lattice in a square
    # Using the golden ratio for uniform distribution
    golden_ratio = (1 + np.sqrt(5)) / 2
    n = N
    
    centers = np.zeros((n, 2))
    for i in range(n):
        # Use the formula for Fibonacci lattice
        # Angle based on golden angle
        angle = 2 * np.pi * i / golden_ratio
        # Radius based on square root of index
        r = np.sqrt((i + 0.5) / n)
        
        # Scale to fit in unit square [-0.5, 0.5]^2
        # The Fibonacci lattice naturally fills a circle, so we scale accordingly
        # Max radius should be 0.5 * sqrt(2) to fit in the square, but we want to use the full square
        # Actually, let's use a different approach: map to square
        # Use a spiral that fills the square
        
        # Better: use the standard Fibonacci lattice in a circle, then scale
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # Scale to fit in [-0.5, 0.5]^2
        # The maximum extent of the Fibonacci lattice in a circle is r_max = sqrt(n/n) = 1
        # But we want it to fill the square, so scale by 0.5 / (sqrt(2)/2) = 0.5 * sqrt(2)
        # Actually, let's just scale to fit in the square
        scale = 0.5 / np.sqrt(2)  # This makes the circle inscribed in the square
        x *= scale * 2  # Scale to [-0.5, 0.5]
        y *= scale * 2
    
    centers[:, 0] = x
    centers[:, 1] = y
    
    # Clip to square
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers


def _hex_lattice_init():
    """Create hexagonal lattice initialization for better starting packing."""
    best_centers = None
    best_r = 0
    
    for rows in range(10, 25):
        cols = int(np.ceil(N / rows))
        dx = 1.0 / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        centers = []
        for row in range(rows):
            for col in range(cols):
                if len(centers) >= N:
                    break
                x = (col + 0.5) * dx - 0.5 + (0.5 * dx if row % 2 == 1 else 0)
                y = (row + 0.5) * dy - 0.5
                centers.append([x, y])
        
        if len(centers) < N:
            continue
        
        centers = np.array(centers[:N])
        centers = np.clip(centers, -0.5, 0.5)
        
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers


def _repulsive_optimization(centers, max_iters=2000, time_limit=30):
    """
    Optimize using repulsive forces between circles with rigid boundary constraints.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    step_size = 0.003
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find nearest neighbor for each point
        tree = cKDTree(current)
        dists, neighbors = tree.query(current, k=2)
        nearest_dists = dists[:, 1]
        nearest_neighbors = neighbors[:, 1]
        
        # Compute repulsive forces
        forces = np.zeros_like(current)
        for i in range(len(current)):
            j = nearest_neighbors[i]
            if i == j:
                continue
            diff = current[j] - current[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                forces[i] += np.random.randn(2) * 0.01
            else:
                # Stronger repulsion when closer than 2*r
                target_dist = 2 * current_r
                if dist < target_dist:
                    force_mag = step_size * (target_dist / dist - 1.0) * 2.0
                else:
                    force_mag = step_size * 0.5
                forces[i] += (diff / dist) * force_mag
        
        # Wall repulsion - push away from walls
        for i in range(len(current)):
            for coord in range(2):
                val = current[i, coord]
                if val < -0.5 + current_r:
                    forces[i, coord] += (0.5 - current_r - abs(val)) * 0.1
                elif val > 0.5 - current_r:
                    forces[i, coord] -= (0.5 - current_r - abs(val)) * 0.1
        
        # Update positions with adaptive step
        candidate = current + forces
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 30 == 0:
                step_size *= 0.85
                if step_size < 1e-6:
                    break
    
    return best


def _simulated_annealing(centers, max_iters=5000, time_limit=40):
    """Simulated annealing to escape local minima."""
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    temp = 0.008
    cooling = 0.997
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        idx = np.random.randint(len(current))
        perturbation = np.random.randn(2) * temp
        candidate = current.copy()
        candidate[idx] += perturbation
        candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        delta = candidate_r - current_r
        
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        
        temp *= cooling
        if temp < 1e-7:
            break
    
    return best


def _refine_with_kdtree(centers, max_iters=3000, time_limit=25):
    """Fine-tuning using KDTree to identify and fix close pairs."""
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    step_size = 0.001
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        tree = cKDTree(current)
        pairs = tree.query_pairs(2 * current_r * 1.02)
        
        if len(pairs) == 0:
            dists, neighbors = tree.query(current, k=2)
            min_dist = np.min(dists[:, 1])
            if min_dist > 2 * current_r * 1.05:
                idx = np.random.randint(len(current))
                perturbation = np.random.randn(2) * step_size
                candidate = current.copy()
                candidate[idx] += perturbation
                candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
                candidate_r = _radius_of_placement(candidate)
                if candidate_r > current_r:
                    current = candidate
                    current_r = candidate_r
                    if current_r > best_r:
                        best_r = current_r
                        best = current.copy()
            continue
        
        min_pair_dist = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(current[i] - current[j])
            if d < min_pair_dist:
                min_pair_dist = d
                min_pair = (i, j)
        
        if min_pair is None:
            continue
        
        i, j = min_pair
        diff = current[i] - current[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
        else:
            direction = diff / dist
        
        move = step_size * 0.5
        candidate = current.copy()
        candidate[i] += direction * move
        candidate[j] -= direction * move
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            step_size *= 0.9
            if step_size < 1e-7:
                break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Try both Fibonacci and hexagonal initializations
    fib_centers = _fibonacci_lattice_init()
    hex_centers = _hex_lattice_init()
    
    fib_r = _radius_of_placement(fib_centers)
    hex_r = _radius_of_placement(hex_centers)
    
    # Start with the better initialization
    if fib_r > hex_r:
        centers = fib_centers.copy()
        best_centers = fib_centers.copy()
        best_r = fib_r
    else:
        centers = hex_centers.copy()
        best_centers = hex_centers.copy()
        best_r = hex_r
    
    # Phase 1: Repulsive optimization
    if time.time() < deadline - 60:
        centers = _repulsive_optimization(centers, max_iters=2000, time_limit=25)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 35:
        centers = _simulated_annealing(centers, max_iters=5000, time_limit=20)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 3: KDTree refinement
    if time.time() < deadline - 15:
        centers = _refine_with_kdtree(centers, max_iters=3000, time_limit=15)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 4: Try the other initialization if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        # Try hexagonal if we started with Fibonacci, or vice versa
        if fib_r > hex_r:
            alt_centers = hex_centers.copy()
        else:
            alt_centers = fib_centers.copy()
        
        # Add small random perturbation
        alt_centers += np.random.randn(N, 2) * 0.003
        alt_centers = np.clip(alt_centers, -0.5, 0.5)
        
        # Quick optimization
        alt_centers = _repulsive_optimization(alt_centers, max_iters=800, time_limit=10)
        alt_r = _radius_of_placement(alt_centers)
        
        if alt_r > best_r:
            best_r = alt_r
            best_centers = alt_centers.copy()
    
    # Final refinement
    if time.time() < deadline - 3:
        best_centers = _refine_with_kdtree(best_centers, max_iters=1500, time_limit=3)
        final_r = _radius_of_placement(best_centers)
        if final_r > best_r:
            best_r = final_r
    
    return best_centers
# EVOLVE-BLOCK-END

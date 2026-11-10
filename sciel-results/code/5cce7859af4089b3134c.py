# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if n != 10:
        return False
    if centers.shape != (10, 2):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if t > 0:
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
    return True

def _score(centers, sides):
    """Compute the score (sum of side lengths)."""
    if not _is_valid(centers, sides):
        return -np.inf
    return np.sum(sides)

def _random_valid_packing():
    """Generate a random valid packing."""
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # Start with a grid and add random perturbations
    base_s = 0.25
    for i in range(10):
        col = i % 4
        row = i // 4
        centers[i, 0] = (col + 0.5) * base_s
        centers[i, 1] = (row + 0.5) * base_s
        sides[i] = base_s * (0.8 + 0.4 * random.random())
    
    # Adjust to fit
    for i in range(10):
        centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
        centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
    
    return centers, sides

def _local_search(centers, sides, max_iters=5000, step_size=0.01, deadline=None):
    """Perform local search to improve the packing."""
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _score(best_centers, best_sides)
    
    if np.isinf(best_score):
        return best_centers, best_sides, best_score
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_score = best_score
    
    for it in range(max_iters):
        if deadline and time.time() > deadline - 5:
            break
        
        # Pick a random square and perturb it
        i = random.randint(0, 9)
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        
        # Perturb center
        new_centers[i, 0] += (random.random() - 0.5) * step_size
        new_centers[i, 1] += (random.random() - 0.5) * step_size
        
        # Perturb side length
        new_sides[i] *= (1 + (random.random() - 0.5) * step_size)
        new_sides[i] = max(0.01, new_sides[i])
        
        # Clamp to bounds
        new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
        new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
        
        new_score = _score(new_centers, new_sides)
        
        if new_score > current_score:
            current_centers = new_centers
            current_sides = new_sides
            current_score = new_score
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                best_score = current_score
    
    return best_centers, best_sides, best_score

def _simulated_annealing(centers, sides, max_iters=10000, start_temp=0.05, end_temp=0.001, deadline=None):
    """Perform simulated annealing to improve the packing."""
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _score(best_centers, best_sides)
    
    if np.isinf(best_score):
        return best_centers, best_sides, best_score
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_score = best_score
    
    for it in range(max_iters):
        if deadline and time.time() > deadline - 5:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Pick a random square and perturb it
        i = random.randint(0, 9)
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        
        # Perturb center
        new_centers[i, 0] += (random.random() - 0.5) * 0.05
        new_centers[i, 1] += (random.random() - 0.5) * 0.05
        
        # Perturb side length
        new_sides[i] *= (1 + (random.random() - 0.5) * 0.1)
        new_sides[i] = max(0.01, new_sides[i])
        
        # Clamp to bounds
        new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
        new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
        
        new_score = _score(new_centers, new_sides)
        
        if np.isinf(new_score):
            # Invalid, reject
            continue
        
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_centers = new_centers
            current_sides = new_sides
            current_score = new_score
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                best_score = current_score
    
    return best_centers, best_sides, best_score

def _constructive_heuristic():
    """Construct a good initial packing using a greedy approach."""
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # Try different configurations
    configs = []
    
    # Config 1: 3 large + 7 small (as in current program)
    s_large = 1.0 / 3.0
    for i in range(3):
        centers[i, 0] = (i + 0.5) * s_large
        centers[i, 1] = 1.0 - s_large / 2
        sides[i] = s_large
    
    s_med = 0.25
    y_med = 2.0 / 3.0 - s_med / 2
    for i in range(4):
        centers[3 + i, 0] = (i + 0.5) * s_med
        centers[3 + i, 1] = y_med
        sides[3 + i] = s_med
    
    s_small = 1.0 / 3.0
    y_small = s_small / 2
    for i in range(3):
        centers[7 + i, 0] = (i + 0.5) * s_small
        centers[7 + i, 1] = y_small
        sides[7 + i] = s_small
    
    configs.append((centers.copy(), sides.copy()))
    
    # Config 2: 2 large + 8 small
    s_large2 = 0.4
    centers2 = np.zeros((10, 2))
    sides2 = np.zeros(10)
    
    # Two large squares in top row
    centers2[0, 0] = 0.2
    centers2[0, 1] = 1.0 - 0.2
    sides2[0] = 0.4
    centers2[1, 0] = 0.6
    centers2[1, 1] = 1.0 - 0.2
    sides2[1] = 0.4
    
    # Remaining space: bottom 0.6 height, full width
    # Plus top-right 0.2x0.4 corner
    # Place 8 small squares
    s_small2 = 0.2
    # Bottom area: 1 x 0.6
    # 5 in a row: 5*0.2 = 1.0, fits
    for i in range(5):
        centers2[2 + i, 0] = (i + 0.5) * 0.2
        centers2[2 + i, 1] = 0.3
        sides2[2 + i] = 0.2
    # Top-right corner: 0.2 x 0.4
    # Place 3 more
    centers2[7, 0] = 0.9
    centers2[7, 1] = 0.8
    sides2[7] = 0.2
    centers2[8, 0] = 0.9
    centers2[8, 1] = 0.6
    sides2[8] = 0.2
    centers2[9, 0] = 0.9
    centers2[9, 1] = 0.4
    sides2[9] = 0.2
    
    configs.append((centers2.copy(), sides2.copy()))
    
    # Config 3: Grid 4x4
    centers3 = np.zeros((10, 2))
    sides3 = np.zeros(10)
    s3 = 0.25
    for i in range(10):
        col = i % 4
        row = i // 4
        centers3[i, 0] = (col + 0.5) * s3
        centers3[i, 1] = (row + 0.5) * s3
        sides3[i] = s3
    configs.append((centers3.copy(), sides3.copy()))
    
    # Config 4: 5+5 two rows
    centers4 = np.zeros((10, 2))
    sides4 = np.zeros(10)
    s4 = 0.2
    for i in range(5):
        centers4[i, 0] = (i + 0.5) * s4
        centers4[i, 1] = 0.75
        sides4[i] = s4
        centers4[5 + i, 0] = (i + 0.5) * s4
        centers4[5 + i, 1] = 0.25
        sides4[5 + i] = s4
    configs.append((centers4.copy(), sides4.copy()))
    
    # Pick the best valid config
    best_c = None
    best_s = None
    best_sc = -np.inf
    for c, s in configs:
        sc = _score(c, s)
        if sc > best_sc:
            best_sc = sc
            best_c = c
            best_s = s
    
    return best_c, best_s

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares
    inside the unit square [0,1]^2.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds deadline
    n = 10
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Step 1: Constructive heuristic
    centers, sides = _constructive_heuristic()
    if centers is not None:
        score = _score(centers, sides)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # Step 2: Local search from the best initial config
    if best_centers is not None and time.time() < deadline - 60:
        centers, sides, score = _local_search(best_centers, best_sides, max_iters=20000, step_size=0.02, deadline=deadline)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # Step 3: Simulated annealing from multiple random starts
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random start
            centers, sides = _random_valid_packing()
            score = _score(centers, sides)
            
            # Local search first
            centers, sides, score = _local_search(centers, sides, max_iters=5000, step_size=0.02, deadline=deadline)
            
            # Then SA
            centers, sides, score = _simulated_annealing(centers, sides, max_iters=5000, deadline=deadline)
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Step 4: Final local search polish
    if best_centers is not None and time.time() < deadline - 5:
        centers, sides, score = _local_search(best_centers, best_sides, max_iters=10000, step_size=0.005, deadline=deadline)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # Return the best result found
    if best_centers is None:
        # Fallback to simple grid
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        s = 0.25
        for i in range(10):
            col = i % 4
            row = i // 4
            centers[i, 0] = (col + 0.5) * s
            centers[i, 1] = (row + 0.5) * s
            sides[i] = s
        return centers, sides
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END

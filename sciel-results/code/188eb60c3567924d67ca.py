"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate(circles, W, H):
    """Check validity and compute sum of radii. circles: (N,3) array [x, y, r]."""
    n = len(circles)
    # Check inside rectangle
    for i in range(n):
        x, y, r = circles[i]
        if x < -1e-9 or y < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return None
        if r < 1e-9:
            return None
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0] - circles[j,0]
            dy = circles[i,1] - circles[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i,2] + circles[j,2] - 1e-9:
                return None
    return np.sum(circles[:, 2])

def _simulated_annealing(W, H, n, seed, max_time):
    """Run SA for a given rectangle. Returns best circles found."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initialize: place circles randomly
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = rng.uniform(0, W)
        circles[i, 1] = rng.uniform(0, H)
        circles[i, 2] = rng.uniform(0.02, 0.1)
    
    # Initial energy: sum of radii (we want to maximize)
    def energy(c):
        """Return -sum(radii) + penalty for violations (we minimize this)."""
        val = -np.sum(c[:, 2])
        # Penalty for being outside
        for i in range(n):
            x, y, r = c[i]
            if x < r: val += 100*(r-x)
            if y < r: val += 100*(r-y)
            if x + r > W: val += 100*(x+r-W)
            if y + r > H: val += 100*(y+r-H)
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i,0]-c[j,0]
                dy = c[i,1]-c[j,1]
                dist = np.sqrt(dx*dx+dy*dy)
                overlap = c[i,2]+c[j,2]-dist
                if overlap > 0:
                    val += 50*overlap
        return val
    
    current_energy = energy(circles)
    best_circles = circles.copy()
    best_energy = current_energy
    best_valid_val = None
    
    T = 1.0
    cooling = 0.9995
    min_T = 1e-6
    
    while T > min_T:
        if time.time() - start_time > max_time:
            break
        
        # Random move: perturb one circle
        idx = rng.randint(n)
        new_circles = circles.copy()
        move_type = rng.randint(3)
        if move_type == 0:
            # Move position
            new_circles[idx, 0] += rng.normal(0, 0.05)
            new_circles[idx, 1] += rng.normal(0, 0.05)
        elif move_type == 1:
            # Grow radius
            new_circles[idx, 2] *= (1 + rng.normal(0, 0.05))
            new_circles[idx, 2] = max(1e-4, new_circles[idx, 2])
        else:
            # Shrink radius
            new_circles[idx, 2] *= (1 - rng.uniform(0, 0.1))
            new_circles[idx, 2] = max(1e-4, new_circles[idx, 2])
        
        # Clamp to rectangle
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, W)
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, H)
        
        new_energy = energy(new_circles)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            circles = new_circles
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_circles = circles.copy()
                # Check if valid
                val = _evaluate(best_circles, W, H)
                if val is not None:
                    best_valid_val = val
        
        T *= cooling
    
    return best_circles, best_valid_val

def _try_aspect_ratio(ratio, n, seed, max_time):
    """Try a specific aspect ratio W/H = ratio."""
    # Perimeter = 2*(W+H) = 4 => W+H = 2
    # W = ratio * H => ratio*H + H = 2 => H = 2/(1+ratio), W = 2*ratio/(1+ratio)
    H = 2.0 / (1.0 + ratio)
    W = 2.0 * ratio / (1.0 + ratio)
    
    circles, val = _simulated_annealing(W, H, n, seed, max_time)
    return circles, val, W, H

def construct():
    n = N
    total_deadline = 120.0  # seconds
    start = time.time()
    
    best_val = 0.0
    best_circles = None
    best_W = 1.0
    best_H = 1.0
    
    # Try a range of aspect ratios
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    # Allocate time: most to ratio=1.0 (square), some to others
    time_budgets = {}
    for r in ratios:
        if r == 1.0:
            time_budgets[r] = 40.0
        elif r in [0.9, 1.1]:
            time_budgets[r] = 20.0
        elif r in [0.8, 1.2]:
            time_budgets[r] = 15.0
        else:
            time_budgets[r] = 8.0
    
    # Sort by priority
    sorted_ratios = sorted(ratios, key=lambda r: -time_budgets[r])
    
    for ratio in sorted_ratios:
        if time.time() - start > total_deadline - 5.0:
            break
        remaining = total_deadline - 5.0 - (time.time() - start)
        t_budget = min(time_budgets[ratio], remaining)
        if t_budget < 1.0:
            break
        
        circles, val, W, H = _try_aspect_ratio(ratio, n, 42, t_budget)
        if val is not None and val > best_val:
            best_val = val
            best_circles = circles
            best_W = W
            best_H = H
    
    # If no valid result found, fall back to simple grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
        best_W = 1.0
        best_H = 1.0
        best_val = np.sum(best_circles[:, 2])
    
    # Final validation
    final_val = _evaluate(best_circles, best_W, best_H)
    if final_val is None:
        # Fallback
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
    
    return best_circles
# EVOLVE-BLOCK-END

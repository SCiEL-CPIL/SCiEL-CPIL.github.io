"""Force-directed simulated annealing for circle packing in unit square."""
N = 26
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time


def _validate_and_fix(arr):
    """Ensure circles are inside [0,1]^2 with non-negative radii."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Clamp radius to be at least 0
    r = np.maximum(r, 1e-6)
    # Clamp centers so circles are inside [0,1]^2
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _penalty(x, y, r):
    """Compute penalty for violations: overlap and boundary."""
    n = len(x)
    penalty = 0.0
    # Boundary penalty: circle must be inside [0,1]^2
    for i in range(n):
        if x[i] < r[i]:
            penalty += (r[i] - x[i]) ** 2
        if x[i] > 1.0 - r[i]:
            penalty += (x[i] - (1.0 - r[i])) ** 2
        if y[i] < r[i]:
            penalty += (r[i] - y[i]) ** 2
        if y[i] > 1.0 - r[i]:
            penalty += (y[i] - (1.0 - r[i])) ** 2
    # Overlap penalty: circles must not overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return penalty


def _force(x, y, r, hardness=100.0):
    """Compute forces on each circle: repulsion from overlaps, attraction to boundaries."""
    n = len(x)
    fx = np.zeros(n)
    fy = np.zeros(n)
    fr = np.zeros(n)
    # Repulsion from overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-12:
                # Repulsive force proportional to overlap
                overlap = min_dist - dist
                force_mag = hardness * overlap
                fx[i] += force_mag * dx / dist
                fy[i] += force_mag * dy / dist
                fx[j] -= force_mag * dx / dist
                fy[j] -= force_mag * dy / dist
                # Also push radii apart slightly
                fr[i] -= 0.1 * hardness * overlap
                fr[j] -= 0.1 * hardness * overlap
    # Boundary forces: push circles inside
    for i in range(n):
        if x[i] < r[i]:
            fx[i] += hardness * (r[i] - x[i])
            fr[i] -= 0.5 * hardness * (r[i] - x[i])
        elif x[i] > 1.0 - r[i]:
            fx[i] -= hardness * (x[i] - (1.0 - r[i]))
            fr[i] -= 0.5 * hardness * (x[i] - (1.0 - r[i]))
        if y[i] < r[i]:
            fy[i] += hardness * (r[i] - y[i])
            fr[i] -= 0.5 * hardness * (r[i] - y[i])
        elif y[i] > 1.0 - r[i]:
            fy[i] -= hardness * (y[i] - (1.0 - r[i]))
            fr[i] -= 0.5 * hardness * (y[i] - (1.0 - r[i]))
    return fx, fy, fr


def _anneal(x, y, r, time_budget=10.0, seed=42):
    """Simulated annealing to find a good packing."""
    rng = np.random.RandomState(seed)
    n = len(x)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    start_time = time.time()
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    steps = 0
    
    best_score = -np.inf
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    while time.time() - start_time < time_budget:
        # Compute forces
        fx, fy, fr = _force(x, y, r, hardness=10.0)
        
        # Propose move: random perturbation
        if rng.random() < 0.7:
            # Move a random circle
            i = rng.randint(n)
            dx = rng.randn() * 0.02 * T
            dy = rng.randn() * 0.02 * T
            x[i] += dx
            y[i] += dy
        else:
            # Change radius of a random circle
            i = rng.randint(n)
            dr = rng.randn() * 0.01 * T
            r[i] = np.clip(r[i] + dr, 0.005, 0.5)
        
        # Clamp to boundaries
        for i in range(n):
            r[i] = max(r[i], 0.005)
            x[i] = np.clip(x[i], 0, 1.0)
            y[i] = np.clip(y[i], 0, 1.0)
        
        # Compute score: sum of radii minus penalty
        score = np.sum(r) - 10.0 * _penalty(x, y, r)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Cool down
        T *= cooling
        steps += 1
    
    return best_x, best_y, best_r


def _polish(x, y, r, time_budget=5.0, seed=42):
    """Local optimization to polish the packing."""
    rng = np.random.RandomState(seed)
    n = len(x)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    start_time = time.time()
    best_score = np.sum(r) - 10.0 * _penalty(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    T = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_budget:
        # Small perturbations
        i = rng.randint(n)
        dx = rng.randn() * 0.005 * T
        dy = rng.randn() * 0.005 * T
        dr = rng.randn() * 0.002 * T
        x[i] += dx
        y[i] += dy
        r[i] = np.clip(r[i] + dr, 0.005, 0.5)
        
        # Clamp
        x[i] = np.clip(x[i], 0, 1.0)
        y[i] = np.clip(y[i], 0, 1.0)
        
        score = np.sum(r) - 10.0 * _penalty(x, y, r)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        T *= cooling
    
    return best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Initial configuration: hexagonal-ish grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side - 0.005
            cells.append([cx, cy, cr])
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = np.sum(r) - 10.0 * _penalty(x, y, r)
    
    # Run multiple annealing trials with different seeds
    seeds = [42, 123, 456, 789, 1024]
    for seed in seeds:
        if time.time() - start_time > deadline - 15.0:
            break
        trial_budget = min(8.0, deadline - (time.time() - start_time) - 5.0)
        if trial_budget < 2.0:
            break
        tx, ty, tr = _anneal(x, y, r, time_budget=trial_budget, seed=seed)
        t_score = np.sum(tr) - 10.0 * _penalty(tx, ty, tr)
        if t_score > best_score:
            best_score = t_score
            best_x, best_y, best_r = tx, ty, tr
    
    # Polish the best result
    if time.time() - start_time < deadline - 5.0:
        polish_budget = min(10.0, deadline - (time.time() - start_time) - 2.0)
        if polish_budget > 1.0:
            px, py, pr = _polish(best_x, best_y, best_r, time_budget=polish_budget, seed=999)
            p_score = np.sum(pr) - 10.0 * _penalty(px, py, pr)
            if p_score > best_score:
                best_x, best_y, best_r = px, py, pr
    
    # Final validation
    result = np.column_stack([best_x, best_y, best_r])
    result = _validate_and_fix(result)
    
    return result
# EVOLVE-BLOCK-END

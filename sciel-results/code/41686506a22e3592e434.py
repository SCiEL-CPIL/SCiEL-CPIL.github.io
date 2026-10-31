"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(20):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _compute_contact_graph(pos, radii, tol=1e-6):
    """
    Compute the contact graph: which circles are touching (or nearly touching).
    Returns a set of (i, j) pairs that are in contact.
    """
    contacts = set()
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] + tol:
                contacts.add((i, j))
    return contacts

def _contact_aware_local_opt(pos, radii, max_iter=100, deadline=None):
    """
    Local optimization that respects the contact graph.
    Only moves circles that are in contact, and only in directions that
    maintain or improve the packing.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute contact graph
        contacts = _compute_contact_graph(pos, radii)
        
        if not contacts:
            break
        
        # For each contact pair, try to increase radii by moving apart
        improved = False
        for (i, j) in contacts:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-12:
                continue
            
            # Direction from i to j
            nx = dx / dist
            ny = dy / dist
            
            # Try moving i in -n direction and j in +n direction
            # to create more space for radius growth
            step = 0.001
            new_pos_i = pos[i].copy()
            new_pos_j = pos[j].copy()
            new_pos_i[0] -= step * nx
            new_pos_i[1] -= step * ny
            new_pos_j[0] += step * nx
            new_pos_j[1] += step * ny
            
            # Check if this is valid (within bounds)
            if (new_pos_i[0] >= radii[i] and new_pos_i[0] <= 1 - radii[i] and
                new_pos_i[1] >= radii[i] and new_pos_i[1] <= 1 - radii[i] and
                new_pos_j[0] >= radii[j] and new_pos_j[0] <= 1 - radii[j] and
                new_pos_j[1] >= radii[j] and new_pos_j[1] <= 1 - radii[j]):
                
                # Try to grow both radii slightly
                growth = 0.0005
                new_radii_i = radii[i] + growth
                new_radii_j = radii[j] + growth
                
                # Check if new radii are valid with the new positions
                valid = True
                for k in range(N):
                    if k == i or k == j:
                        continue
                    # Check distance from i to k
                    dx_ik = new_pos_i[0] - pos[k, 0]
                    dy_ik = new_pos_i[1] - pos[k, 1]
                    dist_ik = np.sqrt(dx_ik*dx_ik + dy_ik*dy_ik)
                    if dist_ik < new_radii_i + radii[k] - 1e-9:
                        valid = False
                        break
                    # Check distance from j to k
                    dx_jk = new_pos_j[0] - pos[k, 0]
                    dy_jk = new_pos_j[1] - pos[k, 1]
                    dist_jk = np.sqrt(dx_jk*dx_jk + dy_jk*dy_jk)
                    if dist_jk < new_radii_j + radii[k] - 1e-9:
                        valid = False
                        break
                
                if valid:
                    pos[i] = new_pos_i
                    pos[j] = new_pos_j
                    radii[i] = new_radii_i
                    radii[j] = new_radii_j
                    improved = True
                    break
        
        if not improved:
            # Try to find any slack and grow
            for i in range(N):
                max_growth = 0.0
                max_growth = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1]) - radii[i]
                
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    slack = dist - radii[i] - radii[j]
                    if slack < max_growth:
                        max_growth = slack
                
                if max_growth > 1e-6:
                    growth = min(max_growth, 0.001)
                    radii[i] += growth
                    improved = True
                    break
        
        if not improved:
            break
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    return best_pos, best_radii

def _differential_evolution_search(pos0, radii0, population_size=30, max_generations=50, deadline=None):
    """
    Differential evolution to explore the space of packings.
    Uses a penalty-based approach for constraints.
    """
    rng = np.random.RandomState(42)
    
    # Initialize population
    population = []
    for i in range(population_size):
        if i == 0:
            pos = pos0.copy()
            radii = radii0.copy()
        else:
            # Perturb the initial solution
            pos = pos0 + rng.uniform(-0.05, 0.05, (N, 2))
            radii = radii0 + rng.uniform(-0.02, 0.02, N)
            pos, radii = _post_process(pos, radii)
        population.append((pos, radii))
    
    best_pos, best_radii = pos0.copy(), radii0.copy()
    best_score = np.sum(radii0)
    
    for gen in range(max_generations):
        if deadline and time.time() > deadline:
            break
        
        # Evaluate fitness for all individuals
        fitness = []
        for pos, radii in population:
            if _is_valid(pos, radii):
                score = np.sum(radii)
            else:
                # Penalty for invalid solutions
                score = -np.sum(radii) - 10.0
            fitness.append(score)
        
        # Find best individual
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_score:
            best_score = fitness[best_idx]
            best_pos = population[best_idx][0].copy()
            best_radii = population[best_idx][1].copy()
        
        # Generate new individuals using differential evolution
        new_population = []
        for i in range(population_size):
            # Select three distinct random individuals
            a, b, c = rng.choice(population_size, size=3, replace=False)
            if len({a, b, c, i}) < 4:
                a, b, c = rng.choice(population_size, size=3, replace=False)
            
            # Differential evolution formula
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            
            trial_pos = population[a][0] + F * (population[b][0] - population[c][0])
            trial_radii = population[a][1] + F * (population[b][1] - population[c][1])
            
            # Crossover
            for k in range(N):
                if rng.random() < CR:
                    continue
                else:
                    trial_pos[k] = population[i][0][k]
                    trial_radii[k] = population[i][1][k]
            
            # Ensure positivity and bounds
            trial_radii = np.maximum(trial_radii, 1e-4)
            trial_pos, trial_radii = _post_process(trial_pos, trial_radii)
            
            new_population.append((trial_pos, trial_radii))
        
        # Replace population with new one
        population = new_population
    
    return best_pos, best_radii

def _initial_greedy_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample random points for better coverage
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98

    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            # Clip to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            pos[k] = [x, y]
            radii[k] = r
            k += 1
            
    # If we have fewer than N, fill remaining with small circles in corners
    while k < N:
        if k == 25:
             pos[k] = [0.5, 0.5] # Center
             radii[k] = 0.05
        else:
             pos[k] = [0.1, 0.1]
             radii[k] = 0.05
        k += 1

    # Ensure validity
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # List of initializers to try
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
    ]
    
    # Try each initializer and run contact-aware local optimization
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 30:
            break
            
        try:
            pos, radii = init_func()
            
            # Check if this initial state is already good
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            
            # Run contact-aware local optimization
            if time.time() > deadline - 25:
                continue
                
            pos_opt, radii_opt = _contact_aware_local_opt(pos, radii, max_iter=200, deadline=deadline - 5)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
        except Exception as e:
            continue
            
    if best_pos is None:
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
        best_score = np.sum(best_radii)

    # Refinement Stage 1: Differential evolution on the best result
    if time.time() < deadline - 30:
        try:
            pos_de, radii_de = _differential_evolution_search(
                best_pos, best_radii, 
                population_size=20, 
                max_generations=30, 
                deadline=deadline - 10
            )
            pos_de, radii_de = _post_process(pos_de, radii_de)
            score_de = np.sum(radii_de)
            if _is_valid(pos_de, radii_de) and score_de > best_score:
                best_score = score_de
                best_pos = pos_de.copy()
                best_radii = radii_de.copy()
        except Exception:
            pass

    # Refinement Stage 2: Contact-aware local optimization on the best result
    if time.time() < deadline - 10:
        try:
            pos_opt, radii_opt = _contact_aware_local_opt(
                best_pos, best_radii, 
                max_iter=500, 
                deadline=deadline - 5
            )
            pos_opt, radii_opt = _post_process(pos_opt, radii_opt)
            score_opt = np.sum(radii_opt)
            if _is_valid(pos_opt, radii_opt) and score_opt > best_score:
                best_score = score_opt
                best_pos = pos_opt.copy()
                best_radii = radii_opt.copy()
        except Exception:
            pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END

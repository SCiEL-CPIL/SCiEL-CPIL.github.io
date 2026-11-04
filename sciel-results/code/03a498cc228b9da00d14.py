"""Circle packing in perimeter-4 rectangle using NSGA-II."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21
DEADLINE = 115.0  # seconds, safety margin from 120s limit


def _perimeter_rect(x, y, r):
    """Compute width and height of bounding rectangle with perimeter 4."""
    # x, y, r are arrays of shape (N,)
    # Bounding box: [min_x, max_x] x [min_y, max_y]
    # Perimeter = 2*(W + H) = 4, so W + H = 2
    # We need to find W, H such that all circles fit.
    # Given circle positions and radii, the minimal bounding box is:
    #   min_x = min(x - r), max_x = max(x + r)
    #   min_y = min(y - r), max_y = max(y + r)
    # But we can also shift/scale. The problem says "variable rectangle of perimeter 4".
    # So we choose the rectangle. The circles must be inside it.
    # The best rectangle for a given set of circles (positions relative to rect) is determined.
    # Let's parameterize: the rectangle has width W and height H with W+H=2.
    # We place circles at (x_i, y_i) with radii r_i.
    # They fit if: min_i(x_i - r_i) >= 0, max_i(x_i + r_i) <= W,
    #              min_i(y_i - r_i) >= 0, max_i(y_i + r_y) <= H
    # To maximize sum of radii, we want large circles.
    # 
    # Alternative approach: fix the rectangle to be W x H with W+H=2.
    # Optimize positions and radii.
    # 
    # Let's use a different parameterization:
    # The rectangle is [0, W] x [0, H] with W + H = 2.
    # We search over W (and H = 2 - W).
    # 
    # Actually, let's think of it as: we have 21 circles in a rectangle of perimeter 4.
    # The rectangle dimensions are part of what we optimize.
    # 
    # For a given configuration of circles (relative positions and radii), 
    # the minimal enclosing rectangle with perimeter 4 can be found.
    # 
    # Simpler: parameterize by (W, x_i, y_i, r_i) for i=1..N with W+H=2, H=2-W.
    # Constraints:
    #   x_i - r_i >= 0, x_i + r_i <= W
    #   y_i - r_i >= 0, y_i + r_i <= H
    #   dist((x_i,y_i),(x_j,y_j)) >= r_i + r_j for i != j
    #   r_i > 0
    # Objective: maximize sum(r_i)
    #
    # This is a non-convex optimization. Let's use NSGA-II.
    
    return None


def _evaluate(x, y, r, W):
    """
    Evaluate a configuration.
    x, y, r: arrays of shape (N,)
    W: width of rectangle, H = 2 - W
    Returns (sum_radii, violation)
    """
    H = 2.0 - W
    
    # Check rectangle bounds
    # x_i - r_i >= 0  =>  r_i <= x_i
    # x_i + r_i <= W  =>  r_i <= W - x_i
    # y_i - r_i >= 0  =>  r_i <= y_i
    # y_i + r_i <= H  =>  r_i <= H - y_i
    
    # For each circle, max allowed radius from bounds:
    max_r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    
    # Clip radii to be feasible w.r.t. bounds
    r_clipped = np.minimum(r, max_r)
    r_clipped = np.maximum(r_clipped, 0.0)
    
    # Check circle-circle constraints
    # dist_ij >= r_i + r_j
    # violation = sum over pairs of max(0, (r_i + r_j) - dist_ij)
    
    # Compute pairwise distances
    pts = np.column_stack([x, y])
    D = cdist(pts, pts)
    
    # Sum of radii pairs
    R = r_clipped[:, None] + r_clipped[None, :]
    
    # Violation for each pair (i,j) with i<j: max(0, R_ij - D_ij)
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    viol_pairs = np.maximum(0.0, R[iu] - D[iu])
    total_viol = np.sum(viol_pairs)
    
    # Also check if any circle is outside (should be handled by clipping, but let's add penalty)
    # Actually, we clipped, so bounds are satisfied. But the clipping changes the radii.
    # The objective should use the actual radii (clipped).
    
    sum_radii = np.sum(r_clipped)
    
    return sum_radii, total_viol


def _nsga2_population_init(pop_size, W_init=1.0):
    """Initialize population."""
    pop = []
    for _ in range(pop_size):
        W = W_init + np.random.uniform(-0.3, 0.3)
        W = np.clip(W, 0.3, 1.7)
        H = 2.0 - W
        
        x = np.random.uniform(0.1, W - 0.1, N)
        y = np.random.uniform(0.1, H - 0.1, N)
        r = np.random.uniform(0.05, 0.15, N)
        pop.append((W, x, y, r))
    return pop


def _nsga2_crossover(parent1, parent2, W_crossover=True):
    """Uniform crossover."""
    W1, x1, y1, r1 = parent1
    W2, x2, y2, r2 = parent2
    
    if W_crossover:
        if np.random.rand() < 0.5:
            W = W1
        else:
            W = W2
    else:
        W = (W1 + W2) / 2.0
    
    mask = np.random.rand(N) < 0.5
    x = np.where(mask, x1, x2)
    y = np.where(mask, y1, y2)
    r = np.where(mask, r1, r2)
    
    return (W, x, y, r)


def _nsga2_mutation(ind, sigma=0.05):
    """Gaussian mutation."""
    W, x, y, r = ind
    W = W + np.random.normal(0, sigma * 0.5)
    W = np.clip(W, 0.3, 1.7)
    H = 2.0 - W
    
    x = x + np.random.normal(0, sigma, N)
    y = y + np.random.normal(0, sigma, N)
    r = r + np.random.normal(0, sigma * 0.5, N)
    
    # Clip to reasonable bounds
    x = np.clip(x, 0.01, W - 0.01)
    y = np.clip(y, 0.01, H - 0.01)
    r = np.clip(r, 0.01, 0.5)
    
    return (W, x, y, r)


def _nsga2_fast_non_dominated_sort(fitnesses):
    """
    Fast non-dominated sort.
    fitnesses: array of shape (pop_size, 2) where [0]=sum_radii (maximize), [1]=violation (minimize)
    Returns: list of fronts, each front is a list of indices
    """
    pop_size = len(fitnesses)
    # Convert to minimization: f0 = -sum_radii, f1 = violation
    f = np.column_stack([-fitnesses[:, 0], fitnesses[:, 1]])
    
    # Dominance: i dominates j if f_i <= f_j and f_i != f_j
    S = [[] for _ in range(pop_size)]  # S[i] = set of indices dominated by i
    n = [0] * pop_size  # n[i] = number of indices that dominate i
    fronts = [[]]
    
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            # Check if i dominates j
            if np.all(f[i] <= f[j]) and np.any(f[i] < f[j]):
                S[i].append(j)
                n[j] += 1
            elif np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                S[j].append(i)
                n[i] += 1
    
    # First front
    for i in range(pop_size):
        if n[i] == 0:
            fronts[0].append(i)
    
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        k += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts


def _nsga2_crowding_distance(front, fitnesses):
    """Compute crowding distance for a front."""
    pop_size = len(front)
    if pop_size <= 2:
        return [float('inf')] * pop_size
    
    f = np.column_stack([-fitnesses[i, 0] for i in front], axis=0)
    # f shape: (front_size, 2)
    
    cd = np.zeros(pop_size)
    
    for m in range(2):
        # Sort by objective m
        order = np.argsort(f[:, m])
        cd[order[0]] = float('inf')
        cd[order[-1]] = float('inf')
        
        f_range = f[order[-1], m] - f[order[0], m]
        if f_range == 0:
            continue
        
        for k in range(1, pop_size - 1):
            cd[order[k]] += (f[order[k + 1], m] - f[order[k - 1], m]) / f_range
    
    return cd


def _nsga2_selection(pop, fitnesses, pop_size):
    """Select next generation using NSGA-II selection."""
    fronts = _nsga2_fast_non_dominated_sort(fitnesses)
    
    next_gen = []
    for front in fronts:
        if len(next_gen) + len(front) <= pop_size:
            next_gen.extend(front)
        else:
            # Need to select from this front
            remaining = pop_size - len(next_gen)
            cd = _nsga2_crowding_distance(front, fitnesses)
            # Sort by crowding distance descending
            order = np.argsort(-cd)
            selected = [front[i] for i in order[:remaining]]
            next_gen.extend(selected)
            break
    
    return next_gen


def construct():
    start_time = time.time()
    
    # Initial population
    pop_size = 60
    pop = _nsga2_population_init(pop_size, W_init=1.0)
    
    # Evaluate initial population
    fitnesses = np.zeros((pop_size, 2))
    for i, (W, x, y, r) in enumerate(pop):
        s, v = _evaluate(x, y, r, W)
        fitnesses[i] = [s, v]
    
    best_fitness = fitnesses.copy()
    best_config = pop.copy()
    
    # Track best valid solution (violation < 1e-6)
    best_valid_sum = -1
    best_valid_config = None
    
    for i in range(pop_size):
        if fitnesses[i, 1] < 1e-6:
            if fitnesses[i, 0] > best_valid_sum:
                best_valid_sum = fitnesses[i, 0]
                best_valid_config = (pop[i][0], pop[i][1].copy(), pop[i][2].copy(), pop[i][3].copy())
    
    gen = 0
    max_gens = 500
    crossover_rate = 0.9
    mutation_rate = 0.3
    
    while gen < max_gens:
        if time.time() - start_time > DEADLINE:
            break
        
        gen += 1
        
        # Selection
        selected_indices = _nsga2_selection(pop, fitnesses, pop_size)
        
        # Crossover and mutation
        children = []
        for i in range(pop_size):
            if np.random.rand() < crossover_rate:
                # Find a partner
                j = np.random.randint(0, pop_size)
                if i == j:
                    j = (j + 1) % pop_size
                child = _nsga2_crossover(pop[selected_indices[i]], pop[selected_indices[j]])
            else:
                child = (pop[selected_indices[i]][0].copy(), 
                         pop[selected_indices[i]][1].copy(), 
                         pop[selected_indices[i]][2].copy(), 
                         pop[selected_indices[i]][3].copy())
            
            if np.random.rand() < mutation_rate:
                child = _nsga2_mutation(child)
            
            children.append(child)
        
        # Evaluate children
        child_fitnesses = np.zeros((pop_size, 2))
        for i, (W, x, y, r) in enumerate(children):
            s, v = _evaluate(x, y, r, W)
            child_fitnesses[i] = [s, v]
        
        # Combine parent and child populations
        combined_pop = pop + children
        combined_fitnesses = np.vstack([fitnesses, child_fitnesses])
        
        # Select next generation
        next_gen_indices = _nsga2_selection(combined_pop, combined_fitnesses, pop_size)
        
        pop = [combined_pop[i] for i in next_gen_indices]
        fitnesses = combined_fitnesses[next_gen_indices]
        
        # Update best
        np.copyto(best_fitness, fitnesses, where=fitnesses[:, 0] > best_fitness[:, 0])
        
        for i in range(pop_size):
            if fitnesses[i, 1] < 1e-6:
                if fitnesses[i, 0] > best_valid_sum:
                    best_valid_sum = fitnesses[i, 0]
                    best_valid_config = (pop[i][0], pop[i][1].copy(), pop[i][2].copy(), pop[i][3].copy())
        
        # Occasionally do local refinement on best valid config
        if best_valid_config is not None and gen % 10 == 0:
            W, x, y, r = best_valid_config
            # Try to improve by scaling radii up slightly and adjusting
            for scale in [1.01, 1.02, 1.05]:
                r_new = r * scale
                s, v = _evaluate(x, y, r_new, W)
                if v < 1e-6 and s > best_valid_sum:
                    best_valid_sum = s
                    best_valid_config = (W, x.copy(), y.copy(), r_new.copy())
    
    # Use best valid config if available, otherwise best overall
    if best_valid_config is not None:
        W, x, y, r = best_valid_config
    else:
        # Find best by sum_radii with minimal violation
        idx = np.argmax(fitnesses[:, 0])
        W, x, y, r = pop[idx]
        # Clip radii to make them valid
        H = 2.0 - W
        max_r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
        r = np.minimum(r, max_r)
        r = np.maximum(r, 0.0)
    
    # Final validation and cleanup
    H = 2.0 - W
    max_r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    r = np.minimum(r, max_r)
    r = np.maximum(r, 0.0)
    
    # Verify disjointness
    pts = np.column_stack([x, y])
    D = cdist(pts, pts)
    R = r[:, None] + r[None, :]
    iu = np.triu_indices(N, 1)
    if np.any(R[iu] > D[iu] + 1e-9):
        # Slightly shrink radii
        viol = np.maximum(0, R[iu] - D[iu])
        # Simple fix: scale down
        scale = 0.95
        r = r * scale
        max_r = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
        r = np.minimum(r, max_r)
        r = np.maximum(r, 0.0)
    
    result = np.column_stack([x, y, r])
    return result


if __name__ == "__main__":
    result = construct()
    print("Result shape:", result.shape)
    print("Sum of radii:", np.sum(result[:, 2]))
    W = np.max(result[:, 0] + result[:, 2])
    H = np.max(result[:, 1] + result[:, 2])
    print(f"Bounding box: W={W:.4f}, H={H:.4f}, Perimeter={2*(W+H):.4f}")

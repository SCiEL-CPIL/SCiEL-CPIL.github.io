"""Circle packing in unit square using NSGA-II style optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _penalty_and_sum(positions, radii, n_circles):
    """
    Compute total penalty (overlap + boundary) and sum of radii.
    positions: (n, 2)
    radii: (n,)
    """
    n = n_circles
    # Boundary violations: circle must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    boundary_viol = np.maximum(0, radii - positions[:, 0]) + \
                    np.maximum(0, radii + positions[:, 0] - 1) + \
                    np.maximum(0, radii - positions[:, 1]) + \
                    np.maximum(0, radii + positions[:, 1] - 1)
    boundary_penalty = np.sum(boundary_viol)
    
    # Pairwise overlap: for each pair i<j, dist < r_i + r_j
    # Overlap amount = (r_i + r_j) - dist if positive
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap * overlap  # quadratic penalty
    
    total_penalty = boundary_penalty + penalty
    sum_radii = np.sum(radii)
    return total_penalty, sum_radii


def _evaluate_solution(individual):
    """
    Evaluate a single individual.
    individual: array of shape (3*n,) where first n are x, next n are y, last n are radii.
    Returns (penalty, sum_radii)
    """
    n = N
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    positions[:, 0] = individual[:n]
    positions[:, 1] = individual[n:2*n]
    radii = individual[2*n:3*n]
    
    # Clamp radii to reasonable range
    radii = np.clip(radii, 0.001, 0.5)
    
    penalty, sum_r = _penalty_and_sum(positions, radii, n)
    return penalty, sum_r


def _nsga2_select(population, fitnesses, pop_size):
    """
    Simple NSGA-II style selection.
    population: list of individuals
    fitnesses: list of (penalty, sum_radii) tuples
    pop_size: target population size
    Returns selected population.
    """
    n_pop = len(population)
    if n_pop <= pop_size:
        return population[:]
    
    # Sort by penalty (primary), then by sum_radii (secondary, descending)
    # We want low penalty and high sum_radii
    indices = list(range(n_pop))
    
    # Non-dominated sorting
    def is_dominated(a, b):
        """a dominates b if a is at least as good in all objectives and strictly better in at least one."""
        # Minimize penalty, maximize sum_radii
        a_pen, a_sum = fitnesses[a]
        b_pen, b_sum = fitnesses[b]
        if a_pen <= b_pen and a_sum >= b_sum:
            if a_pen < b_pen or a_sum > b_sum:
                return True
        return False
    
    # Assign fronts
    fronts = []
    remaining = list(range(n_pop))
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and is_dominated(j, i):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining = [i for i in remaining if i not in front]
    
    # Select from fronts in order
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Take only what we need from this front
            # Sort by crowding distance approximation (just use sum_radii for now)
            front_sorted = sorted(front, key=lambda i: fitnesses[i][1], reverse=True)
            needed = pop_size - len(selected)
            selected.extend(front_sorted[:needed])
            break
    
    return [population[i] for i in selected]


def _crossover(parent1, parent2, crossover_rate=0.8):
    """Blend crossover."""
    if np.random.random() < crossover_rate:
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def _mutate(individual, mutation_rate=0.1, sigma=0.05):
    """Gaussian mutation."""
    n = len(individual)
    mutation_mask = np.random.random(n) < mutation_rate
    if np.any(mutation_mask):
        individual[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * sigma
    # Clamp positions to [0, 1]
    n_c = N
    individual[:n_c] = np.clip(individual[:n_c], 0.0, 1.0)
    individual[n_c:2*n_c] = np.clip(individual[n_c:2*n_c], 0.0, 1.0)
    individual[2*n_c:] = np.clip(individual[2*n_c:], 0.001, 0.5)
    return individual


def construct():
    np.random.seed(42)
    
    n_circles = N
    pop_size = 60
    n_generations = 80
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = np.zeros(3 * n_circles)
        ind[:n_circles] = np.random.uniform(0.05, 0.95, n_circles)
        ind[n_circles:2*n_circles] = np.random.uniform(0.05, 0.95, n_circles)
        ind[2*n_circles:] = np.random.uniform(0.05, 0.2, n_circles)
        population.append(ind)
    
    best_penalty = np.inf
    best_sum = -np.inf
    best_individual = None
    
    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = []
        for ind in population:
            penalty, sum_r = _evaluate_solution(ind)
            fitnesses.append((penalty, sum_r))
        
        # Track best
        for i, (pen, sum_r) in enumerate(fitnesses):
            if pen < best_penalty or (pen == best_penalty and sum_r > best_sum):
                if pen < best_penalty:
                    best_penalty = pen
                    best_sum = sum_r
                    best_individual = population[i].copy()
                elif sum_r > best_sum:
                    best_sum = sum_r
                    best_individual = population[i].copy()
        
        # Selection
        selected_indices = []
        # Non-dominated sorting
        def is_dominated(a, b):
            a_pen, a_sum = fitnesses[a]
            b_pen, b_sum = fitnesses[b]
            if a_pen <= b_pen and a_sum >= b_sum:
                if a_pen < b_pen or a_sum > b_sum:
                    return True
            return False
        
        fronts = []
        remaining = list(range(pop_size))
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j and is_dominated(j, i):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining = [i for i in remaining if i not in front]
        
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size // 2:
                selected.extend(front)
            else:
                needed = pop_size // 2 - len(selected)
                front_sorted = sorted(front, key=lambda i: fitnesses[i][1], reverse=True)
                selected.extend(front_sorted[:max(0, needed)])
                break
        
        # Create new population
        new_population = []
        # Add best individuals
        for idx in selected[:pop_size // 3]:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            i = np.random.randint(0, len(selected))
            j = np.random.randint(0, len(selected))
            if i == j:
                i = (i + 1) % len(selected)
            child1, child2 = _crossover(population[selected[i]], population[selected[j]])
            child1 = _mutate(child1)
            child2 = _mutate(child2)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population[:pop_size]
    
    # Final evaluation of best individual
    if best_individual is not None:
        penalty, sum_r = _evaluate_solution(best_individual)
    
    # Build result from best individual
    if best_individual is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    n = N
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    positions[:, 0] = best_individual[:n]
    positions[:, 1] = best_individual[n:2*n]
    radii = best_individual[2*n:3*n]
    
    # Ensure feasibility: clamp and adjust
    for i in range(n):
        radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        radii[i] = max(radii[i], 0.001)
    
    # Final local refinement with scipy
    def objective(x):
        pos = np.zeros((n, 2))
        rad = np.zeros(n)
        pos[:, 0] = x[:n]
        pos[:, 1] = x[n:2*n]
        rad = x[2*n:3*n]
        rad = np.clip(rad, 0.001, 0.5)
        
        pen = 0.0
        for i in range(n):
            pen += np.maximum(0, rad[i] - pos[i, 0])
            pen += np.maximum(0, rad[i] + pos[i, 0] - 1)
            pen += np.maximum(0, rad[i] - pos[i, 1])
            pen += np.maximum(0, rad[i] + pos[i, 1] - 1)
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = rad[i] + rad[j] - dist
                if overlap > 0:
                    pen += overlap * overlap
        # We want to minimize penalty and maximize sum of radii
        # Use: penalty - sum_radii (so minimizing this is good)
        return pen - np.sum(rad)
    
    # Try local optimization from best individual
    x0 = best_individual.copy()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          bounds=[(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n,
                          options={'maxiter': 500, 'ftol': 1e-10})
        if result.fun < objective(x0):
            x0 = result.x
    except:
        pass
    
    # Build final array
    final_positions = np.zeros((n, 2))
    final_radii = np.zeros(n)
    final_positions[:, 0] = x0[:n]
    final_positions[:, 1] = x0[n:2*n]
    final_radii = x0[2*n:3*n]
    
    # Final clamping for safety
    for i in range(n):
        final_radii[i] = min(final_radii[i], 
                            max(0.001, min(final_positions[i,0], 1-final_positions[i,0],
                                          final_positions[i,1], 1-final_positions[i,1])))
    
    result = np.zeros((n, 3))
    result[:, 0] = final_positions[:, 0]
    result[:, 1] = final_positions[:, 1]
    result[:, 2] = final_radii
    return result

# EVOLVE-BLOCK-END

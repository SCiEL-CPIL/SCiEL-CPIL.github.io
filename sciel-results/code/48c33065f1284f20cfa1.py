import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Targets to try in order of preference
    targets = [17, 16, 15, 14, 13]
    
    best_n = 0
    best_colors = []

    for n in targets:
        if time.time() > deadline - 5: # Leave 5s for final return
            break
            
        # 1. Try algebraic constructions
        alg_colors = try_algebraic(n)
        if alg_colors is not None:
            # Validate just in case
            cycles_np = get_cycles_np(n)
            if is_valid_fast(alg_colors, cycles_np):
                return {'n': n, 'colors': alg_colors}
        
        # 2. Try Genetic Algorithm
        remaining_time = deadline - time.time() - 2
        if remaining_time < 5:
            continue
            
        # Allocate time based on n
        # n=17 is harder, give it more time if available
        if n == 17:
            ga_time = min(remaining_time, 60)
        elif n == 16:
            ga_time = min(remaining_time, 30)
        else:
            ga_time = min(remaining_time, 10)
            
        ga_colors = run_ga(n, ga_time, start_time + ga_time)
        if ga_colors is not None:
            return {'n': n, 'colors': ga_colors}

    # Fallback: Return a small valid coloring if we found nothing better
    # n=5 is always valid?
    # Let's just return n=1 with empty colors or n=2 with 1 color.
    # Actually, any n<=5 is valid because C6 requires 6 vertices.
    # So n=5 is always valid.
    n_fallback = 5
    colors_fallback = [0] * (n_fallback * (n_fallback - 1) // 2)
    return {'n': n_fallback, 'colors': colors_fallback}

def get_cycles_np(n):
    """Generate all 6-cycles in K_n as a numpy array of edge indices."""
    from itertools import combinations, permutations
    
    edges = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges[(i, j)] = idx
            idx += 1
    
    all_cycles = []
    seen = set()
    
    # Optimization: Only iterate unique cycles
    for verts in combinations(range(n), 6):
        v0 = verts[0]
        rest = list(verts[1:])
        
        for perm in permutations(rest):
            # Canonical form: first element < last element to avoid reversal
            if perm[0] > perm[-1]:
                continue
            
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            valid = True
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    valid = False
                    break
                edge_indices.append(edges[(u, v)])
            
            if not valid or len(set(edge_indices)) != 6:
                continue
            
            key = tuple(sorted(edge_indices))
            if key not in seen:
                seen.add(key)
                all_cycles.append(key)
                
    if not all_cycles:
        return np.array([], dtype=np.int32).reshape(0, 6)
        
    return np.array(all_cycles, dtype=np.int32)

def is_valid_fast(colors, cycles_np):
    """Check if coloring has no monochromatic 6-cycles using numpy."""
    if len(cycles_np) == 0:
        return True
        
    colors_np = np.array(colors, dtype=np.int8)
    # Get colors of all edges in all cycles
    # cycles_np shape: (num_cycles, 6)
    # colors_np[cycles_np] shape: (num_cycles, 6)
    edge_colors = colors_np[cycles_np]
    
    # Check if all 6 edges in a cycle have the same color
    # Compare each column to the first column
    is_mono = np.all(edge_colors == edge_colors[:, 0:1], axis=1)
    
    return not np.any(is_mono)

def try_algebraic(n):
    """Try a few algebraic patterns."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    patterns = [
        lambda i, j: (i - j) % 4,
        lambda i, j: (i + j) % 4,
        lambda i, j: (i * j) % 4,
        lambda i, j: (i - j) ** 2 % 4,
        lambda i, j: (i - j) ** 3 % 4,
        lambda i, j: (i ^ j) % 4,
    ]
    
    for p in patterns:
        colors = [p(i, j) for i, j in edges]
        return colors # Caller will validate

def run_ga(n, time_limit, deadline):
    """Run Genetic Algorithm to find valid coloring."""
    num_edges = n * (n - 1) // 2
    cycles_np = get_cycles_np(n)
    
    if len(cycles_np) == 0:
        return [0] * num_edges

    pop_size = 100
    mutation_rate = 0.05
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Start with random, or maybe structured?
        # Random is safer for GA
        ind = [random.randint(0, 3) for _ in range(num_edges)]
        population.append(ind)
    
    best_ind = None
    best_cost = float('inf')
    
    while time.time() < deadline - 1:
        # Evaluate fitness
        fitnesses = []
        for ind in population:
            # Cost = number of monochromatic cycles
            # Using numpy for speed
            colors_np = np.array(ind, dtype=np.int8)
            edge_colors = colors_np[cycles_np]
            is_mono = np.all(edge_colors == edge_colors[:, 0:1], axis=1)
            cost = np.sum(is_mono)
            
            if cost < best_cost:
                best_cost = cost
                best_ind = ind[:]
                if cost == 0:
                    return ind
            
            fitnesses.append(cost)
        
        # Selection (Tournament)
        new_population = []
        for _ in range(pop_size):
            # Pick 2 random, keep better
            i1, i2 = random.sample(range(pop_size), 2)
            if fitnesses[i1] < fitnesses[i2]:
                parent1 = population[i1]
            else:
                parent1 = population[i2]
                
            i1, i2 = random.sample(range(pop_size), 2)
            if fitnesses[i1] < fitnesses[i2]:
                parent2 = population[i1]
            else:
                parent2 = population[i2]
            
            # Crossover (Uniform)
            child = []
            for k in range(num_edges):
                if random.random() < 0.5:
                    child.append(parent1[k])
                else:
                    child.append(parent2[k])
            
            # Mutation
            if random.random() < mutation_rate:
                # Mutate one or few edges
                num_mut = random.randint(1, 3)
                for _ in range(num_mut):
                    idx = random.randint(0, num_edges - 1)
                    child[idx] = random.randint(0, 3)
            
            new_population.append(child)
            
        population = new_population
        
    return best_ind if best_cost == 0 else None

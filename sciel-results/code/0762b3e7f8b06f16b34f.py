"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 17

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state vector (in-place)."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _evaluate_network(comparators, samples=None, sample_size=2048):
    """
    Evaluate a sorting network using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    if samples is None:
        # Generate random Boolean inputs
        samples = np.random.randint(0, 2, size=(sample_size, N))
    
    for sample in samples:
        state = sample.copy()
        for i, j in comparators:
            _apply_comparator(state, i, j)
        # Check if sorted
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _evaluate_network_full(comparators):
    """
    Evaluate a sorting network over ALL 2^17 Boolean inputs (zero-one principle).
    Returns True if the network sorts all inputs.
    """
    for mask in range(1 << N):
        state = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for i, j in comparators:
            _apply_comparator(state, i, j)
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _compute_depth(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # Assign each comparator to a layer (level)
    # A comparator can be at level max(level[i], level[j]) + 1
    level = [0] * N
    depth = 0
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        depth = max(depth, lvl)
    return depth

def _generate_seed_network():
    """Generate the odd-even transposition network as seed."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _mutate(comparators, mutation_rate=0.1):
    """
    Apply mutations to the comparator list.
    Mutations include:
    - Remove a random comparator
    - Swap two comparators
    - Insert a new random comparator
    """
    new_comps = list(comparators)
    n = len(new_comps)
    
    for _ in range(int(mutation_rate * n)):
        op = random.choice(['remove', 'swap', 'insert'])
        if op == 'remove' and n > 0:
            idx = random.randint(0, n - 1)
            new_comps.pop(idx)
            n -= 1
        elif op == 'swap' and n > 1:
            i, j = random.sample(range(n), 2)
            new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
        elif op == 'insert':
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            new_comps.insert(random.randint(0, n), (i, j))
            n += 1
    
    return new_comps

def _crossover(parent1, parent2):
    """Simple crossover: take a random split point."""
    if not parent1 or not parent2:
        return list(parent1 or parent2)
    split = random.randint(0, max(len(parent1), len(parent2)))
    child = parent1[:split] + parent2[split:]
    return child

def _local_search(comparators, max_iters=50):
    """
    Local search: try removing comparators that are redundant.
    A comparator is redundant if removing it still results in a valid sorting network.
    """
    comps = list(comparators)
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        # Try removing each comparator
        for idx in range(len(comps) - 1, -1, -1):
            candidate = comps[:idx] + comps[idx+1:]
            # Quick check with random samples
            if _evaluate_network(candidate, sample_size=512):
                comps = candidate
                changed = True
                break
    return comps

def construct():
    """
    Construct a sorting network using a genetic algorithm approach.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55  # Leave some buffer
    
    # Initialize with seed network
    best = _generate_seed_network()
    best_len = len(best)
    
    # Population
    pop_size = 20
    population = [best]
    for _ in range(pop_size - 1):
        mutated = _mutate(best, mutation_rate=0.2)
        population.append(mutated)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate and select
        scored = []
        for comp in population:
            # Quick evaluation with random samples
            if _evaluate_network(comp, sample_size=1024):
                score = -len(comp)  # Lower is better (we maximize, so negate)
            else:
                score = -10000
            scored.append((score, comp))
        
        scored.sort(key=lambda x: x[0])  # Best first (most negative = fewest comparators)
        
        # Update best
        if scored[0][0] > -10000:  # Valid
            best_comp = scored[0][1]
            if len(best_comp) < best_len:
                # Full verification
                if _evaluate_network_full(best_comp):
                    best = best_comp
                    best_len = len(best)
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best few
        for i in range(min(5, len(scored))):
            if scored[i][0] > -10000:
                new_population.append(scored[i][1])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if len(new_population) < 2:
                parent1 = best
                parent2 = best
            else:
                # Tournament selection
                candidates = random.sample(scored, min(3, len(scored)))
                candidates.sort(key=lambda x: x[0])
                parent1 = candidates[0][1]
                parent2 = candidates[1][1] if len(candidates) > 1 else best
            
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        
        # Local search on best candidate occasionally
        if generation % 5 == 0 and time.time() - start_time < max_time - 10:
            local_result = _local_search(best, max_iters=20)
            if len(local_result) < best_len:
                if _evaluate_network_full(local_result):
                    best = local_result
                    best_len = len(best)
    
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

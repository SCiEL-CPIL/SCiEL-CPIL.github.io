"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import math

_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize 18 bitsets, each of length 2^N.
    bitset[k] has bit i set if the i-th input (0 to 2^N-1) has a 1 in wire k.
    """
    total_bits = 1 << N
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        # Mask for one period: lower half 0s, upper half 1s
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        # Repeat pattern to fill total_bits
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network(comparators):
    """
    Check if network sorts all 2^N Boolean inputs using bitset simulation.
    """
    if not comparators:
        return N <= 1
        
    bitsets = list(_get_initial_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    Uses a randomized order to avoid getting stuck in specific local minima.
    """
    current = list(comparators)
    if not current:
        return current
    
    # Shuffle indices to try removals in a random order
    indices = list(range(len(current)))
    random.shuffle(indices)
    
    # We might need multiple passes if removing one makes another removable?
    # Actually, standard greedy is single pass. But we can do multiple passes if time permits.
    # Let's do a single pass first, then if time allows, another pass on the result.
    
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        
        # Try removing comparator at i
        saved = current[i]
        current.pop(i)
        
        if _is_sorting_network(current):
            # Keep it removed
            pass
        else:
            # Restore
            current.insert(i, saved)
            i += 1
            
    return current

def _ga_step(population, deadline):
    """
    Perform one generation of Genetic Algorithm.
    """
    pop_size = len(population)
    if pop_size == 0:
        return []
    
    # Sort population by fitness (size, lower is better)
    population.sort(key=lambda x: len(x))
    
    new_population = []
    
    # Elitism: Keep the best few
    elite_count = max(1, pop_size // 5)
    for i in range(elite_count):
        new_population.append(population[i])
    
    # Generate rest of population
    while len(new_population) < pop_size:
        if time.time() > deadline:
            break
            
        # Selection: Tournament
        p1 = population[random.randint(0, pop_size - 1)]
        p2 = population[random.randint(0, pop_size - 1)]
        if len(p1) < len(p2):
            parent1 = p1
            parent2 = p2
        else:
            parent1 = p2
            parent2 = p1
            
        # Crossover: Uniform
        # Convert to list of tuples
        child = []
        # We need to handle different lengths. 
        # Strategy: If child gets a comparator from parent1, it's there.
        # To make it simpler, let's do "splice" crossover on the sorted list of unique comparators?
        # No, order matters.
        
        # Simple approach: Randomly pick from parent1 or parent2 for each position up to max len
        max_len = max(len(parent1), len(parent2))
        for i in range(max_len):
            if random.random() < 0.5:
                if i < len(parent1):
                    child.append(parent1[i])
            else:
                if i < len(parent2):
                    child.append(parent2[i])
                    
        # Mutation
        # 1. Add random comparator
        if random.random() < 0.1 and len(child) < 150:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            child.append((i, j))
            
        # 2. Remove random comparator
        if random.random() < 0.1 and len(child) > 0:
            idx = random.randint(0, len(child) - 1)
            child.pop(idx)
            
        # 3. Change random comparator
        if random.random() < 0.1 and len(child) > 0:
            idx = random.randint(0, len(child) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            child[idx] = (i, j)
            
        # Reduce
        # Use a short deadline for reduction to keep GA fast
        reduce_deadline = min(deadline, time.time() + 0.5)
        reduced_child = _greedy_reduce(child, reduce_deadline)
        
        new_population.append(reduced_child)
        
    return new_population

def _sa_refine(network, deadline):
    """
    Simulated Annealing refinement for a specific network.
    """
    current = list(network)
    best = list(current)
    best_size = len(current)
    
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline and T > 0.05:
        r = random.random()
        neighbor = list(current)
        
        if r < 0.5:
            # Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
            else:
                continue
        elif r < 0.7:
            # Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
        else:
            # Change
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                neighbor[idx] = (i, j)
            else:
                continue
                
        if _is_sorting_network(neighbor):
            if len(neighbor) < best_size:
                best = neighbor
                best_size = len(neighbor)
                current = neighbor
            elif len(neighbor) == best_size:
                if random.random() < 0.5:
                    current = neighbor
            else:
                # Accept worse with prob exp(-delta/T)
                delta = len(neighbor) - best_size
                if random.random() < math.exp(-delta / T):
                    current = neighbor
        else:
            # Invalid, reject
            pass
            
        T *= cooling_rate
        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Seed network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    # Initial reduction
    best = _greedy_reduce(seed_network, deadline)
    best_size = len(best)
    
    # Initialize Population
    population = [best]
    pop_size = 20
    
    # Add some random networks
    for _ in range(pop_size - 1):
        rand_net = []
        num_comps = random.randint(50, 100)
        for _ in range(num_comps):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            rand_net.append((i, j))
        reduced_rand = _greedy_reduce(rand_net, min(deadline, time.time() + 0.2))
        population.append(reduced_rand)
        
    # Main Loop
    while time.time() < deadline:
        # GA Step
        new_pop = _ga_step(population, deadline)
        if not new_pop:
            break
        population = new_pop
        
        # Identify best in population
        current_best_in_pop = min(population, key=len)
        
        if len(current_best_in_pop) < best_size:
            best = current_best_in_pop
            best_size = len(best)
            
            # Refine the new best with SA
            refine_deadline = min(deadline, time.time() + 5)
            refined = _sa_refine(best, refine_deadline)
            if len(refined) < best_size:
                best = refined
                best_size = len(best)
                
        # Occasionally, restart population with mutated best to maintain diversity
        if random.random() < 0.1:
            mutated_best = list(best)
            # Add noise
            for _ in range(10):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                mutated_best.append((i, j))
            reduced_mutated = _greedy_reduce(mutated_best, min(deadline, time.time() + 0.5))
            population.append(reduced_mutated)
            population = population[:pop_size] # Keep size constant
            
    return best
# EVOLVE-BLOCK-END

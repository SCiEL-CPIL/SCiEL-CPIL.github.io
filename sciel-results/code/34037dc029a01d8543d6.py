"""HP lattice protein folding using Genetic Algorithm with SAW repair."""
import random
import time
import math

def _compute_energy(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _generate_random_saw(length, dimension):
    """Generate a random self-avoiding walk of given length."""
    if length <= 0:
        return []
    chain = [tuple([0]*dimension)]
    visited = {chain[0]}
    
    for _ in range(length - 1):
        last = chain[-1]
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(last)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in visited:
                    neighbors.append(nb)
        if not neighbors:
            return None # Failed to generate
        next_pos = random.choice(neighbors)
        chain.append(next_pos)
        visited.add(next_pos)
        
    return chain

def _repair_saw(chain, dimension, max_attempts=100):
    """
    Attempt to repair a chain that might have self-intersections or invalid steps.
    Strategy: Find the first index where the walk becomes invalid (self-intersection or non-unit step).
    Truncate the chain at that point and randomly regrow the tail.
    """
    n_target = len(chain)
    if not chain:
        return None
        
    # Validate and find break point
    visited = set()
    valid_prefix = []
    break_idx = 0
    
    for i, p in enumerate(chain):
        # Check unit step
        if i > 0:
            prev = chain[i-1]
            dist = sum(abs(a-b) for a, b in zip(p, prev))
            if dist != 1:
                break_idx = i
                break
        
        # Check self-avoidance
        if p in visited:
            break_idx = i
            break
        
        visited.add(p)
        valid_prefix.append(p)
        
    if len(valid_prefix) == n_target:
        return chain # Already valid
        
    # We need to regrow from valid_prefix
    current_prefix = valid_prefix
    current_visited = set(current_prefix)
    
    for attempt in range(max_attempts):
        tail_len = n_target - len(current_prefix)
        if tail_len == 0:
            return current_prefix
            
        new_tail = [current_prefix[-1]]
        local_visited = set(current_visited)
        success = True
        
        for _ in range(tail_len):
            last = new_tail[-1]
            neighbors = []
            for axis in range(dimension):
                for step in (-1, 1):
                    nb = list(last)
                    nb[axis] += step
                    nb = tuple(nb)
                    if nb not in local_visited:
                        neighbors.append(nb)
                        
            if not neighbors:
                success = False
                break
                
            next_pos = random.choice(neighbors)
            new_tail.append(next_pos)
            local_visited.add(next_pos)
            
        if success:
            return current_prefix + new_tail[1:]
            
        # If regrowth failed, try shortening the prefix slightly to allow more space
        if len(current_prefix) > 2:
            # Remove last 10% of prefix to give more room for tail
            cut = max(1, len(current_prefix) // 10)
            current_prefix = current_prefix[:-cut]
            current_visited = set(current_prefix)
        else:
            # If prefix is too short, start over with a new random prefix
            current_prefix = _generate_random_saw(min(n_target, 10), dimension)
            if not current_prefix:
                return None
            current_visited = set(current_prefix)
            
    return None

def _crossover(parent1, parent2, dimension):
    """
    Crossover two valid chains.
    Pick a random cut point. Take the head from parent1 and tail from parent2.
    This will likely create an invalid chain (step mismatch or self-intersection),
    which is handled by _repair_saw.
    """
    n = len(parent1)
    if n <= 2:
        return parent1[:]
        
    # Choose cut point
    cut = random.randint(1, n - 2)
    
    # Combine
    child = parent1[:cut] + parent2[cut:]
    
    return _repair_saw(child, dimension)

def _mutate(chain, dimension):
    """
    Mutation: Pick a random segment and perturb it.
    Strategy: Pick a start index, keep head, randomly regenerate a middle segment,
    and keep the tail? Or just perturb a small part.
    A common SAW mutation is "bridge move" or "end pivot".
    Here we use a simpler "local regrowth":
    Pick index k. Keep chain[:k]. Regenerate chain[k:] randomly.
    This is essentially a strong end-pivot mutation.
    """
    n = len(chain)
    if n <= 2:
        return chain[:]
        
    # Choose pivot point
    k = random.randint(1, n - 2)
    
    prefix = chain[:k]
    visited = set(prefix)
    
    tail = [prefix[-1]]
    for _ in range(n - k):
        last = tail[-1]
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(last)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in visited:
                    neighbors.append(nb)
        if not neighbors:
            return None # Failed
        next_pos = random.choice(neighbors)
        tail.append(next_pos)
        visited.add(next_pos)
        
    return prefix + tail[1:]

def _genetic_algorithm(sequence, dimension, time_limit):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)], 0

    # Initialize population
    pop_size = 100
    population = []
    
    # Try to generate initial random SAWs
    while len(population) < pop_size:
        chain = _generate_random_saw(n, dimension)
        if chain:
            population.append(chain)
            
    if not population:
        # Fallback: straight line if random fails (rare for small n, but possible for large n in tight space)
        # Actually straight line is a valid SAW.
        population = [[(i, 0, 0) for i in range(n)]]
        
    # Evaluate initial population
    pop_energies = [_compute_energy(sequence, c) for c in population]
    best_chain = population[0]
    best_energy = pop_energies[0]
    for i in range(pop_size):
        if pop_energies[i] < best_energy:
            best_energy = pop_energies[i]
            best_chain = population[i]
            
    start_time = time.time()
    generation = 0
    
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Selection: Tournament selection
        def select_tournament():
            # Pick 2 random individuals, return the better one
            i1, i2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
            if pop_energies[i1] < pop_energies[i2]:
                return population[i1]
            else:
                return population[i2]
                
        new_population = []
        
        # Elitism: Copy best 10%
        elite_size = max(1, pop_size // 10)
        # Sort indices by energy
        indices = sorted(range(pop_size), key=lambda i: pop_energies[i])
        for i in indices[:elite_size]:
            new_population.append(population[i])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            p1 = select_tournament()
            p2 = select_tournament()
            
            # Crossover or Mutation
            if random.random() < 0.7:
                child = _crossover(p1, p2, dimension)
            else:
                child = _mutate(p1, dimension)
                
            if child is None:
                # If repair failed, try to just copy a parent or generate new random
                child = _generate_random_saw(n, dimension)
                if child is None:
                    child = p1 # Fallback to parent
            
            new_population.append(child)
            
        # Evaluate new population
        for i, chain in enumerate(new_population):
            new_population[i] = chain
            pop_energies[i] = _compute_energy(sequence, chain)
            
        # Update best
        for i in range(pop_size):
            if pop_energies[i] < best_energy:
                best_energy = pop_energies[i]
                best_chain = new_population[i]
                
        population = new_population
        
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    # Time budget allocation
    # Total budget ~240s for 26 sequences.
    # 10 seqs of ~48, 6 seqs of ~64-90, 10 seqs of ~100-136.
    # Allocate based on length.
    if n < 50:
        time_budget = 4.0
    elif n < 100:
        time_budget = 8.0
    else:
        time_budget = 12.0
        
    start_time = time.time()
    
    best_chain_global = None
    best_energy_global = 0 # Energy is negative, so 0 is "worse" than any negative energy
    # Actually, initialize to a valid chain's energy
    
    # Run GA
    # We can run multiple GA restarts or just one long GA.
    # Given the time budget, one long GA with elitism is usually stable.
    # However, GA can converge prematurely. Let's do multiple short GAs.
    
    # If time_budget is small, just run one GA for the whole time.
    # If time_budget is large, we can do restarts.
    # For simplicity and robustness, let's run one GA for 95% of the time.
    
    chain, energy = _genetic_algorithm(sequence, dimension, time_budget * 0.95)
    
    if best_energy_global == 0 or energy < best_energy_global:
        best_energy_global = energy
        best_chain_global = chain
        
    if best_chain_global is None:
        # Fallback
        chain = _generate_random_saw(n, dimension)
        if not chain:
            chain = [(i, 0, 0) for i in range(n)]
        best_chain_global = chain
        
    return [list(p) for p in best_chain_global]

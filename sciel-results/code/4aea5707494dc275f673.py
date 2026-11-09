"""HP lattice protein folding using a Memetic Algorithm (Genetic Algorithm + Local Search)."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _is_adjacent(p, q):
    """Check if two points are adjacent (Manhattan distance 1)."""
    return sum(abs(a - b) for a, b in zip(p, q)) == 1


def _compute_energy(sequence, chain):
    """Compute the energy of a chain. Energy = -number of H-H contacts."""
    n = len(sequence)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts


def _generate_random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 1:
        return [(0,) * dimension]
    
    for attempt in range(100):
        chain = [(0,) * dimension]
        occupied = {chain[0]}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    
    # Fallback: straight line (might be invalid if n is large, but serves as a placeholder)
    return [(i,) + (0,) * (dimension - 1) for i in range(n)]


def _local_search(sequence, chain, rng, time_limit=0.5):
    """
    Perform local search (Simulated Annealing) on a given chain to refine it.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return list(chain)
    
    # Ensure chain is a list of tuples for internal processing
    chain = [tuple(p) for p in chain]
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    max_steps = 5000
    start_temp = 5.0
    end_temp = 0.01
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        progress = step / max_steps
        temp = start_temp * math.exp(-5.0 * progress)
        if temp < end_temp:
            temp = end_temp
            
        # Single-residue move
        if n > 2:
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            
            for new_pos in neighbors:
                # Check if new_pos is empty or occupied by i
                if new_pos in where and where[new_pos] != i:
                    continue
                # Check connectivity
                if not _is_adjacent(new_pos, chain[i - 1]):
                    continue
                if not _is_adjacent(new_pos, chain[i + 1]):
                    continue
                
                # Calculate delta energy
                # Remove i from where temporarily
                del where[old_pos]
                where[new_pos] = i
                chain[i] = new_pos
                
                # We need to recompute energy efficiently or just fully if n is small
                # For simplicity and robustness in local search, we recompute fully if n < 50
                # Otherwise, we can do incremental. Given the time limit is short, full recompute is okay for n<100
                # But let's try incremental for speed.
                
                # Contacts lost: H-H pairs involving i or neighbors of i that are no longer neighbors?
                # Actually, only contacts involving i change.
                # Old contacts involving i:
                old_contacts_i = 0
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        j = where.get(q) # Note: where has been updated, so old_pos is not in where
                        # We need to check neighbors of old_pos in the NEW state? No, in the OLD state.
                        # This incremental approach is tricky with the 'where' dict modification.
                        # Let's stick to a slightly safer incremental:
                        # 1. Identify all pairs (i, j) that were contacts before.
                        # 2. Identify all pairs (i, j) that are contacts now.
                        # 3. Delta = New - Old.
                        
                        # Let's revert to full recompute for correctness and simplicity in this short window,
                        # as n is max 136. 136 * 6 lookups is ~800 ops. 5000 steps * 800 = 4M ops. Fast enough.
                        pass 
                
                # Revert for now to use full recompute logic which is safer to implement correctly in a short snippet
                # Actually, let's just recompute full energy. It's O(N).
                new_energy = _compute_energy(sequence, chain)
                delta_energy = new_energy - current_energy
                
                if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / temp):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break
                else:
                    # Revert
                    del where[new_pos]
                    where[old_pos] = i
                    chain[i] = old_pos
                    break # Try next residue or next step? Usually break to pick new i.
            
    return best_chain


def _crossover(seq, parent1, parent2, rng):
    """
    Simple crossover: take a prefix from parent1 and suffix from parent2?
    This is hard for lattice walks because they must connect.
    Instead, we use "Segment Exchange" or just rely on Mutation + Local Search.
    A common GA approach for SAWs is to take a segment from one parent and try to fit it into the other.
    For simplicity and robustness, we will use a "Recombination" strategy:
    1. Pick a cut point.
    2. Take prefix from P1.
    3. Try to extend it using P2's suffix? No, that breaks self-avoidance easily.
    
    Alternative: 
    Since Local Search is strong, we can use a simpler "Crossover" that just creates a new random walk 
    but biases the direction or uses a "Bridge" method.
    
    Let's implement a "Segment Move" based mutation which is more effective for SAWs than standard crossover.
    However, to be a "GA", we need recombination.
    
    Let's try: 
    1. Generate a random walk.
    2. Pick a segment from Parent 1.
    3. Try to insert it into Parent 2? Too complex.
    
    Let's stick to a "Memetic" style where the "Crossover" is actually just generating new candidates 
    via random walks and improving them, but we maintain a population of the best ones.
    Effectively, this is a "Island Model" or "Population Based Local Search".
    
    To satisfy the "Genetic" aspect, we will implement a "Uniform Crossover" on the sequence of moves?
    No, that breaks geometry.
    
    Let's use "Segment Recombination":
    1. Pick two parents A, B.
    2. Pick a segment [i, j] from A.
    3. Try to replace the corresponding segment in B with the shape of A?
    
    Given the complexity, I will implement a "Population of Local Searches" (PLS).
    Each individual is refined by local search.
    "Crossover" will be: Take a random individual, apply a large perturbation (long segment move), and local search.
    This is often more effective than geometric crossover for SAWs.
    """
    pass


def _mutate(sequence, chain, dimension, rng):
    """
    Perform a significant mutation: a long segment move or a random restart of a segment.
    """
    n = len(sequence)
    if n < 5:
        return _generate_random_walk(sequence, dimension, rng)
    
    new_chain = [tuple(p) for p in chain]
    where = {p: i for i, p in enumerate(new_chain)}
    
    # Pick a segment to move
    seg_len = rng.randint(3, min(10, n - 2))
    i_start = rng.randint(1, n - seg_len - 1)
    i_end = i_start + seg_len - 1
    
    # Extract segment
    seg_positions = [new_chain[i_start + k] for k in range(seg_len)]
    
    # Remove segment
    for k in range(seg_len):
        del where[new_chain[i_start + k]]
        
    # Try to reinsert
    anchor_left = new_chain[i_start - 1]
    anchor_right = new_chain[i_end + 1]
    
    # Try to find a new location for the segment
    # We do a random walk of length seg_len starting from a neighbor of anchor_left
    # and check if it ends at a neighbor of anchor_right.
    
    # This is expensive. Let's try a few times.
    for attempt in range(50):
        candidates = [q for q in _neighbours(anchor_left) if q not in where]
        if not candidates:
            continue
        
        start_pos = rng.choice(candidates)
        current_pos = start_pos
        new_seg = [current_pos]
        occupied_temp = {current_pos}
        success = True
        
        for k in range(1, seg_len):
            neighbors = [q for q in _neighbours(current_pos) if q not in where and q not in occupied_temp]
            if not neighbors:
                success = False
                break
            current_pos = rng.choice(neighbors)
            new_seg.append(current_pos)
            occupied_temp.add(current_pos)
            
        if not success:
            continue
            
        # Check if end is adjacent to anchor_right
        if not _is_adjacent(new_seg[-1], anchor_right):
            continue
            
        # Check self-avoidance with rest
        valid = True
        for pos in new_seg:
            if pos in where:
                valid = False
                break
        if not valid:
            continue
            
        # Apply move
        for k in range(seg_len):
            new_chain[i_start + k] = new_seg[k]
            where[new_seg[k]] = i_start + k
            
        return new_chain
        
    # If no valid move found, return original
    return chain


def fold(sequence, dimension):
    """
    Find the best conformation using a Memetic Algorithm.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    # Time budget: 240s total for 26 sequences. ~9s per sequence.
    # We allocate 8.5s for the main loop.
    time_limit_total = 8.5
    start_time = time.time()
    
    # Population size
    pop_size = 20
    
    # Initialize population with random walks
    population = []
    for _ in range(pop_size):
        chain = _generate_random_walk(sequence, dimension, rng)
        # Refine immediately
        chain = _local_search(sequence, chain, rng, time_limit=0.2)
        population.append(chain)
    
    best_chain = max(population, key=lambda c: _compute_energy(sequence, c))
    best_energy = _compute_energy(sequence, best_chain)
    
    # Main Loop
    while time.time() - start_time < time_limit_total:
        # Evaluate population (already refined)
        # Sort by energy
        population.sort(key=lambda c: _compute_energy(sequence, c))
        
        # Update best
        if _compute_energy(sequence, population[0]) < best_energy:
            best_energy = _compute_energy(sequence, population[0])
            best_chain = population[0]
            
        # Selection and Reproduction
        new_population = []
        
        # Elitism: Keep the best 2
        for i in range(min(2, len(population))):
            new_population.append(population[i])
            
        # Fill rest with mutated offspring
        while len(new_population) < pop_size:
            # Pick a parent (tournament selection)
            p1 = rng.choice(population[:5]) # Top 5
            p2 = rng.choice(population[:10]) # Top 10
            
            # Mutate p1
            offspring = _mutate(sequence, p1, dimension, rng)
            
            # Refine offspring with local search
            # Allocate remaining time / remaining individuals
            time_left = time_limit_total - (time.time() - start_time)
            remaining_slots = pop_size - len(new_population)
            if remaining_slots > 0:
                ls_time = max(0.1, time_left / remaining_slots)
            else:
                ls_time = 0.1
                
            offspring = _local_search(sequence, offspring, rng, time_limit=ls_time)
            new_population.append(offspring)
            
        population = new_population
        
    # Final refinement of the best chain
    time_left = time_limit_total - (time.time() - start_time)
    if time_left > 0.5:
        best_chain = _local_search(sequence, best_chain, rng, time_limit=time_left)
        
    # Verify self-avoidance
    seen = set()
    valid = True
    for p in best_chain:
        if p in seen:
            valid = False
            break
        seen.add(p)
        
    # Verify unit steps
    if valid:
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(best_chain[i], best_chain[i - 1]))
            if diff != 1:
                valid = False
                break
                
    if not valid:
        # Fallback
        best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

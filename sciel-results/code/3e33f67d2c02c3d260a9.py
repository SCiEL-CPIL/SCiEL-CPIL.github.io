"""
HP lattice protein folding on the cubic lattice.
Strategy: Genetic Algorithm with splice crossover, error-repair, and local polishing.
This approach explores the space of self-avoiding walks by recombining valid structures
and repairing the resulting invalidities, rather than relying solely on local moves from
single seeds.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill

def _neighbours_list(point):
    """Return list of 6 face-neighbours of point in 3D."""
    x, y, z = point
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours_list(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, bias=1.0):
    """
    Fast initial self-avoiding walk with heuristic bias towards H contacts.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours_list(last)
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            sx, sy, sz = site
            h_score = 0
            free_score = 0
            
            site_neighbors = _neighbours_list(site)
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _is_valid_walk(chain, n):
    """Check if chain is a valid self-avoiding walk of length n."""
    if len(chain) != n:
        return False
    if len(set(chain)) != n:
        return False
    for i in range(n - 1):
        p1, p2 = chain[i], chain[i+1]
        dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        if dist != 1:
            return False
    return True

def _splice_crossover(chain1, chain2, rng):
    """
    Splice two valid walks at a random point to create a child.
    Returns a potentially invalid chain (collisions possible).
    """
    n = len(chain1)
    if n < 4:
        return chain1[:]
    
    # Pick a splice point between 2 and n-2
    k = rng.randint(2, n - 2)
    
    # Child is prefix of chain1 + suffix of chain2
    # We need to translate the suffix of chain2 so that it connects to the end of chain1's prefix
    # The last point of prefix is chain1[k-1]
    # The first point of suffix is chain2[k]
    # We want the new first point of the suffix to be adjacent to chain1[k-1].
    # Ideally, we rotate the suffix so that the vector (chain2[k] - chain2[k-1]) matches some valid direction,
    # or simpler: just translate chain2[k:] so that chain2[k] becomes a neighbor of chain1[k-1].
    
    # Actually, a standard "splicing" in SAWs often involves:
    # Take prefix of A and suffix of B.
    # Translate B's suffix so that B[k] is adjacent to A[k-1].
    
    a_end = chain1[k-1]
    b_start = chain2[k]
    
    # Find a neighbor of a_end that is not a_end itself (obviously) and not the previous node in B (if we were keeping B[k-1], but we are cutting before B[k])
    # We need to map the direction of the step entering B[k] in B (which is B[k]-B[k-1]) to a step entering the new position from A[k-1].
    
    # Let's just pick a random valid neighbor of a_end that is not chain1[k-2] (to avoid immediate backtrack)
    # and use that as the new position for B[k].
    
    neighbors_of_a_end = _neighbours_list(a_end)
    # Filter out the node before a_end in chain1 (if any) to avoid 2-step backtrack
    if k >= 2:
        prev_in_a = chain1[k-2]
        candidates = [p for p in neighbors_of_a_end if p != prev_in_a]
    else:
        candidates = neighbors_of_a_end
    
    if not candidates:
        return chain1[:] # Fallback
    
    new_b_start = rng.choice(candidates)
    
    # Translation vector
    dx = new_b_start[0] - b_start[0]
    dy = new_b_start[1] - b_start[1]
    dz = new_b_start[2] - b_start[2]
    
    # Construct child
    child_prefix = chain1[:k]
    child_suffix = [(x+dx, y+dy, z+dz) for x, y, z in chain2[k:]]
    
    child = child_prefix + child_suffix
    
    return child

def _repair_walk(sequence, chain, rng, max_steps=2000):
    """
    Attempt to repair a potentially invalid walk (collisions or non-unit steps)
    using local slide moves.
    Returns (valid_chain, energy) or (None, None) if failed.
    """
    n = len(chain)
    if n == 0:
        return None, None
    
    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    
    # Check validity
    if not _is_valid_walk(current_chain, n):
        # We need to fix it.
        # Strategy: Identify collisions and try to move one of the conflicting residues.
        # This is a heuristic repair.
        
        occupied = {}
        for i, p in enumerate(current_chain):
            if p in occupied:
                # Collision at p. Move residue i to a free neighbor.
                # Try to move i
                pos = current_chain[i]
                neighbors = _neighbours_list(pos)
                rng.shuffle(neighbors)
                
                # Build current occupied set for quick check
                occ_set = set(current_chain)
                
                moved = False
                for new_pos in neighbors:
                    # Check if new_pos is free
                    if new_pos in occ_set:
                        continue
                    
                    # Check connectivity
                    if i > 0:
                        d1 = abs(current_chain[i-1][0]-new_pos[0]) + abs(current_chain[i-1][1]-new_pos[1]) + abs(current_chain[i-1][2]-new_pos[2])
                        if d1 != 1:
                            continue
                    if i < n - 1:
                        d2 = abs(current_chain[i+1][0]-new_pos[0]) + abs(current_chain[i+1][1]-new_pos[1]) + abs(current_chain[i+1][2]-new_pos[2])
                        if d2 != 1:
                            continue
                    
                    # Move it
                    occ_set.remove(pos)
                    occ_set.add(new_pos)
                    current_chain[i] = new_pos
                    moved = True
                    break
                
                if not moved:
                    # Try moving the other residue involved in collision?
                    # For simplicity, if we can't fix this collision, fail this repair attempt
                    return None, None
            else:
                occupied[p] = i
        
        # After one pass, re-check validity
        if not _is_valid_walk(current_chain, n):
            # Try a few more rounds of random slide moves to shake it out
            for _ in range(max_steps):
                # Pick a random interior residue
                idx = rng.randint(1, n - 2)
                pos = current_chain[idx]
                neighbors = _neighbours_list(pos)
                rng.shuffle(neighbors)
                
                occ_set = set(current_chain)
                
                for new_pos in neighbors:
                    if new_pos in occ_set:
                        continue
                    
                    d1 = abs(current_chain[idx-1][0]-new_pos[0]) + abs(current_chain[idx-1][1]-new_pos[1]) + abs(current_chain[idx-1][2]-new_pos[2])
                    d2 = abs(current_chain[idx+1][0]-new_pos[0]) + abs(current_chain[idx+1][1]-new_pos[1]) + abs(current_chain[idx+1][2]-new_pos[2])
                    
                    if d1 == 1 and d2 == 1:
                        # Valid move
                        occ_set.remove(pos)
                        occ_set.add(new_pos)
                        current_chain[idx] = new_pos
                        break
            
            if not _is_valid_walk(current_chain, n):
                return None, None
            
            current_energy = _energy(sequence, current_chain)
        
    return current_chain, current_energy

def _local_polish(sequence, chain, time_budget, rng):
    """
    Short local search (SA) to polish the chain.
    """
    n = len(chain)
    start_time = time.time()
    
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    T = 1.0
    T_final = 0.1
    # Estimate steps
    steps = 0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_budget:
        # Random slide move
        idx = rng.randint(1, n - 2)
        pos = current_chain[idx]
        neighbors = _neighbours_list(pos)
        new_pos = rng.choice(neighbors)
        
        if new_pos == pos:
            continue
            
        # Check collision
        # Efficient check: only check if new_pos is occupied by someone else
        # We don't have a fast set here, so we use a dictionary if we maintained one, 
        # but for simplicity in this short polish, we can just check the chain.
        # O(N) check is too slow for many steps. Let's use a dict.
        
        # Actually, let's just use a dict for occupied
        # Rebuilding dict every step is O(N). 
        # For a short polish (maybe 1000 steps), O(N) is fine for N=50-100.
        
        occupied = {p: i for i, p in enumerate(current_chain)}
        
        if new_pos in occupied and occupied[new_pos] != idx:
            continue
            
        # Check connectivity
        d1 = abs(current_chain[idx-1][0]-new_pos[0]) + abs(current_chain[idx-1][1]-new_pos[1]) + abs(current_chain[idx-1][2]-new_pos[2])
        d2 = abs(current_chain[idx+1][0]-new_pos[0]) + abs(current_chain[idx+1][1]-new_pos[1]) + abs(current_chain[idx+1][2]-new_pos[2])
        
        if d1 != 1 or d2 != 1:
            continue
            
        # Calculate delta energy
        # Remove old, add new
        # This is expensive to do exactly without a precomputed contact list.
        # Let's just recompute energy. O(N^2) or O(N) with dict.
        # Given the short budget, recomputation is acceptable.
        
        old_chain = current_chain[:]
        current_chain[idx] = new_pos
        new_energy = _energy(sequence, current_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            current_chain = old_chain
            
        T *= cooling_rate
        if T < T_final:
            T = T_final
            
        steps += 1
        if steps > 5000: # Safety cap
            break
            
    return best_chain, best_energy

def _ga_fold(sequence, dimension, time_budget):
    """
    Genetic Algorithm folding.
    """
    n = len(sequence)
    rng = random.Random(54321 + len(sequence))
    
    start_time = time.time()
    
    # 1. Initialize Population
    pop_size = 20
    population = []
    
    for i in range(pop_size):
        bias = 1.0 + i * 0.1
        chain = _greedy_grow(sequence, dimension, rng, bias=bias)
        if chain:
            e = _energy(sequence, chain)
            population.append((chain, e))
            
    if not population:
        # Fallback
        chain = [(i, 0, 0) for i in range(n)]
        return chain
        
    # Sort by energy
    population.sort(key=lambda x: x[1])
    
    best_chain, best_energy = population[0]
    
    # 2. Evolution Loop
    generation = 0
    while time.time() - start_time < time_budget:
        generation += 1
        
        new_population = []
        
        # Elitism: keep top 2
        for i in range(2):
            if i < len(population):
                new_population.append(population[i])
        
        # Generate children
        while len(new_population) < pop_size:
            # Selection: Tournament of 3
            c1 = rng.choice(population[:10])
            c2 = rng.choice(population[:10])
            c3 = rng.choice(population[:10])
            
            # Pick 2 best from tournament
            winners = sorted([c1, c2, c3], key=lambda x: x[1])[:2]
            parent1, parent2 = winners
            
            # Crossover
            child_chain = _splice_crossover(parent1[0], parent2[0], rng)
            
            # Repair
            repaired_chain, repaired_energy = _repair_walk(sequence, child_chain, rng)
            
            if repaired_chain is not None:
                # Polish
                # Allocate small time budget per child
                polish_time = 0.05
                polished_chain, polished_energy = _local_polish(sequence, repaired_chain, polish_time, rng)
                
                if polished_chain:
                    new_population.append((polished_chain, polished_energy))
            else:
                # If repair failed, just use a mutated parent
                # Simple mutation: slide a random residue
                mut_chain = parent1[0][:]
                idx = rng.randint(1, n - 2)
                pos = mut_chain[idx]
                neighbors = _neighbours_list(pos)
                new_pos = rng.choice(neighbors)
                
                # Basic validity check for mutation
                if new_pos not in mut_chain and \
                   abs(mut_chain[idx-1][0]-new_pos[0]) + abs(mut_chain[idx-1][1]-new_pos[1]) + abs(mut_chain[idx-1][2]-new_pos[2]) == 1 and \
                   abs(mut_chain[idx+1][0]-new_pos[0]) + abs(mut_chain[idx+1][1]-new_pos[1]) + abs(mut_chain[idx+1][2]-new_pos[2]) == 1:
                    mut_chain[idx] = new_pos
                    mut_energy = _energy(sequence, mut_chain)
                    new_population.append((mut_chain, mut_energy))
                else:
                    # Keep parent
                    new_population.append(parent1)
        
        population = new_population
        population.sort(key=lambda x: x[1])
        
        if population[0][1] < best_energy:
            best_energy = population[0][1]
            best_chain = population[0][0]
            
        # Check if we've reached the theoretical minimum?
        # Max contacts is bounded. If we find a very low energy, we can stop early?
        # No, just run until time.
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using Genetic Algorithm with repair.
    """
    n = len(sequence)
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 1.0:
        # Emergency fallback
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time per sequence
    # 26 sequences. 325s total.
    # Let's use a fixed 10s per sequence, capped by remaining.
    budget = 10.0
    if remaining - budget < 5.0:
        budget = remaining - 5.0
        
    if budget < 1.0:
        budget = 0.5

    best_chain = _ga_fold(sequence, dimension, budget)
    
    # Final check: ensure self-avoidance
    if len(set(best_chain)) != n:
        # Fallback
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

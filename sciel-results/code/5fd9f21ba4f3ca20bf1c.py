import time
import random
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calc_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _local_search_fixed_steps(sequence, chain, steps, rng):
    """
    Refine a chain using local search with simulated annealing for a fixed number of steps.
    """
    current_chain = chain[:]
    current_energy = _calc_energy(sequence, current_chain)
    occupied = set(current_chain)
    
    t = 1.0
    t_min = 0.01
    cooling_rate = 0.999 # Adjusted for fixed steps
    
    for _ in range(steps):
        # Choose a move
        move_type = rng.random()
        new_chain = None
        new_energy = None
        
        if move_type < 0.7:
            # End move: try to move the last residue
            last_idx = len(current_chain) - 1
            last_pos = current_chain[last_idx]
            prev_pos = current_chain[last_idx - 1]
            
            # Remove last residue
            occupied.remove(last_pos)
            
            candidates = []
            for neighbor in _neighbours(prev_pos):
                if neighbor not in occupied:
                    candidates.append(neighbor)
            
            if candidates:
                new_pos = rng.choice(candidates)
                if new_pos != last_pos:
                    if new_pos not in occupied:
                        new_chain = current_chain[:]
                        new_chain[-1] = new_pos
                        occupied.add(new_pos)
                        new_energy = _calc_energy(sequence, new_chain)
                    else:
                        pass
                else:
                    occupied.add(last_pos)
                    continue
            else:
                occupied.add(last_pos)
                continue
        
        elif move_type < 0.9:
            # 2-End move: try to move the last two residues
            last_idx = len(current_chain) - 1
            second_last_idx = last_idx - 1
            if second_last_idx < 1:
                continue
                
            pos1 = current_chain[second_last_idx]
            pos2 = current_chain[last_idx]
            pos0 = current_chain[second_last_idx - 1]
            
            occupied.remove(pos1)
            occupied.remove(pos2)
            
            candidates = []
            for n1 in _neighbours(pos0):
                if n1 in occupied:
                    continue
                for n2 in _neighbours(n1):
                    if n2 in occupied:
                        continue
                    if n2 == pos0:
                        continue
                    candidates.append((n1, n2))
            
            if candidates:
                n1, n2 = rng.choice(candidates)
                new_chain = current_chain[:]
                new_chain[second_last_idx] = n1
                new_chain[last_idx] = n2
                occupied.add(n1)
                occupied.add(n2)
                new_energy = _calc_energy(sequence, new_chain)
            else:
                occupied.add(pos1)
                occupied.add(pos2)
                continue
        else:
            # Skip or random restart
            continue
            
        if new_chain is None:
            continue
            
        delta_e = new_energy - current_energy
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
        else:
            if rng.random() < math.exp(-delta_e / t):
                current_chain = new_chain
                current_energy = new_energy
        
        t = max(t_min, t * cooling_rate)
        
    return current_chain, current_energy

def _grow_initial(sequence, dimension, rng):
    """
    Generate an initial chain using randomized greedy growth.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using multi-start local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Time budget for this sequence
    seq_time_limit = 10.0
    
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    num_restarts = 0
    steps_per_restart = 2000
    
    while time.time() - start_time < seq_time_limit:
        num_restarts += 1
        rng = random.Random(12345 + num_restarts + len(sequence))
        
        chain = _grow_initial(sequence, dimension, rng)
        
        if chain is None:
            continue
            
        refined_chain, energy = _local_search_fixed_steps(sequence, chain, steps_per_restart, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = refined_chain
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

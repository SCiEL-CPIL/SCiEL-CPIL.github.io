"""HP lattice protein folding using Monte Carlo with End-Pivot Moves."""
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

def _try_end_pivot(chain, k, dimension):
    """
    Try an end-pivot move at index k.
    Keeps chain[0..k] fixed.
    Regenerates chain[k+1..n-1] as a random SAW starting from chain[k].
    Returns new_chain or None if invalid.
    """
    n = len(chain)
    if k >= n - 1:
        return None
        
    fixed_head = chain[:k+1]
    head_set = set(fixed_head)
    
    current_pos = fixed_head[-1]
    new_tail = [current_pos]
    visited_tail = {current_pos}
    
    for _ in range(n - k - 1):
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(current_pos)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in head_set and nb not in visited_tail:
                    neighbors.append(nb)
                    
        if not neighbors:
            return None # Stuck
            
        current_pos = random.choice(neighbors)
        new_tail.append(current_pos)
        visited_tail.add(current_pos)
        
    return fixed_head + new_tail[1:]

def _initial_conformation(sequence, dimension):
    """Generate a valid initial conformation."""
    n = len(sequence)
    # Try to generate a random SAW
    for _ in range(100):
        chain = _generate_random_saw(n, dimension)
        if chain:
            return chain
    # Fallback: straight line
    return [(i, 0, 0) for i in range(n)]

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    # Time budget allocation
    # Total budget ~240s for 26 sequences.
    # 10 seqs of ~48, 6 seqs of ~64-90, 10 seqs of ~100-136.
    # Allocate based on length to balance difficulty.
    if n < 50:
        time_budget = 4.0
    elif n < 100:
        time_budget = 8.0
    else:
        time_budget = 12.0
        
    start_time = time.time()
    
    # Initial state
    chain = _initial_conformation(sequence, dimension)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    temperature = 1.0
    cooling_rate = 0.999
    
    # Monte Carlo loop
    while time.time() - start_time < time_budget * 0.95:
        # Pick random pivot point
        k = random.randint(1, n - 2)
        
        new_chain = _try_end_pivot(chain, k, dimension)
        
        if new_chain is None:
            # Failed to generate a valid tail, skip or cool slightly
            temperature *= cooling_rate
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        temperature *= cooling_rate
        
    return [list(p) for p in best_chain]

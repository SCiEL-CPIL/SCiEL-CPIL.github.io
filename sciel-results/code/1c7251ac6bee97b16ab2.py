"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute the HP energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_fold(sequence, dimension, rng, greedy_factor=1.0):
    """One randomized greedy chain growth."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Compute gain: new H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * greedy_factor + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        best_score = max(o[0] for o in options)
        best_sites = [o[1] for o in options if o[0] == best_score]
        site = rng.choice(best_sites)
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_steps=5000, deadline=None):
    """Improve a conformation via local search (move/flip operations)."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best_chain = list(chain)
    best_energy = current_energy
    
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random residue to try moving
        i = rng.randint(1, n - 2)  # Avoid endpoints for simplicity
        
        # Try all 6 possible moves
        possible_moves = []
        for site in _neighbours(chain[i]):
            if site in occupied:
                # Check if it's the previous or next residue
                if occupied.get(site) == i - 1 or occupied.get(site) == i + 1:
                    continue
                continue
            possible_moves.append(site)
        
        if not possible_moves:
            step += 1
            continue
        
        # Pick a random move
        new_site = rng.choice(possible_moves)
        
        # Check self-avoidance
        if new_site in occupied:
            step += 1
            continue
        
        # Compute energy change
        old_site = chain[i]
        
        # Remove old site
        del occupied[old_site]
        
        # Add new site
        occupied[new_site] = i
        
        # Check if this creates a self-avoidance violation
        valid = True
        for j, p in enumerate(chain):
            if j != i and p == new_site:
                valid = False
                break
        
        if not valid:
            # Revert
            del occupied[new_site]
            occupied[old_site] = i
            step += 1
            continue
        
        new_chain = list(chain)
        new_chain[i] = new_site
        new_energy = _energy(sequence, new_chain)
        
        if new_energy <= best_energy:
            # Accept (even if not better than current, for diversity)
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_chain = list(chain)
                best_energy = new_energy
        else:
            # Revert
            del occupied[new_site]
            occupied[old_site] = i
        
        step += 1
    
    return best_chain, best_energy


def _simulated_annealing(sequence, dimension, rng, start_temp=5.0, end_temp=0.1,
                          max_iter=20000, deadline=None):
    """Simulated annealing for HP protein folding."""
    n = len(sequence)
    
    # Start with a random valid conformation using greedy
    chain = None
    for _ in range(20):
        chain = _greedy_fold(sequence, dimension, rng, greedy_factor=rng.uniform(0.5, 2.0))
        if chain is not None:
            break
    
    if chain is None:
        # Fallback: straight line
        return [(i, 0, 0) for i in range(n)], 0
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    temp = start_temp
    total_iter = max_iter
    step = 0
    
    while step < total_iter:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        
        # Try a random move
        old_site = chain[i]
        possible_moves = [s for s in _neighbours(old_site) 
                         if s not in occupied or occupied.get(s) in (i - 1, i + 1)]
        # Filter out self-avoidance violations
        possible_moves = [s for s in possible_moves if s not in occupied]
        
        if not possible_moves:
            step += 1
            continue
        
        new_site = rng.choice(possible_moves)
        
        # Apply move
        del occupied[old_site]
        occupied[new_site] = i
        
        # Check self-avoidance
        valid = True
        for j, p in enumerate(chain):
            if j != i and p == new_site:
                valid = False
                break
        
        if not valid:
            del occupied[new_site]
            occupied[old_site] = i
            step += 1
            continue
        
        new_chain = list(chain)
        new_chain[i] = new_site
        new_energy = _energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            # Accept
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_chain = list(chain)
                best_energy = new_energy
        else:
            # Revert
            del occupied[new_site]
            occupied[old_site] = i
        
        # Cool down
        cooling = (end_temp / start_temp) ** (1.0 / total_iter)
        temp *= cooling
        step += 1
    
    return best_chain, best_energy


def _hybrid_search(sequence, dimension, deadline):
    """Hybrid approach: greedy initialization + local search refinement."""
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # We want negative energy, so start with 0 as worst
    
    # Multiple greedy starts with different parameters
    for attempt in range(30):
        if deadline is not None and time.time() > deadline:
            break
        
        greedy_factor = 0.5 + attempt * 0.1
        chain = _greedy_fold(sequence, dimension, rng, greedy_factor=greedy_factor)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        # Local search refinement
        refined_chain, refined_energy = _local_search(
            sequence, chain, rng, max_steps=2000, deadline=deadline)
        
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using hybrid search with deadline guard."""
    n = len(sequence)
    
    # Set deadline: 280s for safety margin (evaluator kills at 330s)
    start_time = time.time()
    deadline = start_time + 280.0
    
    # For small sequences, use more intensive search
    if n <= 50:
        # Use simulated annealing with multiple restarts
        best_chain = None
        best_energy = 0
        
        rng = random.Random(12345 + n)
        
        # Try multiple SA restarts
        num_restarts = min(10, max(3, int(280.0 / (n * 0.1))))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            sa_deadline = deadline - 10.0  # Leave 10s buffer
            chain, energy = _simulated_annealing(
                sequence, dimension, rng,
                start_temp=3.0 + restart * 0.5,
                end_temp=0.05,
                max_iter=min(15000, max(5000, 20000 - restart * 1000)),
                deadline=sa_deadline)
            
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        
        if best_chain is None:
            best_chain = [(i, 0, 0) for i in range(n)]
        
        return [list(p) for p in best_chain]
    
    else:
        # For longer sequences, use hybrid approach
        chain, energy = _hybrid_search(sequence, dimension, deadline)
        
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        
        return [list(p) for p in chain]


# EVOLVE-BLOCK-END

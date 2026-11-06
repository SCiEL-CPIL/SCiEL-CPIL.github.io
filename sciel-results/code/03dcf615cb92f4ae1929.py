"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline
_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 170.0

def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, dimension, rng, max_iter=100):
    """
    Fast local search using pivot moves with delta energy calculation.
    """
    if chain is None:
        return None, 1
    
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    where = {p: i for i, p in enumerate(current_chain)}
    
    for _ in range(max_iter):
        if _time_left() < 0.5:
            break
            
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        p = current_chain[i]
        res_i = sequence[i]
        
        best_move = None
        best_delta = 0 # We want negative delta (energy decrease)
        
        for q in _neighbours(p):
            # q must not be occupied by i-1 or i+1
            if q == current_chain[i-1] or q == current_chain[i+1]:
                continue
            
            # q must not be occupied by any other residue
            if q in where:
                continue
            
            # Calculate delta energy
            # Delta E = lost_contacts - gained_contacts
            # If res_i is H, it forms contacts. If P, it doesn't.
            lost = 0
            gained = 0
            
            if res_i == 'H':
                # Lost contacts: neighbors of p that are H and not i-1, i+1
                for neighbor in _neighbours(p):
                    if neighbor == current_chain[i-1] or neighbor == current_chain[i+1]:
                        continue
                    j = where.get(neighbor)
                    if j is not None and sequence[j] == 'H':
                        lost += 1
                
                # Gained contacts: neighbors of q that are H and not i-1, i+1
                for neighbor in _neighbours(q):
                    if neighbor == current_chain[i-1] or neighbor == current_chain[i+1]:
                        continue
                    j = where.get(neighbor)
                    if j is not None and sequence[j] == 'H':
                        gained += 1
            
            delta = lost - gained
            
            if delta < best_delta:
                best_delta = delta
                best_move = q
        
        if best_move is not None and best_delta < 0:
            # Apply move
            current_chain[i] = best_move
            del where[p]
            where[best_move] = i
            current_energy += best_delta
        else:
            # No improving move found. 
            # To avoid getting stuck in local minima, we could try a random valid move 
            # with a small probability (simulated annealing), but for speed and simplicity,
            # we just break. The multiple restarts in _grow provide diversity.
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1 # Unfolded energy is 0, we want negative.
    
    # Phase 1: Generate multiple initial conformations
    max_restarts = 200
    
    for _ in range(max_restarts):
        if _time_left() < 2.0:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local Search on the best chain found
    if _time_left() > 5.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, dimension, rng, max_iter=200)
        if improved_chain is not None and improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

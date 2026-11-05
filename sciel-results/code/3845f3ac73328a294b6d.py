"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=0.0):
    """Randomized greedy chain growth with contact-based scoring."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Count new H-H contacts this placement would create
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            # Count free neighbours for future growth flexibility
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, then randomness
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        if temperature > 0:
            # Simulated annealing: accept worse moves with some probability
            scores = [o[0] for o in options]
            max_score = max(scores)
            min_score = min(scores)
            if max_score > min_score:
                # Boltzmann-like selection
                exp_scores = []
                for s in scores:
                    exp_scores.append(math.exp((s - max_score) / temperature))
                total = sum(exp_scores)
                if total > 0:
                    r = rng.random() * total
                    cum = 0
                    for idx, es in enumerate(exp_scores):
                        cum += es
                        if r <= cum:
                            site = options[idx][1]
                            occupied[site] = i
                            chain.append(site)
                            break
                    else:
                        site = options[-1][1]
                        occupied[site] = i
                        chain.append(site)
                    continue
            
            site = max(options)[1]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_steps=1000):
    """Local search: try to improve by moving individual residues."""
    n = len(sequence)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Build position map
    where = {p: i for i, p in enumerate(chain)}
    
    for step in range(max_steps):
        # Pick a random residue to move
        i = rng.randint(1, n - 2)  # Don't move endpoints
        old_pos = chain[i]
        
        # Try all neighbouring positions
        candidates = []
        for site in _neighbours(old_pos):
            if site in where:
                j = where[site]
                if j == i - 1 or j == i + 1:
                    continue  # Can't move to adjacent in chain
            if site == old_pos:
                continue
            # Check self-avoidance: site must not be occupied by another residue
            if site in where and where[site] != i:
                continue
            candidates.append(site)
        
        if not candidates:
            continue
        
        # Try each candidate
        for new_pos in candidates:
            # Check if new position is valid (not occupied by non-adjacent residue)
            if new_pos in where:
                j = where[new_pos]
                if j != i and abs(j - i) > 1:
                    continue
            
            # Tentatively move
            chain[i] = new_pos
            # Update where map
            del where[old_pos]
            where[new_pos] = i
            
            # Recompute energy (expensive but necessary for correctness)
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                # Revert
                chain[i] = old_pos
                del where[new_pos]
                where[old_pos] = i
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple growth strategies and local search."""
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for 330s wall-clock
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 means no contacts yet
    
    # Strategy 1: Multiple greedy growths with different random seeds
    num_growths = 500
    for attempt in range(num_growths):
        if time.time() > deadline:
            break
        
        chain = _grow(sequence, dimension, rng, temperature=0.0)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # Strategy 2: Local search on the best chain found
    if best_chain is not None and time.time() < deadline - 30:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, max_steps=500)
    
    # Strategy 3: More growths with simulated annealing
    if time.time() < deadline - 60:
        for attempt in range(200):
            if time.time() > deadline:
                break
            temp = 0.5 * (1 - attempt / 200.0) + 0.01
            chain = _grow(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
        
        # Local search on improved chain
        if best_chain is not None and time.time() < deadline - 30:
            best_chain, best_energy = _local_search(sequence, best_chain, rng, max_steps=300)
    
    # Fallback: straight line is always self-avoiding
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

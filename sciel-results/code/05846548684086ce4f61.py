"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 320  # safety margin before 330s kill


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
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """Greedy growth: at each step, pick the site that maximizes H-H contact gain."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        best_site = None
        best_score = -999999
        best_rand = None
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score: immediate H-H contact gain
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 10  # Strong preference for immediate contacts
            
            # Secondary: count of neighboring occupied H residues (potential future contacts)
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and sequence[j] == 'H':
                        score += 1
            
            # Tertiary: free space (to avoid trapping)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            
            # Add small random tiebreaker
            rand_val = rng.random() * 0.01
            total = score + rand_val
            
            if total > best_score or (best_site is None):
                best_score = total
                best_site = site
                best_rand = rand_val
        
        if best_site is None:
            # No valid site - fail
            return None
        
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _local_optimize(sequence, chain, max_iterations=1000):
    """Fast local optimization: try moving each residue to adjacent empty sites."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        if time.time() > _DEADLINE - 5:
            break
        
        for i in range(n):
            if time.time() > _DEADLINE - 5:
                break
            site = chain[i]
            occupied = set(chain)
            occupied.remove(site)
            
            for new_site in _neighbours(site):
                if new_site in occupied:
                    continue
                
                # Check if moving this residue maintains self-avoidance
                # (it does by construction since we removed it)
                new_chain = list(chain)
                new_chain[i] = new_site
                
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
    
    return chain, current_energy


def fold(sequence, dimension):
    """Fold using greedy growth with multiple restarts and local optimization."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Many greedy growth restarts
    start_time = time.time()
    time_budget_growth = min(20.0, (_DEADLINE - start_time - 10) * 0.6)
    
    max_restarts = 500
    restarts_done = 0
    
    for r in range(max_restarts):
        if time.time() > _DEADLINE - 15:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        restarts_done += 1
        
        if chain is not None and len(chain) == n:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_energy = e
                best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local optimization on the best chain
    if time.time() < _DEADLINE - 10:
        best_chain, best_energy = _local_optimize(sequence, best_chain)
    
    # Phase 3: If time remains, do more restarts and optimize
    if time.time() < _DEADLINE - 15:
        for r in range(200):
            if time.time() > _DEADLINE - 10:
                break
            
            chain = _greedy_grow(sequence, dimension, rng)
            if chain is not None and len(chain) == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain
        
        # Optimize the new best
        if time.time() < _DEADLINE - 5:
            best_chain, best_energy = _local_optimize(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

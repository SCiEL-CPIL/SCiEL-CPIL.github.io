"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Deadline guard: 330s wall-clock kill, use 310s as safety margin
_DEADLINE = 310.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
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
    """
    One randomized chain growth.
    Strategy: prefer sites that create new H-H contacts, then maximize free space,
    with small random noise for diversity.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _energy_from_chain(sequence, chain):
    """Fast energy computation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, dimension, rng, time_budget=10.0):
    """
    Local search with 2-opt reversals.
    Uses incremental time budget rather than fixed move count.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy_from_chain(sequence, best_chain)
    
    start = time.time()
    
    while time.time() - start < time_budget:
        if _get_remaining_time() < 2.0:
            break
        
        # 2-opt: pick a < b, reverse chain[a:b+1]
        a = rng.randint(0, n - 2)
        b = rng.randint(a + 1, n - 1)
        
        if b - a == 1:
            continue
        
        new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
        
        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if not valid:
            continue
        
        # Check unit steps at junctions only
        if a > 0:
            diff = sum(abs(new_chain[a-1][d] - new_chain[a][d]) for d in range(dimension))
            if diff != 1:
                valid = False
        if valid and b < n - 1:
            diff = sum(abs(new_chain[b][d] - new_chain[b+1][d]) for d in range(dimension))
            if diff != 1:
                valid = False
        
        if not valid:
            continue
        
        new_energy = _energy_from_chain(sequence, new_chain)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain, best_energy


def _polish(sequence, chain, dimension, rng, time_budget=3.0):
    """
    Final polish: try single-residue moves and short segment reversals.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy_from_chain(sequence, best_chain)
    
    start = time.time()
    
    while time.time() - start < time_budget:
        if _get_remaining_time() < 1.0:
            break
        
        move_type = rng.choice(['single', 'short_rev'])
        
        if move_type == 'single':
            i = rng.randint(1, n - 2)
            current = best_chain[i]
            for site in _neighbours(current):
                if site == best_chain[i-1] or site == best_chain[i+1]:
                    continue
                if site in best_chain:
                    continue
                d1 = sum(abs(best_chain[i-1][d] - site[d]) for d in range(dimension))
                d2 = sum(abs(site[d] - best_chain[i+1][d]) for d in range(dimension))
                if d1 != 1 or d2 != 1:
                    continue
                
                new_chain = best_chain[:i] + [site] + best_chain[i+1:]
                new_energy = _energy_from_chain(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    break
        
        else:  # short_rev
            seg_len = rng.randint(2, 4)
            a = rng.randint(0, n - 1 - seg_len)
            b = a + seg_len
            if b >= n:
                continue
            new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
            
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue
            if a > 0:
                diff = sum(abs(new_chain[a-1][d] - new_chain[a][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
            if valid and b < n - 1:
                diff = sum(abs(new_chain[b][d] - new_chain[b+1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
            if not valid:
                continue
            
            new_energy = _energy_from_chain(sequence, new_chain)
            
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using:
    1. Many randomized greedy growths
    2. Local search (2-opt) on the best candidate
    3. Final polish pass
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Greedy growths
    # Scale restarts based on chain length and remaining time
    if n <= 48:
        n_restarts = 400
    elif n <= 80:
        n_restarts = 200
    else:
        n_restarts = 100
    
    for restart in range(n_restarts):
        if _get_remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Phase 2: Local search on best chain
    if best_chain is not None and _get_remaining_time() > 10.0:
        time_budget = min(15.0, _get_remaining_time() - 5.0)
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, rng, time_budget)
    
    # Phase 3: Final polish pass
    if best_chain is not None and _get_remaining_time() > 4.0:
        time_budget = min(3.0, _get_remaining_time() - 2.0)
        best_chain, best_energy = _polish(sequence, best_chain, dimension, rng, time_budget)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

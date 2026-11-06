"""Improved HP lattice folding: fast growth + efficient local search + time guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy_fast(seq, chain):
    """Compute energy using a position->index dict for O(n) lookup."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth with H-H contact preference."""
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


def _local_search(sequence, chain, rng, max_iters=500):
    """Improve conformation via random segment reversals (U-turns)."""
    n = len(chain)
    if n < 4:
        return chain
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    
    for it in range(max_iters):
        if _remaining() < 1.0:
            break
        # Pick a random segment to reverse
        lo = rng.randint(1, n - 2)
        hi = rng.randint(lo + 1, n - 1)
        if hi - lo < 1:
            continue
        
        # Try reversing segment [lo, hi]
        new_chain = chain[:lo] + list(reversed(chain[lo:hi+1])) + chain[hi+1:]
        
        # Check self-avoidance of the modified segment
        valid = True
        new_set = set(new_chain)
        if len(new_set) != n:
            valid = False
        else:
            # Check unit steps
            for i in range(n - 1):
                a, b = new_chain[i], new_chain[i+1]
                dist = sum(abs(a[k] - b[k]) for k in range(len(a)))
                if dist != 1:
                    valid = False
                    break
        
        if not valid:
            continue
        
        new_energy = _energy_fast(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain


def _local_search2(sequence, chain, rng, max_iters=300):
    """Local search with 2-step moves and small perturbations."""
    n = len(chain)
    if n < 5:
        return chain
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    
    for it in range(max_iters):
        if _remaining() < 1.0:
            break
        # Random reversal of a short segment (length 2-6)
        seg_len = rng.randint(2, min(6, n - 2))
        lo = rng.randint(1, n - 1 - seg_len)
        hi = lo + seg_len
        
        new_chain = chain[:lo] + list(reversed(chain[lo:hi+1])) + chain[hi+1:]
        
        # Quick self-avoidance check
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if not valid:
            continue
        
        # Check unit steps
        for i in range(n - 1):
            a, b = new_chain[i], new_chain[i+1]
            dist = sum(abs(a[k] - b[k]) for k in range(len(a)))
            if dist != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy_fast(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple restarts with local search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    
    # Phase 1: Fast greedy growths to get initial solutions
    num_grow = min(100, max(20, n))
    for _ in range(num_grow):
        if _remaining() < 2.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy_fast(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy_fast(sequence, best)
    
    # Phase 2: Local search on the best and some random solutions
    if _remaining() > 5.0:
        # Local search on current best
        improved = _local_search(sequence, best, rng, max_iters=200)
        improved_energy = _energy_fast(sequence, improved)
        if improved_energy < best_energy:
            best, best_energy = improved, best_energy
    
    # Phase 3: More restarts with local search if time permits
    restarts = 0
    while _remaining() > 3.0:
        restarts += 1
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Quick local search
        improved = _local_search2(sequence, chain, rng, max_iters=100)
        value = _energy_fast(sequence, improved)
        if value < best_energy:
            best, best_energy = improved, value
            # If we found a better solution, do a deeper local search on it
            if _remaining() > 5.0:
                deeper = _local_search(sequence, best, rng, max_iters=150)
                deeper_energy = _energy_fast(sequence, deeper)
                if deeper_energy < best_energy:
                    best, best_energy = deeper, deeper_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

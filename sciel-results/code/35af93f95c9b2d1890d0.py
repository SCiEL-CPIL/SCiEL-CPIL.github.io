"""Improved HP lattice folding: fast growth + efficient local search + time guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == "H":
            # Check 4 neighbours
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                # Count H-H contacts this site would form
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Free space heuristic
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, rng, max_iters=500):
    """Local search via random reversals and small moves. Returns improved chain."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    
    for _ in range(max_iters):
        if _remaining() < 5.0:
            break
        # Try a random reversal
        if n < 4:
            break
        a = rng.randint(0, n - 4)
        b = rng.randint(a + 2, n - 1)
        new_chain = best_chain[:a] + best_chain[a:b + 1][::-1] + best_chain[b + 1:]
        
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
        valid = True
        for i in range(n - 1):
            dx = abs(new_chain[i][0] - new_chain[i + 1][0])
            dy = abs(new_chain[i][1] - new_chain[i + 1][1])
            if dx + dy != 1:
                valid = False
                break
        if not valid:
            continue
        
        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts and local search with time guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # start at 0 (unfolded)
    n = len(sequence)
    
    # Number of restarts based on time available
    # Estimate time per grow: roughly O(n^2) for small n
    # For n=100, one grow is fast; we can do many
    
    num_restarts = 300
    for r in range(num_restarts):
        if _remaining() < 3.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        e = _energy_fast(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain
        
        # Apply local search on the best candidate periodically
        if r % 10 == 0 and best is not None and _remaining() > 10.0:
            # Local search on current best
            improved, imp_e = _local_search(sequence, best, rng, max_iters=100)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved
    
    # Final local search on the overall best if time permits
    if best is not None and _remaining() > 5.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=200)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved
    
    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

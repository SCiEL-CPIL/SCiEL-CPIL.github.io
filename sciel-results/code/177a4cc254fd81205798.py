"""HP lattice protein folding: greedy growth + move-based local search, with a global time guard."""
# EVOLVE-BLOCK-START
import random
import time

_GLOBAL_START = time.time()
# 120 s CPU budget for all sequences; leave a safety margin.
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


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
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, time_limit):
    """Move-based local search. Mutates chain in place; returns best energy found."""
    n = len(chain)
    if n < 4:
        return _energy(sequence, chain)
    
    where = {p: i for i, p in enumerate(chain)}
    best = _energy(sequence, chain)
    deadline = time.time() + time_limit
    
    while time.time() < deadline:
        improved = False
        
        # --- Reversal Moves ---
        # Reverse chain[a:b+1].
        # Valid if chain[a-1] is adjacent to chain[b] AND chain[a] is adjacent to chain[b+1].
        for a in range(1, n - 2):
            for b in range(a + 1, n - 1):
                if time.time() > deadline:
                    break
                p_prev = chain[a - 1]
                p_end = chain[b]
                p_start = chain[a]
                p_next = chain[b + 1]
                
                # Check new junctions are valid (distance 1)
                if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
                    continue
                if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
                    continue
                
                # Apply reversal
                seg = chain[a:b + 1]
                seg.reverse()
                chain[a:b + 1] = seg
                for i, p in enumerate(seg, start=a):
                    where[p] = i
                
                e = _energy(sequence, chain)
                if e < best:
                    best = e
                    improved = True
                else:
                    # Revert
                    seg.reverse()
                    chain[a:b + 1] = seg
                    for i, p in enumerate(seg, start=a):
                        where[p] = i
            if time.time() > deadline:
                break
        if improved:
            continue

        # --- Pivot Moves ---
        # Rotate tail chain[i+1:] around chain[i]
        for i in range(1, n - 2):
            if time.time() > deadline:
                break
            pivot = chain[i]
            tail = chain[i + 1:]
            
            # Points occupied by the prefix (indices 0 to i-1)
            prefix_points = set(chain[:i])
            
            for k in (1, 2, 3):
                if time.time() > deadline:
                    break
                rotated = []
                valid = True
                for p in tail:
                    x = p[0] - pivot[0]
                    y = p[1] - pivot[1]
                    for _ in range(k):
                        x, y = -y, x
                    new_p = (pivot[0] + x, pivot[1] + y)
                    rotated.append(new_p)
                    # Check collision with prefix
                    if new_p in prefix_points:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check junction: rotated[0] must not be chain[i-1]
                if rotated[0] == chain[i - 1]:
                    continue
                
                # Apply
                chain[i + 1:] = rotated
                for j, p in enumerate(rotated, start=i + 1):
                    where[p] = j
                
                e = _energy(sequence, chain)
                if e < best:
                    best = e
                    improved = True
                    break
                else:
                    # Revert
                    chain[i + 1:] = tail
                    for j, p in enumerate(tail, start=i + 1):
                        where[p] = j
            
            if improved:
                break
        
        if not improved:
            break
            
    return best


def fold(sequence, dimension):
    """Best conformation from many randomized greedy growths followed by local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    n = len(sequence)
    # Decide how many growths to attempt based on remaining time.
    while _remaining_time() > 0.5:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Give each chain a small local-search budget.
        ls_time = min(0.2, _remaining_time() * 0.5)
        if ls_time > 0.05:
            _local_search(sequence, chain, ls_time)
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
        # Stop early if we've found a very good conformation.
        if best_energy <= -n // 2:
            break
    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

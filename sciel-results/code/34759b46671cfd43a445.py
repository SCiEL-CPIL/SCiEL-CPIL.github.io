"""HP lattice protein folding on the cubic lattice – stronger local-search design."""
# EVOLVE-BLOCK-START
import random
import time
import heapq
import math

# Global deadline guard: the evaluator kills after 330 s wall-clock.
# We reserve a safety margin of 30 s, so our hard stop is 300 s from import.
_T0 = time.time()
_DEADLINE = _T0 + 300.0


def _check_deadline():
    return time.time() > _DEADLINE


def _neighbours3(p):
    """Yield the 6 face-neighbours of a 3-lattice point (tuple of 3 ints)."""
    x, y, z = p
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _compute_energy(sequence, chain):
    """Compute -number of non-consecutive H-H contacts. chain is list of tuples."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for q in _neighbours3(chain[i]):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """
    Randomized greedy chain growth with simulated-annealing-like noise.
    Returns a self-avoiding walk (list of tuples) or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours3(last):
            if site in occupied:
                continue
            # Count new H-H contacts this placement would create
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours3(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free neighbours (encourages spreading)
            free = 0
            for q in _neighbours3(site):
                if q not in occupied:
                    free += 1
            # Score: prefer high gain, some free space, plus random noise
            noise = rng.gauss(0, temperature * 0.5)
            score = gain * 2.0 + 0.1 * free + noise
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        # Accept top choice with high probability, occasionally pick 2nd or 3rd
        if len(options) > 1 and rng.random() < 0.15:
            idx = min(1 + rng.randint(0, min(2, len(options) - 1)), len(options) - 1)
        else:
            idx = 0
        site = options[idx][1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_iters=5000, time_limit=10.0):
    """
    Simulated annealing local search on the HP chain.
    Moves: pick a random non-terminal residue, try to move it to an adjacent empty site
    (keeping self-avoidance). Accept if energy improves or with probability exp(-dE/T).
    Returns the best chain found.
    """
    n = len(chain)
    if n <= 1:
        return chain
    
    # Build position map
    def build_map(chain):
        return {p: i for i, p in enumerate(chain)}
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    cur_chain = list(chain)
    cur_energy = best_energy
    
    # Initial temperature based on chain length
    T_start = max(0.5, n * 0.05)
    T_end = 0.05
    total_iters = max_iters
    start_time = time.time()
    
    for it in range(total_iters):
        if _check_deadline():
            break
        if it % 500 == 0 and (time.time() - start_time) > time_limit:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (it / max(1, total_iters - 1))
        
        # Pick a random moveable residue (not first or last, and not adjacent to them in a way that breaks connectivity)
        # We move residue k to a neighbouring empty site, then shift the path between k and the new position.
        # Simpler approach: pick two non-adjacent positions and try to swap? No, that breaks self-avoidance easily.
        # Better: single-residue move: pick position k (1 <= k <= n-2), try moving chain[k] to each of its 6 neighbours
        # if that site is empty, and the resulting path from k-1 to k to k+1 is still valid (unit steps).
        # Actually moving a single interior residue is tricky because it must maintain unit-step connectivity.
        # Standard approach: "kink move" or "pivot move". Let's use a simpler "2-opt" style:
        # Reverse a segment of the chain.
        
        # Try a segment reversal (2-opt style)
        if n < 4:
            break
        
        # Pick random segment
        a = rng.randint(1, n - 3)  # start of reversal (not 0)
        b = rng.randint(a + 1, n - 2)  # end of reversal (not n-1)
        
        # Ensure the reversal maintains unit-step connectivity at the cut points
        # After reversing chain[a..b], the new neighbours are:
        #   chain[a-1] -- chain[b]  (was chain[a-1] -- chain[a])
        #   chain[b+1] -- chain[a]  (was chain[b] -- chain[b+1])
        # Check these are unit steps
        pa = chain[a]
        pb = chain[b]
        pa_prev = chain[a - 1]
        pb_next = chain[b + 1]
        
        # Check pa_prev to pb is a unit step
        if abs(pa_prev[0] - pb[0]) + abs(pa_prev[1] - pb[1]) + abs(pa_prev[2] - pb[2]) != 1:
            continue
        # Check pb_next to pa is a unit step
        if abs(pb_next[0] - pa[0]) + abs(pb_next[1] - pa[1]) + abs(pb_next[2] - pa[2]) != 1:
            continue
        
        # Check self-avoidance: no two non-consecutive residues in the new chain share a site
        # We need to verify that the reversed segment doesn't create collisions.
        # Build a set of all positions except the segment, then check the reversed segment against it.
        # And also within the segment, consecutive positions must be distinct (they are, since original was self-avoiding
        # and reversal preserves adjacency structure within the segment).
        
        # Quick check: create set of all positions, remove segment positions, add back in reversed order
        # Actually simpler: just build the new chain and check for duplicates
        new_chain = chain[:a] + list(reversed(chain[a:b+1])) + chain[b+1:]
        
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
        
        # Compute energy change
        new_energy = _compute_energy(sequence, new_chain)
        dE = new_energy - cur_energy
        
        if dE < 0 or rng.random() < math.exp(-dE / max(T, 0.001)):
            cur_chain = new_chain
            cur_energy = new_energy
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_chain = list(cur_chain)
    
    return best_chain


def fold(sequence, dimension):
    """
    Best conformation via multiple greedy growths followed by local search.
    Returns list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if dimension != 3:
        # Fallback for other dimensions: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    rng = random.Random(42 + len(sequence) * 7)
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (energy = 0)
    
    # Phase 1: Generate diverse initial solutions via greedy growth
    num_grows = min(30, max(5, 600 // n))  # Scale with chain length
    initial_chains = []
    
    for _ in range(num_grows):
        if _check_deadline():
            break
        temp = 0.5 + rng.random() * 1.0
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Evaluate all initial chains
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    # Phase 2: Local search on the best initial chains
    # Allocate remaining time for local search
    remaining = _DEADLINE - time.time() - 5.0  # 5s safety margin
    if remaining > 1:
        # Do local search on top few initial chains
        initial_chains.sort(key=lambda c: _compute_energy(sequence, c))
        num_ls = min(5, len(initial_chains))
        time_per_ls = remaining / max(1, num_ls)
        
        for i in range(num_ls):
            if _check_deadline():
                break
            chain = initial_chains[i]
            e = _compute_energy(sequence, chain)
            if e >= best_energy:
                break  # No point searching worse starting points
            
            ls_chain = _local_search(
                sequence, chain, rng,
                max_iters=min(10000, int(time_per_ls * 500)),
                time_limit=max(1.0, time_per_ls - 0.5)
            )
            ls_energy = _compute_energy(sequence, ls_chain)
            if ls_energy < best_energy:
                best_energy = ls_energy
                best_chain = ls_chain
    
    if best_chain is None:
        return [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

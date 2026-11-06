"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search growth (forward) + local optimization (kink moves + simulated annealing)
with deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

_TOTAL_DEADLINE = 320.0  # safety margin from 330s
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: negative of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_growth(sequence, dimension, beam_width=50, rng=None):
    """Beam search growing the chain from start, keeping top-width partial conformations."""
    if rng is None:
        rng = random.Random()
    
    start = (0,) * dimension
    # Each beam entry: (chain_list, occupied_set)
    # We'll store chains as lists of tuples
    beam = [([start], {start})]
    
    n = len(sequence)
    for i in range(1, n):
        candidates = []
        for chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Score: prefer sites that create new H-H contacts
                gain = 0
                if sequence[i] == "H":
                    for nb in _neighbours(site):
                        if nb in occupied:
                            idx = chain.index(nb) if nb in chain else -1
                            # Check if the residue at nb is H and not adjacent
                            if 0 <= idx < len(chain) - 1 and abs(idx - i) > 1 and sequence[idx] == "H":
                                gain += 1
                # Add small randomness for diversity
                score = gain + rng.random() * 0.1
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                candidates.append((score, new_chain, new_occupied))
        
        if not candidates:
            # Dead end - try to salvage or return None
            return None
        
        # Keep top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(c[1], c[2]) for c in candidates[:beam_width]]
    
    # Return the best full chain
    best_chain = None
    best_energy = 0
    for chain, _ in beam:
        if len(chain) == n:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    return best_chain


def _local_optimize(sequence, chain, max_iters=5000, rng=None, time_limit=None):
    """Local optimization using kink moves (pivot moves) and simulated annealing."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 3:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    if time_limit is None:
        time_limit = _time_remaining()
    
    start_t = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() - start_t > time_limit:
            break
        
        # Choose a kink: pick i in [1, n-2], move chain[i] to an adjacent free site
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        
        # Get neighbours of old_pos
        moved = False
        for new_pos in _neighbours(old_pos):
            # Check self-avoidance: new_pos must not be occupied by any other residue
            # and must be unit step from both i-1 and i+1
            if new_pos == current[i-1] or new_pos == current[i+1]:
                # Unit step check
                pass
            else:
                continue
            
            # Check that new_pos is not occupied by any j != i
            occupied = True
            for j in range(n):
                if j != i and current[j] == new_pos:
                    occupied = False
                    break
            if occupied:
                continue
            
            # Also check unit steps: distance from current[i-1] to new_pos is 1, and new_pos to current[i+1] is 1
            def _dist1(a, b):
                return sum(abs(a[k] - b[k]) for k in range(len(a))) == 1
            
            if not _dist1(current[i-1], new_pos) or not _dist1(new_pos, current[i+1]):
                continue
            
            # Try the move
            current[i] = new_pos
            new_energy = _energy(sequence, current)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best = list(current)
            else:
                # Revert
                current[i] = old_pos
            
            moved = True
            break
        
        T = max(T * cooling, T_min)
    
    return best


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
    
    return chain


def fold(sequence, dimension):
    """Best conformation using beam search + local optimization with deadline guard."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Fallback: straight line
    fallback = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    fallback_energy = _energy(sequence, fallback)
    best_chain = list(fallback)
    best_energy = fallback_energy
    
    # Phase 1: Beam search for initial conformations
    time_left = _time_remaining()
    if time_left > 5.0:
        # Adjust beam width based on sequence length and time
        if n <= 50:
            bw = 100
        elif n <= 80:
            bw = 50
        else:
            bw = 30
        
        chain = _beam_growth(sequence, dimension, beam_width=bw, rng=rng)
        if chain is not None and len(chain) == n:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_chain = list(chain)
                best_energy = e
    
    # Phase 2: Local optimization on best chain
    time_left = _time_remaining()
    if time_left > 10.0 and best_chain is not None:
        iters = min(10000, int(time_left * 500))
        optimized = _local_optimize(sequence, best_chain, max_iters=iters, rng=rng,
                                     time_limit=time_left - 5.0)
        if optimized is not None and len(optimized) == n:
            e = _energy(sequence, optimized)
            if e < best_energy:
                best_chain = list(optimized)
                best_energy = e
    
    # Phase 3: If time remains, try a few random walks + local opt
    time_left = _time_remaining()
    if time_left > 15.0 and n <= 80:
        attempts = min(3, int(time_left / 10))
        for _ in range(attempts):
            time_left = _time_remaining()
            if time_left < 10.0:
                break
            rw = _random_walk(sequence, dimension, rng)
            if rw is not None and len(rw) == n:
                e = _energy(sequence, rw)
                if e < best_energy:
                    best_chain = list(rw)
                    best_energy = e
                # Quick local opt
                time_left = _time_remaining()
                if time_left > 8.0:
                    iters = min(3000, int(time_left * 300))
                    opt = _local_optimize(sequence, rw, max_iters=iters, rng=rng,
                                           time_limit=time_left - 5.0)
                    if opt is not None and len(opt) == n:
                        e2 = _energy(sequence, opt)
                        if e2 < best_energy:
                            best_chain = list(opt)
                            best_energy = e2
    
    # Final check: ensure self-avoiding and valid
    if best_chain is not None and len(best_chain) == n:
        # Verify self-avoidance
        if len(set(tuple(p) for p in best_chain)) == n:
            return [list(p) for p in best_chain]
    
    # Ultimate fallback
    return [list(p) for p in fallback]
# EVOLVE-BLOCK-END

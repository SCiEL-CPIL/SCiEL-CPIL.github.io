"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 170.0  # Safety margin for 210s kill

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

def _grow(sequence, dimension, rng, bias=1.0):
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
            score = bias * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _pivot_move(chain, rng):
    """Try a pivot move: rotate a subchain. Returns new chain or None if invalid."""
    n = len(chain)
    if n < 4:
        return None
    pivot = rng.randint(2, n - 2)
    head = chain[:pivot]
    tail = chain[pivot:]
    
    px, py = chain[pivot]
    
    # Try rotation by 90 degrees clockwise: (dx,dy) -> (dy,-dx)
    new_tail = [chain[pivot]]
    for i in range(1, len(tail)):
        dx = tail[i][0] - tail[i-1][0]
        dy = tail[i][1] - tail[i-1][1]
        new_dx = dy
        new_dy = -dx
        new_tail.append((new_tail[-1][0] + new_dx, new_tail[-1][1] + new_dy))
    
    new_chain = head + new_tail
    
    if len(set(new_chain)) != n:
        return None
    
    dx = new_tail[0][0] - head[-1][0]
    dy = new_tail[0][1] - head[-1][1]
    if abs(dx) + abs(dy) != 1:
        return None
    
    return new_chain

def _segment_move(chain, rng):
    """Try a segment move: rotate an internal segment. Returns new chain or None if invalid."""
    n = len(chain)
    if n < 5:
        return None
    
    seg_len = rng.randint(2, min(5, n - 3))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    before = chain[:start]
    segment = chain[start:end]
    after = chain[end:]
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = directions[rng.randint(0, 3)]
    
    new_segment = [(p[0] + dx, p[1] + dy) for p in segment]
    
    new_chain = before + new_segment + after
    
    if len(set(new_chain)) != n:
        return None
    
    dx1 = new_segment[0][0] - before[-1][0]
    dy1 = new_segment[0][1] - before[-1][1]
    if abs(dx1) + abs(dy1) != 1:
        return None
    
    if after:
        dx2 = after[0][0] - new_segment[-1][0]
        dy2 = after[0][1] - new_segment[-1][1]
        if abs(dx2) + abs(dy2) != 1:
            return None
    
    return new_chain

def _local_search(sequence, chain, rng, max_steps=2000):
    """Local search using pivot and segment moves with simulated annealing."""
    current = chain[:]
    current_energy = _energy(sequence, current)
    best = current[:]
    best_energy = current_energy
    
    # Simple simulated annealing
    T = 5.0
    cooling = 0.99
    
    for step in range(max_steps):
        if _time_left() < 5.0:
            break
        
        if step % 2 == 0:
            new_chain = _pivot_move(current, rng)
        else:
            new_chain = _segment_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = current[:]
                best_energy = current_energy
        else:
            delta = current_energy - new_energy
            if delta < 100: # Cap delta for exp
                accept_prob = math.exp(-delta / max(0.1, T))
            else:
                accept_prob = 0.0
            if rng.random() < accept_prob:
                current = new_chain
                current_energy = new_energy
        
        T *= cooling
    
    return best, best_energy

def fold(sequence, dimension):
    """Best conformation using randomized growth + local search with time budget."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Phase 1: Generate a large pool of initial solutions
    # We allocate ~150 seconds to this phase, leaving 20s for refinement
    # Growth is very fast, so we can do thousands of them
    
    best_solutions = []
    best_energy = 1
    
    # Track top 5 solutions to avoid redundant local search
    # We will store (energy, chain)
    
    # Estimate time per growth: for n=100, it's very fast. 
    # Let's just loop until time is up.
    
    while _time_left() > 20.0:
        bias = 0.5 + 0.5 * rng.random()
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_solutions.insert(0, (energy, chain[:]))
            if len(best_solutions) > 5:
                best_solutions.pop()
        else:
            # Check if this is a unique good solution
            # Simple check: if it's significantly better than the worst in our top list, replace
            if len(best_solutions) < 5:
                best_solutions.append((energy, chain[:]))
                best_solutions.sort(key=lambda x: x[0])
            elif energy < best_solutions[-1][0]:
                best_solutions[-1] = (energy, chain[:])
                best_solutions.sort(key=lambda x: x[0])
                
    if not best_solutions:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Phase 2: Refine the top solutions
    final_best = best_solutions[0][1]
    final_energy = best_solutions[0][0]
    
    # We have ~20 seconds left. 
    # Distribute this time among the top solutions.
    # Let's give more time to the best one.
    
    # Sort best_solutions by energy (best is first)
    best_solutions.sort(key=lambda x: x[0])
    
    for i, (e, c) in enumerate(best_solutions):
        if _time_left() < 2.0:
            break
            
        # Allocate steps based on remaining time and number of solutions
        # Rough estimate: 1 step ~ 0.5ms? So 20s ~ 40000 steps total.
        # Let's just run local search until time is low.
        
        # We will run local search on the top 3 solutions if possible
        if i > 2:
            break
            
        # Run local search
        improved, improved_energy = _local_search(sequence, c, rng, max_steps=10000)
        
        if improved_energy < final_energy:
            final_best = improved
            final_energy = improved_energy
            
    return [list(p) for p in final_best]
# EVOLVE-BLOCK-END

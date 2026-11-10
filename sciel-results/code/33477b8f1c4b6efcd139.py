"""HP lattice protein folding with beam search + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def _neighbours(point):
    """Yield all 6 neighbors of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=50, rng=None):
    """
    Beam search: grow chain while keeping top beam_width partial solutions.
    Each state is (chain, occupied_set, energy_so_far).
    Returns list of complete chains.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n == 0:
        return [[]]
    
    # Initial state
    start = (0, 0, 0)
    beam = [([start], {start: 0}, 0)]
    
    for i in range(1, n):
        candidates = []
        for chain, occupied, e in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_chain = chain + [site]
                new_occupied = occupied | {site: i}
                new_e = e
                # Compute new contacts formed by adding residue i
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            new_e -= 1  # one more contact, energy decreases
                candidates.append((new_e, new_chain, new_occupied))
        
        if not candidates:
            break
        
        # Sort by energy (lower is better) and keep top beam_width
        candidates.sort(key=lambda x: x[0])
        beam = [(c[1], c[2], c[0]) for c in candidates[:beam_width]]
        
        if not beam:
            break
    
    # Return all complete chains from beam
    results = [chain for chain, occ, e in beam if len(chain) == n]
    return results

def _sa_polish(sequence, chain, rng, max_iters=5000, t_start=1.0, t_end=0.01):
    """
    Simulated annealing with local moves (rotate subchain).
    Returns best chain found.
    """
    n = len(sequence)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Precompute occupied set for current chain
    def _get_occupied(chain):
        return {p: i for i, p in enumerate(chain)}
    
    cur_occ = _get_occupied(current_chain)
    
    t = t_start
    cooling = (t_start - t_end) / max(1, max_iters)
    
    for it in range(max_iters):
        if it % 100 == 0 and it > 0:
            t = max(t_end, t - cooling * 100)
        
        # Choose a random pivot and direction
        pivot = rng.randint(1, n - 2)
        length = rng.randint(2, min(10, n - pivot - 1))
        
        if length < 2:
            continue
        
        # Try rotating the subchain [pivot:pivot+length] by 90 degrees around one axis
        # We'll try all 6 possible rotations (3 axes x 2 directions)
        improved = False
        for axis in range(3):
            for sign in (-1, 1):
                # Check if rotation is valid (no self-avoidance violation)
                new_chain = list(current_chain)
                sub = current_chain[pivot:pivot+length]
                
                # Rotate subchain around axis through pivot point
                pivot_point = current_chain[pivot]
                rotated_sub = []
                for p in sub:
                    rel = [p[d] - pivot_point[d] for d in range(3)]
                    # Rotate: for axis 0, swap y and z with sign
                    if axis == 0:
                        new_rel = [rel[0], sign * rel[2], -sign * rel[1]]
                    elif axis == 1:
                        new_rel = [sign * rel[2], rel[1], -sign * rel[0]]
                    else:
                        new_rel = [-sign * rel[1], sign * rel[0], rel[2]]
                    new_p = [pivot_point[d] + new_rel[d] for d in range(3)]
                    rotated_sub.append(tuple(new_p))
                
                # Check self-avoidance
                valid = True
                new_occ = {p: i for i, p in enumerate(current_chain[:pivot])}
                for idx, p in enumerate(rotated_sub):
                    if p in new_occ:
                        valid = False
                        break
                    new_occ[p] = pivot + idx
                for idx, p in enumerate(current_chain[pivot+length:]):
                    if p in new_occ:
                        valid = False
                        break
                    new_occ[p] = pivot + length + idx
                
                if not valid:
                    continue
                
                new_chain = current_chain[:pivot] + rotated_sub + current_chain[pivot+length:]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
                    current_chain = new_chain
                    current_energy = new_energy
                    cur_occ = _get_occupied(current_chain)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break  # Accept first valid move
            if improved:
                break
        
        t = max(t_end, t - cooling)
    
    return best_chain

def fold(sequence, dimension):
    """Fold sequence into 3D lattice, maximizing H-H contacts."""
    start_time = time.time()
    deadline = start_time + 240  # 240s budget with safety margin
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Beam search to get diverse initial conformations
    beam_width = 30
    initial_chains = _beam_search(sequence, dimension, beam_width=beam_width, rng=rng)
    
    # Deduplicate
    seen = set()
    unique_chains = []
    for c in initial_chains:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_chains.append(c)
    
    if not unique_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 2: SA polish on each initial conformation
    for i, init_chain in enumerate(unique_chains):
        if time.time() > deadline - 10:
            break
        
        # Determine SA iterations based on remaining time
        remaining = deadline - 10 - time.time()
        max_iters = min(3000, max(500, int(remaining * 500)))
        
        polished = _sa_polish(sequence, init_chain, rng, max_iters=max_iters)
        e = _energy(sequence, polished)
        
        if e < best_energy:
            best_energy = e
            best_chain = polished
    
    if best_chain is None:
        best_chain = unique_chains[0]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

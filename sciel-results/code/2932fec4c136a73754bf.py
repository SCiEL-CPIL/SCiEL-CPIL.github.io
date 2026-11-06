"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search with incremental contact tracking + fast local optimization (SA with Kink/Crank moves).
Deadline guard for 330s wall-clock limit.
"""
import time
import random
import math
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 330.0  # wall-clock seconds
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    if j > i:
                        contacts += 1
    return -contacts


def _beam_search(sequence, dimension, rng, beam_width=100):
    """Beam search over partial chains with contact-aware scoring."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    # State: (score, chain, occupied_set, pos_to_idx)
    # Higher score is better
    # Score = contacts * 1.0 + free_neighbors * 0.1
    beam = [(0, [start], {start}, {start: 0})]
    
    for i in range(1, n):
        if _time_left() < 10:
            return None
        candidates = []
        for score, chain, occupied, pos_to_idx in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Calculate new contacts
                new_contacts = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = pos_to_idx[q]
                            if abs(j - i) > 1 and sequence[j] == "H":
                                new_contacts += 1
                            
                # Calculate free neighbors for the new site
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                
                # Penalize dead ends
                if free == 0:
                    continue # Don't add dead ends to beam
                
                new_score = score + new_contacts * 1.0 + free * 0.1
                
                new_occupied = set(occupied)
                new_occupied.add(site)
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = i
                new_chain = chain + [site]
                
                candidates.append((new_score, new_chain, new_occupied, new_pos_to_idx))
        
        if not candidates:
            return None
        
        # Keep top beam_width candidates
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        beam = candidates[:beam_width]
        
        # Optional: Prune based on diversity? 
        # For now, just keep top scores.
        
    if not beam:
        return None
    
    # Return the best complete chain
    return beam[0][1]


def _kink_move(chain, i, occupied, pos_to_idx, sequence, dimension):
    """Try to move residue i to an adjacent free site (kink move).
    Returns (new_chain, new_occupied, new_pos_to_idx, delta_energy) or None."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    
    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    old_pos = chain[i]
    
    # Remove old_pos from occupied temporarily for checking? 
    # No, we check if the new site is free.
    # The new site must be a neighbor of old_pos.
    # And it must be adjacent to prev_pos and next_pos.
    
    for site in _neighbours(old_pos):
        if site in occupied:
            continue
        
        # Check connectivity
        dist_prev = sum(abs(site[k] - prev_pos[k]) for k in range(dimension))
        dist_next = sum(abs(site[k] - next_pos[k]) for k in range(dimension))
        
        if dist_prev == 1 and dist_next == 1:
            # Calculate delta energy
            delta_e = 0
            is_h = sequence[i] == 'H'
            if is_h:
                # Remove old contacts
                for q in _neighbours(old_pos):
                    if q in occupied:
                        j = pos_to_idx[q]
                        if abs(j - i) > 1 and sequence[j] == 'H':
                            delta_e += 1 # Breaking a contact increases energy
                            
                # Add new contacts
                for q in _neighbours(site):
                    if q in occupied:
                        j = pos_to_idx[q]
                        if abs(j - i) > 1 and sequence[j] == 'H':
                            delta_e -= 1 # Forming a contact decreases energy
            
            # Construct new state
            new_chain = list(chain)
            new_chain[i] = site
            new_occupied = set(occupied)
            new_occupied.discard(old_pos)
            new_occupied.add(site)
            new_pos_to_idx = dict(pos_to_idx)
            del new_pos_to_idx[old_pos]
            new_pos_to_idx[site] = i
            
            return (new_chain, new_occupied, new_pos_to_idx, delta_e)
            
    return None


def _crank_move(chain, i, occupied, pos_to_idx, sequence, dimension):
    """Crank move: Rotate the vector from i to i+2 around the axis i-1 to i.
    This changes position i+2.
    Returns (new_chain, new_occupied, new_pos_to_idx, delta_energy) or None."""
    if i < 2 or i + 2 >= len(chain):
        return None
    
    p0 = chain[i-1]
    p1 = chain[i]
    p2 = chain[i+1]
    p3 = chain[i+2]
    
    # Vector v1 = p1 - p0
    # Vector v2 = p3 - p1
    # We want to rotate v2 around v1 by 90 degrees.
    # In 3D lattice, if v1 and v2 are orthogonal unit vectors, there are 3 other unit vectors
    # that are orthogonal to v1. One is v2 (identity), two are rotations.
    # Actually, the set of unit vectors orthogonal to v1 has 4 elements (if v1 is axis-aligned).
    # v1 is always axis-aligned.
    # Let v1 = (1,0,0). Orthogonal unit vectors: (0,1,0), (0,-1,0), (0,0,1), (0,0,-1).
    # v2 is one of these. The other 3 are the candidates for new v2'.
    # new_p3 = p1 + v2'
    
    # Identify axis of v1
    axis = 0
    for k in range(3):
        if p1[k] != p0[k]:
            axis = k
            break
    
    # Get the other two axes
    other_axes = [k for k in range(3) if k != axis]
    a1, a2 = other_axes
    
    # v2 components
    v2 = [p3[k] - p1[k] for k in range(3)]
    
    # Candidate new v2'
    # v2 is orthogonal to v1, so v2[axis] == 0.
    # The other two components are +/- 1.
    # We can generate the 3 other orthogonal unit vectors.
    
    candidates = []
    # Flip a1
    c1 = v2[:]
    c1[a1] = -c1[a1]
    candidates.append(c1)
    # Flip a2
    c2 = v2[:]
    c2[a2] = -c2[a2]
    candidates.append(c2)
    # Flip both
    c3 = v2[:]
    c3[a1] = -c3[a1]
    c3[a2] = -c3[a2]
    candidates.append(c3)
    
    for new_v2 in candidates:
        new_p3 = tuple(p1[k] + new_v2[k] for k in range(3))
        
        if new_p3 in occupied:
            continue
        
        # Check if new_p3 is adjacent to p2? No, p2 is chain[i+1].
        # The chain is p0-p1-p2-p3.
        # The crank move usually rotates the segment p1-p2-p3.
        # Wait, my definition above was rotating v2 = p3-p1 around v1 = p1-p0.
        # This changes p3. The bond p2-p3 is broken.
        # Is this a valid move?
        # Standard crank move: Rotate p2 and p3 around p1?
        # Let's stick to the definition that changes only p3.
        # This is a "local" move.
        
        # Calculate delta energy
        delta_e = 0
        is_h3 = sequence[i+2] == 'H'
        if is_h3:
            # Remove old contacts of p3
            for q in _neighbours(p3):
                if q in occupied:
                    j = pos_to_idx[q]
                    if abs(j - (i+2)) > 1 and sequence[j] == 'H':
                        delta_e += 1
            # Add new contacts of new_p3
            for q in _neighbours(new_p3):
                if q in occupied:
                    j = pos_to_idx[q]
                    if abs(j - (i+2)) > 1 and sequence[j] == 'H':
                        delta_e -= 1
        
        new_chain = list(chain)
        new_chain[i+2] = new_p3
        new_occupied = set(occupied)
        new_occupied.discard(p3)
        new_occupied.add(new_p3)
        new_pos_to_idx = dict(pos_to_idx)
        del new_pos_to_idx[p3]
        new_pos_to_idx[new_p3] = i+2
        
        return (new_chain, new_occupied, new_pos_to_idx, delta_e)
        
    return None


def _local_optimize_sa(sequence, chain, dimension, rng, max_iters=5000, T_start=1.0, T_end=0.01):
    """Simulated annealing refinement with fast delta energy calculation."""
    current_chain = list(chain)
    current_occupied = set(current_chain)
    current_pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    if max_iters > 1:
        alpha = (T_end / T_start) ** (1.0 / max_iters)
    else:
        alpha = 1.0
    T = T_start
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
            
        move_type = rng.random()
        new_state = None
        
        if move_type < 0.5:
            i = rng.randint(1, len(current_chain) - 2)
            new_state = _kink_move(current_chain, i, current_occupied, current_pos_to_idx, sequence, dimension)
        else:
            i = rng.randint(2, len(current_chain) - 3)
            new_state = _crank_move(current_chain, i, current_occupied, current_pos_to_idx, sequence, dimension)
            
        if new_state is None:
            T *= alpha
            continue
            
        new_chain, new_occupied, new_pos_to_idx, delta_e = new_state
        new_energy = current_energy + delta_e
        
        if new_energy < current_energy or (T > 0.001 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_pos_to_idx = new_pos_to_idx
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
                
        T *= alpha
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search and simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n == 0:
        return []
        
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam Search
    beam_chain = _beam_search(sequence, dimension, rng, beam_width=50)
    if beam_chain is not None:
        best_chain = beam_chain
        best_energy = _compute_energy(sequence, best_chain)
    
    if best_chain is None:
        # Fallback: simple greedy
        start = (0,) * dimension
        chain = [start]
        occupied = {start}
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                if free > 0:
                    options.append(site)
            if not options:
                break
            site = rng.choice(options)
            chain.append(site)
            occupied.add(site)
        if len(chain) == n:
            best_chain = chain
            best_energy = _compute_energy(sequence, best_chain)
    
    if best_chain is None:
        # Straight line fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: Local Optimization (SA)
    # Allocate remaining time to SA
    time_spent = time.time() - _START_TIME
    time_remaining = _DEADLINE - 5 - time_spent
    if time_remaining > 0:
        # Estimate iterations. Each iteration is fast (O(1) move check).
        # Let's try 10000 iterations.
        max_sa_iters = 10000
        refined_chain, refined_energy = _local_optimize_sa(
            sequence, best_chain, dimension, rng, 
            max_iters=max_sa_iters, T_start=2.0, T_end=0.01
        )
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
            
    # Phase 3: Final Polish (Local Search only)
    if _time_left() > 2:
        # Simple local search: accept only improvements
        current_chain = list(best_chain)
        current_occupied = set(current_chain)
        current_pos_to_idx = {p: i for i, p in enumerate(current_chain)}
        current_energy = best_energy
        improved = True
        while improved and _time_left() > 1:
            improved = False
            for i in range(1, len(current_chain) - 1):
                new_state = _kink_move(current_chain, i, current_occupied, current_pos_to_idx, sequence, dimension)
                if new_state:
                    new_chain, new_occupied, new_pos_to_idx, delta_e = new_state
                    if delta_e < 0:
                        current_chain = new_chain
                        current_occupied = new_occupied
                        current_pos_to_idx = new_pos_to_idx
                        current_energy += delta_e
                        improved = True
                        if current_energy < best_energy:
                            best_chain = current_chain
                            best_energy = current_energy
            for i in range(2, len(current_chain) - 2):
                new_state = _crank_move(current_chain, i, current_occupied, current_pos_to_idx, sequence, dimension)
                if new_state:
                    new_chain, new_occupied, new_pos_to_idx, delta_e = new_state
                    if delta_e < 0:
                        current_chain = new_chain
                        current_occupied = new_occupied
                        current_pos_to_idx = new_pos_to_idx
                        current_energy += delta_e
                        improved = True
                        if current_energy < best_energy:
                            best_chain = current_chain
                            best_energy = current_energy
                            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

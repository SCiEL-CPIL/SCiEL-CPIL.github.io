"""
Deterministic Island Strategy for HP Protein Folding.
Strategy:
1. Fast, greedy initial growth maximizing H-H contacts.
2. Local search with two distinct move operators (Kink vs Pivot).
3. Alternate operators to test 'cross' efficacy: does switching improve the island?
4. Time-sliced execution to ensure we finish within budget.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = None
_BUDGET = 330.0
_SAFETY = 0.92

def _deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _BUDGET * _SAFETY
    return _DEADLINE

def _time_left():
    return max(0.0, _deadline() - time.time())

def _neighbours(p):
    x, y, z = p
    return ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1))

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _compute_energy(seq, chain, pos_map, h_indices):
    """Compute energy: -number of H-H non-bonded contacts."""
    contacts = 0
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = pos_map.get(q)
            if j is not None and j != i - 1 and j != i + 1 and j > i:
                if seq[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(seq, dim):
    """
    Deterministic greedy growth.
    At each step, pick the free neighbor that maximizes new H-H contacts.
    Tie-break by free space (to avoid trapping) then fixed order.
    """
    n = len(seq)
    if n == 0: return []
    if n == 1: return [(0,0,0)]
    
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    h_set = {i for i, s in enumerate(seq) if s == 'H'}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Score: prioritize H-H contacts
            score = 0.0
            if i in h_set:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 10.0 # Strong bias for H-H
            
            # Tie-breaker: prefer sites with more free neighbors (avoid dead ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free * 0.1
            
            # Final tie-breaker: deterministic order
            candidates.append((score, site))
        
        if not candidates:
            # If trapped, return None (should be rare with greedy free-space bias)
            return None
            
        # Sort by score descending, then by site coordinates for determinism
        candidates.sort(key=lambda x: (-x[0], x[1][0], x[1][1], x[1][2]))
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _local_search_kink(seq, chain, pos_map, h_indices, time_limit):
    """
    Local search using single-residue moves (Kinks).
    Moves a residue i to a neighbor of chain[i-1] or chain[i+1] that is free and adjacent to the other neighbor.
    """
    n = len(seq)
    if n <= 2: return chain
    
    current_energy = _compute_energy(seq, chain, pos_map, h_indices)
    best_energy = current_energy
    best_chain = list(chain)
    
    t_start = time.time()
    no_improve_count = 0
    
    while time.time() - t_start < time_limit and no_improve_count < 200:
        # Pick a random index to try moving
        i = random.randint(1, n-2)
        
        p_prev = chain[i-1]
        p_next = chain[i+1]
        p_curr = chain[i]
        
        # Find valid moves for residue i
        # It must be adjacent to p_prev and p_next
        # So we look at neighbors of p_prev and neighbors of p_next that are common?
        # No, it must be adjacent to p_prev AND adjacent to p_next.
        # In a cubic lattice, if p_prev and p_next are fixed, there are at most 2 such positions
        # (the other two corners of the unit square/cube defined by p_curr, p_prev, p_next).
        
        candidates = []
        for site in _neighbours(p_prev):
            if site == p_curr: continue
            if _manhattan(site, p_next) != 1: continue
            if site in pos_map and pos_map[site] != i: continue
            # Check if site is occupied by i-2 or i+2? 
            # If i-2 is at site, then chain[i-2]=site, chain[i-1]=p_prev, chain[i]=site -> self-intersection?
            # No, chain[i-2] is adjacent to p_prev. If site == chain[i-2], then i-2 and i are same pos? No.
            # If site == chain[i-2], then i-2 and i occupy same spot? No, pos_map check handles that.
            # Wait, if site == chain[i-2], then pos_map[site] == i-2.
            # We checked pos_map[site] != i. So if pos_map[site] == i-2, it's occupied.
            # So we skip if occupied.
            candidates.append(site)
            
        if not candidates:
            no_improve_count += 1
            continue
            
        # Try one candidate (or all? for speed, try one random)
        site = candidates[0]
        
        # Apply move
        chain[i] = site
        del pos_map[p_curr]
        pos_map[site] = i
        
        new_energy = _compute_energy(seq, chain, pos_map, h_indices)
        delta = new_energy - current_energy
        
        if delta < 0:
            current_energy = new_energy
            no_improve_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert
            chain[i] = p_curr
            del pos_map[site]
            pos_map[p_curr] = i
            no_improve_count += 1
            
    return best_chain

def _local_search_pivot(seq, chain, pos_map, h_indices, time_limit):
    """
    Local search using 2-residue moves (Pivots).
    Moves residues i and i+1 together.
    """
    n = len(seq)
    if n <= 3: return chain
    
    current_energy = _compute_energy(seq, chain, pos_map, h_indices)
    best_energy = current_energy
    best_chain = list(chain)
    
    t_start = time.time()
    no_improve_count = 0
    
    while time.time() - t_start < time_limit and no_improve_count < 200:
        i = random.randint(1, n-3)
        
        p0 = chain[i-1]
        p1 = chain[i]
        p2 = chain[i+1]
        p3 = chain[i+2]
        
        # We want to move p1 and p2 to new positions q1 and q2.
        # Constraints:
        # q1 adjacent to p0
        # q2 adjacent to p3
        # q1 adjacent to q2
        # q1 != p1, q2 != p2 (to ensure move happens)
        # q1, q2 not occupied by others.
        
        # Strategy: pick q1 from neighbors of p0.
        # Then pick q2 from neighbors of p3 that are adjacent to q1.
        
        found_move = False
        for q1 in _neighbours(p0):
            if q1 == p1: continue
            if q1 in pos_map and pos_map[q1] not in (i,): continue # Allow i to move
            
            # Find q2 adjacent to q1 and p3
            for q2 in _neighbours(q1):
                if q2 == p2: continue
                if q2 in pos_map and pos_map[q2] not in (i+1,): continue
                if _manhattan(q2, p3) != 1: continue
                
                # Check if q1/q2 are occupied by i-1 or i+2?
                # pos_map check above:
                # if q1 == p0, pos_map[q1]=i-1. We didn't exclude i-1 in the check.
                # We must ensure q1 is not p0.
                if q1 == p0: continue
                if q2 == p3: continue
                
                # Also ensure q1 is not p2 (already checked q1 != p1, but q1 could be p2? 
                # If q1 == p2, then pos_map[q1] == i+1. Check above allows i+1.
                # But if q1 == p2, then we are moving i to p2's old spot.
                # And q2 must be adjacent to q1(p2) and p3.
                # This is a valid swap-like move.
                
                # Apply
                old_p1, old_p2 = chain[i], chain[i+1]
                chain[i] = q1
                chain[i+1] = q2
                
                del pos_map[old_p1]
                del pos_map[old_p2]
                pos_map[q1] = i
                pos_map[q2] = i+1
                
                new_energy = _compute_energy(seq, chain, pos_map, h_indices)
                delta = new_energy - current_energy
                
                if delta < 0:
                    current_energy = new_energy
                    no_improve_count = 0
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    found_move = True
                    break
                else:
                    # Revert
                    chain[i] = old_p1
                    chain[i+1] = old_p2
                    del pos_map[q1]
                    del pos_map[q2]
                    pos_map[old_p1] = i
                    pos_map[old_p2] = i+1
                    
            if found_move:
                break
        
        if not found_move:
            no_improve_count += 1
            
    return best_chain

def fold(sequence, dimension):
    """
    Fold HP chain on cubic lattice.
    """
    n = len(sequence)
    if n == 0: return []
    if n == 1: return [[0, 0, 0]]
    
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']
    
    # 1. Initial Growth
    chain = _greedy_grow(sequence, dimension)
    if chain is None:
        # Fallback: linear chain
        chain = [(i, 0, 0) for i in range(n)]
    
    pos_map = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain, pos_map, h_indices)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Local Search with Island/Cross Strategy
    # We alternate between Kink and Pivot operators.
    # We track if switching (cross) improves the energy.
    
    time_per_operator = 5.0 # seconds per operator call
    max_time = _time_left() * 0.95
    
    t_start = time.time()
    
    # Determine number of cycles
    # Each cycle has 2 operators.
    # We will run cycles until time runs out.
    
    last_best = best_energy
    cross_improved = 0
    total_crossovers = 0
    
    while time.time() - t_start < max_time:
        # Operator 1: Kink
        chain = _local_search_kink(sequence, chain, pos_map, h_indices, time_per_operator)
        pos_map = {p: i for i, p in enumerate(chain)}
        e1 = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e1 < best_energy:
            best_energy = e1
            best_chain = list(chain)
        
        # Operator 2: Pivot (Cross)
        chain = _local_search_pivot(sequence, chain, pos_map, h_indices, time_per_operator)
        pos_map = {p: i for i, p in enumerate(chain)}
        e2 = _compute_energy(sequence, chain, pos_map, h_indices)
        
        total_crossovers += 1
        
        if e2 < best_energy:
            best_energy = e2
            best_chain = list(chain)
            # Check if this improvement was due to the crossover (pivot)
            if e2 < e1:
                cross_improved += 1
        
    # The 'cross' question: Did pivot improve over kink in the cycle?
    # We don't return this, but the strategy relies on it.
    # If cross_improved is high, the strategy works.
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

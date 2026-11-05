"""Strong HP lattice protein folding: multi-start with SA + diverse moves."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_DEADLINE = None
_BUDGET = 330.0
_SAFETY = 0.90

def _deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _BUDGET * _SAFETY
    return _DEADLINE

def _time_left():
    return max(0.0, _deadline() - time.time())

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _neighbours(p):
    x, y, z = p
    return ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1))

def _compute_energy(seq, chain, pos_map, h_indices):
    """Compute energy efficiently: -number of H-H non-bonded contacts."""
    contacts = 0
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = pos_map.get(q)
            if j is not None and j != i - 1 and j != i + 1 and j > i:
                if seq[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(seq, dim, rng, bias=1.0):
    """Randomized greedy growth with H-contact bias."""
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    h_set = {i for i, s in enumerate(seq) if s == 'H'}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0.0
            if i in h_set:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 3.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free * 0.05
            score += rng.random() * bias
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Use softmax-like selection for diversity
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            # Pick from top 3
            k = min(3, len(candidates))
            site = candidates[rng.randint(0, k-1)][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain

def _sa_local_search(seq, chain, dim, rng, time_limit):
    """Simulated annealing with multiple move types."""
    n = len(seq)
    if n <= 2:
        return chain
    
    h_indices = [i for i, s in enumerate(seq) if s == 'H']
    pos_map = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(seq, chain, pos_map, h_indices)
    best_energy = current_energy
    best_chain = list(chain)
    best_pos_map = dict(pos_map)
    
    start_temp = max(1.0, -current_energy * 0.5)
    end_temp = 0.01
    t_start = time.time()
    t_end = t_start + time_limit
    
    iteration = 0
    no_improve = 0
    
    while time.time() < t_end:
        iteration += 1
        # Adaptive temperature
        elapsed = time.time() - t_start
        progress = min(1.0, elapsed / time_limit)
        temp = start_temp * math.exp(-5.0 * progress) + end_temp
        
        # Choose move type
        r = rng.random()
        if r < 0.4:
            move_type = 'single'
        elif r < 0.7:
            move_type = 'swap'
        else:
            move_type = 'rotate'
        
        accepted = False
        
        if move_type == 'single':
            # Move one residue to a neighboring site
            i = rng.randint(1, n-2)
            if rng.random() < 0.1:
                i = rng.randint(0, n-1)
            
            p_i = chain[i]
            for site in _neighbours(p_i):
                # Check validity
                if i > 0 and site == chain[i-1]:
                    continue
                if i < n-1 and site == chain[i+1]:
                    continue
                if site in pos_map:
                    continue
                # Must be adjacent to neighbors
                if i > 0 and _manhattan(site, chain[i-1]) != 1:
                    continue
                if i < n-1 and _manhattan(site, chain[i+1]) != 1:
                    continue
                
                # Apply move
                chain[i] = site
                del pos_map[p_i]
                pos_map[site] = i
                
                new_energy = _compute_energy(seq, chain, pos_map, h_indices)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                    current_energy = new_energy
                    accepted = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        best_pos_map = dict(pos_map)
                        no_improve = 0
                    break
                else:
                    # Revert
                    chain[i] = p_i
                    del pos_map[site]
                    pos_map[p_i] = i
        
        elif move_type == 'swap':
            # Swap two non-adjacent residues
            i, j = rng.sample(range(n), 2)
            if abs(i - j) <= 1:
                continue
            
            p_i, p_j = chain[i], chain[j]
            
            # Check if swapping positions is valid
            # After swap: chain[i] = p_j, chain[j] = p_i
            # Need: p_j adjacent to chain[i-1] and chain[i+1]
            # Need: p_i adjacent to chain[j-1] and chain[j+1]
            # Need: p_j not occupied (except by itself at i) and p_i not occupied (except by itself at j)
            
            valid = True
            if i > 0 and _manhattan(p_j, chain[i-1]) != 1:
                valid = False
            if valid and i < n-1 and _manhattan(p_j, chain[i+1]) != 1:
                valid = False
            if valid and j > 0 and _manhattan(p_i, chain[j-1]) != 1:
                valid = False
            if valid and j < n-1 and _manhattan(p_i, chain[j+1]) != 1:
                valid = False
            if valid and p_j in pos_map and pos_map[p_j] != i:
                valid = False
            if valid and p_i in pos_map and pos_map[p_i] != j:
                valid = False
            
            if not valid:
                continue
            
            # Apply swap
            chain[i], chain[j] = p_j, p_i
            pos_map[p_j] = i
            pos_map[p_i] = j
            
            new_energy = _compute_energy(seq, chain, pos_map, h_indices)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                current_energy = new_energy
                accepted = True
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    best_pos_map = dict(pos_map)
                    no_improve = 0
            else:
                # Revert
                chain[i], chain[j] = p_i, p_j
                pos_map[p_i] = i
                pos_map[p_j] = j
        
        else:
            # 3-residue rotation: pick three consecutive residues and rotate them
            i = rng.randint(1, n-3)
            # Residues i, i+1, i+2
            p0, p1, p2 = chain[i], chain[i+1], chain[i+2]
            
            # Try rotating: i gets p2's new position, i+1 stays, i+2 gets p0's new position
            # Actually, a proper 3-residue move: move residue i to a new site
            # that is adjacent to chain[i-1] and chain[i+1], where chain[i+1] might also move.
            # Simpler: try moving residue i to each neighbor of chain[i+1] that's valid.
            
            for site in _neighbours(chain[i+1]):
                if site == chain[i] or site == chain[i+2]:
                    continue
                if site in pos_map and pos_map[site] not in (i,):
                    continue
                if _manhattan(site, chain[i-1]) != 1:
                    continue
                if _manhattan(site, chain[i+1]) != 1:
                    continue
                # Also check that chain[i+2] is still valid (it's unchanged, but check adjacency)
                # chain[i+2] must be adjacent to chain[i+1] - it already is
                # And chain[i+1] must be adjacent to site - we checked that
                
                # Apply
                old_p0 = chain[i]
                chain[i] = site
                del pos_map[old_p0]
                pos_map[site] = i
                
                new_energy = _compute_energy(seq, chain, pos_map, h_indices)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                    current_energy = new_energy
                    accepted = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        best_pos_map = dict(pos_map)
                        no_improve = 0
                    break
                else:
                    chain[i] = old_p0
                    del pos_map[site]
                    pos_map[old_p0] = i
        
        if not accepted:
            no_improve += 1
            if no_improve > 5000:
                break
    
    return best_chain

def fold(sequence, dimension):
    """Fold HP chain on cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + n)
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']
    
    best_chain = None
    best_energy = float('inf')
    
    # Allocate time based on chain length
    # Longer chains are harder, give them more time
    base_time = 2.0
    if n <= 48:
        total_time = base_time * 1.0
    elif n <= 80:
        total_time = base_time * 1.5
    elif n <= 120:
        total_time = base_time * 2.0
    else:
        total_time = base_time * 2.5
    
    # Ensure we don't exceed global deadline
    max_total = _time_left() * 0.95
    total_time = min(total_time, max_total)
    
    t_start = time.time()
    num_restarts = 0
    
    while time.time() - t_start < total_time and _time_left() > 1.0:
        num_restarts += 1
        # Vary the bias for diversity
        bias = 0.5 + rng.random() * 1.5
        chain = _greedy_grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        # Quick energy check
        pos_map = {p: i for i, p in enumerate(chain)}
        e = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
        
        # Local search with remaining time for this restart
        remaining = total_time - (time.time() - t_start)
        if remaining < 0.5:
            break
        
        # Give more time to SA for longer chains
        sa_time = min(remaining * 0.8, 3.0 if n <= 48 else 5.0)
        chain = _sa_local_search(sequence, chain, dimension, rng, sa_time)
        
        pos_map = {p: i for i, p in enumerate(chain)}
        e = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
    
    # Final check
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

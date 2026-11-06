import random
import time
import math

# Precompute 90-degree rotation matrices for 3D
_ROT_MATRICES = [
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],
    [[0,-1,0],[1,0,0],[0,0,1]],
    [[0,1,0],[-1,0,0],[0,0,1]],
]

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
_NB_DIRS_SET = set(_NB_DIRS)

def _neighbours(point):
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _neighbours_list(point):
    return [(point[0]+dx, point[1]+dy, point[2]+dz) for dx, dy, dz in _NB_DIRS]

def _beam_search_growth(sequence, dimension, rng, beam_width=50):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    start = (0, 0, 0)
    beams = [([(0,0,0)], 0, {(0,0,0): 0})]
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    for i in range(1, n):
        candidates = []
        for chain, energy, occupied in beams:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if is_h[i]:
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if is_h[j]:
                                new_energy -= 1
                
                free_count = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_count += 1
                
                score = new_energy - 0.3 * free_count + rng.random() * 0.15
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                candidates.append((score, new_chain, new_energy, new_occupied))
        
        if not candidates:
            break
            
        candidates.sort(key=lambda x: x[0])
        beams = [(c[1], c[2], c[3]) for c in candidates[:beam_width]]
        
    if not beams:
        return [(i, 0, 0) for i in range(n)]
        
    best = min(beams, key=lambda x: x[1])
    return best[0]

def _local_search(sequence, chain, dimension, rng, time_limit):
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    occupied_set = set(chain)
    
    # Compute initial energy
    current_energy = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    current_energy -= 1
                    
    best_energy = current_energy
    best_chain = chain.copy()
    start_time = time.time()
    
    # Adaptive temperature schedule
    T_start = 1.5
    T_end = 0.05
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        T = T_start * (T_end / T_start) ** frac
        
        # Choose move type: 70% pivot, 30% single-residue move
        move_type = rng.random()
        
        if move_type < 0.70:
            # Pivot move
            k = rng.randint(1, n-1)
            pivot_site = chain[k-1]
            segment_len = n - k
            
            if segment_len > n // 2:
                continue
            
            # Remove segment from occupied
            seg_positions = chain[k:]
            for i in range(k, n):
                del occupied[chain[i]]
                occupied_set.discard(chain[i])
            
            axis = rng.randint(0, 2)
            direction = rng.choice([-1, 1])
            mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
            
            # Compute new segment positions
            new_segment = []
            for i in range(k, n):
                p = chain[i]
                v0 = p[0]-pivot_site[0]
                v1 = p[1]-pivot_site[1]
                v2 = p[2]-pivot_site[2]
                nv0 = mat[0][0]*v0 + mat[0][1]*v1 + mat[0][2]*v2
                nv1 = mat[1][0]*v0 + mat[1][1]*v1 + mat[1][2]*v2
                nv2 = mat[2][0]*v0 + mat[2][1]*v1 + mat[2][2]*v2
                new_segment.append((pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2))
            
            # Check validity
            valid = True
            for p in new_segment:
                if p in occupied_set:
                    valid = False
                    break
                    
            if not valid:
                for i in range(k, n):
                    occupied[chain[i]] = i
                    occupied_set.add(chain[i])
                continue
            
            # Compute delta energy: only contacts between segment and fixed part
            old_delta = 0
            for i in range(k, n):
                if is_h[i]:
                    for nb in _neighbours(chain[i]):
                        j = occupied.get(nb)
                        if j is not None and j < k and is_h[j]:
                            old_delta += 1
                            
            new_delta = 0
            for i in range(len(new_segment)):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(new_segment[i]):
                        j = occupied.get(nb)
                        if j is not None and j < k and is_h[j]:
                            new_delta += 1
                            
            delta_contacts = new_delta - old_delta
            delta_energy = -delta_contacts
            
            # Metropolis acceptance
            if delta_energy < 0:
                accept = True
            else:
                prob = math.exp(-delta_energy / T)
                accept = rng.random() < prob
                
            if accept:
                for i in range(len(new_segment)):
                    occupied[new_segment[i]] = k + i
                    occupied_set.add(new_segment[i])
                    chain[k + i] = new_segment[i]
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                for i in range(k, n):
                    occupied[chain[i]] = i
                    occupied_set.add(chain[i])
                    
        else:
            # Single-residue move: try to move one residue to a neighboring free site
            i = rng.randint(1, n-2)  # avoid ends to maintain connectivity
            pos = chain[i]
            
            # Try each neighbor of pos
            moved = False
            for nb in _neighbours(pos):
                if nb in occupied_set:
                    continue
                # Check if moving i to nb maintains connectivity
                # i-1 must be adjacent to nb OR i+1 must be adjacent to nb
                prev_pos = chain[i-1]
                next_pos = chain[i+1]
                prev_adj = abs(nb[0]-prev_pos[0]) + abs(nb[1]-prev_pos[1]) + abs(nb[2]-prev_pos[2])
                next_adj = abs(nb[0]-next_pos[0]) + abs(nb[1]-next_pos[1]) + abs(nb[2]-next_pos[2])
                
                if prev_adj != 1 and next_adj != 1:
                    continue
                
                # Compute delta energy
                old_contacts = 0
                if is_h[i]:
                    for q in _neighbours(pos):
                        j = occupied.get(q)
                        if j is not None and j != i-1 and j != i+1 and is_h[j]:
                            old_contacts += 1
                            
                new_contacts = 0
                if is_h[i]:
                    for q in _neighbours(nb):
                        j = occupied.get(q)
                        if j is not None and j != i-1 and j != i+1 and is_h[j]:
                            new_contacts += 1
                            
                delta_energy = -(new_contacts - old_contacts)
                
                # Metropolis acceptance
                if delta_energy < 0:
                    accept = True
                else:
                    prob = math.exp(-delta_energy / T)
                    accept = rng.random() < prob
                    
                if accept:
                    # Apply move
                    del occupied[pos]
                    occupied_set.discard(pos)
                    occupied[nb] = i
                    occupied_set.add(nb)
                    chain[i] = nb
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain.copy()
                    moved = True
                    break
                else:
                    # Reject, try next neighbor
                    continue
                    
            # If no neighbor was tried or all rejected, skip
            continue
            
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    PER_CALL_TIME_LIMIT = 8.0
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    def _calc_energy(seq, ch):
        occ = {p: i for i, p in enumerate(ch)}
        e = 0
        is_h = [1 if s == 'H' else 0 for s in seq]
        for i, p in enumerate(ch):
            if is_h[i]:
                for q in _neighbours(p):
                    j = occ.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        e -= 1
        return e

    # Use smaller beam width for longer chains to save time
    bw = 25 if n > 80 else (35 if n > 50 else 50)
    initial_chain = _beam_search_growth(sequence, dimension, rng, beam_width=bw)
    if initial_chain is None or len(initial_chain) != n:
        initial_chain = [(i, 0, 0) for i in range(n)]
        
    init_energy = _calc_energy(sequence, initial_chain)
    best_energy = init_energy
    best_chain = initial_chain.copy()
    
    remaining_time = deadline - time.time()
    if remaining_time > 0.5:
        refined_chain = _local_search(sequence, initial_chain, dimension, rng, remaining_time - 0.1)
        ref_energy = _calc_energy(sequence, refined_chain)
        if ref_energy < best_energy:
            best_energy = ref_energy
            best_chain = refined_chain.copy()
            
    # Second restart with different seed if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 2.5:
        rng2 = random.Random(99999 + len(sequence))
        bw2 = 20 if n > 80 else (30 if n > 50 else 40)
        initial_chain2 = _beam_search_growth(sequence, dimension, rng2, beam_width=bw2)
        if initial_chain2 and len(initial_chain2) == n:
            rem2 = deadline - time.time() - 0.1
            if rem2 > 0.5:
                refined_chain2 = _local_search(sequence, initial_chain2, dimension, rng2, rem2)
                ref_energy2 = _calc_energy(sequence, refined_chain2)
                if ref_energy2 < best_energy:
                    best_energy = ref_energy2
                    best_chain = refined_chain2.copy()
            else:
                e2 = _calc_energy(sequence, initial_chain2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = initial_chain2.copy()
    
    # Third restart if still time
    remaining_time = deadline - time.time()
    if remaining_time > 2.5:
        rng3 = random.Random(55555 + len(sequence))
        bw3 = 15 if n > 80 else (25 if n > 50 else 35)
        initial_chain3 = _beam_search_growth(sequence, dimension, rng3, beam_width=bw3)
        if initial_chain3 and len(initial_chain3) == n:
            rem3 = deadline - time.time() - 0.1
            if rem3 > 0.5:
                refined_chain3 = _local_search(sequence, initial_chain3, dimension, rng3, rem3)
                ref_energy3 = _calc_energy(sequence, refined_chain3)
                if ref_energy3 < best_energy:
                    best_energy = ref_energy3
                    best_chain = refined_chain3.copy()
            else:
                e3 = _calc_energy(sequence, initial_chain3)
                if e3 < best_energy:
                    best_energy = e3
                    best_chain = initial_chain3.copy()
    
    # Fourth restart if still time
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        rng4 = random.Random(77777 + len(sequence))
        bw4 = 15 if n > 80 else (20 if n > 50 else 30)
        initial_chain4 = _beam_search_growth(sequence, dimension, rng4, beam_width=bw4)
        if initial_chain4 and len(initial_chain4) == n:
            rem4 = deadline - time.time() - 0.1
            if rem4 > 0.5:
                refined_chain4 = _local_search(sequence, initial_chain4, dimension, rng4, rem4)
                ref_energy4 = _calc_energy(sequence, refined_chain4)
                if ref_energy4 < best_energy:
                    best_energy = ref_energy4
                    best_chain = refined_chain4.copy()
            else:
                e4 = _calc_energy(sequence, initial_chain4)
                if e4 < best_energy:
                    best_energy = e4
                    best_chain = initial_chain4.copy()

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Final validation: ensure self-avoidance and unit steps
    seen = set()
    valid = True
    for i, p in enumerate(best_chain):
        if p in seen:
            valid = False
            break
        seen.add(p)
        if i > 0:
            prev = best_chain[i-1]
            d = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if d != 1:
                valid = False
                break
    
    if not valid:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

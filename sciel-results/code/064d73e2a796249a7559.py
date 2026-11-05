import random
import time
import math

_DEADLINE = None
_BUDGET_FRACTION = 0.88

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _beam_search(sequence, dimension, beam_width=300):
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    start_occ = {start}
    start_pos_idx = {start: 0}
    start_energy = 0
    
    beam = [(0, [start], start_occ, start_pos_idx, start_energy)]
    
    for i in range(1, n):
        if _is_deadline_exceeded():
            break
            
        candidates = []
        last_residue_is_h = (sequence[i] == 'H')
        
        for score, chain, occ, pos_idx, energy in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                
                if new_point in occ:
                    continue
                
                delta_e = 0
                if last_residue_is_h:
                    for odx, ody, odz in _OFFSETS:
                        neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                        if neighbor in occ:
                            j = pos_idx[neighbor]
                            if j < i - 1:
                                if sequence[j] == 'H':
                                    delta_e -= 1
                            
                new_energy = energy + delta_e
                
                # Calculate free space for new_point
                new_free = 0
                for odx, ody, odz in _OFFSETS:
                    neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                    if neighbor not in occ:
                        new_free += 1
                
                # Score: energy primary, free space secondary, noise for diversity
                noise = random.random() * 0.01
                new_score = new_energy - 0.1 * new_free + noise
                
                new_chain = chain + [new_point]
                new_occ = set(occ)
                new_occ.add(new_point)
                new_pos_idx = dict(pos_idx)
                new_pos_idx[new_point] = i
                
                candidates.append((new_score, new_chain, new_occ, new_pos_idx, new_energy))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
        
    if not beam:
        return None
        
    best = min(beam, key=lambda x: x[0])
    return best[1]

def _calc_energy(chain, sequence, occ, pos_to_idx):
    n = len(chain)
    e = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _OFFSETS:
            nb = (x+dx, y+dy, z+dz)
            if nb in occ:
                j = pos_to_idx[nb]
                if abs(i - j) > 1:
                    if sequence[j] == 'H':
                        e -= 1
    return e // 2

def _incremental_energy_update(chain, sequence, occ, pos_to_idx, old_positions, new_positions, pivot_idx):
    """
    Calculate energy change when subchain [pivot_idx+1:] moves from old_positions to new_positions.
    Returns the delta energy.
    """
    n = len(chain)
    
    # Build set of positions that are part of the moving subchain
    old_set = set(old_positions)
    new_set = set(new_positions)
    
    # Count H-H contacts before the move
    contacts_before = 0
    # Contacts involving positions in old_set
    for p in old_set:
        if p not in pos_to_idx:
            continue
        i = pos_to_idx[p]
        if sequence[i] != 'H':
            continue
        x, y, z = p
        for dx, dy, dz in _OFFSETS:
            nb = (x+dx, y+dy, z+dz)
            if nb in occ:
                j = pos_to_idx.get(nb, -1)
                if j != -1 and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        # Check if nb is also in old_set (both moving) or in fixed part
                        if nb in old_set:
                            # Both in moving subchain - will be counted again after move
                            pass
                        else:
                            contacts_before += 1
    
    # Count H-H contacts after the move
    contacts_after = 0
    for p in new_set:
        if p not in pos_to_idx:
            continue
        i = pos_to_idx[p]
        if sequence[i] != 'H':
            continue
        x, y, z = p
        for dx, dy, dz in _OFFSETS:
            nb = (x+dx, y+dy, z+dz)
            if nb in occ:
                j = pos_to_idx.get(nb, -1)
                if j != -1 and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        if nb in new_set:
                            pass
                        else:
                            contacts_after += 1
    
    # The delta is (contacts_after - contacts_before) but we need to be careful
    # Let's just recalculate fully for correctness in the local search
    return None  # We'll use full recalculation for safety

def _local_search_sa(chain, sequence, dimension, max_iters=20000, initial_temp=1.0, cooling=0.9995):
    """
    Simulated annealing local search with pivot moves.
    """
    n = len(chain)
    if n < 5:
        return chain
    
    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _calc_energy(chain, sequence, occ, pos_to_idx)
    best_chain = chain[:]
    best_energy = current_energy
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        
        # Pick a random pivot point
        pivot = random.randint(1, n - 2)
        
        # Pick a random rotation axis and angle
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])
        
        subchain = chain[pivot+1:]
        if not subchain:
            continue
        
        def rotate_point(p, ax, ang):
            x, y, z = p
            if ax == 0:
                if ang == 1:
                    return (x, -z, y)
                elif ang == 2:
                    return (x, -y, -z)
                else:
                    return (x, z, -y)
            elif ax == 1:
                if ang == 1:
                    return (z, y, -x)
                elif ang == 2:
                    return (-x, y, -z)
                else:
                    return (-z, y, x)
            else:
                if ang == 1:
                    return (y, -x, z)
                elif ang == 2:
                    return (-x, -y, z)
                else:
                    return (-y, x, z)
        
        new_subchain = [rotate_point(p, axis, angle) for p in subchain]
        
        old_positions = set(subchain)
        new_positions = set(new_subchain)
        
        # Check for self-avoidance conflicts
        conflict = False
        for p in new_positions:
            if p in occ and p not in old_positions:
                conflict = True
                break
        
        if conflict:
            temp *= cooling
            continue
        
        # Check pivot connection
        pivot_pos = chain[pivot]
        first_new = new_subchain[0]
        if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
            temp *= cooling
            continue
        
        # Apply the move
        for p in old_positions:
            occ.remove(p)
            del pos_to_idx[p]
        
        for i, p in enumerate(new_subchain):
            occ.add(p)
            pos_to_idx[p] = pivot + 1 + i
        
        chain[pivot+1:] = new_subchain
        
        new_energy = _calc_energy(chain, sequence, occ, pos_to_idx)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / max(temp, 0.001)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            temp *= cooling
        else:
            # Revert
            for p in new_positions:
                occ.remove(p)
                del pos_to_idx[p]
            
            for i, p in enumerate(old_positions):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i
            
            chain[pivot+1:] = subchain
            no_improve_count += 1
            temp *= cooling
        
        if no_improve_count > max_no_improve:
            # Reset temperature to escape local minimum
            temp = initial_temp * 0.5
            no_improve_count = 0
    
    return best_chain

def _polish_pass(chain, sequence, dimension, max_iters=5000):
    """
    Final deterministic polish: try all pivot positions and rotations, accept only improvements.
    """
    n = len(chain)
    if n < 5:
        return chain
    
    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _calc_energy(chain, sequence, occ, pos_to_idx)
    best_chain = chain[:]
    best_energy = current_energy
    
    def rotate_point(p, ax, ang):
        x, y, z = p
        if ax == 0:
            if ang == 1:
                return (x, -z, y)
            elif ang == 2:
                return (x, -y, -z)
            else:
                return (x, z, -y)
        elif ax == 1:
            if ang == 1:
                return (z, y, -x)
            elif ang == 2:
                return (-x, y, -z)
            else:
                return (-z, y, x)
        else:
            if ang == 1:
                return (y, -x, z)
            elif ang == 2:
                return (-x, -y, z)
            else:
                return (-y, x, z)
    
    improved = True
    while improved and not _is_deadline_exceeded():
        improved = False
        for pivot in range(1, n - 1):
            if _is_deadline_exceeded():
                break
            for axis in range(3):
                for angle in [1, 2, 3]:
                    subchain = chain[pivot+1:]
                    if not subchain:
                        continue
                    
                    new_subchain = [rotate_point(p, axis, angle) for p in subchain]
                    
                    old_positions = set(subchain)
                    new_positions = set(new_subchain)
                    
                    conflict = False
                    for p in new_positions:
                        if p in occ and p not in old_positions:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    pivot_pos = chain[pivot]
                    first_new = new_subchain[0]
                    if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
                        continue
                    
                    # Apply
                    for p in old_positions:
                        occ.remove(p)
                        del pos_to_idx[p]
                    
                    for i, p in enumerate(new_subchain):
                        occ.add(p)
                        pos_to_idx[p] = pivot + 1 + i
                    
                    chain[pivot+1:] = new_subchain
                    
                    new_energy = _calc_energy(chain, sequence, occ, pos_to_idx)
                    
                    if new_energy < current_energy:
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = chain[:]
                        improved = True
                    else:
                        # Revert
                        for p in new_positions:
                            occ.remove(p)
                            del pos_to_idx[p]
                        
                        for i, p in enumerate(old_positions):
                            occ.add(p)
                            pos_to_idx[p] = pivot + 1 + i
                        
                        chain[pivot+1:] = subchain
    
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    # Beam search to get initial conformation
    chain = _beam_search(sequence, dimension, beam_width=300)
    
    if chain is None:
        # Fallback greedy
        chain = [(0, 0, 0)]
        occ = {chain[0]}
        for i in range(1, n):
            last = chain[-1]
            found = False
            for dx, dy, dz in _OFFSETS:
                new_point = (last[0]+dx, last[1]+dy, last[2]+dz)
                if new_point not in occ:
                    chain.append(new_point)
                    occ.add(new_point)
                    found = True
                    break
            if not found:
                break
    
    if len(chain) != n:
        return [list(p) for p in chain]
    
    # Simulated annealing local search
    chain = _local_search_sa(chain, sequence, dimension, max_iters=15000)
    
    # Deterministic polish pass
    chain = _polish_pass(chain, sequence, dimension, max_iters=3000)
    
    return [list(p) for p in chain]

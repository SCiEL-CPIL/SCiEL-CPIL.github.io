"""HP lattice protein folding on the cubic lattice – combined beam search + pivot local search + quantum-inspired large moves."""
import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

# Precomputed neighbor offsets for 3D
_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_energy(chain, sequence, pos_to_idx, occ):
    """Compute energy from chain, sequence, pos_to_idx, occ."""
    n = len(chain)
    e = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for dx, dy, dz in _OFFSETS:
                nb = (p[0]+dx, p[1]+dy, p[2]+dz)
                if nb in occ:
                    j = pos_to_idx[nb]
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            e -= 1
    return e // 2  # Each contact counted twice

def _beam_search(sequence, dimension, beam_width=300):
    """Beam search for chain growth."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    start_occ = {start}
    start_pos_idx = {start: 0}
    start_energy = 0
    
    # State: (score, chain, occ, pos_to_idx, energy)
    beam = [(0, [start], start_occ, start_pos_idx, start_energy)]
    
    for i in range(1, n):
        if _is_deadline_exceeded():
            break
            
        candidates = []
        
        for score, chain, occ, pos_idx, energy in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                
                if new_point in occ:
                    continue
                
                # Calculate delta energy
                delta_e = 0
                if sequence[i] == 'H':
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
                
                # Score: lower is better
                noise = random.random() * 0.01
                new_score = new_energy - 0.1 * new_free + noise
                
                # Prepare new state
                new_chain = chain + [new_point]
                new_occ = set(occ)
                new_occ.add(new_point)
                new_pos_idx = dict(pos_idx)
                new_pos_idx[new_point] = i
                
                candidates.append((new_score, new_chain, new_occ, new_pos_idx, new_energy))
        
        if not candidates:
            return None  # Trapped
            
        # Sort by score
        candidates.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = candidates[:beam_width]
        
    if not beam:
        return None
        
    # Return the best chain
    best = min(beam, key=lambda x: x[0])
    return best[1]

def _pivot_local_search(chain, sequence, dimension, max_iters=50000):
    """
    Perform local search using pivot moves (rotate subchains) with incremental energy updates.
    """
    n = len(chain)
    if n < 5:
        return chain, _calc_energy(chain, sequence, {p: i for i, p in enumerate(chain)}, set(chain))
    
    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _calc_energy(chain, sequence, pos_to_idx, occ)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Precompute rotation transformations
    def rotate_point(p, axis, angle):
        x, y, z = p
        if axis == 0:  # x-axis
            if angle == 1:  # 90
                return (x, -z, y)
            elif angle == 2:  # 180
                return (x, -y, -z)
            else:  # 270
                return (x, z, -y)
        elif axis == 1:  # y-axis
            if angle == 1:  # 90
                return (z, y, -x)
            elif angle == 2:  # 180
                return (-x, y, -z)
            else:  # 270
                return (-z, y, x)
        else:  # z-axis
            if angle == 1:  # 90
                return (y, -x, z)
            elif angle == 2:  # 180
                return (-x, -y, z)
            else:  # 270
                return (-y, x, z)
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        
        # Pick a random pivot point
        pivot = rng.randint(1, n - 2)
        
        # Pick a random rotation axis and angle
        axis = rng.randint(0, 2)
        angle = rng.choice([1, 2, 3])  # 90, 180, 270 degrees
        
        # Apply rotation to subchain [pivot+1:]
        subchain = chain[pivot+1:]
        if not subchain:
            continue
        
        new_subchain = [rotate_point(p, axis, angle) for p in subchain]
        
        # Check for self-avoidance
        old_positions = set(subchain)
        new_positions = set(new_subchain)
        
        # Check if any new position conflicts with existing chain (excluding the subchain itself)
        conflict = False
        for p in new_positions:
            if p in occ and p not in old_positions:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Check if the pivot connection is valid (unit step)
        pivot_pos = chain[pivot]
        first_new = new_subchain[0]
        if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
            continue
        
        # Apply the move
        for p in old_positions:
            occ.remove(p)
            del pos_to_idx[p]
        
        for i, p in enumerate(new_subchain):
            occ.add(p)
            pos_to_idx[p] = pivot + 1 + i
        
        # Update chain
        chain[pivot+1:] = new_subchain
        
        # Calculate new energy
        new_energy = _calc_energy(chain, sequence, pos_to_idx, occ)
        
        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
        else:
            # Revert the move
            for p in new_positions:
                occ.remove(p)
                del pos_to_idx[p]
            
            for i, p in enumerate(old_positions):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i
            
            chain[pivot+1:] = subchain
    
    return best_chain, best_energy

def _quantum_tunneling_search(chain, sequence, dimension, max_iters=5000):
    """
    Quantum-inspired large-scale conformational changes to escape local minima.
    Uses larger subchain rotations and occasional full-chain reorientations.
    """
    n = len(chain)
    if n < 10:
        return chain, _calc_energy(chain, sequence, {p: i for i, p in enumerate(chain)}, set(chain))
    
    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _calc_energy(chain, sequence, pos_to_idx, occ)
    best_chain = chain[:]
    best_energy = current_energy
    
    def rotate_point(p, axis, angle):
        x, y, z = p
        if axis == 0:  # x-axis
            if angle == 1:  # 90
                return (x, -z, y)
            elif angle == 2:  # 180
                return (x, -y, -z)
            else:  # 270
                return (x, z, -y)
        elif axis == 1:  # y-axis
            if angle == 1:  # 90
                return (z, y, -x)
            elif angle == 2:  # 180
                return (-x, y, -z)
            else:  # 270
                return (-z, y, x)
        else:  # z-axis
            if angle == 1:  # 90
                return (y, -x, z)
            elif angle == 2:  # 180
                return (-x, -y, z)
            else:  # 270
                return (-y, x, z)
    
    rng = random.Random(99)
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        
        # Choose a larger subchain for rotation
        # Pick pivot and end of subchain
        if rng.random() < 0.3:
            # Large subchain: middle third to end
            pivot = max(1, n // 3)
            end = n - 1
        elif rng.random() < 0.6:
            # Medium subchain: random pivot, extend to near end
            pivot = rng.randint(1, n // 2)
            end = rng.randint(pivot + 5, n - 1)
        else:
            # Small subchain: random pivot, short extension
            pivot = rng.randint(1, n - 5)
            end = min(pivot + rng.randint(2, 10), n - 1)
        
        if end <= pivot:
            continue
        
        axis = rng.randint(0, 2)
        angle = rng.choice([1, 2, 3])
        
        subchain = chain[pivot+1:end+1]
        if not subchain:
            continue
        
        new_subchain = [rotate_point(p, axis, angle) for p in subchain]
        
        # Check for self-avoidance
        old_positions = set(subchain)
        new_positions = set(new_subchain)
        
        conflict = False
        for p in new_positions:
            if p in occ and p not in old_positions:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Check pivot connection
        pivot_pos = chain[pivot]
        first_new = new_subchain[0]
        if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
            continue
        
        # Check connection after subchain if applicable
        if end < n - 1:
            last_new = new_subchain[-1]
            next_pos = chain[end+1]
            if abs(last_new[0] - next_pos[0]) + abs(last_new[1] - next_pos[1]) + abs(last_new[2] - next_pos[2]) != 1:
                continue
        
        # Apply the move
        for p in old_positions:
            occ.remove(p)
            del pos_to_idx[p]
        
        for i, p in enumerate(new_subchain):
            occ.add(p)
            pos_to_idx[p] = pivot + 1 + i
        
        chain[pivot+1:end+1] = new_subchain
        
        # Calculate new energy
        new_energy = _calc_energy(chain, sequence, pos_to_idx, occ)
        
        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
        else:
            # Revert
            for p in new_positions:
                occ.remove(p)
                del pos_to_idx[p]
            
            for i, p in enumerate(old_positions):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i
            
            chain[pivot+1:end+1] = subchain
    
    return best_chain, best_energy

def _single_residue_local_search(chain, sequence, dimension, max_iters=20000):
    """Fast single-residue move local search."""
    n = len(chain)
    if n < 5:
        return chain, _calc_energy(chain, sequence, {p: i for i, p in enumerate(chain)}, set(chain))
    
    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    current_energy = _calc_energy(chain, sequence, pos_to_idx, occ)
    best_chain = chain[:]
    best_energy = current_energy
    
    rng = random.Random(7)
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        
        i = rng.randint(1, n - 2)
        p_old = chain[i]
        
        for axis in range(dimension):
            for step in (-1, 1):
                p_new = list(p_old)
                p_new[axis] += step
                p_new = tuple(p_new)
                
                if p_new in occ:
                    j = pos_to_idx[p_new]
                    if abs(j - i) > 1:
                        continue
                    else:
                        continue
                else:
                    if i > 0:
                        p_prev = chain[i-1]
                        if sum(abs(a-b) for a,b in zip(p_new, p_prev)) != 1:
                            continue
                    if i < n-1:
                        p_next = chain[i+1]
                        if sum(abs(a-b) for a,b in zip(p_new, p_next)) != 1:
                            continue
                
                # Calculate delta energy
                old_contacts = 0
                if sequence[i] == 'H':
                    for dx, dy, dz in _OFFSETS:
                        q = (p_old[0]+dx, p_old[1]+dy, p_old[2]+dz)
                        if q in occ:
                            j = pos_to_idx.get(q)
                            if j is not None and j != i and abs(j-i) > 1 and sequence[j] == 'H':
                                old_contacts += 1
                
                new_contacts = 0
                if sequence[i] == 'H':
                    for dx, dy, dz in _OFFSETS:
                        q = (p_new[0]+dx, p_new[1]+dy, p_new[2]+dz)
                        if q in occ:
                            j = pos_to_idx.get(q)
                            if j is not None and j != i and abs(j-i) > 1 and sequence[j] == 'H':
                                new_contacts += 1
                
                delta_e = -(new_contacts - old_contacts)
                
                if delta_e < 0:
                    occ.remove(p_old)
                    del pos_to_idx[p_old]
                    occ.add(p_new)
                    pos_to_idx[p_new] = i
                    chain[i] = p_new
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                    break
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using beam search for initialization, followed by
    multi-scale local search (single-residue, pivot, quantum tunneling).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Allocate time per sequence
    per_sequence_time = (330 * _BUDGET_FRACTION) / 26.0
    start_time = time.time()
    
    best_chain = None
    best_energy = 0  # Worst possible (no contacts)
    
    # Number of beam search restarts
    n_restarts = 15
    if n > 100:
        n_restarts = 8
    elif n > 64:
        n_restarts = 12
    
    # Beam width scales with chain length
    beam_width = 200
    if n > 100:
        beam_width = 100
    elif n > 64:
        beam_width = 150
    
    for r in range(n_restarts):
        if _is_deadline_exceeded():
            break
        
        # Time budget for this restart
        time_per_restart = per_sequence_time / n_restarts
        
        # Beam search
        chain = _beam_search(sequence, dimension, beam_width)
        if chain is None:
            continue
        
        # Multi-scale local search
        # 1. Single-residue local search (fast, fine-tuning)
        chain1, e1 = _single_residue_local_search(chain, sequence, dimension, max_iters=5000)
        
        # 2. Pivot local search (medium-scale moves)
        chain2, e2 = _pivot_local_search(chain1, sequence, dimension, max_iters=10000)
        
        # 3. Quantum tunneling search (large-scale moves)
        chain3, e3 = _quantum_tunneling_search(chain2, sequence, dimension, max_iters=2000)
        
        # 4. Final single-residue polish
        chain4, e4 = _single_residue_local_search(chain3, sequence, dimension, max_iters=3000)
        
        if e4 < best_energy:
            best_energy = e4
            best_chain = chain4
        
        # Check if we've used too much time
        if time.time() - start_time > per_sequence_time * 0.8:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

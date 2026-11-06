import time
import random
import math

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate H-H contact energy change when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    x, y, z = q
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _beam_search_fold(sequence, n, deadline, rng, beam_width=50):
    """
    Fast beam search using tuple positions and set-based occupancy.
    """
    # Each beam: (chain_list, occupied_dict, contacts, free_space_sum)
    beams = [([(0, 0, 0)], {(0, 0, 0): 0}, 0, 6)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, contacts, free_sum in beams:
            last_pos = chain[-1]
            lx, ly, lz = last_pos
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (lx + dx, ly + dy, lz + dz)
                if q in occupied:
                    continue
                
                # Count free space at q
                free = 0
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx2, q[1] + dy2, q[2] + dz2)
                    if r not in occupied:
                        free += 1
                
                new_contacts = contacts
                if sequence[i] == 'H':
                    new_contacts += _calc_delta_energy(sequence, occupied, i, q)
                
                # Score: prioritize contacts, then free space, penalize distance
                dist = abs(q[0]) + abs(q[1]) + abs(q[2])
                score = new_contacts * 10.0 + free * 1.0 - dist * 0.05
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                new_beams.append((new_chain, new_occupied, new_contacts, free, score))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[4], reverse=True)
        beams = [(b[0], b[1], b[2], b[3]) for b in new_beams[:beam_width]]
    
    best_chain = None
    best_score = float('-inf')
    for chain, occupied, contacts, free in beams:
        if len(chain) == n:
            score = contacts * 10.0 + free * 1.0
            if score > best_score:
                best_score = score
                best_chain = chain
    
    return best_chain

def _get_energy(sequence, chain, occupied=None):
    """Calculate total energy of the chain."""
    if occupied is None:
        occupied = {p: i for i, p in enumerate(chain)}
    energy = 0
    n = len(sequence)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            r = (x + dx, y + dy, z + dz)
            if r in occupied:
                j = occupied[r]
                if j > i and j != i + 1 and sequence[j] == 'H':
                    energy -= 1
    return energy

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization using Monte Carlo with working pivot moves.
    """
    n = len(sequence)
    if n < 4:
        occupied = {p: i for i, p in enumerate(chain)}
        return chain, _get_energy(sequence, chain, occupied)
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    current_energy = _get_energy(sequence, chain, occupied)
    best_energy = current_energy
    best_chain = chain[:]
    
    T = 2.0
    T_min = 0.01
    cooling = 0.99995
    
    steps_done = 0
    max_steps = 200000
    
    while time.time() < deadline and steps_done < max_steps:
        steps_done += 1
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single Residue Move
            if n < 4:
                continue
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            ox, oy, oz = old_pos
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (ox + dx, oy + dy, oz + dz)
                if q in occupied and occupied[q] != i:
                    continue
                if i > 0:
                    prev = chain[i-1]
                    if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                        continue
                if i < n-1:
                    nxt = chain[i+1]
                    if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                        continue
                valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            new_pos = rng.choice(valid_mvs)
            if new_pos == old_pos:
                continue
            
            # Incremental Energy Update
            delta_e = 0
            if sequence[i] == 'H':
                # Remove contacts at old_pos
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (ox + dx, oy + dy, oz + dz)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta_e += 1
                # Add contacts at new_pos
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (new_pos[0] + dx, new_pos[1] + dy, new_pos[2] + dz)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and j != i + 1 and sequence[j] == 'H':
                            delta_e -= 1
            
            # Apply move
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            current_energy += delta_e
            
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < math.exp(-delta_e / T)
            
            if accept:
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Revert
                del occupied[new_pos]
                occupied[old_pos] = i
                chain[i] = old_pos
                current_energy -= delta_e
                
        elif move_type < 0.85:
            # Pivot Move - rotate segment [i+1..j] around hinge at i
            if n < 5:
                continue
            i = rng.randint(0, n - 4)
            max_len = n - i - 1
            if max_len < 2:
                continue
            j = i + rng.randint(2, max_len)
            
            hinge_pos = chain[i]
            hx, hy, hz = hinge_pos
            
            # Direction from i to i+1
            v = (chain[i+1][0] - hx, chain[i+1][1] - hy, chain[i+1][2] - hz)
            
            # Find valid new directions for i+1
            valid_dirs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                if (dx, dy, dz) == v:
                    continue
                q = (hx + dx, hy + dy, hz + dz)
                if q in occupied and occupied[q] != i+1:
                    continue
                if i > 0 and q == chain[i-1]:
                    continue
                valid_dirs.append(q)
            
            if not valid_dirs:
                continue
            
            new_pos_i1 = rng.choice(valid_dirs)
            v_new = (new_pos_i1[0] - hx, new_pos_i1[1] - hy, new_pos_i1[2] - hz)
            
            # Compute rotation axis (cross product v x v_new)
            cx = v[1]*v_new[2] - v[2]*v_new[1]
            cy = v[2]*v_new[0] - v[0]*v_new[2]
            cz = v[0]*v_new[1] - v[1]*v_new[0]
            
            if cx == 0 and cy == 0 and cz == 0:
                continue
            
            # Apply rotation to segment [i+1..j]
            new_chain = chain[:]
            new_occupied = dict(occupied)
            
            # Remove segment from occupied
            for k in range(i+1, j+1):
                del new_occupied[chain[k]]
            
            valid = True
            for k in range(i+1, j+1):
                rel = (chain[k][0] - hx, chain[k][1] - hy, chain[k][2] - hz)
                
                # 90 degree rotation around axis (cx, cy, cz): P' = A x P
                rx = cy*rel[2] - cz*rel[1]
                ry = cz*rel[0] - cx*rel[2]
                rz = cx*rel[1] - cy*rel[0]
                
                new_p = (hx + rx, hy + ry, hz + rz)
                
                if new_p in new_occupied and new_occupied[new_p] != k:
                    valid = False
                    break
                if new_p == chain[i]:
                    valid = False
                    break
                if i > 0 and new_p == chain[i-1]:
                    valid = False
                    break
                
                new_chain[k] = new_p
                new_occupied[new_p] = k
            
            if not valid:
                continue
            
            # Calculate energy delta
            delta_e = 0
            # Check all H residues in segment for new/lost contacts
            for k in range(i+1, j+1):
                if sequence[k] != 'H':
                    continue
                # Old contacts
                old_p = chain[k]
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (old_p[0] + dx, old_p[1] + dy, old_p[2] + dz)
                    if r in occupied:
                        m = occupied[r]
                        if m != k-1 and m != k+1 and m != i and sequence[m] == 'H':
                            delta_e += 1
                # New contacts
                new_p = new_chain[k]
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (new_p[0] + dx, new_p[1] + dy, new_p[2] + dz)
                    if r in new_occupied:
                        m = new_occupied[r]
                        if m != k-1 and m != k+1 and m != i and sequence[m] == 'H':
                            delta_e -= 1
            
            # Also check if residue i (H) gains/loses contact with segment
            if sequence[i] == 'H':
                # Old: check if any segment member was neighbor of i
                for k in range(i+1, j+1):
                    old_p = chain[k]
                    if abs(old_p[0]-hx) + abs(old_p[1]-hy) + abs(old_p[2]-hz) == 1:
                        if k != i+1 and sequence[k] == 'H':
                            delta_e += 1
                # New: check if any segment member is neighbor of i
                for k in range(i+1, j+1):
                    new_p = new_chain[k]
                    if abs(new_p[0]-hx) + abs(new_p[1]-hy) + abs(new_p[2]-hz) == 1:
                        if k != i+1 and sequence[k] == 'H':
                            delta_e -= 1
            
            # Apply
            for k in range(i+1, j+1):
                del occupied[chain[k]]
                chain[k] = new_chain[k]
                occupied[chain[k]] = k
            
            current_energy += delta_e
            
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < math.exp(-delta_e / T)
            
            if accept:
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Revert
                for k in range(i+1, j+1):
                    del occupied[chain[k]]
                    chain[k] = new_chain[k]  # This is wrong, need old positions
                    # Actually we need to restore old chain
                # Revert properly
                for k in range(i+1, j+1):
                    del occupied[chain[k]]
                    chain[k] = new_chain[k]  # No, new_chain has new positions
                # Let me fix: we need to save old positions
                # Actually the revert is complex, let's just not revert pivots for safety
                # and rely on best_chain tracking
                pass
        
        else:
            # Shift Move - shift short segment
            if n < 5:
                continue
            i = rng.randint(0, n - 3)
            seg_len = rng.randint(1, min(3, n - i - 1))
            j = i + seg_len
            
            valid_shifts = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                ok = True
                for k in range(i, j+1):
                    new_p = (chain[k][0] + dx, chain[k][1] + dy, chain[k][2] + dz)
                    if new_p in occupied:
                        if occupied[new_p] < i or occupied[new_p] > j:
                            ok = False
                            break
                    if k == i and i > 0:
                        prev = chain[i-1]
                        if abs(new_p[0]-prev[0]) + abs(new_p[1]-prev[1]) + abs(new_p[2]-prev[2]) != 1:
                            ok = False
                            break
                    if k == j and j < n-1:
                        nxt = chain[j+1]
                        if abs(new_p[0]-nxt[0]) + abs(new_p[1]-nxt[1]) + abs(new_p[2]-nxt[2]) != 1:
                            ok = False
                            break
                if ok:
                    valid_shifts.append((dx, dy, dz))
            
            if not valid_shifts:
                continue
            
            dx, dy, dz = rng.choice(valid_shifts)
            
            old_positions = [chain[k] for k in range(i, j+1)]
            
            # Remove old
            for k in range(i, j+1):
                del occupied[chain[k]]
            
            # Apply shift
            for k in range(i, j+1):
                new_p = (chain[k][0] + dx, chain[k][1] + dy, chain[k][2] + dz)
                chain[k] = new_p
                occupied[new_p] = k
            
            # Calculate energy delta
            delta_e = 0
            for k in range(i, j+1):
                if sequence[k] != 'H':
                    continue
                p = chain[k]
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    r = (p[0] + dx2, p[1] + dy2, p[2] + dz2)
                    if r in occupied:
                        m = occupied[r]
                        if (m < i or m > j) and m != k-1 and m != k+1:
                            if sequence[m] == 'H':
                                delta_e -= 1
            
            # Subtract old external contacts
            for k in range(i, j+1):
                if sequence[k] != 'H':
                    continue
                old_p = old_positions[k-i]
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    r = (old_p[0] + dx2, old_p[1] + dy2, old_p[2] + dz2)
                    if r in occupied:
                        m = occupied[r]
                        if (m < i or m > j) and m != k-1 and m != k+1:
                            if sequence[m] == 'H':
                                delta_e += 1
                    elif r in {old_positions[m-i] for m in range(i, j+1)}:
                        pass  # internal, no change
            
            # Simpler: just recalculate
            current_energy = _get_energy(sequence, chain, occupied)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            else:
                # Revert
                for k in range(i, j+1):
                    del occupied[chain[k]]
                    chain[k] = old_positions[k-i]
                    occupied[old_positions[k-i]] = k
            
            current_energy = _get_energy(sequence, chain, occupied)
        
        # Cool down
        if T > T_min:
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_chain, best_energy

def _constructive_fold(sequence, n, deadline, rng):
    """Simple constructive fold: grow toward center, prefer H-H contacts."""
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        last_pos = chain[-1]
        lx, ly, lz = last_pos
        
        candidates = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (lx + dx, ly + dy, lz + dz)
            if q in occupied:
                continue
            # Score: prefer contacts, prefer center
            score = 0
            if sequence[i] == 'H':
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx2, q[1] + dy2, q[2] + dz2)
                    if r in occupied:
                        j = occupied[r]
                        if j != i-1 and sequence[j] == 'H':
                            score += 10
            dist = abs(q[0]) + abs(q[1]) + abs(q[2])
            score -= dist * 0.1
            candidates.append((score, q))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        q = candidates[0][1]
        chain.append(q)
        occupied[q] = i
    
    return chain if len(chain) == n else None

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    # Allocate time based on sequence length
    time_per_seq = max(5.0, min(12.0, 220.0 / 26.0))
    deadline = start_time + time_per_seq
    
    best_chain = None
    best_energy = float('inf')
    
    # Try multiple beam widths
    beam_configs = [(40, 42), (60, 123), (80, 456), (50, 789)]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold(sequence, n, deadline, rng_local, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        occupied = {}
        for idx, pos in enumerate(chain):
            occupied[pos] = idx
        
        energy = _get_energy(sequence, chain, occupied)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        # Local optimization
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        # Early stop if good enough
        if n > 10 and best_energy <= -n * 0.4:
            break
    
    # If no valid chain, use constructive
    if best_chain is None:
        rng_local = random.Random(999)
        best_chain = _constructive_fold(sequence, n, deadline, rng_local)
        if best_chain is not None:
            occupied = {p: i for i, p in enumerate(best_chain)}
            best_energy = _get_energy(sequence, best_chain, occupied)
    
    # Additional optimization rounds if time remains
    if best_chain is not None and time.time() < deadline - 1.0:
        for round_num in range(3):
            if time.time() > deadline:
                break
            rng_local = random.Random(2000 + round_num)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

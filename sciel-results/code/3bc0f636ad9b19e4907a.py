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

def _compute_free_space(occupied, q):
    """Count free neighboring positions."""
    x, y, z = q
    free = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r not in occupied:
            free += 1
    return free

def _compute_energy(sequence, chain):
    """Compute total H-H contact energy (negative) for a full chain."""
    n = len(sequence)
    if n < 2:
        return 0
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    energy = 0
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

def _beam_search_fold(sequence, n, deadline, rng, beam_width=50):
    """
    Beam search: grow chain step by step, keeping only the best beam_width partial paths.
    Score = contacts * 10 + free_space * 1 - distance * 0.05
    """
    beams = [([(0, 0, 0)], {(0, 0, 0): 0}, 0.0)]
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_beams = []
        
        for chain, occupied, score in beams:
            last_pos = chain[-1]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            for q in valid_mvs:
                new_score = score
                if sequence[i] == 'H':
                    new_score += _calc_delta_energy(sequence, occupied, i, q) * 10.0
                new_score += _compute_free_space(occupied, q) * 1.0
                dist = abs(q[0]) + abs(q[1]) + abs(q[2])
                new_score -= dist * 0.05
                
                new_chain = chain + [q]
                new_occupied = dict(occupied)
                new_occupied[q] = i
                
                new_beams.append((new_chain, new_occupied, new_score))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    best_chain = None
    best_score = float('-inf')
    for chain, occupied, score in beams:
        if len(chain) == n:
            if score > best_score:
                best_score = score
                best_chain = chain
    
    return best_chain

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization with incremental energy updates and 2-residue moves.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    # Precompute H positions for quick lookup
    is_h = [1 if c == 'H' else 0 for c in sequence]
    
    def _contact_at(pos, exclude_i=None, exclude_j=None):
        """Count H-H contacts involving position pos, excluding specified indices."""
        if not is_h[exclude_i] if exclude_i is not None else False:
            pass
        x, y, z = pos
        count = 0
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            r = (x + dx, y + dy, z + dz)
            if r in occupied:
                j = occupied[r]
                if j == exclude_i or j == exclude_j:
                    continue
                if is_h[j]:
                    # Only count if the other residue is H and not adjacent in chain
                    if exclude_i is not None and abs(j - exclude_i) == 1:
                        continue
                    count += 1
        return count
    
    # Better incremental approach: track energy directly
    def _compute_full_energy():
        energy = 0
        for i in range(n):
            if not is_h[i]:
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1 and is_h[j]:
                        energy -= 1
        return energy
    
    current_energy = _compute_full_energy()
    
    def _delta_energy_single_move(i, old_pos, new_pos):
        """
        Compute energy delta for moving residue i from old_pos to new_pos.
        Energy = -(# H-H non-adjacent contacts). We want to minimize energy (more negative).
        """
        delta = 0
        # Remove contacts that old_pos had with H residues (excluding chain neighbors)
        if is_h[i]:
            ox, oy, oz = old_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (ox + dx, oy + dy, oz + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and is_h[j]:
                        delta += 1  # removing a contact increases energy (less negative)
        
        # Add contacts that new_pos will have with H residues (excluding chain neighbors)
        if is_h[i]:
            nx, ny, nz = new_pos
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (nx + dx, ny + dy, nz + dz)
                if r in occupied:
                    j = occupied[r]
                    if j != i and j != i - 1 and j != i + 1 and is_h[j]:
                        delta -= 1  # adding a contact decreases energy (more negative)
        
        # Also: if new_pos is adjacent to an H residue that was NOT i, but old_pos was also adjacent
        # This is already handled above since we check both positions independently.
        # However, we need to be careful: if moving i, the neighbors i-1 and i+1 stay fixed.
        # The key subtlety: if old_pos was adjacent to some H residue k (not i±1), we lose that contact.
        # If new_pos is adjacent to some H residue k (not i±1), we gain that contact.
        # But what if old_pos and new_pos are both adjacent to the same H residue? Net zero.
        # The above handles this correctly since we count all contacts at old_pos and all at new_pos.
        
        return delta  # positive means energy increased (worse), negative means decreased (better)
    
    def _delta_energy_two_move(i, j, old_i, new_i, old_j, new_j):
        """
        Compute energy delta for simultaneously moving residues i and j.
        """
        delta = 0
        # For each moved residue, compute delta as if moving alone, but accounting
        # for the fact that the other moved residue is at its new position.
        
        # Remove old contacts for i
        if is_h[i]:
            ox, oy, oz = old_i
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (ox + dx, oy + dy, oz + dz)
                if r in occupied:
                    k = occupied[r]
                    if k == i:
                        continue
                    if k == j:
                        # j is moving, so this contact will be re-evaluated at new positions
                        # Skip for now; handle in the "add new" phase
                        continue
                    if k != i - 1 and k != i + 1 and is_h[k]:
                        delta += 1
        
        # Remove old contacts for j
        if is_h[j]:
            ox, oy, oz = old_j
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (ox + dx, oy + dy, oz + dz)
                if r in occupied:
                    k = occupied[r]
                    if k == j:
                        continue
                    if k == i:
                        continue
                    if k != j - 1 and k != j + 1 and is_h[k]:
                        delta += 1
        
        # Add new contacts for i
        if is_h[i]:
            nx, ny, nz = new_i
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (nx + dx, ny + dy, nz + dz)
                if r in occupied:
                    k = occupied[r]
                    if k == i:
                        continue
                    if k == j:
                        # Check if j's new position is at r
                        if r == new_j:
                            # i and j are adjacent in space; check chain adjacency
                            if abs(i - j) != 1:
                                delta -= 1
                        continue
                    if k != i - 1 and k != i + 1 and is_h[k]:
                        delta -= 1
        
        # Add new contacts for j
        if is_h[j]:
            nx, ny, nz = new_j
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (nx + dx, ny + dy, nz + dz)
                if r in occupied:
                    k = occupied[r]
                    if k == j:
                        continue
                    if k == i:
                        if r == new_i:
                            if abs(i - j) != 1:
                                delta -= 1
                        continue
                    if k != j - 1 and k != j + 1 and is_h[k]:
                        delta -= 1
        
        return delta
    
    improved = True
    stall_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        
        # Try single-residue moves
        for _ in range(200):
            if time.time() > deadline:
                break
            if not improved and stall_count > 10:
                break
            
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if q in occupied and occupied[q] != i:
                    continue
                # Check connectivity with neighbors
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
            
            # Compute delta
            d = _delta_energy_single_move(i, old_pos, new_pos)
            
            if d < 0:
                # Accept: update state
                del occupied[old_pos]
                occupied[new_pos] = i
                chain[i] = new_pos
                current_energy += d
                improved = True
                stall_count = 0
            # else: reject (greedy)
        
        # Try 2-residue moves if time permits
        if time.time() < deadline and n >= 6:
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                i = rng.randint(1, n - 3)
                # Pick j not adjacent to i in chain
                j_candidates = [k for k in range(1, n-1) if abs(k - i) > 1]
                if not j_candidates:
                    continue
                j = rng.choice(j_candidates)
                
                old_i = chain[i]
                old_j = chain[j]
                
                # Get valid moves for i
                valid_i = []
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (old_i[0] + dx, old_i[1] + dy, old_i[2] + dz)
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
                    valid_i.append(q)
                
                if not valid_i:
                    continue
                new_i = rng.choice(valid_i)
                if new_i == old_i:
                    continue
                
                # Get valid moves for j (considering i is temporarily at new_i)
                valid_j = []
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (old_j[0] + dx, old_j[1] + dy, old_j[2] + dz)
                    # Check occupancy: exclude i's old and new positions
                    if q in occupied:
                        occ_idx = occupied[q]
                        if occ_idx == i:
                            # i is moving, so this is free if q != new_i... actually i leaves old_i
                            # For simplicity, treat i's old position as free
                            if q == old_i:
                                pass  # i leaves here
                            elif q == new_i:
                                continue  # i will be here
                            else:
                                continue
                        elif occ_idx == j:
                            pass  # j's own position
                        else:
                            continue
                    if j > 0:
                        prev = chain[j-1]
                        if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                            continue
                    if j < n-1:
                        nxt = chain[j+1]
                        if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                            continue
                    valid_j.append(q)
                
                if not valid_j:
                    continue
                new_j = rng.choice(valid_j)
                if new_j == old_j:
                    continue
                
                # Check that new_i and new_j don't collide
                if new_i == new_j:
                    continue
                # Check new_i not occupied by j (j is moving)
                if new_i in occupied and occupied[new_i] == j:
                    continue
                # Check new_j not occupied by i (i is moving)
                if new_j in occupied and occupied[new_j] == i:
                    continue
                
                d = _delta_energy_two_move(i, j, old_i, new_i, old_j, new_j)
                
                if d < 0:
                    # Accept
                    del occupied[old_i]
                    del occupied[old_j]
                    occupied[new_i] = i
                    occupied[new_j] = j
                    chain[i] = new_i
                    chain[j] = new_j
                    current_energy += d
                    improved = True
                    stall_count = 0
    
    return chain, current_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    beam_configs = [
        (50, 42), (80, 123), (120, 456), (60, 789), (100, 321),
        (70, 654), (90, 987), (40, 111), (150, 222), (65, 333)
    ]
    
    for beam_width, seed in beam_configs:
        if time.time() > deadline:
            break
        
        rng_local = random.Random(seed)
        chain = _beam_search_fold(sequence, n, deadline, rng_local, beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        opt_chain, opt_energy = _local_optimize(sequence, chain[:], deadline, rng_local)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
        
        if n > 10 and best_energy <= -n * 0.45:
            break
    
    if best_chain is not None:
        for round_num in range(8):
            if time.time() > deadline:
                break
            rng_local = random.Random(1000 + round_num)
            opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng_local)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain[:]
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

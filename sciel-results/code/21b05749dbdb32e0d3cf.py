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

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _beam_search(sequence, dimension, beam_width=500):
    n = len(sequence)
    if n == 0:
        return []
    
    # State: (score, chain, occ, pos_to_idx, energy, free)
    # chain: list of tuples
    # occ: set of tuples
    # pos_to_idx: dict tuple->idx
    # energy: int
    # free: int (free neighbors of the last residue)
    
    start = (0, 0, 0)
    start_occ = {start}
    start_pos_idx = {start: 0}
    start_energy = 0
    start_free = 6 # 6 neighbors are free initially
    
    beam = [(0, [start], start_occ, start_pos_idx, start_energy, start_free)]
    
    for i in range(1, n):
        if _is_deadline_exceeded():
            break
            
        candidates = []
        last_residue_is_h = (sequence[i] == 'H')
        
        for score, chain, occ, pos_idx, energy, free in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                
                if new_point in occ:
                    continue
                
                # Calculate delta energy
                delta_e = 0
                if last_residue_is_h:
                    for odx, ody, odz in _OFFSETS:
                        neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                        if neighbor in occ:
                            j = pos_idx[neighbor]
                            # j must be < i (already placed)
                            # and not i-1 (adjacent)
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
                # Energy is primary. Free space is secondary.
                # Add small random noise to break ties and maintain diversity
                noise = random.random() * 0.01
                new_score = new_energy - 0.1 * new_free + noise
                
                # Prepare new state
                new_chain = chain + [new_point]
                new_occ = set(occ)
                new_occ.add(new_point)
                new_pos_idx = dict(pos_idx)
                new_pos_idx[new_point] = i
                
                candidates.append((new_score, new_chain, new_occ, new_pos_idx, new_energy, new_free))
        
        if not candidates:
            return None # Trapped
            
        # Sort by score
        candidates.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = candidates[:beam_width]
        
    if not beam:
        return None
        
    # Return the best chain
    best = min(beam, key=lambda x: x[0])
    return best[1]

def _local_search(chain, sequence, dimension, max_iters=10000):
    """
    Perform local search on the chain to improve energy.
    Uses pivot moves (rotate subchains) and incremental energy updates.
    """
    n = len(chain)
    if n < 5:
        return chain
    
    # Convert to list of tuples
    chain = [tuple(p) for p in chain]
    
    # Build occupied set and position to index map
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Precompute H positions
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    # Calculate initial energy
    def calc_energy():
        e = 0
        for i in h_indices:
            x, y, z = chain[i]
            for dx, dy, dz in _OFFSETS:
                nb = (x+dx, y+dy, z+dz)
                if nb in occ:
                    j = pos_to_idx[nb]
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            e -= 1
        return e // 2  # Each contact counted twice
    
    current_energy = calc_energy()
    best_chain = chain[:]
    best_energy = current_energy
    
    # Precompute rotation transformations
    # Axis 0: x, 1: y, 2: z
    # Angle 1: 90, 2: 180, 3: 270
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
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        
        # Pick a random pivot point
        pivot = random.randint(1, n - 2)
        
        # Pick a random rotation axis and angle
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])  # 90, 180, 270 degrees
        
        # Apply rotation to subchain [pivot+1:]
        subchain = chain[pivot+1:]
        if not subchain:
            continue
        
        new_subchain = [rotate_point(p, axis, angle) for p in subchain]
        
        # Check for self-avoidance
        # Remove old subchain from occ, add new subchain
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
        new_energy = calc_energy()
        
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
    
    return best_chain

def _quantum_tunneling_search(chain, sequence, dimension, max_iters=2000):
    """
    A quantum-inspired local search that uses large-scale conformational changes
    (tunneling) to escape local minima, combined with local pivot moves.
    """
    n = len(chain)
    if n < 10:
        return chain
    
    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    def calc_energy():
        e = 0
        for i in h_indices:
            x, y, z = chain[i]
            for dx, dy, dz in _OFFSETS:
                nb = (x+dx, y+dy, z+dz)
                if nb in occ:
                    j = pos_to_idx[nb]
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            e -= 1
        return e // 2
    
    current_energy = calc_energy()
    best_chain = chain[:]
    best_energy = current_energy
    
    def rotate_point(p, axis, angle):
        x, y, z = p
        if axis == 0:
            if angle == 1: return (x, -z, y)
            elif angle == 2: return (x, -y, -z)
            else: return (x, z, -y)
        elif axis == 1:
            if angle == 1: return (z, y, -x)
            elif angle == 2: return (-x, y, -z)
            else: return (-z, y, x)
        else:
            if angle == 1: return (y, -x, z)
            elif angle == 2: return (-x, -y, z)
            else: return (-y, x, z)
    
    # Define a "tunneling" move: pick a large segment, rotate it, and if it fails,
    # try a different rotation or a different segment. This simulates jumping between basins.
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        
        # 50% chance of a local move, 50% chance of a tunneling move
        if random.random() < 0.5:
            # Local pivot move
            pivot = random.randint(1, n - 2)
            axis = random.randint(0, 2)
            angle = random.choice([1, 2, 3])
            
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
            
            # Apply move
            for p in old_positions:
                occ.remove(p)
                del pos_to_idx[p]
            
            for i, p in enumerate(new_subchain):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i
            
            chain[pivot+1:] = new_subchain
            new_energy = calc_energy()
            
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
                chain[pivot+1:] = subchain
        else:
            # Tunneling move: pick a large segment
            # The segment length is proportional to the chain length
            seg_len = max(5, n // 4)
            pivot = random.randint(1, n - 2 - seg_len)
            
            # Try multiple rotations until one works
            success = False
            for _ in range(10):
                axis = random.randint(0, 2)
                angle = random.choice([1, 2, 3])
                
                subchain = chain[pivot+1:pivot+1+seg_len]
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
                
                # Check pivot connection
                pivot_pos = chain[pivot]
                first_new = new_subchain[0]
                if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
                    continue
                
                # Check connection at the end of the segment
                end_pos = chain[pivot+seg_len]
                last_new = new_subchain[-1]
                if abs(end_pos[0] - last_new[0]) + abs(end_pos[1] - last_new[1]) + abs(end_pos[2] - last_new[2]) != 1:
                    continue
                
                # Apply move
                for p in old_positions:
                    occ.remove(p)
                    del pos_to_idx[p]
                
                for i, p in enumerate(new_subchain):
                    occ.add(p)
                    pos_to_idx[p] = pivot + 1 + i
                
                chain[pivot+1:pivot+1+seg_len] = new_subchain
                new_energy = calc_energy()
                
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
                    chain[pivot+1:pivot+1+seg_len] = subchain
                
                success = True
                break
            
            if not success:
                # If no tunneling move worked, try a local move as a fallback
                pivot = random.randint(1, n - 2)
                axis = random.randint(0, 2)
                angle = random.choice([1, 2, 3])
                
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
                
                for p in old_positions:
                    occ.remove(p)
                    del pos_to_idx[p]
                
                for i, p in enumerate(new_subchain):
                    occ.add(p)
                    pos_to_idx[p] = pivot + 1 + i
                
                chain[pivot+1:] = new_subchain
                new_energy = calc_energy()
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = chain[:]
                else:
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
    
    # Try beam search with a reasonable width
    chain = _beam_search(sequence, dimension, beam_width=300)
    
    if chain is None:
        # Fallback to greedy if beam search fails
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
    
    # Apply local search to improve
    if len(chain) == n:
        # Use a larger iteration count for local search if time permits
        # Estimate time per iteration is roughly constant, but we have a deadline guard inside
        # For shorter chains, we can afford more iterations
        # For longer chains, fewer iterations but each is more complex
        # Let's scale max_iters slightly with chain length but cap it
        base_iters = 2000
        if n < 50:
            max_ls_iters = 5000
            max_qs_iters = 2000
        elif n < 100:
            max_ls_iters = 3000
            max_qs_iters = 1500
        else:
            max_ls_iters = 2000
            max_qs_iters = 1000
            
        chain = _local_search(chain, sequence, dimension, max_iters=max_ls_iters)
        
        # Apply quantum tunneling search
        if not _is_deadline_exceeded():
            chain = _quantum_tunneling_search(chain, sequence, dimension, max_iters=max_qs_iters)
    
    return [list(p) for p in chain]

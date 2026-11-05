import random
import time
import sys
import math
from collections import deque

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Return the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos):
    """
    Compute the change in energy when moving residue i from old_pos to new_pos.
    pos_dict maps position -> index.
    """
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    
    # Remove contacts with old_pos
    for q in _neighbours(old_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta += 1  # Losing a contact increases energy by 1
            
    # Add contacts with new_pos
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1  # Gaining a contact decreases energy by 1
            
    return delta

def _calculate_segment_delta(sequence, chain, pos_dict, start, end, new_positions):
    """
    Calculate energy delta for moving a segment [start:end] to new_positions.
    Assumes new_positions is a list of tuples of length end-start.
    Assumes the move is valid (self-avoiding).
    """
    n = len(chain)
    delta = 0
    old_positions = chain[start:end]
    
    # 1. Remove old contacts involving the segment
    # Contacts are (i, j) where |i-j|>1 and they are neighbors.
    # We need to remove any contact where i is in [start, end] or j is in [start, end].
    
    # To do this efficiently, we iterate over the segment and its neighbors in the OLD chain.
    # But pos_dict is for the CURRENT chain.
    
    # Let's build a set of indices in the segment for quick check
    seg_indices = set(range(start, end))
    
    # Iterate over all residues in the segment
    for i in range(start, end):
        if sequence[i] != 'H':
            continue
        old_pos = chain[i]
        for q in _neighbours(old_pos):
            j = pos_dict.get(q)
            if j is not None and j != i and abs(j - i) > 1:
                # Contact exists in old chain
                # If j is NOT in the segment, this contact is lost.
                # If j IS in the segment, this contact might be preserved or changed.
                # Actually, rigid rotation preserves internal distances.
                # So if i and j are both in the segment, and they were neighbors, they will still be neighbors?
                # NO. Rotation around a pivot changes the absolute positions, but the relative vectors between i and j are preserved.
                # So if i and j were neighbors (Manhattan dist 1), they remain neighbors.
                # Thus, internal contacts are preserved.
                # We only care about contacts BETWEEN the segment and the REST of the chain.
                
                if j not in seg_indices:
                    # Contact with outside is lost
                    if sequence[j] == 'H':
                        delta += 1

    # 2. Add new contacts involving the segment
    # We need to check new positions against the REST of the chain.
    # The rest of the chain is fixed.
    
    # Build a set of new positions for the segment for quick lookup? 
    # No, we iterate over the new positions and check neighbors in the FIXED part.
    
    # We need to know which indices are in the fixed part.
    # Fixed part indices: [0, start) U (end, n)
    
    for k, new_pos in enumerate(new_positions):
        i = start + k
        if sequence[i] != 'H':
            continue
            
        for q in _neighbours(new_pos):
            j = pos_dict.get(q)
            if j is not None:
                # j is in the current (old) chain
                # If j is in the segment, it's an internal contact (preserved, so ignore for delta? 
                # Wait, if j is in the segment, pos_dict has the OLD position. 
                # The new position of j is new_positions[j-start].
                # We are checking if new_pos (of i) is neighbor to OLD pos of j.
                # This is incorrect. We must check if new_pos (of i) is neighbor to NEW pos of j.
                
                # If j is in the segment:
                if j in seg_indices:
                    # Check if new_pos is neighbor to new_pos_j
                    new_pos_j = new_positions[j - start]
                    if abs(new_pos[0] - new_pos_j[0]) + abs(new_pos[1] - new_pos_j[1]) == 1:
                         # They are still neighbors. Internal contact preserved.
                         # But wait, did they lose a contact with each other? No, rigid motion.
                         # Did they gain? No.
                         # So internal contacts don't contribute to delta.
                         pass
                else:
                    # j is in the fixed part. Its position is pos_dict[j] (which is the same as chain[j]).
                    # Check if new_pos is neighbor to chain[j]
                    if abs(new_pos[0] - chain[j][0]) + abs(new_pos[1] - chain[j][1]) == 1:
                         # New contact formed
                         if abs(i - j) > 1 and sequence[j] == 'H':
                             delta -= 1

    return delta

def _try_pivot_move(sequence, chain, pos_dict, i, L, rng):
    """
    Try to rotate segment [i+1, i+L] around i.
    Returns (success, new_chain, delta, new_pos_dict) or (False, None, 0, None)
    """
    n = len(chain)
    if i < 0 or i >= n - L:
        return False, None, 0, None
    
    # Segment is i+1 to i+L
    start_idx = i + 1
    end_idx = i + L
    
    if start_idx >= n:
        return False, None, 0, None
        
    pivot = chain[i]
    segment = chain[start_idx:end_idx]
    
    # Directions for rotation: 90, 180, 270
    # We can simulate rotation by transforming coordinates.
    # Let's try all 3 rotations.
    
    # Optimization: Pre-compute the "free" space? No, just check collision.
    # The rest of the chain is chain[:i] + chain[end_idx:]
    # We need to check if any new position collides with the rest.
    
    # To speed up collision check, we can use a set of occupied positions 
    # excluding the segment itself.
    
    # However, creating a set every time is O(N). 
    # We maintain `pos_dict` globally. 
    # For collision check, we can iterate new positions and check `pos_dict`.
    # If `pos_dict.get(new_pos)` is not None, and the index is NOT in [start_idx, end_idx], it's a collision.
    
    rotations = [
        # 90 deg clockwise: (x, y) -> (y, -x) relative to pivot
        lambda x, y: (pivot[0] + (y - pivot[1]), pivot[1] - (x - pivot[0])),
        # 180 deg
        lambda x, y: (2 * pivot[0] - x, 2 * pivot[1] - y),
        # 270 deg clockwise (or 90 counter-clockwise)
        lambda x, y: (pivot[0] - (y - pivot[1]), pivot[1] + (x - pivot[0]))
    ]
    
    # Randomly pick a rotation
    rot_func = rng.choice(rotations)
    
    new_segment = [rot_func(x, y) for x, y in segment]
    
    # Check validity
    # 1. Self-avoidance with rest of chain
    for new_pos in new_segment:
        idx = pos_dict.get(new_pos)
        if idx is not None:
            # If idx is in the segment, it's okay (internal)
            # But wait, if two different residues in the segment map to the same new_pos?
            # Rigid rotation is an isometry, so distinct points remain distinct.
            # So we only need to check against indices OUTSIDE the segment.
            if idx < start_idx or idx >= end_idx:
                return False, None, 0, None
            
    # 2. Check connectivity
    # The segment is moved as a whole. 
    # Connection at start: chain[i] to new_segment[0]
    # Connection at end: new_segment[-1] to chain[end_idx]
    
    d1 = abs(pivot[0] - new_segment[0][0]) + abs(pivot[1] - new_segment[0][1])
    if d1 != 1:
        return False, None, 0, None
        
    if end_idx < n:
        next_pos = chain[end_idx]
        d2 = abs(new_segment[-1][0] - next_pos[0]) + abs(new_segment[-1][1] - next_pos[1])
        if d2 != 1:
            return False, None, 0, None
    # If end_idx == n, no next connection needed.

    # Calculate Delta
    delta = _calculate_segment_delta(sequence, chain, pos_dict, start_idx, end_idx, new_segment)
    
    # Construct new chain
    new_chain = chain[:start_idx] + new_segment + chain[end_idx:]
    
    # Update pos_dict
    # Remove old segment positions
    for pos in segment:
        del pos_dict[pos]
    # Add new segment positions
    for k, pos in enumerate(new_segment):
        pos_dict[pos] = start_idx + k
        
    return True, new_chain, delta, pos_dict

def _local_search(sequence, chain, deadline, rng):
    """
    Local search with Single, Pivot, and Segment moves.
    """
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 4.0
    cooling = 0.9995
    min_T = 0.05
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        
        # Move selection
        # For short chains, pivots are very effective.
        # For long chains, single moves are cheap.
        
        r = rng.random()
        
        if r < 0.5:
            # Single move
            i = rng.randint(1, n - 2)
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            old_pos = current_chain[i]
            
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
            
            if not candidates:
                continue
                
            new_pos = rng.choice(candidates)
            delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                del pos_dict[old_pos]
                current_chain[i] = new_pos
                pos_dict[new_pos] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        elif r < 0.8:
            # Pivot move
            # Choose pivot i
            i = rng.randint(1, n - 2)
            # Choose segment length L (short to keep O(L) low)
            max_L = min(8, n - i - 1)
            if max_L < 1:
                continue
            L = rng.randint(1, max_L)
            
            success, new_chain, delta, new_pos_dict = _try_pivot_move(sequence, current_chain, pos_dict, i, L, rng)
            
            if success:
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    pos_dict = new_pos_dict
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        else:
            # Segment re-insertion (Cut & Paste) - simplified version
            # Cut a small segment and try to re-insert it at a random valid location.
            # This is complex to implement correctly with O(N) validation.
            # Instead, let's do a "Swap" of two segments? No, that breaks connectivity.
            # Let's stick to Pivot and Single. 
            # Maybe try a longer pivot?
            i = rng.randint(1, n - 2)
            max_L = min(15, n - i - 1)
            if max_L < 1:
                continue
            L = rng.randint(5, max_L)
            
            success, new_chain, delta, new_pos_dict = _try_pivot_move(sequence, current_chain, pos_dict, i, L, rng)
            
            if success:
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    pos_dict = new_pos_dict
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        if no_improve_count > max_no_improve:
            T = max(T * 1.5, 2.0)
            no_improve_count = 0

    return best_chain, best_energy

def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation
        # Simple random walk
        init_chain = None
        for _ in range(5):
            init_chain = _random_walk(sequence, rng)
            if init_chain:
                break
        
        if init_chain is None:
            init_chain = [(i, 0) for i in range(n)]
        
        init_energy = _energy(sequence, init_chain)
        
        # Run local search
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy

def _random_walk(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in dirs:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        if not candidates:
            return None
        new_pos = rng.choice(candidates)
        chain.append(new_pos)
        occupied.add(new_pos)
    return chain

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    n = len(sequence)
    
    # Total budget: 120s for all sequences. 
    # Safety margin: 11s per sequence.
    
    deadline = start_time + 11.0
    
    if n == 0:
        return []
    
    # Number of restarts
    if n <= 30:
        n_restarts = 40
    elif n <= 50:
        n_restarts = 30
    elif n <= 70:
        n_restarts = 20
    else:
        n_restarts = 15
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

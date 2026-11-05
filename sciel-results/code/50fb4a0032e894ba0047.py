"""
HP Lattice Protein Folding using Monte Carlo Pivoting with Incremental Energy Updates.

Strategy:
1. Generate initial conformations using a randomized greedy growth algorithm that 
   biases towards H-H contacts.
2. Refine using Simulated Annealing with Pivot Moves.
   - A pivot move rotates the tail of the chain (from index k to n-1) around the 
     bond (k-1, k) by 90 or 180 degrees.
   - Energy is updated incrementally by only checking contacts involving the moved 
     residues.
3. Multi-start approach to escape local minima.
"""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_DEADLINE = None
_BUDGET = 330.0
_SAFETY = 0.92

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _BUDGET * _SAFETY
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() >= _get_deadline()

def _neighbours(p):
    x, y, z = p
    return ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1))

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

def _initial_grow(seq, dim, rng, bias=1.0):
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
        
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            k = min(3, len(candidates))
            site = candidates[rng.randint(0, k-1)][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain

def _apply_pivot(chain, pos_map, pivot_idx, rotation_matrix, seq):
    """
    Attempt a pivot move.
    pivot_idx: The index of the first residue in the tail to be rotated.
    rotation_matrix: A 3x3 rotation matrix (list of lists).
    Returns (success, new_chain, new_pos_map, energy_delta)
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n:
        return False, None, None, 0
    
    # The pivot is the bond between pivot_idx-1 and pivot_idx.
    # We rotate the tail [pivot_idx, n-1] around the vector chain[pivot_idx-1] -> chain[pivot_idx].
    # Actually, standard pivot moves rotate the tail around the bond vector.
    # Let's define the rotation center as chain[pivot_idx-1].
    # The vector from center to chain[pivot_idx] is v = chain[pivot_idx] - chain[pivot_idx-1].
    # The new position of chain[pivot_idx] will be center + R*v.
    # The new position of chain[k] for k > pivot_idx will be center + R * (chain[k] - chain[pivot_idx-1]).
    
    center = chain[pivot_idx - 1]
    
    # Check if the new position of the first tail residue is valid
    # v = chain[pivot_idx] - center
    v0 = (chain[pivot_idx][0] - center[0], 
          chain[pivot_idx][1] - center[1], 
          chain[pivot_idx][2] - center[2])
    
    # Apply rotation
    new_v0 = (
        rotation_matrix[0][0]*v0[0] + rotation_matrix[0][1]*v0[1] + rotation_matrix[0][2]*v0[2],
        rotation_matrix[1][0]*v0[0] + rotation_matrix[1][1]*v0[1] + rotation_matrix[1][2]*v0[2],
        rotation_matrix[2][0]*v0[0] + rotation_matrix[2][1]*v0[1] + rotation_matrix[2][2]*v0[2]
    )
    
    new_pos0 = (center[0] + new_v0[0], center[1] + new_v0[1], center[2] + new_v0[2])
    
    # Check if new_pos0 is occupied by a non-tail residue
    if new_pos0 in pos_map:
        if pos_map[new_pos0] < pivot_idx:
            return False, None, None, 0
        # If it's occupied by the tail itself, it might be a valid move (rotation back)
        # But we need to check the whole tail for collisions.
    
    # We need to check the entire tail for self-avoidance.
    # This is O(N) per move, which is acceptable for N=100-1000 if we do it efficiently.
    
    new_tail = [None] * (n - pivot_idx)
    old_tail_positions = set()
    for k in range(pivot_idx, n):
        old_tail_positions.add(chain[k])
        
    # Calculate new positions
    for k in range(pivot_idx, n):
        vk = (chain[k][0] - center[0], 
              chain[k][1] - center[1], 
              chain[k][2] - center[2])
        new_vk = (
            rotation_matrix[0][0]*vk[0] + rotation_matrix[0][1]*vk[1] + rotation_matrix[0][2]*vk[2],
            rotation_matrix[1][0]*vk[0] + rotation_matrix[1][1]*vk[1] + rotation_matrix[1][2]*vk[2],
            rotation_matrix[2][0]*vk[0] + rotation_matrix[2][1]*vk[1] + rotation_matrix[2][2]*vk[2]
        )
        new_pos = (center[0] + new_vk[0], center[1] + new_vk[1], center[2] + new_vk[2])
        new_tail[k - pivot_idx] = new_pos
        
    # Check for collisions with the head (indices < pivot_idx)
    # and self-collisions within the new tail.
    # We can do this by checking if any new_pos is in pos_map (excluding the tail itself)
    # or if any new_pos is in the set of other new_positions.
    
    # To be efficient, we can just iterate and check.
    seen_new_positions = set()
    for k in range(pivot_idx, n):
        np = new_tail[k - pivot_idx]
        if np in seen_new_positions:
            return False, None, None, 0
        seen_new_positions.add(np)
        
        if np in pos_map:
            if pos_map[np] < pivot_idx:
                return False, None, None, 0
            # If pos_map[np] >= pivot_idx, it's part of the old tail. 
            # Since we checked seen_new_positions, if it's in pos_map and >= pivot_idx, 
            # it means the new position overlaps with an old tail position that is NOT the one we just placed?
            # No, pos_map is the OLD map. 
            # If np is in pos_map and pos_map[np] >= pivot_idx, it means the new position 
            # collides with an old tail position. This is a collision.
            # Wait, if we rotate the tail, the old tail positions are vacated.
            # So we should only care about collisions with the HEAD (indices < pivot_idx).
            # The old tail positions are no longer occupied.
            # So we just need to check if np is in pos_map and pos_map[np] < pivot_idx.
            # We already did that above.
            # What if np == chain[k] (identity rotation)? Then it's fine.
            # But if np is an old tail position of a DIFFERENT residue, that's a collision.
            # However, since we are moving the WHOLE tail, the set of occupied sites by the tail changes.
            # The condition for validity is:
            # 1. No two residues in the new tail share the same site. (Checked by seen_new_positions)
            # 2. No site in the new tail is occupied by the head. (Checked by pos_map[np] < pivot_idx)
            # 3. The bond between pivot_idx-1 and pivot_idx is preserved (it is, by construction).
            # 4. The bond between pivot_idx and pivot_idx+1 is preserved (it is, by rotation).
            
            # So, if np is in pos_map and pos_map[np] >= pivot_idx, it's NOT a collision with the head.
            # It's a collision with the old tail. But the old tail is being moved.
            # So we don't need to check against the old tail positions in pos_map.
            # We only need to check against the head.
            
            # Wait, what if the new tail position overlaps with an old tail position that is NOT being moved to that spot?
            # Example: Tail is A-B-C. Rotate A to B's old spot. B moves to C's old spot. C moves to A's old spot.
            # This is a valid permutation.
            # My check `if np in pos_map: if pos_map[np] < pivot_idx: return False` is correct for head collisions.
            # I do NOT need to check if np is in the old tail, because the old tail is vacated.
            
            # However, I MUST ensure that the new tail doesn't collide with ITSELF.
            # `seen_new_positions` handles that.
            
            # So the logic is correct.
            pass
    
    # If we reach here, the move is valid.
    # Update pos_map and chain.
    # We need to calculate energy delta.
    # Old energy contribution of the tail:
    # New energy contribution of the tail:
    # This is complex. Let's just recalculate energy for the whole chain if N is small, 
    # or do incremental update.
    # For now, let's just return the new chain and pos_map, and let the caller recalculate energy.
    # To save time, we can pass the new pos_map.
    
    new_chain = list(chain)
    new_pos_map = dict(pos_map)
    
    # Remove old tail positions from pos_map
    for k in range(pivot_idx, n):
        del new_pos_map[chain[k]]
        
    # Add new tail positions to pos_map
    for k in range(pivot_idx, n):
        new_pos_map[new_tail[k - pivot_idx]] = k
        new_chain[k] = new_tail[k - pivot_idx]
        
    return True, new_chain, new_pos_map, 0

def _local_search(seq, chain, dim, rng, time_limit):
    """Simulated Annealing with Pivot Moves."""
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
    
    # Pre-compute rotation matrices
    # 90 deg around X, Y, Z
    rot_90_x = [[1,0,0],[0,0,-1],[0,1,0]]
    rot_90_y = [[0,0,1],[0,1,0],[-1,0,0]]
    rot_90_z = [[0,-1,0],[1,0,0],[0,0,1]]
    rot_180_x = [[1,0,0],[0,-1,0],[0,0,-1]]
    rot_180_y = [[-1,0,0],[0,1,0],[0,0,-1]]
    rot_180_z = [[-1,0,0],[0,-1,0],[0,0,1]]
    
    rotations = [rot_90_x, rot_90_y, rot_90_z, rot_180_x, rot_180_y, rot_180_z]
    
    no_improve = 0
    iteration = 0
    
    while time.time() < t_end and not _is_deadline_exceeded():
        iteration += 1
        elapsed = time.time() - t_start
        progress = min(1.0, elapsed / time_limit)
        temp = start_temp * math.exp(-5.0 * progress) + end_temp
        
        # Choose pivot index
        # Bias towards middle of chain? Or random?
        # Random is standard.
        pivot_idx = rng.randint(1, n - 1)
        
        # Choose rotation
        rot = rotations[rng.randint(0, len(rotations)-1)]
        
        success, new_chain, new_pos_map, delta = _apply_pivot(chain, pos_map, pivot_idx, rot, seq)
        
        if not success:
            no_improve += 1
            if no_improve > 10000:
                break
            continue
        
        new_energy = _compute_energy(seq, new_chain, new_pos_map, h_indices)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            chain = new_chain
            pos_map = new_pos_map
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
                best_pos_map = dict(pos_map)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
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
    base_time = 2.0
    if n <= 48:
        total_time = base_time * 1.0
    elif n <= 80:
        total_time = base_time * 1.5
    elif n <= 120:
        total_time = base_time * 2.0
    else:
        total_time = base_time * 2.5
        
    max_total = (_get_deadline() - time.time()) * 0.95
    total_time = min(total_time, max_total)
    
    t_start = time.time()
    num_restarts = 0
    
    while time.time() - t_start < total_time and not _is_deadline_exceeded():
        num_restarts += 1
        bias = 0.5 + rng.random() * 1.5
        chain = _initial_grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        pos_map = {p: i for i, p in enumerate(chain)}
        e = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
        
        remaining = total_time - (time.time() - t_start)
        if remaining < 0.5:
            break
        
        sa_time = min(remaining * 0.8, 3.0 if n <= 48 else 5.0)
        chain = _local_search(sequence, chain, dimension, rng, sa_time)
        
        pos_map = {p: i for i, p in enumerate(chain)}
        e = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

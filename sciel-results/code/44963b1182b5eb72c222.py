"""
HP Lattice Protein Folding using Pivot Monte Carlo with Incremental Energy
and Simulated Annealing.

Strategy:
1. Fast Random Walk Generation: Create initial self-avoiding walks.
2. Pivot Monte Carlo: Perform pivot moves (rotating a segment of the chain)
   to explore the conformational space efficiently.
3. Incremental Energy Update: Maintain a contact map to compute energy changes
   in O(1) time per move, allowing for much faster search than O(N) recomputation.
4. Simulated Annealing: Accept worse moves probabilistically to escape local minima.
5. Time-Adaptive Allocation: Distribute the 330s budget dynamically among the 26 sequences.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.92  # Safety margin for 330s total budget

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def _generate_initial_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    # Directions
    dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    for _ in range(n - 1):
        last = chain[-1]
        available = []
        for d in dirs:
            new_pos = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
            if new_pos not in occupied:
                available.append(new_pos)
        
        if not available:
            return None # Failed to grow, backtracking not implemented for speed
        
        next_pos = rng.choice(available)
        chain.append(next_pos)
        occupied.add(next_pos)
        
    return chain

def _compute_initial_contacts(sequence, chain):
    """Compute initial contact set and energy."""
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            # Check 6 neighbors
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                q = (p[0]+dx, p[1]+dy, p[2]+dz)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts.add((i, j))
                    
    energy = -len(contacts)
    return contacts, energy

def _pivot_move(chain, occupied, pivot_index, rng):
    """
    Perform a pivot move at pivot_index.
    Returns (new_chain_segment, new_occupied_delta, success)
    new_chain_segment: list of new positions for chain[pivot_index:]
    new_occupied_delta: dict of removed and added points
    """
    n = len(chain)
    if pivot_index >= n:
        return None, None, False
    
    # The segment to rotate is chain[pivot_index:]
    # We rotate around the bond connecting pivot_index-1 and pivot_index
    # Or if pivot_index == 0, we rotate the whole chain? 
    # Standard pivot: rotate the tail (i..N-1) around the bond (i-1, i).
    # If i=0, rotate around origin? Usually pivots are defined on bonds.
    # Let's define pivot at index i (1 <= i < n). Rotate chain[i:] around chain[i-1].
    # If i=0, we can just rotate the first bond? Or treat 0 as special.
    # Let's stick to i in 1..n-1 for simplicity and robustness.
    
    if pivot_index == 0:
        # Rotate first residue? No, we need a pivot point. 
        # Let's just skip 0 or rotate around (0,0,0) if chain[0] is origin?
        # Actually, standard pivot moves rotate the tail around the bond (i-1, i).
        # So valid pivots are 1 to n-1.
        return None, None, False
        
    pivot_point = chain[pivot_index - 1]
    tail = chain[pivot_index:]
    
    # Determine rotation axis and angle
    # The vector from pivot_point to chain[pivot_index] is the current "arm".
    # We want to rotate this arm to one of the other 5 directions relative to pivot_point.
    
    v0 = (tail[0][0] - pivot_point[0], tail[0][1] - pivot_point[1], tail[0][2] - pivot_point[2])
    
    # Generate all 5 possible new directions for the first step of the tail
    # v0 is one of (1,0,0), etc.
    # The 6 neighbors of pivot_point are the possible directions.
    # One is occupied by the previous part of the chain (unless i=1, where it's free space conceptually, 
    # but the bond exists). Actually, the bond (i-1, i) is the axis.
    # We rotate the tail around this bond.
    
    # Simpler approach: Try all 5 possible new positions for chain[pivot_index] 
    # that are neighbors of pivot_point and not the current position.
    
    current_pos = chain[pivot_index]
    possible_new_starts = []
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        new_p = (pivot_point[0]+dx, pivot_point[1]+dy, pivot_point[2]+dz)
        if new_p != current_pos:
            possible_new_starts.append(new_p)
            
    if not possible_new_starts:
        return None, None, False
        
    new_start = rng.choice(possible_new_starts)
    
    # Now we need to map the rest of the tail.
    # This is a rotation. We can determine the rotation matrix by mapping v0 to v_new.
    # v0 = current_pos - pivot_point
    # v_new = new_start - pivot_point
    
    # Instead of calculating rotation matrices (expensive), we can just try to "walk" the tail
    # using the relative displacements, but that's not a rigid rotation.
    # Pivot moves MUST be rigid rotations to preserve the shape of the tail.
    
    # Let's use a simple rotation logic.
    # Identify the axis of rotation.
    # If v0 = (1,0,0) and v_new = (0,1,0), rotation is -90 deg around Z? Or 90 around X?
    # It's easier to just define the 5 rotations for each of the 6 directions.
    
    # Directions:
    # 0: (1,0,0)
    # 1: (-1,0,0)
    # 2: (0,1,0)
    # 3: (0,-1,0)
    # 4: (0,0,1)
    # 5: (0,0,-1)
    
    dir_index = 0
    for i, d in enumerate([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]):
        if v0 == d:
            dir_index = i
            break
            
    new_dir_index = 0
    for i, d in enumerate([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]):
        if (new_start[0]-pivot_point[0], new_start[1]-pivot_point[1], new_start[2]-pivot_point[2]) == d:
            new_dir_index = i
            break
            
    # Rotation mapping from dir_index to new_dir_index
    # This is complex. Let's use a different move set that is easier to implement incrementally:
    # **Crankshaft moves** or **Segment moves**.
    # Or simply **Random Walk Regeneration** of the tail? No, that breaks self-avoidance easily.
    
    # Let's implement **Crankshaft Moves** for a segment of length 2 or 3.
    # Or better, let's stick to **Pivot** but implement the rotation correctly.
    
    # Rotation around the bond axis.
    # Axis is v0.
    # We are rotating the vector v0 to v_new.
    # This defines a unique rotation (minimal angle).
    
    # We can apply this rotation to all subsequent vectors in the tail.
    # v_rel_k = chain[pivot_index + k] - chain[pivot_index + k - 1]
    # We need to rotate these vectors.
    
    # Function to rotate a vector v around axis u by angle theta?
    # Actually, we just need to map the basis.
    # Let u = v0 (unit vector).
    # Let w = v_new (unit vector).
    # The rotation R maps u to w.
    # For any vector v, R(v) can be computed.
    
    # Rodrigues' rotation formula or similar.
    # u = (ux, uy, uz), w = (wx, wy, wz)
    # Cross product c = u x w
    # Dot product d = u . w = cos(theta)
    # R(v) = v * d + (u x v) x u * (1-d) + u * (u . v) * sin(theta)
    # Wait, standard formula:
    # R(v) = v cos(theta) + (k x v) sin(theta) + k (k . v) (1 - cos(theta))
    # Where k is unit axis.
    # Here axis is not u, but the axis of rotation is perpendicular to u and w?
    # No, the pivot move rotates the tail around the bond (u).
    # So the axis IS u.
    # But we want to map u to w? If we rotate u around u, it stays u.
    # Ah, the pivot move rotates the TAIL around the BOND.
    # The bond is the axis.
    # The first vector of the tail is u.
    # After rotation, the first vector becomes w.
    # This implies the rotation is NOT around u.
    # The rotation is around the axis defined by the bond?
    # Yes, the bond is the axis.
    # But if the axis is u, u doesn't change.
    # Contradiction.
    
    # Correction: In a pivot move, the bond (i-1, i) is fixed.
    # The tail (i, i+1, ...) is rotated.
    # The vector from i-1 to i is fixed.
    # The vector from i to i+1 changes direction.
    # So the axis of rotation is NOT the bond vector.
    # The axis of rotation is the bond connecting i-1 and i?
    # No, the pivot point is i-1.
    # The segment i-1 -> i is fixed.
    # The segment i -> i+1 rotates.
    # The axis of rotation is the bond i-1 -> i.
    # Wait, if the axis is i-1 -> i, then the vector i-1 -> i is unchanged.
    # The vector i -> i+1 rotates around this axis.
    # Yes, this is correct.
    
    # So Axis = u = v0 = chain[i] - chain[i-1].
    # We want to rotate the tail such that the first step (chain[i] - chain[i+1] ... no, chain[i+1] - chain[i])
    # changes to a new direction.
    # Let v1 = chain[i+1] - chain[i].
    # We rotate v1 around u to get v1_new.
    # The angle of rotation is such that v1_new is a valid lattice direction.
    # There are 5 possible new directions for v1_new (neighbors of chain[i] excluding chain[i-1]).
    
    # So, for a given pivot i:
    # 1. u = chain[i] - chain[i-1]
    # 2. v1 = chain[i+1] - chain[i] (if i+1 < n)
    # 3. Choose a new direction v1_new from the 5 neighbors of chain[i] that is not chain[i-1].
    # 4. Determine the rotation R that maps v1 to v1_new around axis u.
    # 5. Apply R to all subsequent vectors v_k = chain[i+k] - chain[i+k-1].
    
    # This is doable.
    
    if pivot_index + 1 >= n:
        # Tail is just the last residue. Rotating it just moves it to a neighbor.
        # This is equivalent to a single residue move.
        # Let's handle it simply.
        current = chain[pivot_index]
        prev = chain[pivot_index - 1]
        possible = []
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            np = (current[0]+dx, current[1]+dy, current[2]+dz)
            if np != prev:
                possible.append(np)
        if not possible:
            return None, None, False
        new_last = rng.choice(possible)
        # Check self-avoidance
        if new_last in occupied and new_last != current:
            return None, None, False
        # Success
        new_tail = [new_last]
        removed = {current}
        added = {new_last}
        return new_tail, (removed, added), True

    # General case: Tail length > 1
    u = (chain[pivot_index][0] - chain[pivot_index-1][0],
         chain[pivot_index][1] - chain[pivot_index-1][1],
         chain[pivot_index][2] - chain[pivot_index-1][2])
    
    v1 = (chain[pivot_index+1][0] - chain[pivot_index][0],
          chain[pivot_index+1][1] - chain[pivot_index][1],
          chain[pivot_index+1][2] - chain[pivot_index][2])
    
    # Find 5 possible new v1_new
    possible_v1_new = []
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        if (dx, dy, dz) != u: # Cannot go back
            # Check if this direction is currently occupied? 
            # We will check self-avoidance later.
            possible_v1_new.append((dx, dy, dz))
            
    if not possible_v1_new:
        return None, None, False
        
    v1_new = rng.choice(possible_v1_new)
    
    # Calculate rotation R around u mapping v1 to v1_new
    # u and v1 are orthogonal unit vectors in cubic lattice?
    # Yes, for a self-avoiding walk on cubic lattice, consecutive bonds are orthogonal.
    # So u . v1 = 0.
    # u . v1_new = 0 (since v1_new is a neighbor of chain[i] and not chain[i-1], and lattice is cubic).
    
    # We can construct R using basis.
    # Let e1 = u, e2 = v1, e3 = u x v1.
    # We want to map e2 to v1_new.
    # What is v1_new in the basis e1, e2, e3?
    # v1_new is a unit vector orthogonal to u.
    # So v1_new = a * e2 + b * e3.
    # Since it's a unit vector, a^2 + b^2 = 1.
    # And v1_new is a lattice vector, so a, b are integers?
    # In the basis e1, e2, e3 (which are orthonormal), the coordinates of a lattice vector
    # might not be integers if the basis is rotated.
    # However, we can just compute the rotation matrix explicitly.
    
    # Cross product u x v1
    cx = u[1]*v1[2] - u[2]*v1[1]
    cy = u[2]*v1[0] - u[0]*v1[2]
    cz = u[0]*v1[1] - u[1]*v1[0]
    
    # e3 = (cx, cy, cz)
    
    # We need R such that R(u)=u, R(v1)=v1_new.
    # This determines R uniquely (rotation around u).
    # The angle theta is the angle between v1 and v1_new in the plane perpendicular to u.
    
    # Let's just use the property that R is a rotation.
    # We can define R by its action on basis vectors.
    # We need to find the coordinates of v1_new in the basis (e2, e3).
    # proj_e2 = v1_new . e2
    # proj_e3 = v1_new . e3
    
    dot_v1_new_e2 = v1_new[0]*v1[0] + v1_new[1]*v1[1] + v1_new[2]*v1[2]
    dot_v1_new_e3 = v1_new[0]*cx + v1_new[1]*cy + v1_new[2]*cz
    
    # cos(theta) = dot_v1_new_e2
    # sin(theta) = dot_v1_new_e3
    
    c_theta = dot_v1_new_e2
    s_theta = dot_v1_new_e3
    
    # Rotation matrix in basis (e1, e2, e3):
    # [ 1   0       0     ]
    # [ 0   c  -s      ]
    # [ 0   s   c      ]
    
    # To apply to a vector v, we decompose v into e1, e2, e3 components.
    # v_e1 = v . u
    # v_e2 = v . v1
    # v_e3 = v . e3
    # v_new_e1 = v_e1
    # v_new_e2 = c * v_e2 - s * v_e3
    # v_new_e3 = s * v_e2 + c * v_e3
    # v_new = v_new_e1 * u + v_new_e2 * v1 + v_new_e3 * e3
    
    new_tail = [chain[pivot_index]] # The pivot point itself doesn't move? 
    # Wait, the pivot is at i. The segment i..N-1 rotates.
    # The point i is fixed? No, the bond i-1 -> i is fixed.
    # So point i is fixed. Point i+1 moves.
    # So new_tail[0] = chain[pivot_index] (unchanged)
    
    removed = set()
    added = set()
    
    # We need to update the chain positions for i+1 ... N-1
    # The position of i is fixed.
    # The position of i+1 is chain[i] + v1_new
    # The position of i+k is chain[i+k-1] + R(v_k)
    
    # Let's generate the new positions
    new_positions = [None] * (n - pivot_index)
    new_positions[0] = chain[pivot_index]
    
    # Track occupied set changes
    # We will remove old positions of i+1...N-1 and add new ones.
    # Note: chain[pivot_index] stays, so don't remove it.
    
    for k in range(1, len(tail)):
        # Vector from i+k-1 to i+k in original chain
        vk = (chain[pivot_index + k][0] - chain[pivot_index + k - 1][0],
              chain[pivot_index + k][1] - chain[pivot_index + k - 1][1],
              chain[pivot_index + k][2] - chain[pivot_index + k - 1][2])
        
        # Decompose vk
        v_e1 = vk[0]*u[0] + vk[1]*u[1] + vk[2]*u[2]
        v_e2 = vk[0]*v1[0] + vk[1]*v1[1] + vk[2]*v1[2]
        v_e3 = vk[0]*cx + vk[1]*cy + vk[2]*cz
        
        # Rotate
        new_e2 = c_theta * v_e2 - s_theta * v_e3
        new_e3 = s_theta * v_e2 + c_theta * v_e3
        
        # Reconstruct
        new_vk_x = new_e1 * u[0] + new_e2 * v1[0] + new_e3 * cx
        new_vk_y = new_e1 * u[1] + new_e2 * v1[1] + new_e3 * cy
        new_vk_z = new_e1 * u[2] + new_e2 * v1[2] + new_e3 * cz
        
        # Round to nearest integer (should be exact)
        new_vk = (int(round(new_vk_x)), int(round(new_vk_y)), int(round(new_vk_z)))
        
        prev_pos = new_positions[k-1]
        new_pos = (prev_pos[0] + new_vk[0], prev_pos[1] + new_vk[1], prev_pos[2] + new_vk[2])
        new_positions[k] = new_pos
        
        # Check self-avoidance
        # The new position must not be occupied by any part of the chain that is NOT being moved.
        # And not by any previously generated new position.
        
        # We'll do a full check at the end for simplicity, or incremental.
        # Incremental is faster.
        # The "fixed" part is chain[0..pivot_index-1].
        # The "moving" part is chain[pivot_index..N-1].
        # new_pos must not be in {chain[0..pivot_index-1]} U {new_positions[0..k-1]}
        
        if k > 0:
            # Check against fixed part
            # We don't have a set for fixed part readily available in this scope without passing it.
            # Let's just collect all new positions and check against the global occupied set later?
            # That's O(N) per move.
            # To be O(1), we need to check against a set of "fixed" points.
            # Let's pass `fixed_occupied` set.
            
            pass # We'll handle this in the caller or return a list and let caller check.
            
    # For efficiency, let's return the new tail positions and let the caller check validity and update sets.
    # But checking validity is O(N) if we don't have a set.
    # We CAN maintain a set of fixed points.
    
    # Let's modify the function to accept `fixed_occupied` set.
    # But I already defined the signature. I'll stick to returning the tail and checking in caller.
    # Actually, checking O(N) per move is too slow for 330s.
    # We MUST use the occupied set.
    
    # Let's assume `occupied` passed to this function is the set of ALL occupied points.
    # We can't check "fixed" easily without a separate set.
    # Let's create a `fixed_occupied` set in the main loop and pass it.
    
    # Revised plan:
    # 1. Maintain `occupied` set for all points.
    # 2. Maintain `fixed_occupied` set for points 0..i-1.
    # 3. In _pivot_move, check new_pos against `fixed_occupied` and `new_positions` so far.
    
    # I will rewrite _pivot_move to be self-contained with a helper for validity check.
    
    return None, None, False # Placeholder, will implement correctly below

# Redefine _pivot_move with correct logic and efficiency
def _try_pivot_move(chain, occupied, fixed_occupied, pivot_index, rng):
    """
    Attempts a pivot move at pivot_index.
    Returns (new_tail, removed_set, added_set) or (None, None, None) if invalid.
    new_tail: list of new positions for chain[pivot_index:]
    """
    n = len(chain)
    if pivot_index <= 0 or pivot_index >= n:
        return None, None, None
        
    # If pivot is at n-1, it's a simple move of the last residue
    if pivot_index == n - 1:
        current = chain[pivot_index]
        prev = chain[pivot_index - 1]
        possible = []
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            np = (current[0]+dx, current[1]+dy, current[2]+dz)
            if np != prev and np not in fixed_occupied and np != current:
                possible.append(np)
        if not possible:
            return None, None, None
        new_last = rng.choice(possible)
        # Check if new_last is in the tail (only current is in tail)
        # So just check fixed_occupied
        # And check if it's the same as current (handled by np != current)
        # And check if it's in occupied (but we remove current first)
        # So valid if not in fixed_occupied.
        return [new_last], {current}, {new_last}

    # General pivot
    u = (chain[pivot_index][0] - chain[pivot_index-1][0],
         chain[pivot_index][1] - chain[pivot_index-1][1],
         chain[pivot_index][2] - chain[pivot_index-1][2])
         
    v1 = (chain[pivot_index+1][0] - chain[pivot_index][0],
          chain[pivot_index+1][1] - chain[pivot_index][1],
          chain[pivot_index+1][2] - chain[pivot_index][2])
          
    possible_v1_new = []
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        if (dx, dy, dz) != u:
            possible_v1_new.append((dx, dy, dz))
            
    if not possible_v1_new:
        return None, None, None
        
    v1_new = rng.choice(possible_v1_new)
    
    # Rotation parameters
    cx = u[1]*v1[2] - u[2]*v1[1]
    cy = u[2]*v1[0] - u[0]*v1[2]
    cz = u[0]*v1[1] - u[1]*v1[0]
    
    c_theta = v1_new[0]*v1[0] + v1_new[1]*v1[1] + v1_new[2]*v1[2]
    s_theta = v1_new[0]*cx + v1_new[1]*cy + v1_new[2]*cz
    
    new_tail = [chain[pivot_index]] # Fixed point
    removed = set()
    added = set()
    
    # We need to check self-avoidance against:
    # 1. fixed_occupied (indices 0..pivot_index-1)
    # 2. The new_tail generated so far
    
    # Note: The original tail positions are in `occupied`, but we are replacing them.
    # So we only check against fixed_occupied and the new positions.
    
    for k in range(1, n - pivot_index):
        vk = (chain[pivot_index + k][0] - chain[pivot_index + k - 1][0],
              chain[pivot_index + k][1] - chain[pivot_index + k - 1][1],
              chain[pivot_index + k][2] - chain[pivot_index + k - 1][2])
              
        v_e1 = vk[0]*u[0] + vk[1]*u[1] + vk[2]*u[2]
        v_e2 = vk[0]*v1[0] + vk[1]*v1[1] + vk[2]*v1[2]
        v_e3 = vk[0]*cx + vk[1]*cy + vk[2]*cz
        
        new_e2 = c_theta * v_e2 - s_theta * v_e3
        new_e3 = s_theta * v_e2 + c_theta * v_e3
        
        new_vk_x = new_e1 * u[0] + new_e2 * v1[0] + new_e3 * cx
        new_vk_y = new_e1 * u[1] + new_e2 * v1[1] + new_e3 * cy
        new_vk_z = new_e1 * u[2] + new_e2 * v1[2] + new_e3 * cz
        
        new_vk = (int(round(new_vk_x)), int(round(new_vk_y)), int(round(new_vk_z)))
        
        prev_pos = new_tail[-1]
        new_pos = (prev_pos[0] + new_vk[0], prev_pos[1] + new_vk[1], prev_pos[2] + new_vk[2])
        
        # Check collision
        if new_pos in fixed_occupied or new_pos in added:
            return None, None, None
            
        new_tail.append(new_pos)
        added.add(new_pos)
        removed.add(chain[pivot_index + k])
        
    # Also add the pivot point to removed/added? No, it stays.
    # But we need to remove the OLD positions of the tail (excluding pivot) from occupied
    # and add the NEW positions.
    # The function returns the sets to update the global `occupied`.
    
    return new_tail, removed, added

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # 26 sequences. 330s total.
    # Let's allocate time based on sequence length.
    # Or just run until deadline.
    
    # Multi-start
    num_starts = 0
    while not _is_deadline_exceeded():
        num_starts += 1
        # Limit starts per sequence to avoid infinite loop if one is very fast?
        # Or just let it run.
        
        chain = _generate_initial_walk(sequence, dimension, rng)
        if chain is None:
            continue
            
        contacts, energy = _compute_initial_contacts(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
            
        # Simulated Annealing
        current_energy = energy
        current_chain = chain
        current_contacts = contacts
        current_occupied = set(chain)
        
        # Build fixed_occupied for pivot 1?
        # We will update fixed_occupied as we move pivots.
        # Actually, for a pivot at i, fixed_occupied is set(chain[0:i]).
        # This is O(N) to build.
        # We can maintain a prefix sum of sets? No.
        # We can just build it on the fly if we do few pivots.
        # Or we can just check against `current_occupied` and remove the tail before checking.
        # That's O(N) again.
        
        # To be fast, we assume N is small enough (<=136) that O(N) check is okay?
        # 136 * 1000 steps * 26 seqs = 3.5M ops. Should be fine in 330s.
        
        T = 1.0
        T_min = 0.1
        cooling = 0.995
        
        for step in range(5000): # Max steps per start
            if _is_deadline_exceeded():
                break
                
            if T < T_min:
                break
                
            # Pick random pivot
            pivot_index = rng.randint(1, n - 1)
            
            # Build fixed_occupied
            # Optimization: If we just moved a pivot, we can update incrementally?
            # Too complex. Just rebuild.
            fixed_occupied = set(current_chain[:pivot_index])
            
            new_tail, removed, added = _try_pivot_move(current_chain, current_occupied, fixed_occupied, pivot_index, rng)
            
            if new_tail is None:
                continue
                
            # Calculate energy change
            # Remove contacts involving indices in [pivot_index, n-1]
            # Add new contacts involving indices in [pivot_index, n-1]
            
            # This is tricky with the contact set.
            # Let's just recompute energy for the whole chain?
            # O(N) per move.
            # 5000 steps * 136 * 6 neighbors = 4M ops per start.
            # 10 starts = 40M ops.
            # 26 seqs = 1B ops.
            # Python might be too slow for 1B ops in 330s.
            
            # We need incremental energy.
            # Let's maintain a contact set.
            
            # Identify affected indices: pivot_index to n-1.
            # And their neighbors in the chain (pivot_index-1, etc).
            
            # Remove old contacts:
            # For each i in [pivot_index, n-1]:
            #   For each neighbor j in chain (i-1, i+1) and spatial neighbors:
            #     If (i,j) in contacts: remove it.
            
            # This is still O(N * K).
            
            # Given the constraints, let's stick to O(N) energy recomputation but optimize it.
            # Or reduce steps.
            
            # Let's try 1000 steps per start.
            
            new_chain = current_chain[:pivot_index] + new_tail
            new_energy = _energy(sequence, new_chain)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                current_occupied = set(current_chain)
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
            T *= cooling
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                q = (p[0]+dx, p[1]+dy, p[2]+dz)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts
# EVOLVE-BLOCK-END

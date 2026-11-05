"""HP lattice protein folding using Pivot Monte Carlo with incremental energy updates."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.92  # Use 92% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _initial_grow(sequence, dimension, rng):
    """
    Fast greedy growth with look-ahead to maximize contacts.
    Returns a list of tuples representing the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Precompute H indices for faster checks
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        
        # Candidates are the 6 neighbors of the last residue
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Score: maximize immediate H-H contacts + free space (to avoid trapping)
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and is_h[j]:
                            gain += 1
            
            # Count free neighbors to prefer open spaces
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Heuristic: strongly prefer contact, weakly prefer free space, random tie-breaker
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _pivot_mc_fold(sequence, chain, dimension, rng, time_limit=5.0):
    """
    Improve chain using Pivot Monte Carlo with incremental energy updates.
    """
    n = len(sequence)
    if n <= 2:
        return chain
    
    # State variables
    pos = [tuple(p) for p in chain]
    pos_map = {p: i for i, p in enumerate(pos)}
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Current energy (number of H-H contacts)
    # Initialize by scanning
    current_contacts = 0
    for i in range(n):
        if is_h[i]:
            for q in _neighbours(pos[i]):
                j = pos_map.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    current_contacts += 1
    
    best_contacts = current_contacts
    best_pos = list(pos)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.05
    # Estimate iterations based on time, but we will loop until time limit
    # We need a rough estimate for alpha if we use geometric cooling over a fixed number of steps.
    # Instead, we use a time-based cooling or just a large fixed number of steps.
    # Let's assume ~10000 steps take ~1s. So in 5s we can do ~50000 steps.
    # But we don't know the speed. Let's just use a dynamic loop.
    
    # To make cooling work without knowing total steps, we can use a simple
    # "cooling every K steps" or just a very long geometric schedule.
    # Let's use a fixed number of "logical" steps and map them to time.
    # Or simpler: use a constant T that decays slowly.
    # Actually, standard SA needs a schedule. Let's assume 20000 steps.
    max_steps = 20000
    alpha = (T_end / T_start) ** (1.0 / max_steps)
    T = T_start
    
    start_time = time.time()
    step_count = 0
    
    while step_count < max_steps:
        if _is_deadline_exceeded() or (time.time() - start_time > time_limit):
            break
            
        # 1. Choose pivot point
        # Uniform random pivot in [1, n-1]
        pivot = rng.randint(1, n - 1)
        
        # 2. Choose rotation axis (0, 1, or 2)
        axis = rng.randint(0, dimension - 1)
        
        # 3. Choose rotation angle (pi/2, pi, 3pi/2)
        # Represented as 90, 180, 270 degrees
        angle = rng.choice([90, 180, 270])
        
        # 4. Determine which part of the chain moves
        # The segment from pivot to n-1 moves.
        # We need to check if the new positions are valid.
        
        # Optimization: Only try if the pivot has at least 1 free neighbor in the rotation plane?
        # No, just try and check collisions.
        
        # Get the moving segment
        moving_len = n - pivot
        if moving_len == 1:
            # Single residue pivot: just rotate that one residue
            # New pos = rotate(pos[pivot] - pos[pivot-1], axis, angle) + pos[pivot-1]
            # Actually, standard pivot: rotate the vector from pivot-1 to pivot, and everything after.
            # Let's stick to the standard definition:
            # The chain is cut at pivot. The segment [pivot..n-1] is rotated around the bond (pivot-1, pivot).
            # Wait, standard pivot MC:
            # Pick pivot i. The segment i..n-1 is rotated around the bond (i-1, i).
            # So the residue at i-1 is fixed. The residue at i moves.
            
            # Let's implement the move:
            # 1. Identify the vector v = pos[pivot] - pos[pivot-1]
            # 2. Rotate v around 'axis' by 'angle' to get v'
            # 3. New pos[pivot] = pos[pivot-1] + v'
            # 4. New pos[k] for k > pivot = pos[k-1] + (pos[k] - pos[k-1]) rotated?
            # No, the entire segment is rigid.
            # The relative vectors between residues in the moving segment remain the same.
            # Only the orientation of the first bond (pivot-1, pivot) changes, and the rest follows.
            # Actually, it's easier to think:
            # The moving segment is a rigid body.
            # The pivot point is pos[pivot-1].
            # The vector from pivot-1 to pivot is v.
            # We rotate v to v'.
            # The rest of the chain is attached to pivot.
            # So we need to apply the same rotation to all vectors in the moving segment relative to the pivot?
            # No, the rotation is around the bond axis? No, around an axis passing through the pivot point?
            # Standard definition: Rotation around the bond (pivot-1, pivot) is NOT standard.
            # Standard Pivot: Rotation around an axis defined by the bond (pivot-1, pivot) is not possible in 3D lattice for arbitrary angles.
            # Correct Standard Pivot:
            # Pick pivot i.
            # Pick an axis (x, y, or z) passing through pos[i].
            # Rotate the segment i..n-1 around this axis by 90, 180, or 270 degrees.
            # This keeps pos[i] fixed? No, usually the pivot point is the one that stays fixed, or the bond before it?
            # Let's check literature: "Pivot Monte Carlo... a pivot point is chosen... the segment of the chain following the pivot is rotated... around an axis passing through the pivot point".
            # So pos[pivot] stays fixed?
            # If pos[pivot] is fixed, and we rotate the rest, the bond (pivot-1, pivot) might break if pivot-1 moves? No, pivot-1 is in the fixed part.
            # So the bond (pivot-1, pivot) changes direction.
            # This is valid as long as the new position of pivot+1 is not occupied.
            
            # Let's use this definition:
            # Fixed: pos[0] ... pos[pivot]
            # Moving: pos[pivot+1] ... pos[n-1]
            # Axis: passes through pos[pivot], parallel to one of the coordinate axes.
            # Angle: 90, 180, 270.
            
            pass # Logic below
            
        # Implementation of the move:
        # 1. Check if the move is trivial (e.g., rotating 180 around an axis that is the bond axis? No, axis is global x/y/z).
        
        # Determine new positions for the moving segment
        # We need to rotate the vectors relative to pos[pivot]
        # Let P = pos[pivot]
        # For k in pivot+1 to n-1:
        #   vec = pos[k] - P
        #   new_vec = rotate(vec, axis, angle)
        #   new_pos[k] = P + new_vec
        
        # We must check:
        # 1. new_pos[k] is not in pos_map (except for the fixed part? No, the fixed part is 0..pivot. The moving part is pivot+1..n-1.
        #    The new positions must not collide with ANY part of the chain, including the fixed part and the old positions of the moving part that are being vacated).
        #    Actually, we are replacing pos[pivot+1..n-1].
        #    So new_pos[k] must not be in {pos[0]...pos[pivot]} U {new_pos[j] for j != k}.
        
        # Let's collect the new positions first.
        new_pos = [None] * n
        # Fixed part
        for k in range(pivot + 1):
            new_pos[k] = pos[k]
        
        P = pos[pivot]
        
        # Helper to rotate a vector
        def rotate_vec(v, axis, angle):
            # v is a tuple (x, y, z)
            # axis is 0, 1, or 2
            # angle is 90, 180, 270
            # We rotate in the plane perpendicular to the axis.
            # If axis=0 (x), rotate y,z.
            # If axis=1 (y), rotate x,z.
            # If axis=2 (z), rotate x,y.
            
            # Copy v
            rv = list(v)
            
            if angle == 180:
                # Negate the two components perpendicular to axis
                for a in range(dimension):
                    if a != axis:
                        rv[a] = -rv[a]
                return tuple(rv)
            
            if angle == 90:
                # (x, y, z) -> depends on axis
                # Standard right-hand rule?
                # Let's define:
                # Axis 0 (x): y' = z, z' = -y
                # Axis 1 (y): x' = z, z' = -x
                # Axis 2 (z): x' = y, y' = -x
                # This is a consistent 90 degree rotation.
                
                if axis == 0:
                    rv[1], rv[2] = rv[2], -rv[1]
                elif axis == 1:
                    rv[0], rv[2] = rv[2], -rv[0]
                elif axis == 2:
                    rv[0], rv[1] = rv[1], -rv[0]
                return tuple(rv)
            
            if angle == 270:
                # Inverse of 90
                if axis == 0:
                    rv[1], rv[2] = -rv[2], rv[1]
                elif axis == 1:
                    rv[0], rv[2] = -rv[2], rv[0]
                elif axis == 2:
                    rv[0], rv[1] = -rv[1], rv[0]
                return tuple(rv)
            
            return tuple(rv)
        
        valid = True
        # Set of occupied positions in the FIXED part (0..pivot)
        fixed_occupied = set(new_pos[0:pivot+1])
        
        # Calculate new positions for moving part
        moving_new = {}
        for k in range(pivot + 1, n):
            vec = tuple(pos[k][a] - P[a] for a in range(dimension))
            new_vec = rotate_vec(vec, axis, angle)
            new_p = tuple(P[a] + new_vec[a] for a in range(dimension))
            
            # Check collision with fixed part
            if new_p in fixed_occupied:
                valid = False
                break
            
            # Check collision with other moving parts (we'll do this in a set)
            if new_p in moving_new:
                valid = False
                break
            
            moving_new[k] = new_p
            new_pos[k] = new_p
        
        if not valid:
            step_count += 1
            T *= alpha
            continue
        
        # Check collision with the OLD positions of the moving part that are NOT being kept?
        # The old positions of the moving part are being vacated.
        # So we only care about collisions with the FIXED part and within the NEW moving part.
        # We already checked both.
        
        # Calculate Delta Energy
        # Energy = - (Number of H-H contacts)
        # We need to count H-H contacts in the NEW configuration vs OLD.
        # Only contacts involving the moving segment (pivot+1..n-1) can change.
        # Contacts between fixed-fixed are unchanged.
        # Contacts between fixed-moving can change.
        # Contacts between moving-moving can change.
        
        # Efficient Delta Calculation:
        # 1. Remove all contacts involving the old moving segment (pivot+1..n-1).
        # 2. Add all contacts involving the new moving segment (pivot+1..n-1).
        
        # Helper to count contacts for a specific set of indices
        # We need to be careful not to double count.
        
        # Let's count contacts for the moving segment in the OLD and NEW configurations.
        # A contact is (i, j) with i < j, |i-j| > 1, and is_h[i]==1, is_h[j]==1, and pos[i] adjacent to pos[j].
        
        # We will sum up contacts where at least one index is in the moving segment.
        
        def count_moving_contacts(positions, pos_map, moving_indices):
            # moving_indices is a set of indices
            # We count contacts (i, j) such that i in moving OR j in moving.
            # To avoid double counting, we can just iterate through moving indices and check neighbors.
            # For each i in moving:
            #   for each neighbor q of pos[i]:
            #     j = pos_map.get(q)
            #     if j is not None and j != i and abs(i-j) > 1 and is_h[i] and is_h[j]:
            #        # Is this contact counted?
            #        # If both i and j are in moving, it might be counted twice (once for i, once for j).
            #        # If one is in moving and one is fixed, it is counted once (when iterating the moving one).
            #        # So we need to handle this.
            #        # Simpler: Count all contacts involving moving indices.
            #        # If both are moving, count 0.5? No, integers.
            #        # Let's just count pairs (i, j) with i < j.
            #        # Iterate i in moving.
            #        # For each neighbor j of i:
            #        #   if j > i and abs(i-j)>1 and is_h[i] and is_h[j]:
            #        #      count += 1
            #        # This counts each contact exactly once if we enforce i < j.
            
            count = 0
            for i in moving_indices:
                if not is_h[i]:
                    continue
                for q in _neighbours(positions[i]):
                    j = pos_map.get(q)
                    if j is not None:
                        if j > i and abs(i - j) > 1 and is_h[j]:
                            count += 1
            return count
        
        # Old contacts involving moving segment
        # We need the OLD pos_map.
        old_pos_map = pos_map
        moving_indices = set(range(pivot + 1, n))
        
        old_moving_contacts = count_moving_contacts(pos, old_pos_map, moving_indices)
        
        # New contacts involving moving segment
        # We need the NEW pos_map.
        # The fixed part is the same. The moving part is new.
        # new_pos_map = {p: i for i, p in enumerate(new_pos)}
        # But we can construct it partially or just use a dict for the moving part + fixed part.
        # Actually, we can just update the global pos_map if we accept, but for delta we need to check validity first.
        # We already checked validity (no collisions).
        # So we can compute the new contacts.
        
        # Construct new_pos_map for efficiency
        new_pos_map = {}
        for i in range(pivot + 1):
            new_pos_map[pos[i]] = i
        for k in range(pivot + 1, n):
            new_pos_map[new_pos[k]] = k
            
        new_moving_contacts = count_moving_contacts(new_pos, new_pos_map, moving_indices)
        
        delta_contacts = new_moving_contacts - old_moving_contacts
        # Energy delta = - delta_contacts
        delta_energy = -delta_contacts
        
        # Acceptance
        if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
            # Accept
            # Update state
            for k in range(pivot + 1, n):
                pos[k] = new_pos[k]
            pos_map = new_pos_map
            current_contacts += delta_contacts
            
            if current_contacts > best_contacts:
                best_contacts = current_contacts
                best_pos = list(pos)
        
        step_count += 1
        T *= alpha
        
    return best_pos

def fold(sequence, dimension):
    """Best conformation using multi-start greedy growth and Pivot Monte Carlo."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0 # Energy is negative contacts, so 0 is best (no contacts? No, min energy is most negative)
    # Wait, energy = -contacts. So we want to MINIMIZE energy (make it more negative).
    # So best_energy should start at 0 (or a large positive number) and we look for smaller values.
    # Let's use best_contacts to be clearer.
    best_contacts = -1
    
    # Time budget per sequence
    # Total 330s. 26 sequences. ~12s per sequence.
    # We will allocate time dynamically.
    
    # Multi-start
    # Let's try 10 restarts, each with a time limit.
    # Or just one long run?
    # Pivot MC is good at escaping local minima, so one long run might be better than many short ones.
    # But greedy growth is stochastic.
    # Let's do 5 restarts, each with ~2s limit.
    
    n_restarts = 5
    time_per_restart = 2.0
    
    for restart in range(n_restarts):
        if _is_deadline_exceeded():
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Improve with Pivot MC
        improved_chain = _pivot_mc_fold(sequence, chain, dimension, rng, time_limit=time_per_restart)
        
        # Calculate final contacts
        pos_map = {p: i for i, p in enumerate(improved_chain)}
        is_h = [1 if s == 'H' else 0 for s in sequence]
        contacts = 0
        for i in range(len(sequence)):
            if is_h[i]:
                for q in _neighbours(improved_chain[i]):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        contacts += 1
        
        if contacts > best_contacts:
            best_contacts = contacts
            best = improved_chain
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

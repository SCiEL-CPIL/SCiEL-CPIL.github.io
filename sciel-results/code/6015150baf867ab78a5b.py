import random
import time
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _rotate_vector(v, axis):
    """
    Rotate vector v by 90 degrees around axis.
    In 3D, if axis is a unit vector along one of the principal axes, 
    we can use simple permutation/sign changes.
    Since the chain is on a cubic lattice, the bond vector 'axis' 
    is always one of (1,0,0), (-1,0,0), (0,1,0), etc.
    """
    # Determine which axis it is
    ax = abs(axis[0])
    ay = abs(axis[1])
    az = abs(axis[2])
    
    # The rotation must be a 90 degree rotation.
    # There are 3 possible 90-degree rotations around a given axis.
    # We will return the 3 possible rotated vectors.
    # Note: v is the vector from k-1 to k. We rotate the segment k..N-1.
    # The vector v itself stays fixed relative to k-1, but the rest of the chain rotates.
    # Actually, it's easier to think of rotating the coordinates of the segment relative to the pivot point.
    
    # Let's define the rotation matrices for 90 deg around X, Y, Z.
    # R_x: (x, y, z) -> (x, -z, y) or (x, z, -y)
    # R_y: (x, y, z) -> (z, y, -x) or (-z, y, x)
    # R_z: (x, y, z) -> (y, -x, z) or (-y, x, z)
    
    rotations = []
    if ax == 1:
        # Axis is X. Rotate Y-Z plane.
        # Original direction of bond is +/- X.
        # The segment extends from k.
        # We rotate the vectors of the segment relative to k.
        # Let's just return the 3 possible rotated versions of the segment vectors.
        pass # Handled in _try_pivot

def _try_pivot(chain, k, occupied, sequence):
    """
    Try a pivot move at index k.
    Rotates chain[k:] around the bond (chain[k-1], chain[k]).
    Returns list of (new_chain_segment, delta_energy, new_occupied_set_changes) or None if no valid moves.
    """
    N = len(chain)
    if k < 1 or k > N - 2:
        return []
    
    pivot = chain[k]
    prev = chain[k-1]
    
    # Vector from pivot to prev is fixed.
    # We need to rotate the segment chain[k], chain[k+1], ... chain[N-1]
    # around the axis defined by (prev, pivot).
    
    # The axis is the vector prev -> pivot.
    axis = tuple(pivot[i] - prev[i] for i in range(3))
    
    # There are 3 distinct 90-degree rotations.
    # We will generate the new positions for the segment.
    # To do this efficiently, we can rotate the relative coordinates.
    # Let relative_coord[i] = chain[k+i] - pivot.
    # We rotate these relative coords.
    
    rel_coords = []
    for i in range(k, N):
        p = chain[i]
        rel = tuple(p[j] - pivot[j] for j in range(3))
        rel_coords.append(rel)
        
    # Generate 3 rotations
    # Axis is one of the 6 unit vectors.
    # We can determine the rotation based on the axis.
    
    # Let's define rotation functions for the 3 principal axes.
    # If axis is along X: (dx, dy, dz) -> (dx, -dz, dy) and (dx, dz, -dy)
    # If axis is along Y: (dx, dy, dz) -> (dz, dy, -dx) and (-dz, dy, dx)
    # If axis is along Z: (dx, dy, dz) -> (dy, -dx, dz) and (-dy, dx, dz)
    
    # Note: The direction of the axis (+ or -) doesn't change the set of rotated points, 
    # just the direction of rotation, but since we try both +/- 90, we get the same set of 3 orientations (original + 2 rotated).
    # Actually, rotating by +90 and -90 gives 2 new orientations. The original is the 3rd.
    
    rotated_sets = []
    
    # Determine which axis
    ax, ay, az = abs(axis[0]), abs(axis[1]), abs(axis[2])
    
    if ax == 1:
        # Rotate around X
        # Option 1: +90
        new_rel1 = [(r[0], -r[2], r[1]) for r in rel_coords]
        # Option 2: -90
        new_rel2 = [(r[0], r[2], -r[1]) for r in rel_coords]
        rotated_sets.append(new_rel1)
        rotated_sets.append(new_rel2)
    elif ay == 1:
        # Rotate around Y
        new_rel1 = [(r[2], r[1], -r[0]) for r in rel_coords]
        new_rel2 = [(-r[2], r[1], r[0]) for r in rel_coords]
        rotated_sets.append(new_rel1)
        rotated_sets.append(new_rel2)
    elif az == 1:
        # Rotate around Z
        new_rel1 = [(r[1], -r[0], r[2]) for r in rel_coords]
        new_rel2 = [(-r[1], r[0], r[2]) for r in rel_coords]
        rotated_sets.append(new_rel1)
        rotated_sets.append(new_rel2)
        
    valid_moves = []
    
    # The set of occupied points in the fixed part (0 to k-1)
    fixed_occupied = set(chain[:k])
    
    for new_rel in rotated_sets:
        # Construct new absolute coordinates
        new_segment = []
        valid = True
        # Check for self-intersection within the new segment
        seen_in_segment = set()
        
        for i, rel in enumerate(new_rel):
            new_p = (pivot[0] + rel[0], pivot[1] + rel[1], pivot[2] + rel[2])
            
            # Check collision with fixed part
            if new_p in fixed_occupied:
                valid = False
                break
            
            # Check collision with other parts of the new segment
            if new_p in seen_in_segment:
                valid = False
                break
            
            seen_in_segment.add(new_p)
            new_segment.append(new_p)
            
        if not valid:
            continue
            
        # If valid, this is a candidate move.
        # We need to calculate delta energy.
        # We only need to check contacts involving indices k-1 to N-1.
        # Specifically, contacts between:
        # 1. Fixed part (0..k-1) and New part (k..N-1)
        # 2. Within New part (k..N-1)
        # 3. The bond (k-1, k) is preserved, so no change there.
        
        # Let's calculate the energy change.
        # Old energy contributions involving indices >= k-1:
        # We need to subtract old contacts and add new contacts.
        # This is tricky to do incrementally without storing the contact map.
        # For simplicity and correctness, if the chain is short (< 50), full re-calc is fast.
        # For longer chains, we should optimize.
        # Given N <= 136, full re-calc of energy is O(N * 6) ~ 800 ops.
        # Pivot check is O(N).
        # So full re-calc is acceptable if we do it only for valid moves.
        
        # Let's store the candidate for later evaluation.
        valid_moves.append((k, new_segment, new_rel))
        
    return valid_moves

def _calculate_energy(chain, sequence):
    """Calculate total energy (negative number of H-H contacts)."""
    if not chain:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _initial_chain(sequence, dimension=3):
    """Generate a random compact self-avoiding walk."""
    N = len(sequence)
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    # Simple random growth with backtracking if stuck (limited)
    # To ensure compactness, we bias towards directions that keep the chain close to the origin.
    
    for i in range(1, N):
        last = chain[-1]
        options = []
        for n in _neighbours(last):
            if n in occupied:
                continue
            # Bias: prefer directions that reduce distance to origin or stay close
            dist = sum(x*x for x in n)
            options.append((dist, n))
            
        if not options:
            # Stuck. This is bad. 
            # For robustness, we could return None and retry initialization, 
            # but a straight line is always valid.
            # However, a straight line has 0 contacts.
            # Let's try a spiral or just a line if we get stuck often.
            # Given the high success rate of random walks in 3D for N<100, 
            # we will just pick a random valid neighbor if available, 
            # or if not, we have to backtrack. 
            # To keep it simple, we'll just generate a random walk and if it fails, 
            # we'll fall back to a straight line in the main function.
            return None
            
        # Sort by distance to origin to encourage compactness
        options.sort(key=lambda x: x[0])
        # Pick from the top few to maintain some randomness
        # Pick randomly from the best 50%
        half = max(1, len(options) // 2)
        chosen = random.choice(options[:half])
        next_p = chosen[1]
        chain.append(next_p)
        occupied[next_p] = i
        
    return chain

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 300  # 300s limit to be safe
    
    best_chain = None
    best_energy = float('inf')
    
    N = len(sequence)
    
    # 1. Generate Initial Chain
    # Try a few times to get a good compact start
    initial_chain = None
    for _ in range(10):
        c = _initial_chain(sequence, dimension)
        if c:
            initial_chain = c
            break
            
    if initial_chain is None:
        # Fallback: straight line
        initial_chain = [(i, 0, 0) for i in range(N)]
        
    best_chain = initial_chain
    best_energy = _calculate_energy(initial_chain, sequence)
    
    # 2. Simulated Annealing
    # We will run SA on the best chain found so far.
    # We might do multiple restarts if time permits.
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999 # Adjust based on steps
    
    # Estimate steps per second to determine total steps?
    # Better to just run until deadline.
    
    current_chain = initial_chain
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Pre-compute H positions for faster energy delta? 
    # For now, full re-calc is simple.
    
    step = 0
    T = T_start
    
    while time.time() < deadline:
        # Pick a move type
        # Pivot moves are good for global rearrangement
        # Crankshaft moves are good for local packing
        
        move_type = random.random()
        if move_type < 0.5:
            # Pivot
            k = random.randint(1, N - 2)
            moves = _try_pivot(current_chain, k, occupied, sequence)
            if moves:
                k_move, new_segment, _ = random.choice(moves)
                # Construct new full chain
                new_chain = current_chain[:k_move] + new_segment
                new_energy = _calculate_energy(new_chain, sequence)
                delta_e = new_energy - current_energy
            else:
                continue
        else:
            # Crankshaft (simplified: rotate 2 residues)
            # Pick k such that k-1, k, k+1, k+2 exist
            if N < 4:
                continue
            k = random.randint(1, N - 3)
            # Axis is (k-1, k)
            pivot = current_chain[k]
            prev = current_chain[k-1]
            axis = tuple(pivot[i] - prev[i] for i in range(3))
            
            # Rotate segment k, k+1
            # Rel coords
            rel0 = tuple(current_chain[k][i] - pivot[i] for i in range(3))
            rel1 = tuple(current_chain[k+1][i] - pivot[i] for i in range(3))
            
            # Generate rotations
            ax, ay, az = abs(axis[0]), abs(axis[1]), abs(axis[2])
            rotated_sets = []
            if ax == 1:
                rotated_sets.append(((rel0[0], -rel0[2], rel0[1]), (rel1[0], -rel1[2], rel1[1])))
                rotated_sets.append(((rel0[0], rel0[2], -rel0[1]), (rel1[0], rel1[2], -rel1[1])))
            elif ay == 1:
                rotated_sets.append(((rel0[2], rel0[1], -rel0[0]), (rel1[2], rel1[1], -rel1[0])))
                rotated_sets.append(((-rel0[2], rel0[1], rel0[0]), (-rel1[2], rel1[1], rel1[0])))
            elif az == 1:
                rotated_sets.append(((rel0[1], -rel0[0], rel0[2]), (rel1[1], -rel1[0], rel1[2])))
                rotated_sets.append(((-rel0[1], rel0[0], rel0[2]), (-rel1[1], rel1[0], rel1[2])))
                
            if not rotated_sets:
                continue
                
            new_rel0, new_rel1 = random.choice(rotated_sets)
            new_p0 = (pivot[0] + new_rel0[0], pivot[1] + new_rel0[1], pivot[2] + new_rel0[2])
            new_p1 = (pivot[0] + new_rel1[0], pivot[1] + new_rel1[1], pivot[2] + new_rel1[2])
            
            # Check validity
            # Fixed: 0..k-1 and k+2..N-1
            fixed_occupied = set(current_chain[:k]) | set(current_chain[k+2:])
            
            if new_p0 in fixed_occupied or new_p1 in fixed_occupied or new_p0 == new_p1:
                continue
                
            # Also check if new_p0 is adjacent to prev? 
            # The bond (k-1, k) is the pivot axis. 
            # new_p0 is the new position of k. 
            # It must be adjacent to prev. 
            # Since we rotated around the axis, the distance is preserved.
            # But we must ensure it doesn't collide with k-1 (it can't, distance is 1).
            
            new_chain = current_chain[:k] + [new_p0, new_p1] + current_chain[k+2:]
            new_energy = _calculate_energy(new_chain, sequence)
            delta_e = new_energy - current_energy
            
        # Acceptance
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            # Accept
            current_chain = new_chain
            current_energy = new_energy
            occupied = {p: i for i, p in enumerate(current_chain)}
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        step += 1
        
        # Optional: Restart if stuck?
        # For now, just keep going.
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

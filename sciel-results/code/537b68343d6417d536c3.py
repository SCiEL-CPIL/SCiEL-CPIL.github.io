import random
import time
import math

# Precompute 90-degree rotation matrices for 3D
# Axis 0 (X): rotate Y,Z. Axis 1 (Y): rotate X,Z. Axis 2 (Z): rotate X,Y.
# We need rotations around the bond axis.
# The bond is from P_prev to P_curr. The vector is V = P_curr - P_prev.
# We rotate the rest of the chain around this axis.
# Since the lattice is cubic, V is one of (1,0,0), (0,1,0), (0,0,1) or their negatives.
# We can precompute the rotation for each of the 6 possible bond directions.

def _get_rotation_matrix(bond_dir):
    """
    Returns a 3x3 rotation matrix for a 90 degree rotation around the axis defined by bond_dir.
    bond_dir is a tuple (dx, dy, dz) where exactly one component is +/-1.
    We rotate such that the chain continues in a "right-hand rule" sense or similar, 
    but since we try multiple rotations, we can just pick one valid 90 deg rotation.
    Actually, there are 3 possible 90 degree rotations (clockwise, counter-clockwise, and 180? No, 90 deg steps).
    For a given axis, there are 3 distinct 90-degree rotations (90, 180, 270).
    Let's generate all 3 for a given axis.
    """
    # Identify axis
    if bond_dir[0] != 0:
        axis = 0
    elif bond_dir[1] != 0:
        axis = 1
    else:
        axis = 2
        
    # Standard rotation matrices for 90 deg around X, Y, Z
    # R_x(90): [[1,0,0],[0,0,-1],[0,1,0]]
    # R_y(90): [[0,0,1],[0,1,0],[-1,0,0]]
    # R_z(90): [[0,-1,0],[1,0,0],[0,0,1]]
    
    mats = []
    if axis == 0:
        # Rotations around X axis
        # 90 deg
        mats.append(((1,0,0), (0,0,-1), (0,1,0)))
        # 180 deg
        mats.append(((1,0,0), (0,-1,0), (0,0,-1)))
        # 270 deg (-90)
        mats.append(((1,0,0), (0,0,1), (0,-1,0)))
    elif axis == 1:
        # Rotations around Y axis
        mats.append(((0,0,1), (0,1,0), (-1,0,0)))
        mats.append((-1,0,0), (0,1,0), (0,0,1)) # Wait, 180 around Y: x->-x, z->-z. [[-1,0,0],[0,1,0],[0,0,-1]]
        mats.append(((0,0,-1), (0,1,0), (1,0,0)))
    else:
        # Rotations around Z axis
        mats.append(((0,-1,0), (1,0,0), (0,0,1)))
        mats.append((-1,0,0), (0,-1,0), (0,0,1)) # 180 around Z: x->-x, y->-y. [[-1,0,0],[0,-1,0],[0,0,1]]
        mats.append(((0,1,0), (-1,0,0), (0,0,1)))

    return mats

# To be safe and simple, let's just use the 6 basic 90-degree rotations around the 3 axes.
# If the bond is along X, we rotate around X.
# If the bond is along Y, we rotate around Y.
# If the bond is along Z, we rotate around Z.
# We will try 3 rotations (90, 180, 270) for the specific axis.

def _apply_rotation(point, pivot, mat):
    """
    Rotate point around pivot by matrix mat.
    mat is a tuple of 3 tuples (rows).
    """
    x = point[0] - pivot[0]
    y = point[1] - pivot[1]
    z = point[2] - pivot[2]
    
    nx = mat[0][0]*x + mat[0][1]*y + mat[0][2]*z
    ny = mat[1][0]*x + mat[1][1]*y + mat[1][2]*z
    nz = mat[2][0]*x + mat[2][1]*y + mat[2][2]*z
    
    return (pivot[0] + nx, pivot[1] + ny, pivot[2] + nz)

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

def _neighbours(point):
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _calc_energy(sequence, chain):
    """Calculate total energy of a chain."""
    n = len(chain)
    if n < 2:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    is_h = [1 if s == 'H' else 0 for s in sequence]
    energy = 0
    # Count H-H contacts
    # To avoid double counting, only count if j > i + 1
    for i in range(n):
        if is_h[i]:
            p = chain[i]
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    energy -= 1
    return energy

def _beam_search_growth(sequence, dimension, rng, beam_width=100):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    start = (0, 0, 0)
    # Each beam element: (chain, energy, occupied_dict)
    # We store occupied as a dict for O(1) lookup
    beams = [([(0,0,0)], 0, {(0,0,0): 0})]
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    for i in range(1, n):
        candidates = []
        for chain, energy, occupied in beams:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if is_h[i]:
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if is_h[j]:
                                new_energy -= 1
                
                # Heuristic: prefer lower energy (more negative) and compactness
                # Compactness proxy: distance from origin? Or just energy.
                # Let's add a small random noise to break ties
                score = new_energy + rng.random() * 0.5
                
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                candidates.append((score, new_chain, new_energy, new_occupied))
        
        if not candidates:
            break
            
        # Sort by score (lowest is best, i.e., most negative energy)
        candidates.sort(key=lambda x: x[0])
        beams = [(c[1], c[2], c[3]) for c in candidates[:beam_width]]
        
    if not beams:
        # Fallback: straight line (might be invalid if n > 1? No, straight line is always valid)
        return [(i, 0, 0) for i in range(n)]
        
    best = min(beams, key=lambda x: x[1])
    return best[0]

def _local_search_sa(sequence, chain, dimension, rng, time_limit):
    """
    Simulated Annealing with segment rotations.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Calculate initial energy
    current_energy = _calc_energy(sequence, chain)
    
    best_energy = current_energy
    best_chain = chain.copy()
    
    start_time = time.time()
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # We don't know the number of steps in advance, so we use time-based cooling
    # Or we can estimate steps. Let's do time-based.
    # Actually, simple cooling: T = T_start * exp(-k * t)
    
    while time.time() - start_time < time_limit:
        # Choose a move
        # 1. Suffix rotation
        # 2. Internal segment rotation
        
        move_type = rng.random()
        if move_type < 0.5:
            # Suffix rotation
            # Choose k in [1, n-1]
            k = rng.randint(1, n-1)
            # Pivot is chain[k-1], axis is bond k-1 to k
            pivot = chain[k-1]
            bond_dir = (chain[k][0] - pivot[0], chain[k][1] - pivot[1], chain[k][2] - pivot[2])
            
            # Determine axis
            if bond_dir[0] != 0: axis = 0
            elif bond_dir[1] != 0: axis = 1
            else: axis = 2
            
            # Choose one of 3 rotations
            rot_idx = rng.randint(0, 2)
            
            # Get rotation matrix
            # We need to handle the sign of the bond? 
            # Rotation around X axis is the same regardless of +X or -X direction?
            # Yes, the axis is the line. The direction matters for "clockwise" vs "counter".
            # But we are trying all 3 (90, 180, 270), so it covers all possibilities.
            
            if axis == 0:
                mats = [
                    ((1,0,0), (0,0,-1), (0,1,0)), # 90
                    ((1,0,0), (0,-1,0), (0,0,-1)), # 180
                    ((1,0,0), (0,0,1), (0,-1,0))  # 270
                ]
            elif axis == 1:
                mats = [
                    ((0,0,1), (0,1,0), (-1,0,0)), # 90
                    ((-1,0,0), (0,1,0), (0,0,-1)), # 180
                    ((0,0,-1), (0,1,0), (1,0,0))  # 270
                ]
            else:
                mats = [
                    ((0,-1,0), (1,0,0), (0,0,1)), # 90
                    ((-1,0,0), (0,-1,0), (0,0,1)), # 180
                    ((0,1,0), (-1,0,0), (0,0,1))  # 270
                ]
            
            mat = mats[rot_idx]
            
            # Extract segment
            segment = chain[k:]
            
            # Remove from occupied
            for p in segment:
                del occupied[p]
            
            # Apply rotation
            new_segment = []
            for p in segment:
                new_segment.append(_apply_rotation(p, pivot, mat))
            
            # Check validity
            valid = True
            for p in new_segment:
                if p in occupied:
                    valid = False
                    break
            
            if not valid:
                # Restore
                for i, p in enumerate(segment):
                    occupied[p] = k + i
                continue
            
            # Calculate Delta Energy
            # Only contacts between segment and fixed part change.
            # Also, contacts within segment remain the same (relative positions unchanged).
            # Contacts within fixed part remain the same.
            
            old_contacts = 0
            for i, p in enumerate(segment):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(p):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            if is_h[nb_idx]:
                                old_contacts += 1
                                
            new_contacts = 0
            for i, p in enumerate(new_segment):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(p):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            if is_h[nb_idx]:
                                new_contacts += 1
                                
            delta_e = -(new_contacts - old_contacts)
            
            # Acceptance
            accept = False
            if delta_e < 0:
                accept = True
            else:
                # Simple cooling: T decreases over time
                # t is elapsed time
                elapsed = time.time() - start_time
                # Map elapsed to T
                # Let's use a simple exponential decay
                # T = T_start * exp(-elapsed / tau)
                # Choose tau such that T drops significantly in time_limit
                tau = time_limit / 5.0 
                T = T_start * math.exp(-elapsed / tau)
                if T > 0:
                    prob = math.exp(-delta_e / T)
                    accept = rng.random() < prob
            
            if accept:
                # Update state
                for i, p in enumerate(new_segment):
                    occupied[p] = k + i
                    chain[k + i] = p
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                # Restore
                for i, p in enumerate(segment):
                    occupied[p] = k + i
        else:
            # Internal Segment Rotation
            # Choose start k and end j
            # k in [1, n-2], j in [k+1, n-1]
            # To keep it simple, let's rotate a segment of random length
            # Or just try suffix again? Suffix is a specific case of internal.
            # Let's try a shorter segment rotation.
            # Pick k in [1, n-3]
            # Pick length L in [1, n-k-1]
            # Rotate chain[k:k+L] around bond k-1, k
            
            if n < 4:
                continue
                
            k = rng.randint(1, n-3)
            max_len = n - k
            L = rng.randint(1, max_len)
            j = k + L # exclusive end index
            
            pivot = chain[k-1]
            bond_dir = (chain[k][0] - pivot[0], chain[k][1] - pivot[1], chain[k][2] - pivot[2])
            
            if bond_dir[0] != 0: axis = 0
            elif bond_dir[1] != 0: axis = 1
            else: axis = 2
            
            rot_idx = rng.randint(0, 2)
            
            if axis == 0:
                mats = [
                    ((1,0,0), (0,0,-1), (0,1,0)),
                    ((1,0,0), (0,-1,0), (0,0,-1)),
                    ((1,0,0), (0,0,1), (0,-1,0))
                ]
            elif axis == 1:
                mats = [
                    ((0,0,1), (0,1,0), (-1,0,0)),
                    ((-1,0,0), (0,1,0), (0,0,-1)),
                    ((0,0,-1), (0,1,0), (1,0,0))
                ]
            else:
                mats = [
                    ((0,-1,0), (1,0,0), (0,0,1)),
                    ((-1,0,0), (0,-1,0), (0,0,1)),
                    ((0,1,0), (-1,0,0), (0,0,1))
                ]
            
            mat = mats[rot_idx]
            
            segment = chain[k:j]
            
            for p in segment:
                del occupied[p]
            
            new_segment = []
            for p in segment:
                new_segment.append(_apply_rotation(p, pivot, mat))
            
            valid = True
            for p in new_segment:
                if p in occupied:
                    valid = False
                    break
            
            if not valid:
                for i, p in enumerate(segment):
                    occupied[p] = k + i
                continue
            
            old_contacts = 0
            for i, p in enumerate(segment):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(p):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            if is_h[nb_idx]:
                                old_contacts += 1
                                
            new_contacts = 0
            for i, p in enumerate(new_segment):
                idx = k + i
                if is_h[idx]:
                    for nb in _neighbours(p):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            if is_h[nb_idx]:
                                new_contacts += 1
                                
            delta_e = -(new_contacts - old_contacts)
            
            accept = False
            if delta_e < 0:
                accept = True
            else:
                elapsed = time.time() - start_time
                tau = time_limit / 5.0 
                T = T_start * math.exp(-elapsed / tau)
                if T > 0:
                    prob = math.exp(-delta_e / T)
                    accept = rng.random() < prob
            
            if accept:
                for i, p in enumerate(new_segment):
                    occupied[p] = k + i
                    chain[k + i] = p
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                for i, p in enumerate(segment):
                    occupied[p] = k + i

    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    # Global deadline management
    # The task says 240s CPU budget for ALL sequences.
    # We don't know the total number of sequences, but we assume we are called per sequence.
    # To be safe, we should track time globally if possible, but the interface is stateless.
    # However, the prompt says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # This implies each call is independent? "240 s CPU budget for all sequences together".
    # If it's a subprocess per sequence, we can't share state.
    # But usually "sandboxed subprocess" for the whole evaluation means one process running all sequences?
    # "The evaluator checks... fold is called once per benchmark sequence in a sandboxed subprocess with a 240 s CPU budget for all sequences together."
    # This phrasing is ambiguous. "Subprocess" singular? Or "subprocesses" plural?
    # If it's one subprocess for all, we can use a global timer.
    # If it's one per sequence, we must limit each call.
    # Given "240s for all sequences together", it's likely one process.
    # But `fold` is the entry point.
    # Let's assume we should limit each call to a reasonable amount, e.g., 240/26 ~ 9s per sequence.
    # To be robust, let's use a global deadline if we can detect it, or a fixed per-call limit.
    # Since we can't easily detect if it's the same process, let's use a fixed per-call limit of 8 seconds.
    # 26 * 8 = 208s < 240s. This is safe.
    
    PER_CALL_TIME_LIMIT = 8.0
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple restarts
    # Use remaining time for restarts
    # Strategy: 
    # 1. Initial beam search
    # 2. SA refinement
    # 3. If time left, repeat with different seed
    
    num_restarts = 5 # Max restarts if time permits
    
    for r in range(num_restarts):
        if time.time() > deadline - 0.5:
            break
            
        # Determine time for this restart
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed - 0.5
        if time_remaining <= 0:
            break
            
        # Allocate time for initialization and SA
        # Init is fast, SA takes the rest
        init_time = 0.5
        sa_time = time_remaining - init_time
        
        if sa_time < 0.1:
            break
            
        seed = 12345 + len(sequence) + r * 100
        rng_r = random.Random(seed)
        
        # 1. Beam Search
        init_start = time.time()
        initial_chain = _beam_search_growth(sequence, dimension, rng_r, beam_width=100)
        if not initial_chain or len(initial_chain) != n:
            initial_chain = [(i, 0, 0) for i in range(n)]
            
        init_elapsed = time.time() - init_start
        if init_elapsed > init_time:
            # Skip SA if init took too long? Or just proceed with less SA time
            pass
            
        # 2. SA
        sa_budget = sa_time - init_elapsed
        if sa_budget > 0.1:
            refined_chain = _local_search_sa(sequence, initial_chain, dimension, rng_r, sa_budget)
        else:
            refined_chain = initial_chain
            
        e = _calc_energy(sequence, refined_chain)
        if e < best_energy:
            best_energy = e
            best_chain = refined_chain.copy()
            
        # Check if we found a very good solution? 
        # If best_energy is close to theoretical min? Hard to know.
        # Just continue until time runs out.

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

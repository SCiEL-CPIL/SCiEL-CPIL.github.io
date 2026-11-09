import math
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    # O(N) energy calculation
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Greedy: prefer H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            # Small randomization to avoid determinism
            options.append((gain + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def fold(sequence, dimension):
    rng = random.Random(42)
    N = len(sequence)
    
    # Get initial chain
    chain = _initial_grow(sequence, dimension, rng)
    if chain is None:
        # Fallback straight line
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    
    start_time = time.time()
    # Allocate ~8 seconds per call to be safe for 26 calls in 300s
    deadline = start_time + 8.0
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Choose move type
        if rng.random() < 0.5:
            # Pivot
            if N < 3: break
            p = rng.randint(1, N-1)
            axis = rng.randint(0, dimension-1)
            direction = rng.choice([-1, 1])
            
            # Rotate chain[p:] around chain[p-1]
            pivot = chain[p-1]
            # Get the vector from pivot to chain[p]
            v = [chain[p][i] - pivot[i] for i in range(dimension)]
            
            # Rotate v by 90 deg in the plane defined by 'axis' and another axis
            # To simplify, just swap two axes and flip one sign?
            # Standard 90 deg rotation around 'axis' in 3D:
            # If axis=0 (x), rotate y-z plane.
            # New y = old z, New z = -old y (for +90)
            # Let's implement a general 90 deg rotation around 'axis'
            
            # Actually, simpler: Pick two axes a1, a2. Rotate in that plane.
            # Let's just pick a random axis to rotate AROUND, and a random perpendicular axis.
            # Or just use the simple "slide" and "pivot" from literature.
            
            # Let's try a simpler Pivot: 
            # Pick p. Pick a new direction for chain[p] that is different from old.
            # Then propagate? No, that's not a rigid body pivot.
            
            # Rigid Body Pivot:
            # The segment chain[p:] rotates around chain[p-1].
            # We need to define the rotation.
            # In 3D, there are 6 possible 90-degree rotations for the vector chain[p]-chain[p-1].
            # Old vector v.
            # New vector v' must be a unit vector, orthogonal to v? No, just a valid step.
            # But for rigid body, the *entire* tail rotates.
            # This requires transforming all points in chain[p:].
            
            # Transformation:
            # Let pivot be P.
            # Let old head be H.
            # Vector V = H - P.
            # We want a new vector V' such that V' is a unit step, and V' != V, and V' is not -V (usually).
            # Actually, any of the 5 other unit steps can be the new first step of the tail.
            # But the rotation must be rigid.
            # In 3D, rotating V to V' is not unique (rotation axis is V x V').
            # However, for HP lattice, we can just try all 5 valid next steps for the new head,
            # and check if the *entire* tail, when "shifted" to match the new head, is valid.
            # This is not a true rotation, it's a "translation" of the tail to fit the new head.
            # This is valid if we treat the tail as a rigid body that is translated.
            # Wait, if I translate the tail, the internal geometry is preserved.
            # So, pick new head H_new.
            # Delta = H_new - H.
            # New tail = [p + Delta for p in chain[p:]].
            # Check collisions for the new tail.
            
            H_old = chain[p]
            H_new = None
            valid_new_heads = []
            for site in _neighbours(pivot):
                if site == H_old:
                    continue
                if site in occupied:
                    # Check if site is part of the tail we are moving?
                    # If site is in chain[p:], it will be moved, so it's "free" for the new head?
                    # No, the new head is the first element of the moved tail.
                    # The rest of the tail is shifted.
                    # So we need to check if the *shifted* tail collides with the *static* part (chain[:p]).
                    
                    # Let's pre-calculate the shift
                    # We don't know H_new yet.
                    pass
            
            # This logic is getting complicated. Let's stick to Slides for simplicity and reliability.
            # Slides are easier to validate.
            
            # Slide Move
            # Pick i, L.
            # Shift chain[i:i+L] by delta.
            
            i = rng.randint(1, N-1)
            max_len = min(5, N - i)
            L = rng.randint(1, max_len)
            axis = rng.randint(0, dimension-1)
            direction = rng.choice([-1, 1])
            
            delta = [0]*dimension
            delta[axis] = direction
            
            # Check if the moved segment collides with the rest of the chain
            # The moved segment is chain[i : i+L]
            # The static part is chain[:i] + chain[i+L:]
            
            # We need to check each point in the moved segment against the static occupied set.
            # But we must exclude the points of the moved segment itself from the "occupied" check,
            # because they are moving.
            
            # Let's create a set of static points
            # static_points = set(chain[:i]) | set(chain[i+L:])
            # This is O(N). 
            # Better: 
            # For each point p in chain[i:i+L], new_p = p + delta.
            # Check if new_p is in occupied.
            # If new_p is in occupied, check if occupied[new_p] is in [i, i+L-1].
            # If yes, it's part of the moving segment (collision within segment? No, rigid move preserves internal geometry).
            # So if new_p is in occupied AND occupied[new_p] is NOT in [i, i+L-1], it's a collision.
            
            valid = True
            new_positions = []
            for idx in range(i, i+L):
                old_p = chain[idx]
                new_p = tuple(old_p[k] + delta[k] for k in range(dimension))
                new_positions.append(new_p)
                if new_p in occupied:
                    occ_idx = occupied[new_p]
                    if not (i <= occ_idx < i+L):
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Check if the first step of the segment (i to i+1) is still connected?
            # Yes, rigid move preserves internal connections.
            # Check if the connection to the previous part (i-1 to i) is still valid?
            # Old: chain[i-1] -> chain[i]. New: chain[i-1] -> chain[i] + delta.
            # Distance must be 1.
            # chain[i-1] and chain[i] are neighbors.
            # chain[i] + delta is a neighbor of chain[i].
            # Is chain[i-1] a neighbor of chain[i] + delta?
            # Not necessarily.
            # Example: chain[i-1]=(0,0,0), chain[i]=(1,0,0). delta=(0,1,0).
            # new chain[i]=(1,1,0). dist((0,0,0), (1,1,0)) = sqrt(2). Invalid.
            
            # So we must check the boundary connections.
            # Connection i-1 -> i
            prev = chain[i-1]
            new_head = new_positions[0]
            dist_sq = sum((prev[k] - new_head[k])**2 for k in range(dimension))
            if dist_sq != 1:
                continue
                
            # Connection i+L-1 -> i+L
            if i+L < N:
                next_part = chain[i+L]
                new_tail = new_positions[-1]
                dist_sq = sum((next_part[k] - new_tail[k])**2 for k in range(dimension))
                if dist_sq != 1:
                    continue
            
            # Move is valid geometrically.
            # Apply move
            for idx in range(i, i+L):
                old_p = chain[idx]
                new_p = new_positions[idx-i]
                del occupied[old_p]
                occupied[new_p] = idx
                chain[idx] = new_p
            
            new_energy = _energy(sequence, chain)
            
            # Accept/Reject
            if new_energy < current_energy or rng.random() < math.exp((current_energy - new_energy) / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Undo
                for idx in range(i, i+L):
                    new_p = chain[idx]
                    old_p = new_positions[idx-i]
                    del occupied[new_p]
                    occupied[old_p] = idx
                    chain[idx] = old_p
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return [list(p) for p in best_chain]

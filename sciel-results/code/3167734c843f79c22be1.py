# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_optimize(sequence, chain, dimension, rng, steps):
    """Improve chain using pivot moves that increase H-H contacts."""
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    n = len(chain)
    
    # Precompute contact map or just use energy diff? 
    # For simplicity and speed, we can just recalc energy if a move is accepted, 
    # or check delta. Recalcing is O(N). N is up to 136. 
    # Steps up to 1000? 1000 * 136 * 6 (neighbours) is ~800k ops. Very fast.
    
    for _ in range(steps):
        if n < 4:
            break
        pivot_idx = rng.randint(1, n - 2)
        # Axis of rotation
        axis = rng.randint(0, dimension - 1)
        # Angle: 90, 180, 270
        angle = rng.choice([1, -1])
        
        # The pivot is at chain[pivot_idx].
        # The tail is chain[pivot_idx+1:]
        # We rotate the tail around the vector connecting chain[pivot_idx-1] and chain[pivot_idx]?
        # Standard pivot move: rotate the sub-chain [k..N] around the bond (k-1, k).
        
        p_prev = chain[pivot_idx - 1]
        p_pivot = chain[pivot_idx]
        
        # Vector v = p_pivot - p_prev
        v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        
        # We need to rotate the tail around the axis defined by v?
        # No, pivot moves in lattice proteins usually rotate the tail around one of the 3 axes passing through the pivot atom, 
        # such that the bond (pivot-1, pivot) remains valid? 
        # Actually, the bond (pivot-1, pivot) is fixed. The rest of the chain rotates.
        # The rotation must be around an axis perpendicular to the bond (pivot-1, pivot) to keep the bond intact?
        # Or simply: The bond (pivot-1, pivot) is the hinge.
        # In 3D lattice, if the bond is along X, we can rotate the tail around X by 90, 180, 270.
        
        # Find the axis of the bond
        bond_axis = -1
        for i in range(dimension):
            if v[i] != 0:
                bond_axis = i
                break
        
        # The rotation axes are the two other axes
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        if not rot_axes:
            continue
            
        r_axis = rng.choice(rot_axes)
        
        # Rotate tail
        new_tail = []
        # The pivot point is the center of rotation for the tail?
        # The tail starts at pivot_idx + 1.
        # The point at pivot_idx is the hinge.
        # We rotate the vector from p_pivot to each tail point?
        # No, we rotate the entire tail segment around the axis passing through p_pivot and parallel to r_axis.
        
        # Let's construct the new tail
        # For each point in tail:
        # new_point = p_pivot + R * (point - p_pivot)
        
        # Rotation matrix for 90 deg around axis r_axis
        # If r_axis == 0 (X), rotate Y->Z, Z->-Y (for angle 1) or Y->-Z, Z->Y (for angle -1)
        
        def rotate_point(point, center, axis, angle):
            p = list(point)
            c = list(center)
            # Translate to origin
            p[0] -= c[0]
            p[1] -= c[1]
            p[2] -= c[2]
            
            if axis == 0: # Rotate around X
                if angle == 1:
                    p[1], p[2] = -p[2], p[1]
                else:
                    p[1], p[2] = p[2], -p[1]
            elif axis == 1: # Rotate around Y
                if angle == 1:
                    p[0], p[2] = p[2], -p[0]
                else:
                    p[0], p[2] = -p[2], p[0]
            elif axis == 2: # Rotate around Z
                if angle == 1:
                    p[0], p[1] = -p[1], p[0]
                else:
                    p[0], p[1] = p[1], -p[0]
            
            # Translate back
            p[0] += c[0]
            p[1] += c[1]
            p[2] += c[2]
            return tuple(p)
        
        # Check if rotation is valid
        # 1. First point of new tail must be adjacent to p_pivot (it will be, by rotation of a lattice vector)
        # 2. No collisions with head (0..pivot_idx-1) or new tail itself.
        
        head_set = set(chain[:pivot_idx]) # Exclude pivot, as it's fixed
        # Wait, the head is chain[0..pivot_idx-1]. The pivot is chain[pivot_idx].
        # The new tail points must not be in head_set.
        # Also, the new tail must be self-avoiding.
        
        valid = True
        new_tail_points = []
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig_point = chain[i]
            new_point = rotate_point(orig_point, p_pivot, r_axis, angle)
            
            # Check collision with head
            if new_point in head_set:
                valid = False
                break
            # Check collision with already placed new tail
            if new_point in occupied_tail:
                valid = False
                break
            
            # Check if step from previous new point is unit
            if new_tail_points:
                prev_new = new_tail_points[-1]
                dist = sum(abs(new_point[k] - prev_new[k]) for k in range(dimension))
                if dist != 1:
                    valid = False
                    break
            
            occupied_tail.add(new_point)
            new_tail_points.append(new_point)
        
        if not valid:
            continue
            
        # Check collision with the pivot itself?
        # The first new point is adjacent to pivot. Pivot is not in head_set.
        # But is the first new point the same as some head point? Checked above.
        
        # Construct new chain
        candidate_chain = chain[:pivot_idx + 1] + new_tail_points
        
        # Calculate energy
        new_energy = _energy(sequence, candidate_chain)
        
        if new_energy < best_energy:
            best_chain = candidate_chain
            best_energy = new_energy
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths and local optimizations."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    start_time = time.time()
    deadline = start_time + 300 # Safety margin
    
    n_restarts = 50
    # If chain is long, reduce restarts to allow more optimization time?
    # Or just keep it simple.
    
    for _ in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local optimization
        # Scale steps with chain length
        opt_steps = 100 + len(sequence)
        optimized_chain, opt_energy = _pivot_optimize(sequence, chain, dimension, rng, opt_steps)
        
        if optimized_chain is not None:
            if best is None or opt_energy < best_energy:
                best = optimized_chain
                best_energy = opt_energy

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

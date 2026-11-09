"""HP lattice protein folding on the cubic lattice.
Combines randomized greedy growth with local tail re-growth refinement
and a deadline guard to stay within the wall-clock budget.
"""
import random
import time

# Deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = time.time()
_DEADLINE = time.time() + 315.0


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute the total energy of the chain. Energy = -num_HH_contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _pivot_move(sequence, chain, i, rng):
    """
    Try a pivot move at position i: rotate the tail starting at i around
    the bond from i-1 to i. Returns (new_chain, new_energy) or (None, None).
    """
    n = len(chain)
    if i < 2 or i >= n - 1:
        return None, None
    
    prefix = chain[:i]
    tail = chain[i:]
    
    # The pivot is at chain[i-1], and we rotate around the axis from chain[i-1] to chain[i]
    pivot = chain[i-1]
    axis_start = chain[i]
    
    # Try all 3 possible rotations (90, 180, 270 degrees) around the axis
    # Axis vector
    ax = axis_start[0] - pivot[0]
    ay = axis_start[1] - pivot[1]
    az = axis_start[2] - pivot[2]
    
    # For a unit axis vector, the 90-degree rotation matrix can be derived
    # We'll try all 3 non-trivial rotations
    rotations = []
    
    # Rotation by 90 degrees around axis (ax, ay, az)
    # Using the rotation matrix formula
    a = ax; b = ay; c = az
    # cos(90)=0, sin(90)=1
    # R = [[0, -c, b, a], [c, 0, -a, b], [-b, a, 0, c]] + identity * cos
    # For 90 deg: R = [[0, -c, b], [c, 0, -a], [-b, a, 0]] + 0*I
    # Actually let's use the standard formula:
    # R = I*cos + (1-cos)*[u]x + [u]x^2
    # For 90 deg: cos=0, sin=1
    # R = [u]x + [u]x^2
    
    # Let's just try the 3 possible new positions for the first tail segment
    # The first tail segment goes from chain[i-1] to chain[i]
    # After rotation, chain[i] should move to a new position that is:
    # 1. Adjacent to chain[i-1]
    # 2. Not equal to chain[i] (no-op)
    # 3. The rotation should be a 90, 180, or 270 degree turn
    
    # Find all neighbours of pivot that are not in the prefix and not chain[i]
    occupied_prefix = {p for p in prefix}
    
    candidate_ends = []
    for site in _neighbours(pivot):
        if site == chain[i]:
            continue
        if site in occupied_prefix:
            continue
        # Check that site is not collinear with the pivot-axis (for 180 deg it's opposite)
        # For 90/270 deg: dot product of (site-pivot) and (axis_start-pivot) should be 0
        # For 180 deg: site should be pivot - (axis_start - pivot) = 2*pivot - axis_start
        sx = site[0] - pivot[0]
        sy = site[1] - pivot[1]
        sz = site[2] - pivot[2]
        dot = sx*ax + sy*ay + sz*az
        if dot == 0:
            # 90 or 270 degree rotation
            candidate_ends.append((site, 90))
        elif dot == -1:
            # 180 degree rotation
            candidate_ends.append((site, 180))
    
    if not candidate_ends:
        return None, None
    
    # Pick a random rotation
    site, angle = candidate_ends[rng.randint(0, len(candidate_ends)-1)]
    
    # Apply the rotation to the entire tail
    # For a rotation around axis (ax, ay, az) by angle:
    # We need to rotate each point in the tail around the axis line through pivot
    
    # Rotation matrix for angle around unit axis (a,b,c):
    # For 90 deg: 
    #   new = (b*(c*py - az*px) + ... ) - this gets complex
    # Let's use a simpler approach: for each point in tail, compute its offset from pivot,
    # rotate the offset, add back pivot
    
    # For 90 deg rotation around axis (ax, ay, az):
    # Using Rodrigues' rotation formula:
    # v_rot = v*cos + (u x v)*sin + u*(u dot v)*(1-cos)
    # For 90 deg: cos=0, sin=1
    # v_rot = (u x v) + u*(u dot v)
    
    # For 180 deg: cos=-1, sin=0
    # v_rot = -v + 2*u*(u dot v)
    
    new_tail = []
    for pt in tail:
        px = pt[0] - pivot[0]
        py = pt[1] - pivot[1]
        pz = pt[2] - pivot[2]
        
        if angle == 90:
            # v_rot = (u x v) + u*(u dot v)
            # u x v = (ay*pz - az*py, az*px - ax*pz, ax*py - ay*px)
            uxv_x = ay*pz - az*py
            uxv_y = az*px - ax*pz
            uxv_z = ax*py - ay*px
            udv = ax*px + ay*py + az*pz
            rx = uxv_x + ax*udv
            ry = uxv_y + ay*udv
            rz = uxv_z + az*udv
        elif angle == 180:
            # v_rot = -v + 2*u*(u dot v)
            udv = ax*px + ay*py + az*pz
            rx = -px + 2*ax*udv
            ry = -py + 2*ay*udv
            rz = -pz + 2*az*udv
        else:
            return None, None
        
        new_pt = (pivot[0] + rx, pivot[1] + ry, pivot[2] + rz)
        new_tail.append(new_pt)
    
    new_chain = prefix + new_tail
    
    # Check self-avoidance
    seen = set()
    for pt in new_chain:
        if pt in seen:
            return None, None
        seen.add(pt)
    
    # Check unit steps
    for k in range(len(new_chain)-1):
        diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(3))
        if diff != 1:
            return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy


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


def _regrow_tail(sequence, chain, i, rng):
    """
    Re-grow the tail from index i onwards, given the prefix up to i-1.
    Returns (new_chain, new_energy) or (None, None) if stuck.
    """
    prefix = chain[:i]
    occupied = {p: idx for idx, p in enumerate(prefix)}
    
    new_chain = list(prefix)
    
    for k in range(i, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None, None
        
        site = max(options)[1]
        occupied[site] = k
        new_chain.append(site)
    
    energy = _energy(sequence, new_chain)
    return new_chain, energy


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    Phase 1: Multiple randomized greedy growths.
    Phase 2: Local optimization via tail re-growth.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Start with 0 (no contacts)
    
    # Phase 1: Multiple greedy growths
    # Scale number of growths with time available and sequence length
    time_remaining = _DEADLINE - time.time()
    # For shorter sequences, we can do more growths
    base_growths = 500 if n < 50 else 200
    num_growths = max(50, min(1000, int(base_growths * (time_remaining / 300.0))))
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 10:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line is always self-avoiding
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local optimization via tail re-growth and pivot moves
    current_chain = list(best_chain)
    current_energy = best_energy
    
    time_remaining = _DEADLINE - time.time()
    # Budget for local optimization
    max_local_steps = max(100, min(8000, int(time_remaining * 60)))
    
    for step in range(max_local_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose between pivot move and tail re-growth
        if rng.random() < 0.5:
            # Pivot move
            if n >= 4:
                i = rng.randint(2, n - 2)
                new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng)
            else:
                new_chain, new_energy = None, None
        else:
            # Tail re-growth
            if rng.random() < 0.3:
                i = rng.randint(1, max(1, n // 4))
            elif rng.random() < 0.5:
                i = rng.randint(n // 4, n // 2)
            else:
                i = rng.randint(n // 2, n - 1)
            
            i = min(i, n - 1)
            new_chain, new_energy = _regrow_tail(sequence, current_chain, i, rng)
        
        if new_chain is None:
            continue
        
        # Accept if energy improves or with small probability (simulated annealing-like)
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        elif rng.random() < 0.05:
            # Occasionally accept worse solutions to escape local optima
            current_chain = new_chain
            current_energy = new_energy
    
    # Ensure we return the best chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    # Final verification: ensure self-avoidance and unit steps
    seen = set()
    valid = True
    for pt in best_chain:
        if pt in seen:
            valid = False
            break
        seen.add(pt)
    if valid:
        for k in range(len(best_chain)-1):
            diff = sum(abs(best_chain[k][d] - best_chain[k+1][d]) for d in range(3))
            if diff != 1:
                valid = False
                break
    
    if not valid:
        # Fallback to a simple valid chain
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

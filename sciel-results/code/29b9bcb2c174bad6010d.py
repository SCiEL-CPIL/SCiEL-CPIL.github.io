"""HP lattice protein folding on cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 280.0  # safety margin before 300s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _check_deadline():
    return _elapsed() > _DEADLINE


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute HP energy: -count of H-H non-bond contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    if j > i:  # count each pair once
                        contacts += 1
    return -contacts


def _is_self_avoiding(chain):
    return len(set(chain)) == len(chain)


def _grow_spiral(sequence, dimension, rng):
    """Grow a chain preferring compact/spiral patterns."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    # Initial direction: +x
    prev_dir = (1, 0, 0) if dimension >= 3 else (1, 0)
    
    for i in range(1, n):
        # Generate candidate moves: prefer turns that create compactness
        candidates = []
        last = chain[-1]
        for axis in range(dimension):
            for step in (-1, 1):
                if axis == 0 and step == -1:
                    continue  # don't go back along initial direction for first few
                q = list(last)
                q[axis] += step
                q = tuple(q)
                if q in occupied:
                    continue
                # Score: prefer moves that are near existing chain (compact)
                dist_to_origin = sum(abs(c) for c in q)
                # Count nearby occupied sites (compactness)
                nearby = 0
                for nn in _neighbours(q):
                    if nn in occupied:
                        nearby += 1
                # H-H contact potential
                gain = 0
                if sequence[i] == 'H':
                    for nn in _neighbours(q):
                        j = occupied.get(nn)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                score = gain * 2.0 + nearby * 0.5 - dist_to_origin * 0.05 + rng.random() * 0.3
                candidates.append((score, q))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        # Pick from top few with some randomness
        top_k = min(3, len(candidates))
        chosen = candidates[rng.randint(0, top_k - 1)][1]
        occupied[chosen] = i
        chain.append(chosen)
    
    return chain


def _grow_greedy(sequence, dimension, rng):
    """Original greedy growth with H-H contact preference."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
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


def _pivot_move(chain, dimension, rng):
    """Perform a 3D pivot move: rotate a subchain around a bond axis.
    Returns new chain or None if invalid."""
    n = len(chain)
    if n < 3:
        return None
    
    # Pick a pivot bond index (between i-1 and i)
    i = rng.randint(1, n - 1)
    # The pivot axis is the bond from chain[i-1] to chain[i]
    a = chain[i - 1]
    b = chain[i]
    axis_vec = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    
    # For unit steps, axis_vec is one of the 6 axis-aligned unit vectors
    # The rotation axis is along this bond
    # We need to rotate the subchain chain[i:] around the axis through a and b
    
    # Determine the axis direction
    ax, ay, az = axis_vec
    
    # The rotation is around the line through a in direction (ax,ay,az)
    # We rotate points chain[i], chain[i+1], ..., chain[n-1]
    
    # For axis-aligned bonds, the rotation is 90, 180, or 270 degrees
    # around the bond axis. The other two axes form a plane.
    
    # Find the two perpendicular axes
    if ax != 0:
        # Bond along x-axis; rotate in yz plane
        perp_axes = [1, 2]
    elif ay != 0:
        perp_axes = [0, 2]
    else:
        perp_axes = [0, 1]
    
    p1, p2 = perp_axes
    
    # Try rotations by 90, 180, 270 degrees
    # For a point relative to pivot point a:
    # new_y = y*cos - z*sin, new_z = y*sin + z*cos (for x-axis bond)
    
    for angle_deg in (90, 180, 270):
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        new_chain = list(chain[:i])  # keep prefix
        new_occupied = set(chain[:i])
        
        valid = True
        for j in range(i, n):
            pt = chain[j]
            # Translate to pivot origin a
            dx = pt[0] - a[0]
            dy = pt[1] - a[1]
            dz = pt[2] - a[2]
            
            # Rotate in the plane perpendicular to bond axis
            if ax != 0:
                # x-axis bond: rotate y,z
                new_dy = dy * cos_a - dz * sin_a
                new_dz = dy * sin_a + dz * cos_a
                new_dx = dx
            elif ay != 0:
                # y-axis bond: rotate x,z
                new_dx = dx * cos_a - dz * sin_a
                new_dz = dx * sin_a + dz * cos_a
                new_dy = dy
            else:
                # z-axis bond: rotate x,y
                new_dx = dx * cos_a - dy * sin_a
                new_dy = dx * sin_a + dy * cos_a
                new_dz = dz
            
            # Check if integer (should be for 90/180/270)
            if abs(new_dx - round(new_dx)) > 1e-9 or \
               abs(new_dy - round(new_dy)) > 1e-9 or \
               abs(new_dz - round(new_dz)) > 1e-9:
                valid = False
                break
            
            new_pt = (a[0] + int(round(new_dx)),
                      a[1] + int(round(new_dy)),
                      a[2] + int(round(new_dz)))
            
            if new_pt in new_occupied:
                valid = False
                break
            
            new_occupied.add(new_pt)
            new_chain.append(new_pt)
        
        if valid:
            return new_chain
    
    return None


def _sa_refine(sequence, chain, dimension, rng, max_iters, deadline_check):
    """Simulated annealing refinement with pivot moves."""
    n = len(sequence)
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = max(1.0, abs(current_energy) * 0.5 + 1.0)
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    T = T_start
    for it in range(max_iters):
        if deadline_check():
            break
        if it % 500 == 0 and it > 0:
            T *= cooling ** 500
        
        new_chain = _pivot_move(current, dimension, rng)
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        
        T *= cooling
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation via multiple growth strategies + SA refinement."""
    global _start_time
    _start_time = time.time()  # reset for this call
    
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 1  # energy is negative, so start high
    
    # Count H residues for scaling effort
    n_h = sequence.count('H')
    
    # Phase 1: Generate diverse initial conformations
    num_growths = min(50, max(10, n_h * 2))
    
    for attempt in range(num_growths):
        if _check_deadline():
            break
        
        if attempt % 2 == 0:
            chain = _grow_spiral(sequence, dimension, rng)
        else:
            chain = _grow_greedy(sequence, dimension, rng)
        
        if chain is None:
            continue
        
        if not _is_self_avoiding(chain):
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Phase 2: SA refinement on best and some other good conformations
    if best_chain is not None:
        # Refine the best
        sa_iters = min(5000, max(500, n * 50))
        refined, refined_e = _sa_refine(sequence, best_chain, dimension, rng, sa_iters, _check_deadline)
        if refined_e < best_energy:
            best_energy = refined_e
            best_chain = refined
    
    # Phase 3: If time remains, do more SA on different starting points
    if not _check_deadline() and best_chain is not None:
        # Try a few more SA rounds from perturbed versions
        for extra in range(5):
            if _check_deadline():
                break
            # Perturb the best chain with a few random pivots
            perturbed = list(best_chain)
            for _ in range(3):
                new_c = _pivot_move(perturbed, dimension, rng)
                if new_c is not None:
                    perturbed = new_c
                    break
            
            if _is_self_avoiding(perturbed):
                sa_iters2 = min(3000, max(300, n * 30))
                refined2, refined_e2 = _sa_refine(sequence, perturbed, dimension, rng, sa_iters2, _check_deadline)
                if refined_e2 < best_energy:
                    best_energy = refined_e2
                    best_chain = refined2
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

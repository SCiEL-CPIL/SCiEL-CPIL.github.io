"""HP lattice protein folding on the cubic lattice – pivot-move SA with spatial grid."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 315.0  # 315s hard limit, 15s safety margin


def _time_left():
    return _HARD_DEADLINE - time.time()


def _neighbours(p):
    """Yield the 6 face-neighbours of a 3D lattice point (tuple of 3 ints)."""
    x, y, z = p
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


class _SpatialGrid:
    """Hash-based occupancy grid for O(1) lookups."""
    def __init__(self):
        self._map = {}

    def set(self, point, idx):
        self._map[point] = idx

    def get(self, point):
        return self._map.get(point)

    def remove(self, point):
        self._map.pop(point, None)

    def __contains__(self, point):
        return point in self._map

    def __len__(self):
        return len(self._map)


def _compute_energy(sequence, chain, grid=None):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    if grid is None:
        grid = _SpatialGrid()
        for i, p in enumerate(chain):
            grid.set(p, i)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = grid.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, grid, i, old_p, new_p):
    """
    Compute energy delta when moving residue i from old_p to new_p.
    Returns the change in energy (negative = improvement).
    """
    delta = 0
    # Contacts lost at old_p
    if sequence[i] == 'H':
        for q in _neighbours(old_p):
            j = grid.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta -= 1
    # Contacts gained at new_p
    if sequence[i] == 'H':
        for q in _neighbours(new_p):
            j = grid.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta += 1
    return delta


def _grow_greedy(sequence, dimension=3):
    """
    Greedy growth: place each residue to maximize local H-H contacts.
    Returns a valid self-avoiding chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    chain = [(0, 0, 0)]
    grid = _SpatialGrid()
    grid.set((0, 0, 0), 0)

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    rng = random.Random(42)

    for i in range(1, n):
        best_dir = None
        best_score = float('-inf')
        shuffled = directions[:]
        rng.shuffle(shuffled)
        for dx, dy, dz in shuffled:
            np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][1] + dz)
            if np_ in grid:
                continue
            score = 0.0
            if sequence[i] == 'H':
                for q in _neighbours(np_):
                    j = grid.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 3.0
            # Prefer compactness
            norm = abs(np_[0]) + abs(np_[1]) + abs(np_[2])
            score -= norm * 0.05
            score += rng.random() * 0.3
            if score > best_score:
                best_score = score
                best_dir = np_
        if best_dir is None:
            # Fallback: any free neighbor
            for dx, dy, dz in directions:
                np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
                if np_ not in grid:
                    best_dir = np_
                    break
        if best_dir is None:
            # Stuck – return what we have (shouldn't happen for reasonable n)
            break
        chain.append(best_dir)
        grid.set(best_dir, i)

    return chain


def _grow_compact(sequence, dimension=3):
    """
    Compact growth: try to keep the chain tightly packed.
    Uses a different random seed for diversity.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    chain = [(0, 0, 0)]
    grid = _SpatialGrid()
    grid.set((0, 0, 0), 0)

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    rng = random.Random(137)

    for i in range(1, n):
        candidates = []
        for dx, dy, dz in directions:
            np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
            if np_ not in grid:
                candidates.append(np_)
        if not candidates:
            break
        # Score by compactness and H contacts
        best_dir = None
        best_score = float('-inf')
        for np_ in candidates:
            score = 0.0
            norm = abs(np_[0]) + abs(np_[1]) + abs(np_[2])
            score -= norm * 0.2
            if sequence[i] == 'H':
                for q in _neighbours(np_):
                    j = grid.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 2.0
            score += rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_dir = np_
        chain.append(best_dir)
        grid.set(best_dir, i)

    return chain


def _random_walk_init(sequence, dimension=3, seed=0):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    chain = [(0, 0, 0)]
    grid = _SpatialGrid()
    grid.set((0, 0, 0), 0)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    rng = random.Random(seed)

    for i in range(1, n):
        dirs = directions[:]
        rng.shuffle(dirs)
        placed = False
        for dx, dy, dz in dirs:
            np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
            if np_ not in grid:
                chain.append(np_)
                grid.set(np_, i)
                placed = True
                break
        if not placed:
            return None
    return chain


def _pivot_move(sequence, chain, grid, pivot_idx, direction, rng):
    """
    Attempt a pivot move: rotate the segment chain[pivot_idx+1:] 
    around the bond (chain[pivot_idx], chain[pivot_idx+1]).
    Returns (new_chain, new_grid, energy_delta) or (None, None, None) if invalid.
    """
    n = len(chain)
    if pivot_idx < 0 or pivot_idx >= n - 1:
        return None, None, None

    p0 = chain[pivot_idx]
    p1 = chain[pivot_idx + 1]

    # Determine rotation axis and find valid rotation
    # The bond vector is p1 - p0. We rotate the tail around this axis.
    # In 3D cubic lattice, a 90-degree rotation around the bond axis 
    # maps the tail to one of up to 4 possible positions (or fewer if blocked).

    # For simplicity, we try all 6 neighbor directions from p1 (excluding p0)
    # and see if the entire tail can be placed there via a rigid rotation/translation.
    # Actually, a proper pivot move rotates around the bond. Let's implement it properly.

    # Bond vector
    bx = p1[0] - p0[0]
    by = p1[1] - p0[1]
    bz = p1[2] - p0[2]

    # Find two vectors perpendicular to the bond to define rotation
    # We need to rotate the tail segment. The tail is chain[pivot_idx+1:]
    tail = chain[pivot_idx + 1:]
    tail_len = len(tail)
    if tail_len == 0:
        return None, None, None

    # For a 90-degree rotation around the bond axis:
    # We need to find a new position for tail[0] (which is p1) such that
    # it's adjacent to p0 and the rest of the tail fits.
    # Actually, p1 stays fixed in a pivot move around the bond (p0, p1).
    # Wait, no: in a pivot move, we rotate the segment AFTER the pivot bond.
    # The pivot bond is (chain[pivot_idx], chain[pivot_idx+1]).
    # We rotate chain[pivot_idx+1:] around the axis defined by the bond.
    # So chain[pivot_idx] stays fixed, chain[pivot_idx+1] moves.

    # Let's try: for each of the 6 neighbors of p0 (excluding p1), 
    # try to place the entire tail starting at that neighbor with a rigid transformation.
    # This is expensive. Instead, let's use a simpler "segment move":
    # Pick a segment and try to translate/rotate it.

    # Simpler approach: try moving a small segment (length 2-5) to a new position.
    # For now, let's implement a "2-opt" style move: reverse a segment.
    # Actually, let's do a proper pivot: rotate tail around bond by 90 degrees.

    # To rotate around the bond axis (bx,by,bz) by 90 degrees:
    # We need an orthonormal basis. Let's find a perpendicular vector.
    if bx != 0:
        # Bond is along x. Perpendicular vectors: (0,1,0) and (0,0,1)
        perp1 = (0, 1, 0)
        perp2 = (0, 0, 1)
    elif by != 0:
        perp1 = (1, 0, 0)
        perp2 = (0, 0, 1)
    else:
        perp1 = (1, 0, 0)
        perp2 = (0, 1, 0)

    # Cross product to get the third axis
    def cross(a, b):
        return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])

    # Rotation by 90 degrees around bond axis:
    # For a point v relative to p1, new_v = perp1 * (v dot perp2) - perp2 * (v dot perp1)
    # (This is a 90-degree rotation in the plane spanned by perp1 and perp2)

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    # Try rotating the tail. The new position of tail[k] is:
    # p1 + rot(tail[k] - p1) where rot is 90-degree rotation around bond axis
    # But we need to check if all new positions are free.

    # Compute new tail positions
    new_tail = [p1]  # p1 stays? No, in pivot move p1 moves.
    # Actually, in a pivot move around bond (p0, p1), p1 is the pivot point.
    # The segment after p1 rotates. So p1 stays, and tail[1:] rotates around p1.
    # Let me redefine: pivot_idx is the index of the pivot atom.
    # We rotate chain[pivot_idx+1:] around the bond (chain[pivot_idx], chain[pivot_idx+1]).
    # So chain[pivot_idx] and chain[pivot_idx+1] stay fixed? No, that's not a move.
    
    # Standard pivot move: pick a bond, rotate one side around it.
    # Let's say we rotate the tail (from pivot_idx+1 onward) around the bond at pivot_idx.
    # The bond is between chain[pivot_idx] and chain[pivot_idx+1].
    # We keep chain[0..pivot_idx] fixed, and rotate chain[pivot_idx+1..n-1] around the bond axis.
    # chain[pivot_idx+1] moves to a new position adjacent to chain[pivot_idx].

    # So: p0 = chain[pivot_idx] stays. p1 = chain[pivot_idx+1] moves to a new neighbor of p0.
    # The rest of the tail follows via rigid rotation.

    # Try each of the 6 neighbors of p0 (excluding current p1)
    best_new_tail = None
    best_delta = 0

    for new_p1 in _neighbours(p0):
        if new_p1 == p1:
            continue
        # Check if new_p1 is free (it will be, since we're removing p1)
        # Compute new tail via rigid rotation
        # Rotation: 90 degrees around bond axis (p0 to p1 direction)
        # New bond direction: p0 to new_p1
        # We need to map the old tail to the new tail via this rotation.
        
        # For simplicity, let's just try to place the tail greedily from new_p1.
        # This is a "greedy pivot" – not a true rigid rotation, but effective.
        
        # Check if new_p1 is occupied (excluding p1 which we'll remove)
        if new_p1 in grid:
            # It might be p1 itself or another atom
            idx = grid.get(new_p1)
            if idx is not None and idx != pivot_idx + 1:
                continue  # Blocked by another atom
        
        # Try to extend the tail from new_p1
        new_chain_prefix = chain[:pivot_idx + 1]
        new_tail = [new_p1]
        temp_grid = _SpatialGrid()
        for k, p in enumerate(new_chain_prefix):
            temp_grid.set(p, k)
        
        valid = True
        for k in range(1, tail_len):
            old_rel = (tail[k][0] - tail[k-1][0], tail[k][1] - tail[k-1][1], tail[k][2] - tail[k-1][2])
            # Try all 6 directions from new_tail[-1]
            placed = False
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                cand = (new_tail[-1][0] + dx, new_tail[-1][1] + dy, new_tail[-1][2] + dz)
                if cand not in temp_grid:
                    new_tail.append(cand)
                    temp_grid.set(cand, pivot_idx + 1 + k)
                    placed = True
                    break
            if not placed:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check that the new tail doesn't collide with the prefix
        for p in new_tail:
            if p in temp_grid and temp_grid.get(p) <= pivot_idx:
                valid = False
                break
        if not valid:
            continue
        
        # Compute energy delta
        # Remove old tail from grid
        old_energy_contacts = 0
        for k, p in enumerate(tail):
            if sequence[pivot_idx + 1 + k] == 'H':
                for q in _neighbours(p):
                    j = grid.get(q)
                    if j is not None and j != pivot_idx + 1 + k and abs(j - (pivot_idx + 1 + k)) > 1 and sequence[j] == 'H':
                        old_energy_contacts += 1
        
        new_energy_contacts = 0
        for k, p in enumerate(new_tail):
            if sequence[pivot_idx + 1 + k] == 'H':
                for q in _neighbours(p):
                    j = temp_grid.get(q)
                    if j is not None and j != pivot_idx + 1 + k and abs(j - (pivot_idx + 1 + k)) > 1 and sequence[j] == 'H':
                        new_energy_contacts += 1
        
        delta = (new_energy_contacts - old_energy_contacts) * (-1)  # Energy = -contacts
        
        if best_new_tail is None or delta < best_delta:
            best_delta = delta
            best_new_tail = new_tail
    
    if best_new_tail is None:
        return None, None, None
    
    # Apply the move
    new_chain = chain[:pivot_idx + 1] + best_new_tail
    new_grid = _SpatialGrid()
    for i, p in enumerate(new_chain):
        new_grid.set(p, i)
    
    return new_chain, new_grid, best_delta


def _simulated_annealing(sequence, chain, time_limit=5.0, seed=0):
    """
    Simulated annealing with pivot moves.
    """
    n = len(chain)
    if n < 4:
        return chain

    grid = _SpatialGrid()
    for i, p in enumerate(chain):
        grid.set(p, i)
    
    current_energy = _compute_energy(sequence, chain, grid)
    best_energy = current_energy
    best_chain = list(chain)
    best_grid = _SpatialGrid()
    for i, p in enumerate(chain):
        best_grid.set(p, i)

    current_chain = list(chain)
    current_grid = grid

    start_time = time.time()
    rng = random.Random(seed)

    T = 1.5
    T_min = 0.05
    # Adaptive cooling
    cooling_rate = 0.9995

    h_indices = [i for i in range(n) if sequence[i] == 'H']

    while time.time() - start_time < time_limit and _time_left() > 2.0:
        T = max(T_min, T * cooling_rate)

        # Choose move type: pivot move
        # Pick a random pivot index (avoid ends)
        pivot_idx = rng.randint(0, n - 2)

        result = _pivot_move(sequence, current_chain, current_grid, pivot_idx, None, rng)
        if result is None or result[0] is None:
            continue

        new_chain, new_grid, delta = result
        new_energy = current_energy + delta

        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_chain = new_chain
            current_grid = new_grid
            current_energy = new_energy

            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
                best_grid = _SpatialGrid()
                for i, p in enumerate(new_chain):
                    best_grid.set(p, i)

    return best_chain


def _local_search(sequence, chain, time_limit=2.0, seed=0):
    """
    Local search using pivot moves (greedy).
    """
    n = len(chain)
    if n < 4:
        return chain

    grid = _SpatialGrid()
    for i, p in enumerate(chain):
        grid.set(p, i)
    
    current_energy = _compute_energy(sequence, chain, grid)
    current_chain = list(chain)
    current_grid = grid

    start_time = time.time()
    rng = random.Random(seed)

    improved = True
    while improved and time.time() - start_time < time_limit and _time_left() > 2.0:
        improved = False
        for pivot_idx in range(0, n - 1):
            if time.time() - start_time > time_limit or _time_left() < 2.0:
                break
            result = _pivot_move(sequence, current_chain, current_grid, pivot_idx, None, rng)
            if result is None or result[0] is None:
                continue
            new_chain, new_grid, delta = result
            if delta < 0:
                current_chain = new_chain
                current_grid = new_grid
                current_energy += delta
                improved = True

    return current_chain


def fold(sequence, dimension):
    """
    Fold the HP sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    if _time_left() < 3.0:
        # Emergency: return a straight line
        return [[i, 0, 0] for i in range(n)]

    # Generate diverse initial conformations
    inits = []
    c1 = _grow_greedy(sequence, dimension)
    if len(c1) == n:
        inits.append(c1)
    c2 = _grow_compact(sequence, dimension)
    if len(c2) == n:
        inits.append(c2)
    for s in [777, 888, 999]:
        c3 = _random_walk_init(sequence, dimension, seed=s)
        if c3 is not None and len(c3) == n:
            inits.append(c3)

    if not inits:
        return [[i, 0, 0] for i in range(n)]

    best_chain = None
    best_energy = float('inf')

    # Time budget per sequence
    seq_budget = max(1.0, min(12.0, _time_left() / 26.0))

    for idx, init_chain in enumerate(inits):
        if _time_left() < 2.0:
            break

        # Local search
        ls_chain = _local_search(sequence, init_chain, time_limit=min(1.5, seq_budget * 0.3), seed=1000 + idx)
        ls_energy = _compute_energy(sequence, ls_chain)
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain

        # Simulated annealing
        if _time_left() > 3.0:
            sa_chain = _simulated_annealing(sequence, ls_chain, time_limit=min(4.0, seq_budget * 0.6), seed=2000 + idx)
            sa_energy = _compute_energy(sequence, sa_chain)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain

    if best_chain is None:
        best_chain = inits[0]

    # Final validity check
    if best_chain and len(best_chain) == n:
        seen = set()
        valid = True
        for i, p in enumerate(best_chain):
            pt = tuple(p)
            if pt in seen:
                valid = False
                break
            seen.add(pt)
            if i > 0:
                prev = tuple(best_chain[i - 1])
                dist = abs(pt[0] - prev[0]) + abs(pt[1] - prev[1]) + abs(pt[2] - prev[2])
                if dist != 1:
                    valid = False
                    break
        if not valid:
            return [[i, 0, 0] for i in range(n)]

    return [list(p) for p in best_chain]

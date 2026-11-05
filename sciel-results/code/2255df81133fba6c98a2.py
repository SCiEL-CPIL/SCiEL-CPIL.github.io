"""HP lattice protein folding using space-filling initialization + local search."""
import random
import time

# EVOLVE-BLOCK-START
import random
import time

_NEIGHBOR_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _neighbors2d(point):
    x, y = point
    for dx, dy in _NEIGHBOR_DIRS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute HP energy: -1 per non-adjacent H-H contact."""
    n = len(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for nb in _neighbors2d(chain[i]):
            j = pos_to_idx.get(nb)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # each contact counted twice


def _snake_conformation(n, rng):
    """Generate a compact snake/grid conformation of length n."""
    # Find a grid size that fits n residues
    # Use a rectangular grid, fill in snake pattern
    import math
    w = int(math.ceil(math.sqrt(n)))
    h = int(math.ceil(n / w))
    # Make sure we can fit n in w x h grid
    if w * h < n:
        w += 1
    if w * h < n:
        h += 1
    
    points = []
    for row in range(h):
        if row % 2 == 0:
            for col in range(w):
                points.append((col, row))
        else:
            for col in range(w - 1, -1, -1):
                points.append((col, row))
    points = points[:n]
    
    # Validate: consecutive points must be adjacent
    valid = True
    for i in range(len(points) - 1):
        dx = abs(points[i][0] - points[i+1][0])
        dy = abs(points[i][1] - points[i+1][1])
        if dx + dy != 1:
            valid = False
            break
    
    if not valid:
        # Fallback: simple line
        return [(i, 0) for i in range(n)]
    
    return points


def _hilbert_conformation(n, rng):
    """Generate a Hilbert curve based conformation for better packing."""
    import math
    # Use a square grid
    side = int(math.ceil(math.sqrt(n)))
    while side * side < n:
        side += 1
    
    # Generate Hilbert curve points
    def d2xy(d, n):
        """Convert Hilbert index d to (x,y) on an n x n grid."""
        rx = 0
        ry = 0
        t = d
        x = 0
        y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            # Rotate
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return (x, y)
    
    points = []
    for d in range(n):
        x, y = d2xy(d, side)
        points.append((x, y))
    
    # Validate
    valid = True
    for i in range(len(points) - 1):
        dx = abs(points[i][0] - points[i+1][0])
        dy = abs(points[i][1] - points[i+1][1])
        if dx + dy != 1:
            valid = False
            break
    
    if not valid:
        return _snake_conformation(n, rng)
    
    return points


def _is_valid_conformation(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i-1][0] - p[0])
            dy = abs(chain[i-1][1] - p[1])
            if dx + dy != 1:
                return False
    return True


def _pivot_move(chain, i, j, rng):
    """
    Attempt a pivot move: rotate the sub-chain from i to j.
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or j >= n - 1 or j <= i:
        return None
    
    prefix = chain[:i]
    segment = chain[i:j+1]
    suffix = chain[j+1:]
    
    # The segment must be rotated so that the new endpoint connects to prefix[i-1]
    # and the new start connects to suffix[j+1]
    # We try rotating the segment by 90, 180, 270 degrees around the pivot point (chain[i-1])
    
    pivot = chain[i-1]
    
    # Try different rotations
    for angle in (90, -90, 180):
        # Rotate segment relative to pivot
        new_segment = []
        for p in segment:
            dx = p[0] - pivot[0]
            dy = p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:  # 180
                ndx, ndy = -dx, -dy
            new_segment.append((pivot[0] + ndx, pivot[1] + ndy))
        
        new_chain = prefix + new_segment + suffix
        
        if _is_valid_conformation(new_chain):
            return new_chain
    
    return None


def _single_move(chain, i, rng):
    """
    Attempt to move a single residue (not first or last) to an adjacent empty spot.
    This is a limited move - only works if we can shift the tail.
    For simplicity, we'll use a different approach: try moving residue i to each
    of its 4 neighbors, and check if the rest of the chain can still be connected.
    This is complex, so let's use a simpler local search.
    """
    # Instead, let's do a simple "kink" move: try to flip a small segment
    # Or better: use a different local search strategy
    
    # Let's try: pick a position, try to move it to an adjacent empty cell,
    # and see if the chain remains connected (this is hard for arbitrary i)
    
    # Simpler: just try pivot moves and small segment rotations
    return None


def _local_search(sequence, chain, rng, max_iterations=5000, time_limit=None, start_time=None):
    """
    Local search using pivot moves. Hill climbing with occasional random restarts.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    iterations = 0
    while iterations < max_iterations:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Pick a random pivot point
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        new_chain = _pivot_move(current, i, j, rng)
        if new_chain is None:
            iterations += 1
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy <= current_energy:
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(current)
                best_energy = new_energy
        
        iterations += 1
    
    return best


def _crankshaft_move(chain, i, j, rng):
    """
    Crankshaft move: rotate a sub-chain while keeping endpoints fixed.
    This is more complex than pivot. For now, use pivot moves.
    """
    return None


def _generate_initial_conformations(sequence, rng, count=5):
    """Generate multiple diverse initial conformations."""
    n = len(sequence)
    conformations = []
    
    # Snake pattern
    conformations.append(_snake_conformation(n, rng))
    
    # Hilbert pattern
    conformations.append(_hilbert_conformation(n, rng))
    
    # Random growth (like the original, but as one of several)
    for _ in range(count - 2):
        chain = _greedy_grow(sequence, rng)
        if chain is not None:
            conformations.append(chain)
    
    return conformations


def _greedy_grow(sequence, rng):
    """Randomized greedy growth (from original, improved)."""
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for dx, dy in _NEIGHBOR_DIRS:
            candidate = (last[0] + dx, last[1] + dy)
            if candidate in occupied:
                continue
            # Score: prefer H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for nb in _neighbors2d(candidate):
                    if nb in occupied:
                        # Check if the residue at nb is H and not adjacent in chain
                        # We need to track which index is at which position
                        pass
            # Just use random tie-breaking
            options.append((rng.random(), candidate))
        
        if not options:
            return None
        
        options.sort(key=lambda x: -x[0])
        chain.append(options[0][1])
        occupied.add(chain[-1])
    
    return chain


def fold(sequence, dimension):
    """
    Fold an HP sequence into a low-energy conformation on the square lattice.
    Returns a list of [x, y] points.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    total_time_limit = 180.0  # Leave margin before 210s kill
    
    rng = random.Random(42 + len(sequence))
    
    # Generate initial conformations
    initial_confs = _generate_initial_conformations(sequence, rng, count=5)
    
    best_chain = None
    best_energy = 0
    
    for conf in initial_confs:
        if not _is_valid_conformation(conf):
            continue
        e = _energy(sequence, conf)
        if best_chain is None or e < best_energy:
            best_chain = list(conf)
            best_energy = e
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Local search from best initial conformation
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 2:
        improved = _local_search(
            sequence, best_chain, rng,
            max_iterations=100000,
            time_limit=remaining_time - 1.0,
            start_time=start_time
        )
        improved_energy = _energy(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    # If time remains, try more local search from other initial conformations
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 5:
        for conf in initial_confs[1:]:
            if time.time() - start_time > total_time_limit - 3:
                break
            if not _is_valid_conformation(conf):
                continue
            e = _energy(sequence, conf)
            if e < best_energy:
                best_chain = list(conf)
                best_energy = e
            
            remaining = total_time_limit - (time.time() - start_time) - 1.0
            if remaining > 1:
                improved = _local_search(
                    sequence, best_chain, rng,
                    max_iterations=50000,
                    time_limit=remaining,
                    start_time=start_time
                )
                improved_energy = _energy(sequence, improved)
                if improved_energy < best_energy:
                    best_chain = improved
                    best_energy = improved_energy
    
    # Final: ensure validity
    if not _is_valid_conformation(best_chain):
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

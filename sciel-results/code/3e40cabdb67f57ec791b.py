"""HP lattice protein folding: skeleton-based initialization + pivot-move refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE_FRACTION = 0.90  # return by 90% of the 120s budget to be safe
_TOTAL_BUDGET = 120.0


def _neighbours_2d(x, y):
    """Yield the 4 lattice neighbours of (x,y)."""
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _generate_serpentine(n, width):
    """
    Generate a serpentine (boustrophedon) Hamiltonian path on a grid
    that visits exactly n cells. Returns list of (x, y) tuples.
    width is the number of columns; rows are ceil(n / width).
    """
    rows = (n + width - 1) // width
    path = []
    for r in range(rows):
        if r % 2 == 0:
            for c in range(width):
                if len(path) < n:
                    path.append((c, r))
        else:
            for c in range(width - 1, -1, -1):
                if len(path) < n:
                    path.append((c, r))
    return path


def _generate_skeletons(n):
    """
    Generate several diverse serpentine skeletons of length n.
    Returns a list of lists of (x, y) tuples.
    """
    skeletons = []
    # Try a range of widths
    for width in range(2, n + 1):
        if width > 20:  # keep grids reasonable
            break
        path = _generate_serpentine(n, width)
        if len(path) == n:
            # Check self-avoidance (should be guaranteed by construction)
            if len(set(path)) == n:
                skeletons.append(path)
    # Also try a few "spiral-like" variants by shifting
    for width in [3, 5, 7, 11, 13]:
        if width < n:
            path = _generate_serpentine(n, width)
            if len(path) == n and len(set(path)) == n:
                # Shift to center around origin
                min_x = min(p[0] for p in path)
                min_y = min(p[1] for p in path)
                shifted = [(p[0] - min_x, p[1] - min_y) for p in path]
                skeletons.append(shifted)
    return skeletons


def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count of non-consecutive H-H neighbours)."""
    n = len(chain)
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if sequence[i] == 'H':
            for (nx, ny) in _neighbours_2d(x, y):
                j = where.get((nx, ny))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice


def _compute_energy_fast(sequence, chain):
    """Same as _compute_energy but avoids the //2 by only counting j > i+1."""
    n = len(chain)
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if sequence[i] == 'H':
            for (nx, ny) in _neighbours_2d(x, y):
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _pivot_move(sequence, chain, i, rng, time_limit, start_time):
    """
    Try all 3 possible pivot rotations at position i (i is the pivot,
    rotating the sub-chain i+1...n-1). Returns (new_chain, new_energy) if
    a valid move was made, or (None, None) if no valid move.
    """
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None, None
    
    # Get the sub-chain to rotate
    fixed = chain[:i+1]
    sub = chain[i+1:]
    
    # The pivot point is chain[i]
    px, py = chain[i]
    
    # The vector from pivot to the next point in sub
    vx, vy = sub[0][0] - px, sub[0][1] - py
    
    # Three possible rotations: 90, 180, 270 degrees
    # Original direction is (vx, vy)
    # 90 deg CCW: (-vy, vx)
    # 180 deg: (-vx, -vy)
    # 270 deg CCW (or 90 CW): (vy, -vx)
    
    rotations = [
        (-vy, vx),   # 90 CCW
        (-vx, -vy),  # 180
        (vy, -vx),   # 90 CW
    ]
    
    # Filter out the original direction
    valid_rotations = [r for r in rotations if r != (vx, vy)]
    
    if not valid_rotations:
        return None, None
    
    # Try each rotation
    for (dx, dy) in valid_rotations:
        # Compute the new sub-chain by rotating around the pivot
        # The first point in sub moves to (px+dx, py+dy)
        # The rest of the sub-chain is translated so that the first point
        # is at (px+dx, py+dy), preserving relative positions
        
        # Actually, for a pivot move, we rotate the entire sub-chain
        # around the pivot point. Let's do it properly.
        
        # The sub-chain points relative to pivot:
        rel = [(sx - px, sy - py) for (sx, sy) in sub]
        
        # Apply rotation to each relative point
        # Rotation matrix for 90 CCW: (x,y) -> (-y, x)
        # For 180: (x,y) -> (-x, -y)
        # For 90 CW: (x,y) -> (y, -x)
        
        if (dx, dy) == (-vy, vx):  # 90 CCW
            new_rel = [(-ry, rx) for (rx, ry) in rel]
        elif (dx, dy) == (-vx, -vy):  # 180
            new_rel = [(-rx, -ry) for (rx, ry) in rel]
        else:  # 90 CW
            new_rel = [(ry, -rx) for (rx, ry) in rel]
        
        # Convert back to absolute coordinates
        new_sub = [(px + rx, py + ry) for (rx, ry) in new_rel]
        
        # Check self-avoidance: the new sub-chain must not overlap with fixed
        # (excluding the pivot which is shared)
        fixed_set = set(fixed[:-1])  # exclude pivot
        valid = True
        seen = set()
        for (nx, ny) in new_sub:
            if (nx, ny) in fixed_set:
                valid = False
                break
            if (nx, ny) in seen:
                valid = False
                break
            seen.add((nx, ny))
        
        if not valid:
            continue
        
        new_chain = fixed + new_sub
        new_energy = _compute_energy_fast(sequence, new_chain)
        return new_chain, new_energy
    
    return None, None


def _local_search(sequence, chain, rng, time_limit, start_time):
    """
    Perform local search with pivot moves and simulated annealing.
    Returns the best chain found.
    """
    n = len(chain)
    best_chain = list(chain)
    best_energy = _compute_energy_fast(sequence, best_chain)
    current_energy = best_energy
    current_chain = list(chain)
    
    # Simulated annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random pivot position
        i = rng.randint(1, n - 2)
        
        new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng, time_limit, start_time)
        
        if new_chain is None:
            continue
        
        delta = new_energy - current_energy
        
        if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T = max(min_T, T * cooling)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the HP sequence into a low-energy conformation on the square lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    start_time = time.time()
    time_limit = _TOTAL_BUDGET * _DEADLINE_FRACTION - 5.0  # 5s safety margin
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    rng = random.Random(42 + n)
    
    # Generate diverse skeletons
    skeletons = _generate_skeletons(n)
    
    # If no skeletons generated, use a simple straight line
    if not skeletons:
        straight = [(i, 0) for i in range(n)]
        return [list(p) for p in straight]
    
    # For each skeleton, do local search refinement
    best_chain = None
    best_energy = 0  # worst possible energy
    
    # Sort skeletons by initial energy to prioritize good ones
    skeleton_energies = []
    for skel in skeletons:
        e = _compute_energy_fast(sequence, skel)
        skeleton_energies.append((e, skel))
    
    skeleton_energies.sort()  # best (most negative) first
    
    # Process skeletons, spending more time on better ones
    for idx, (init_energy, skel) in enumerate(skeleton_energies):
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time based on rank: top skeletons get more time
        remaining = time_limit - (time.time() - start_time)
        if remaining <= 0:
            break
        
        # Give top 3 skeletons more time
        if idx < 3:
            alloc = min(remaining * 0.3, 30.0)
        elif idx < 6:
            alloc = min(remaining * 0.15, 15.0)
        else:
            alloc = min(remaining * 0.05, 5.0)
        
        if alloc < 1.0:
            break
        
        refined_chain, refined_energy = _local_search(
            sequence, skel, rng, alloc, start_time
        )
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # If we didn't find any valid chain, fall back to straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Ensure output is in the correct format: list of lists of integers
    result = [list(p) for p in best_chain]
    
    # Final validation: check self-avoidance and unit steps
    seen = set()
    for i, (x, y) in enumerate(result):
        if (x, y) in seen:
            # Self-avoidance violated; fall back to straight line
            result = [[i, 0] for i in range(n)]
            break
        seen.add((x, y))
        if i > 0:
            px, py = result[i-1]
            if abs(x - px) + abs(y - py) != 1:
                # Unit step violated; fall back to straight line
                result = [[i, 0] for i in range(n)]
                break
    
    return result
# EVOLVE-BLOCK-END

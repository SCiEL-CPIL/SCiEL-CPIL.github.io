"""HP lattice protein folding on the cubic lattice.
Combines randomized greedy growth with local tail re-growth refinement
and a deadline guard to stay within the wall-clock budget.
"""
import random
import time
import math

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
    
    # The axis direction is from pivot to axis_start (unit vector)
    ax = axis_start[0] - pivot[0]
    ay = axis_start[1] - pivot[1]
    az = axis_start[2] - pivot[2]
    
    def rotate_90(p, pivot_pt, ax_dir):
        """Rotate point p around axis through pivot_pt with direction ax_dir by +90 degrees."""
        px, py, pz = p[0] - pivot_pt[0], p[1] - pivot_pt[1], p[2] - pivot_pt[2]
        a, b, c = ax_dir
        
        if a != 0:  # axis along x
            sign = 1 if a > 0 else -1
            return (p[0], pivot_pt[1] + sign * (-pz), pivot_pt[2] + sign * py)
        elif b != 0:  # axis along y
            sign = 1 if b > 0 else -1
            return (pivot_pt[0] + sign * pz, p[1], pivot_pt[2] + sign * (-px))
        else:  # axis along z
            sign = 1 if c > 0 else -1
            return (pivot_pt[0] + sign * py, pivot_pt[1] + sign * (-px), p[2])
    
    def rotate_180(p, pivot_pt):
        """Rotate point p around pivot_pt by 180 degrees."""
        return (2*pivot_pt[0] - p[0], 2*pivot_pt[1] - p[1], 2*pivot_pt[2] - p[2])
    
    def rotate_270(p, pivot_pt, ax_dir):
        """Rotate point p around axis through pivot_pt with direction ax_dir by -90 degrees."""
        px, py, pz = p[0] - pivot_pt[0], p[1] - pivot_pt[1], p[2] - pivot_pt[2]
        a, b, c = ax_dir
        
        if a != 0:  # axis along x
            sign = 1 if a > 0 else -1
            return (p[0], pivot_pt[1] + sign * pz, pivot_pt[2] + sign * (-py))
        elif b != 0:  # axis along y
            sign = 1 if b > 0 else -1
            return (pivot_pt[0] + sign * (-pz), p[1], pivot_pt[2] + sign * px)
        else:  # axis along z
            sign = 1 if c > 0 else -1
            return (pivot_pt[0] + sign * (-py), pivot_pt[1] + sign * px, p[2])
    
    # Try 3 rotations: 90, 180, 270
    for rot in range(1, 4):
        new_tail = []
        valid = True
        occupied = {p: idx for idx, p in enumerate(prefix)}
        
        for j, p in enumerate(tail):
            if rot == 1:
                np_ = rotate_90(p, pivot, (ax, ay, az))
            elif rot == 2:
                np_ = rotate_180(p, pivot)
            else:
                np_ = rotate_270(p, pivot, (ax, ay, az))
            
            if np_ in occupied:
                valid = False
                break
            occupied[np_] = i + j
            new_tail.append(np_)
        
        if not valid:
            continue
        
        new_chain = prefix + new_tail
        new_energy = _energy(sequence, new_chain)
        
        return new_chain, new_energy
    
    return None, None


def _energy(sequence, chain):
    """Compute the total energy of the chain. Energy = -num_HH_contacts."""
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
    
    # Phase 2: Local optimization via tail re-growth
    # Try to improve the best chain by re-growing tails at various positions
    current_chain = list(best_chain)
    current_energy = best_energy
    
    time_remaining = _DEADLINE - time.time()
    # Budget for local optimization
    max_local_steps = max(100, min(5000, int(time_remaining * 50)))
    
    # Temperature for simulated annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    for step in range(max_local_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Randomly choose between pivot move and tail re-growth
        use_pivot = rng.random() < 0.4
        
        if use_pivot:
            # Pick a random position for pivot move
            i = rng.randint(2, n - 2)
            new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng)
        else:
            # Pick a random position to re-grow the tail from
            # Bias towards earlier positions for more impact, but also try later ones
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
        
        # Simulated annealing acceptance
        delta = new_energy - current_energy
        if delta < 0:
            # Always accept improvements
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Accept with probability exp(-delta / T)
            import math
            if T > T_min and rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
        
        # Cool down
        T = max(T_min, T * cooling_rate)
    
    # Final high-precision polish pass: use only pivot moves with tight tolerance
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 15:
        polish_steps = max(50, min(2000, int(time_remaining * 30)))
        T_polish = 0.5
        T_polish_min = 0.01
        cooling_rate_polish = 0.9995
        
        for step in range(polish_steps):
            if time.time() > _DEADLINE - 3:
                break
            
            i = rng.randint(2, n - 2)
            new_chain, new_energy = _pivot_move(sequence, current_chain, i, rng)
            if new_chain is None:
                continue
            
            delta = new_energy - current_energy
            if delta < 0:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            else:
                import math
                if T_polish > T_polish_min and rng.random() < math.exp(-delta / T_polish):
                    current_chain = new_chain
                    current_energy = new_energy
            
            T_polish = max(T_polish_min, T_polish * cooling_rate_polish)
    
    # Ensure we return the best chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

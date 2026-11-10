"""HP lattice protein folding on the cubic lattice.
Uses simulated annealing with connectivity-preserving moves and time-bounded search.
"""
import random
import time
import math

# Global deadline guard: total wall-clock budget with safety margin
_TOTAL_BUDGET = 300.0  # seconds
_DEADLINE = time.time() + _TOTAL_BUDGET - 10.0  # 10s safety margin


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-consecutive, adjacent lattice sites)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    # Count each pair only once: only count if j > i
                    if j > i:
                        contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk (greedy with backtracking)."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    return chain


def _initial_chain(sequence, dimension, rng):
    """Generate an initial self-avoiding chain. Try random walks, fall back to spiral."""
    n = len(sequence)
    for _ in range(50):
        chain = _random_walk(sequence, dimension, rng)
        if chain is not None:
            return chain
    # Fallback: simple spiral in xy-plane
    chain = []
    occupied = set()
    x, y, z = 0, 0, 0
    dx, dy = 1, 0
    for i in range(n):
        if (x, y, z) in occupied:
            break
        chain.append((x, y, z))
        occupied.add((x, y, z))
        # Spiral direction changes
        if i > 0:
            # Determine next step in a simple pattern
            if i % (2 * (i // 2 + 1)) == 0 and i > 0:
                pass
        # Simple: move in a pattern
        if i == 0:
            dx, dy = 1, 0
        elif i == 1:
            dx, dy = 0, 1
        elif i == 2:
            dx, dy = -1, 0
        elif i == 3:
            dx, dy = -1, 0
        elif i == 4:
            dx, dy = 0, -1
        elif i == 5:
            dx, dy = 0, -1
        elif i == 6:
            dx, dy = 1, 0
        elif i == 7:
            dx, dy = 1, 0
        elif i == 8:
            dx, dy = 1, 0
        elif i == 9:
            dx, dy = 0, 1
        elif i == 10:
            dx, dy = 0, 1
        elif i == 11:
            dx, dy = -1, 0
        elif i == 12:
            dx, dy = -1, 0
        elif i == 13:
            dx, dy = -1, 0
        elif i == 14:
            dx, dy = 0, -1
        elif i == 15:
            dx, dy = 0, -1
        elif i == 16:
            dx, dy = 0, -1
        elif i == 17:
            dx, dy = 1, 0
        elif i == 18:
            dx, dy = 1, 0
        elif i == 19:
            dx, dy = 1, 0
        else:
            # Continue spiraling
            pass
        x += dx
        y += dy
    # If spiral failed, use a straight line
    if len(chain) < n:
        chain = [(i, 0, 0) for i in range(n)]
    return chain


def _sa_fold(sequence, dimension, rng, time_limit):
    """Simulated annealing for HP chain folding."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]

    # Generate initial chain
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None or len(chain) < n:
        chain = [(i, 0, 0) for i in range(n)]

    # Build occupancy map
    occupied = {p: i for i, p in enumerate(chain)}

    # Compute initial energy
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    # Annealing schedule
    t_start = time.time()
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995

    while time.time() < t_start + time_limit and T > T_min:
        # Pick a random move: pick a random segment to move
        # Try to move a single residue (pivot move) or a small segment
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move: rotate a segment of length 2-5
            seg_len = rng.randint(2, min(5, n - 1))
            start_idx = rng.randint(0, n - 1 - seg_len)
            
            # Get the segment
            segment = chain[start_idx:start_idx + seg_len + 1]
            # The pivot point is at start_idx
            pivot = segment[0]
            
            # Try all 6 possible directions for the first step of the moved segment
            # Actually, for a pivot move, we rotate the entire segment around the pivot
            # This is complex; let's use a simpler approach: move a single bead
            
            # Skip for now, use single-bead moves
            pass
        
        # Single bead move: pick a random interior bead and try to move it to an adjacent empty site
        idx = rng.randint(1, n - 2)  # Interior beads only
        old_pos = chain[idx]
        
        # Find empty neighbours
        candidates = []
        for nb in _neighbours(old_pos):
            if nb not in occupied:
                # Check that moving this bead doesn't create a bond violation
                # The bead at idx is connected to idx-1 and idx+1
                # After move, it must still be adjacent to both
                prev_pos = chain[idx - 1]
                next_pos = chain[idx + 1]
                # Check adjacency
                if sum(abs(a - b) for a, b in zip(nb, prev_pos)) == 1 and \
                   sum(abs(a - b) for a, b in zip(nb, next_pos)) == 1:
                    candidates.append(nb)
        
        if not candidates:
            T *= cooling_rate
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute energy change
        # Remove old contacts involving this bead
        old_contacts = 0
        if sequence[idx] == "H":
            for nb in _neighbours(old_pos):
                j = occupied.get(nb)
                if j is not None and abs(j - idx) > 1 and sequence[j] == "H":
                    if j > idx:
                        old_contacts += 1
        
        # Add new contacts
        new_contacts = 0
        if sequence[idx] == "H":
            for nb in _neighbours(new_pos):
                j = occupied.get(nb)
                if j is not None and abs(j - idx) > 1 and sequence[j] == "H":
                    if j > idx:
                        new_contacts += 1
        
        delta_E = -(new_contacts - old_contacts)
        
        # Accept or reject
        if delta_E <= 0 or rng.random() < math.exp(-delta_E / T):
            # Accept move
            del occupied[old_pos]
            occupied[new_pos] = idx
            chain[idx] = new_pos
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain


def _greedy_fold(sequence, dimension, rng):
    """Greedy chain growth with local optimization."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    # Greedy growth
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites with H-H contacts, then free space
            score = 0.0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 2.0
            # Free space bonus
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            score += rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            # Try to find any valid site by backtracking (simplified: just fail)
            return None
        
        options.sort(key=lambda x: -x[0])
        # Sometimes pick a suboptimal option for diversity
        if rng.random() < 0.2 and len(options) > 1:
            idx = rng.randint(0, min(3, len(options) - 1))
            site = options[idx][1]
        else:
            site = options[0][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain


def fold(sequence, dimension):
    """Fold an HP sequence on the cubic lattice.
    
    Args:
        sequence: string of 'H' and 'P' characters
        dimension: 3 for cubic lattice
    
    Returns:
        List of len(sequence) lattice points (lists of length dimension)
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    # Time budget per sequence
    per_seq_time = min(10.0, max(1.0, 250.0 / 26))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worst
    
    # Try multiple strategies
    strategies = [
        ("greedy", _greedy_fold),
        ("sa", _sa_fold),
    ]
    
    for name, func in strategies:
        if time.time() > _DEADLINE:
            break
        
        t_start = time.time()
        time_left = per_seq_time - (time.time() - t_start)
        
        if name == "greedy":
            chain = func(sequence, dimension, rng)
            if chain is not None and len(chain) == n:
                e = _compute_energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain
        elif name == "sa":
            if time_left > 1.0:
                chain = func(sequence, dimension, rng, time_left)
                if chain is not None and len(chain) == n:
                    e = _compute_energy(sequence, chain)
                    if e < best_energy:
                        best_energy = e
                        best_chain = chain
    
    # If no valid chain found, use a straight line
    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

"""New approach: local search with simulated annealing on HP lattice."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the HP contact energy: -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _valid_move(sequence, chain, i, new_pos, occupied_set):
    """Check if placing residue i at new_pos is valid (self-avoiding)."""
    if new_pos in occupied_set:
        return False
    # Check unit step from previous residue
    if i > 0:
        prev = chain[i - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return False
    # Check unit step to next residue
    if i < len(chain) - 1:
        nxt = chain[i + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return False
    return True


def _local_search(sequence, chain, max_iter=50000):
    """Local search with random restarts and simulated annealing."""
    n = len(chain)
    if n <= 1:
        return chain

    # Build initial energy
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    rng = random.Random(42 + n)

    for iteration in range(max_iter):
        if T < T_min:
            break

        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)

        # Try all 4 possible moves for this residue
        current_pos = chain[i]
        moves = []
        for new_pos in _neighbours(current_pos):
            # Check if this move is valid
            occupied = set(chain)
            occupied.remove(current_pos)
            if new_pos in occupied:
                continue
            # Check connectivity with neighbors
            if i > 0:
                prev = chain[i - 1]
                if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                    continue
            if i < n - 1:
                nxt = chain[i + 1]
                if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                    continue
            moves.append(new_pos)

        if not moves:
            continue

        new_pos = rng.choice(moves)

        # Compute new energy
        # Remove old position, add new position
        chain[i] = new_pos
        new_energy = _compute_energy(sequence, chain)
        delta_e = new_energy - current_energy

        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            chain[i] = current_pos

        T *= cooling

    return best_chain


def _initial_chain(sequence, dimension):
    """Create an initial self-avoiding chain using a simple heuristic."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0)}

    # Use a spiral-like pattern for initial conformation
    direction = (1, 0)  # start going right
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    for i in range(1, n):
        placed = False
        for d in dirs:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                # Prefer to turn to maintain compactness
                if i < n - 1:
                    # Rotate direction 90 degrees clockwise
                    dirs = [(d[1], -d[0])] + dirs[1:]
                placed = True
                break

        if not placed:
            # Fallback: try all 4 directions
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    placed = True
                    break

        if not placed:
            # If stuck, just extend in a straight line (shouldn't happen often)
            new_pos = (chain[-1][0] + 1, chain[-1][1])
            while new_pos in occupied:
                new_pos = (new_pos[0] + 1, new_pos[1])
            chain.append(new_pos)
            occupied.add(new_pos)

    return chain


def fold(sequence, dimension):
    """Find a low-energy conformation for the given HP sequence."""
    if dimension != 2:
        raise ValueError("Only 2D is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + n)

    # Try multiple initial conformations and local searches
    best_chain = None
    best_energy = float('inf')

    # Strategy 1: Simple spiral initialization
    chain1 = _initial_chain(sequence, dimension)
    chain1 = _local_search(sequence, chain1, max_iter=30000)
    e1 = _compute_energy(sequence, chain1)
    if e1 < best_energy:
        best_energy = e1
        best_chain = chain1

    # Strategy 2: Random walk initialization
    for attempt in range(5):
        # Generate a random self-avoiding walk
        chain2 = [(0, 0)]
        occupied = {(0, 0)}
        success = True
        for i in range(1, n):
            options = []
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (chain2[-1][0] + d[0], chain2[-1][1] + d[1])
                if new_pos not in occupied:
                    options.append(new_pos)
            if not options:
                success = False
                break
            new_pos = rng.choice(options)
            chain2.append(new_pos)
            occupied.add(new_pos)

        if not success:
            continue

        chain2 = _local_search(sequence, chain2, max_iter=20000)
        e2 = _compute_energy(sequence, chain2)
        if e2 < best_energy:
            best_energy = e2
            best_chain = chain2

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
    else:
        best_chain = [[p[0], p[1]] for p in best_chain]

    return best_chain
# EVOLVE-BLOCK-END

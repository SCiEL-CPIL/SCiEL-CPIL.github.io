"""HP lattice protein folding using Hamiltonian path skeletons with local search."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = 210.0  # wall-clock limit
_START_TIME = time.time()


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


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


def _generate_hamiltonian_path(width, height):
    """Generate a Hamiltonian path on a width x height grid using a snake pattern."""
    path = []
    for row in range(height):
        if row % 2 == 0:
            for col in range(width):
                path.append((col, row))
        else:
            for col in range(width - 1, -1, -1):
                path.append((col, row))
    return path


def _try_fit_sequence_to_grid(sequence, width, height):
    """Try to map the sequence onto a width x height grid via Hamiltonian path.
    Returns a chain (list of points) if the grid has exactly len(sequence) cells,
    or None if not."""
    n = len(sequence)
    if width * height != n:
        return None
    path = _generate_hamiltonian_path(width, height)
    # Shift to non-negative coordinates
    min_x = min(p[0] for p in path)
    min_y = min(p[1] for p in path)
    chain = [(p[0] - min_x, p[1] - min_y) for p in path]
    return chain


def _find_grid_dimensions(n):
    """Find grid dimensions (width, height) such that width * height == n,
    preferring square-ish grids. Returns a list of (width, height) pairs."""
    dims = []
    for w in range(1, int(n**0.5) + 1):
        if n % w == 0:
            h = n // w
            dims.append((w, h))
            if w != h:
                dims.append((h, w))
    # Sort by how square-ish (minimize max(w,h)/min(w,h))
    dims.sort(key=lambda d: max(d) / max(min(d), 1))
    return dims


def _pivot_move(chain, i, j, sequence, occupied, where):
    """Try a pivot move: rotate the subchain from i to j around chain[i-1].
    Returns new chain if valid, else None."""
    if i == 0:
        # Rotate around the start: try all 4 orientations of the segment
        segment = chain[i:j+1]
        # Try rotating the segment in different directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_segment = [segment[0]]
            for k in range(1, len(segment)):
                prev = new_segment[-1]
                # Try to extend in the pivot direction
                cand = (prev[0] + dx, prev[1] + dy)
                if cand in occupied and cand != chain[i-1] if i > 0 else True:
                    # Check if it's the attachment point
                    if i > 0 and cand == chain[i-1]:
                        pass
                    else:
                        continue
                # Check self-avoidance
                if cand in occupied and cand != chain[i-1] if i > 0 else (cand in occupied):
                    continue
                new_segment.append(cand)
            # Verify the whole new chain is valid
            new_chain = chain[:i] + new_segment + chain[j+1:]
            new_occupied = set(new_chain)
            if len(new_occupied) != len(new_chain):
                continue
            # Verify unit steps
            valid = True
            for k in range(len(new_chain) - 1):
                diff = abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1])
                if diff != 1:
                    valid = False
                    break
            if not valid:
                continue
            return new_chain
        return None
    else:
        # Standard pivot: rotate segment i..j around chain[i-1]
        pivot = chain[i-1]
        segment = chain[i:j+1]
        # Try all 4 rotations of the segment around the pivot
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_segment = [segment[0]]
            valid = True
            for k in range(1, len(segment)):
                prev = new_segment[-1]
                cand = (prev[0] + dx, prev[1] + dy)
                # Check if cand is in occupied (excluding the segment itself and pivot)
                if cand in occupied:
                    # Allow if it's the pivot point or part of the segment being moved
                    if cand == pivot:
                        pass
                    else:
                        valid = False
                        break
                new_segment.append(cand)
            if not valid:
                continue
            new_chain = chain[:i] + new_segment + chain[j+1:]
            # Check self-avoidance
            new_occupied = set(new_chain)
            if len(new_occupied) != len(new_chain):
                continue
            # Check unit steps
            valid = True
            for k in range(len(new_chain) - 1):
                diff = abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1])
                if diff != 1:
                    valid = False
                    break
            if not valid:
                continue
            return new_chain
        return None


def _local_search(sequence, chain, max_iters=5000):
    """Local search using pivot moves and 2-opt style moves."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    rng = random.Random(42 + len(sequence))
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        # Try a pivot move at random position
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, min(i + rng.randint(2, min(10, n - i)), n - 1))
        occupied = set(current_chain)
        new_chain = _pivot_move(current_chain, i, j, sequence, occupied, None)
        if new_chain is not None:
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
            elif rng.random() < 0.1:  # occasional random jump
                current_chain = new_chain
                current_energy = new_energy
        else:
            # Try a simple move: try to move a single residue
            k = rng.randint(1, n - 2)
            occupied = set(current_chain)
            occupied.discard(current_chain[k])
            for site in _neighbours(current_chain[k]):
                if site in occupied:
                    continue
                # Check unit steps with neighbors
                if k > 0:
                    diff = abs(site[0] - current_chain[k-1][0]) + abs(site[1] - current_chain[k-1][1])
                    if diff != 1:
                        continue
                if k < n - 1:
                    diff = abs(site[0] - current_chain[k+1][0]) + abs(site[1] - current_chain[k+1][1])
                    if diff != 1:
                        continue
                new_chain = list(current_chain)
                new_chain[k] = site
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_chain = list(current_chain)
                        best_energy = current_energy
                break
    
    return best_chain, best_energy


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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


def fold(sequence, dimension):
    """Fold a sequence using multiple strategies: Hamiltonian skeletons + local search + greedy."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Hamiltonian path skeletons
    dims = _find_grid_dimensions(n)
    for w, h in dims[:5]:  # Try up to 5 grid shapes
        if _time_remaining() < 10.0:
            break
        chain = _try_fit_sequence_to_grid(sequence, w, h)
        if chain is None:
            continue
        # Apply local search
        improved_chain, improved_energy = _local_search(sequence, chain, max_iters=3000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Strategy 2: Greedy growth with random restarts
    rng = random.Random(12345 + len(sequence))
    num_greedy = max(50, min(200, int(_time_remaining() * 10)))
    for _ in range(num_greedy):
        if _time_remaining() < 5.0:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
        # Local search on the greedy result
        if _time_remaining() > 10.0:
            improved_chain, improved_energy = _local_search(sequence, chain, max_iters=1000)
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Final local search polish
    if best_chain is not None and _time_remaining() > 5.0:
        polished_chain, polished_energy = _local_search(best_chain, best_energy if best_chain else 0, max_iters=2000)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

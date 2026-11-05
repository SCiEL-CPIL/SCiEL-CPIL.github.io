"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline: 200s wall-clock (safety margin before 210s kill)
_DEADLINE = time.time() + 200.0


def _neighbours(point):
    """Yield the 4 neighbours of a lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP contact energy: -number of H-H non-consecutive neighbours."""
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


def _local_search(sequence, chain, rng, max_iter=5000):
    """Pivot-move local search to improve a conformation."""
    n = len(chain)
    if n < 4:
        return chain
    occupied = {p: i for i, p in enumerate(chain)}
    best_energy = _energy(sequence, chain)
    current = list(chain)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        # Pick a random pivot index (not too close to ends)
        pivot = rng.randint(2, n - 3)
        old_pos = current[pivot]
        # Try moving the pivot to a neighbouring site
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            # Check self-avoidance: new_pos must not be adjacent to any other chain segment
            # except the immediate neighbours (pivot-1, pivot+1)
            valid = True
            for q in _neighbours(new_pos):
                idx = occupied.get(q)
                if idx is not None and abs(idx - pivot) > 1:
                    valid = False
                    break
            if not valid:
                continue
            # Perform the move
            current[pivot] = new_pos
            occupied[new_pos] = pivot
            del occupied[old_pos]
            
            # Compute energy delta
            # Remove old contacts involving pivot
            old_contacts = 0
            if sequence[pivot] == "H":
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and j > pivot + 1 and sequence[j] == "H":
                        old_contacts += 1
                    elif j is not None and j < pivot - 1 and sequence[j] == "H":
                        old_contacts += 1
                # Also check if any H at new_pos neighbours had contacts with old_pos
                # This is complex; just recompute energy for simplicity in small cases
                # For efficiency, let's just compute full energy
            new_energy = _energy(sequence, current)
            
            if new_energy < best_energy:
                best_energy = new_energy
            elif new_energy > best_energy:
                # Revert
                current[pivot] = old_pos
                occupied[old_pos] = pivot
                del occupied[new_pos]
            else:
                # Accept (neutral move)
                pass
    return current


def _generate_skeleton(n, dimension, rng):
    """Generate a space-filling curve skeleton of length n."""
    # Use a simple serpentine pattern
    # Determine grid dimensions
    width = int(math.ceil(math.sqrt(n)))
    height = int(math.ceil(n / width))
    points = []
    for row in range(height):
        for col in range(width):
            if len(points) >= n:
                break
            if row % 2 == 0:
                points.append((col, row))
            else:
                points.append((width - 1 - col, row))
    # Shuffle slightly to avoid deterministic bias
    return points[:n]


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    # Strategy 1: Multiple greedy growths with local search
    num_growths = min(50, max(10, 300 // max(1, n // 10)))
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Apply local search
        chain = _local_search(sequence, chain, rng, max_iter=2000)
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Strategy 2: Skeleton-based approach for longer sequences
    if n > 20 and time.time() < _DEADLINE - 20:
        skeleton = _generate_skeleton(n, dimension, rng)
        # Map sequence onto skeleton and do local search
        chain = list(skeleton)
        chain = _local_search(sequence, chain, rng, max_iter=5000)
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Fallback: straight line
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

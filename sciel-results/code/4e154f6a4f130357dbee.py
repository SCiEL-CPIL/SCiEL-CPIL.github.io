"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -1 per non-consecutive H-H contact."""
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
    """One randomized greedy chain growth."""
    start = (0, 0)
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


def _is_valid(sequence, chain):
    """Check self-avoidance and unit steps."""
    if len(chain) != len(sequence):
        return False
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def _local_search(sequence, chain, rng, max_iterations=5000):
    """Simple local search with single-residue moves and reversals."""
    if not _is_valid(sequence, chain):
        return chain, _energy(sequence, chain)
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    n = len(current)
    if n < 3:
        return best, best_energy
    
    occupied = {p: i for i, p in enumerate(current)}
    
    for iteration in range(max_iterations):
        if _remaining_time() < 2.0:
            break
        
        # Try a reversal move
        if n >= 3:
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            if j - i >= 2:
                # Check if reversal is valid
                new_chain = current[:i+1] + current[i+1:j+1][::-1] + current[j+1:]
                # Check self-avoidance for the reversed segment
                ok = True
                # The segment from i+1 to j is reversed
                # We need to check that the reversed segment doesn't hit other parts
                fixed_set = set(current[:i+1]) | set(current[j+1:])
                for k in range(i+1, j+1):
                    if new_chain[k] in fixed_set:
                        ok = False
                        break
                if ok:
                    # Also check that the connections still work
                    # i to i+1: current[i] to current[j] (was current[i+1])
                    # j to j+1: current[i+1] (now at j) to current[j+1]
                    dx1 = abs(current[i][0] - current[j][0]) + abs(current[i][1] - current[j][1])
                    dx2 = abs(current[i+1][0] - current[j+1][0]) + abs(current[i+1][1] - current[j+1][1])
                    if dx1 == 1 and dx2 == 1:
                        new_energy = _energy(sequence, new_chain)
                        if new_energy <= current_energy or rng.random() < 0.1:
                            current = new_chain
                            current_energy = new_energy
                            if current_energy < best_energy:
                                best = list(current)
                                best_energy = current_energy
                            occupied = {p: idx for idx, p in enumerate(current)}
        
        # Try a simple move: rotate one residue
        if n >= 2:
            i = rng.randint(1, n - 1)
            old_pos = current[i]
            for new_pos in _neighbours(old_pos):
                if new_pos == current[i-1] or new_pos == current[i+1]:
                    continue
                if new_pos in occupied:
                    continue
                # Check that the new position doesn't cause issues
                # Remove old, add new
                test_chain = list(current)
                test_chain[i] = new_pos
                if _is_valid(sequence, test_chain):
                    new_energy = _energy(sequence, test_chain)
                    if new_energy <= current_energy or rng.random() < 0.05:
                        current = test_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best = list(current)
                            best_energy = current_energy
                        occupied = {p: idx for idx, p in enumerate(current)}
                        break
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple growth strategies and local search."""
    if dimension != 2:
        # Fallback for non-2D
        return [[i, 0] for i in range(len(sequence))]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    n = len(sequence)
    
    # Phase 1: Multiple greedy growths
    num_growths = 300 if n <= 50 else 150
    for _ in range(num_growths):
        if _remaining_time() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        if not _is_valid(sequence, chain):
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Straight line fallback
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    if _remaining_time() > 3.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, max_iterations=10000)
    
    # Phase 3: Try a few more growths and local search if time permits
    if _remaining_time() > 10.0:
        for _ in range(50):
            if _remaining_time() < 5.0:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            if not _is_valid(sequence, chain):
                continue
            chain, energy = _local_search(sequence, chain, rng, max_iterations=5000)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

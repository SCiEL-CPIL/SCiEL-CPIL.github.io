"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
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


def _reverse_move(chain, i, j):
    """Try reversing the segment from i to j (inclusive). Return new chain or None."""
    if j - i < 2:
        return None
    # Check self-avoidance: the reversed segment must not collide with itself or outside
    # Simple check: verify no collisions in the new positions
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    # Check unit steps
    for k in range(len(new_chain) - 1):
        diff = sum(abs(a - b) for a, b in zip(new_chain[k], new_chain[k+1]))
        if diff != 1:
            return None
    return new_chain


def _rotate_move(chain, i, j, rng):
    """Try rotating the segment from i to j by 90 degrees around chain[i]. Return new chain or None."""
    if j - i < 2:
        return None
    center = chain[i]
    # Rotation directions: 90, 180, 270 degrees
    for angle in (1, 2, 3):
        new_chain = chain[:i]
        for k in range(i, j+1):
            dx = chain[k][0] - center[0]
            dy = chain[k][1] - center[1]
            if angle == 1:  # 90 deg CCW
                nx, ny = -dy, dx
            elif angle == 2:  # 180 deg
                nx, ny = -dx, -dy
            else:  # 270 deg CCW
                nx, ny = dy, -dx
            new_chain.append((center[0] + nx, center[1] + ny))
        new_chain.extend(chain[j+1:])
        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        if not valid:
            continue
        # Check unit steps
        for k in range(len(new_chain) - 1):
            diff = sum(abs(a - b) for a, b in zip(new_chain[k], new_chain[k+1]))
            if diff != 1:
                valid = False
                break
        if valid:
            return new_chain
    return None


def _anneal(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing to improve a chain."""
    start_time = time.time()
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    temperature = 5.0
    cooling = 0.995
    
    while time.time() - start_time < time_budget and temperature > 0.01:
        # Choose move type
        move_type = rng.random()
        if move_type < 0.6:
            # Reverse move
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            new_chain = _reverse_move(chain, i, j)
        else:
            # Rotate move
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            new_chain = _rotate_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < exp(-delta / temperature):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        temperature *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Time allocation: use more time for longer sequences
    base_time = 2.0
    scale = max(1, n // 20)
    total_budget = min(_remaining_time(), base_time * scale)
    
    if total_budget < 1.0:
        # Not enough time for annealing, just do greedy
        best, best_energy = None, 1
        for _ in range(50):
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
        if best is None:
            best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]
    
    # Generate initial chains and anneal the best ones
    num_growths = min(20, max(5, n // 5))
    candidates = []
    for _ in range(num_growths):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    if not candidates:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]
    
    # Evaluate and sort candidates
    candidates_with_energy = []
    for chain in candidates:
        e = _energy(sequence, chain)
        candidates_with_energy.append((e, chain))
    candidates_with_energy.sort(key=lambda x: x[0])
    
    # Anneal top candidates with time budget
    num_to_anneal = min(len(candidates_with_energy), max(1, int(total_budget / 3.0)))
    time_per_anneal = total_budget / max(1, num_to_anneal)
    
    best_chain = candidates_with_energy[0][1]
    best_energy = candidates_with_energy[0][0]
    
    for idx in range(num_to_anneal):
        if _remaining_time() < 5.0:
            break
        e, chain = candidates_with_energy[idx]
        if e >= best_energy and idx > 0:
            break  # No point annealing worse candidates
        
        anneal_budget = min(time_per_anneal, _remaining_time() - 5.0)
        if anneal_budget < 0.5:
            break
        
        improved_chain, improved_energy = _anneal(sequence, chain, dimension, rng, anneal_budget)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

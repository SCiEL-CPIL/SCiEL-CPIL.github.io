"""HP lattice protein folding - local search with Monte Carlo moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 295  # safety margin within 330s wall-clock


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


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk as initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        if not candidates:
            return None
        site = rng.choice(candidates)
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, pivot, rng, dimension):
    """Apply a pivot move: rotate the tail after pivot."""
    n = len(chain)
    if pivot >= n - 1:
        return None
    
    pivot_pt = chain[pivot]
    tail = chain[pivot + 1:]
    
    # The bond from pivot to first tail point
    bond = [tail[0][a] - pivot_pt[a] for a in range(dimension)]
    bond_axis = None
    for a in range(dimension):
        if bond[a] != 0:
            bond_axis = a
            break
    if bond_axis is None:
        return None
    
    # Choose rotation axis perpendicular to bond
    other_axes = [a for a in range(dimension) if a != bond_axis]
    rot_axis = rng.choice(other_axes)
    
    # Rotate tail by 90 degrees around rot_axis through pivot_pt
    new_tail = []
    for pt in tail:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        new_rel = list(rel)
        if rot_axis == 0:
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_tail.append(new_pt)
    
    # Check self-avoidance
    occupied = set(chain[:pivot + 1])
    for pt in new_tail:
        if pt in occupied:
            return None
        occupied.add(pt)
    
    return chain[:pivot + 1] + new_tail


def _local_search(sequence, chain, rng, time_budget):
    """Monte Carlo local search to improve conformation energy."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 2.0
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        # Choose pivot point
        pivot = rng.randint(0, n - 2)
        new_chain = _pivot_move(current, pivot, rng, 3)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0:
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            if temp > 0.05:
                prob = math.exp(-delta / temp)
                if rng.random() < prob:
                    current = new_chain
                    current_energy = new_energy
        
        temp *= cooling
    
    return best


def fold(sequence, dimension):
    """Best conformation using local search from multiple initial conformations."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    remaining = _DEADLINE - time.time()
    if remaining < 2.0:
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Time budget for this sequence
    total_budget = min(12.0, 2.0 + 0.12 * n)
    total_budget = min(total_budget, remaining * 0.85)
    
    best_chain = None
    best_energy = 0
    
    n_initial = max(3, min(8, int(total_budget / 1.5)))
    
    for init_i in range(n_initial):
        if time.time() > _DEADLINE - 5:
            break
        
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            continue
        
        if len(chain) != n or len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if sum(abs(a - b) for a, b in zip(chain[i], chain[i + 1])) != 1:
                valid = False
                break
        if not valid:
            continue
        
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        time_left = _DEADLINE - time.time()
        if time_left < 3.0:
            break
        search_budget = min(2.5, time_left * 0.75 / max(1, n_initial - init_i))
        improved = _local_search(sequence, chain, rng, search_budget)
        imp_energy = _energy(sequence, improved)
        if imp_energy < best_energy:
            best_chain = improved
            best_energy = imp_energy
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

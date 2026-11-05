"""HP lattice protein folding with tail-regeneration SA."""
# EVOLVE-BLOCK-START
import random
import time

_ABS_DEADLINE = None

def _neighbours(point):
    d = len(point)
    for axis in range(d):
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

def _regrow_tail(sequence, dimension, prefix, occupied_prefix, rng):
    """Randomly regrow the tail from the end of prefix. Returns list or None."""
    chain = list(prefix)
    occupied = set(occupied_prefix)
    for i in range(len(prefix), len(sequence)):
        last = chain[-1]
        options = [s for s in _neighbours(last) if s not in occupied]
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
    return chain

def _pivot_move(sequence, dimension, chain, rng):
    """Rotate a sub-chain at a pivot; return new chain or None if invalid."""
    n = len(chain)
    if n < 3:
        return None
    pivot = rng.randint(1, n - 2)
    old_pos = chain[pivot + 1]
    prefix_set = set(chain[:pivot + 1])
    candidates = [s for s in _neighbours(chain[pivot]) if s not in prefix_set]
    if not candidates:
        return None
    new_pos = rng.choice(candidates)
    if new_pos == old_pos:
        return None
    dx = new_pos[0] - old_pos[0]
    dy = new_pos[1] - old_pos[1]
    dz = new_pos[2] - old_pos[2]
    new_chain = chain[:pivot + 1] + [new_pos]
    occupied_set = set(new_chain)
    for i in range(pivot + 2, n):
        shifted = (chain[i][0] + dx, chain[i][1] + dy, chain[i][2] + dz)
        if shifted in occupied_set:
            return None
        occupied_set.add(shifted)
        new_chain.append(shifted)
    return new_chain

def _tail_regen_move(sequence, dimension, chain, rng):
    """Cut at a random pivot and regenerate the entire tail (tunneling move)."""
    n = len(chain)
    if n < 4:
        return None
    min_cut = max(1, n // 4)
    max_cut = n - 3
    if min_cut >= max_cut:
        return None
    pivot = rng.randint(min_cut, max_cut)
    prefix = chain[:pivot + 1]
    occupied_prefix = set(prefix)
    new_chain = _regrow_tail(sequence, dimension, prefix, occupied_prefix, rng)
    return new_chain

def _sa_fold(sequence, dimension, rng, time_limit):
    """Simulated annealing with tail-regeneration moves."""
    global _ABS_DEADLINE
    n = len(sequence)
    # Initial fold: random walk
    chain = None
    for _ in range(50):
        chain = _regrow_tail(sequence, dimension, [(0, 0, 0)], {(0, 0, 0)}, rng)
        if chain is not None:
            break
    if chain is None:
        chain = [(i, 0, 0) for i in range(n)]
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 2.0
    T_min = 0.01
    
    while time.time() - start_time < time_limit:
        if time.time() > _ABS_DEADLINE:
            break
        if rng.random() < 0.3:
            new_chain = _tail_regen_move(sequence, dimension, chain, rng)
        else:
            new_chain = _pivot_move(sequence, dimension, chain, rng)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        T *= 0.999
        if T < T_min:
            T = T_min
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation via multi-start SA with tail regeneration."""
    global _ABS_DEADLINE
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    _ABS_DEADLINE = time.time() + 300.0  # safety margin below 330s kill
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    starts = 0
    while time.time() < _ABS_DEADLINE - 5:
        remaining = _ABS_DEADLINE - time.time() - 2
        if remaining < 0.5:
            break
        if starts == 0:
            t_start = min(remaining * 0.3, 30)
        elif starts < 5:
            t_start = min(remaining * 0.15, 20)
        else:
            t_start = min(remaining * 0.1, 15)
        t_start = max(t_start, 0.5)
        
        chain, energy = _sa_fold(sequence, dimension, rng, t_start)
        if chain is not None:
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        starts += 1
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

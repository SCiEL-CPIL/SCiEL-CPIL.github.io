"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 310  # safety margin before 330s kill


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


def _grow(sequence, dimension, rng, bias=0.0):
    """One randomized greedy chain growth with pocket-aware scoring."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
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
            # Penalize dead-ends: if free == 0, this is a trap
            dead_end_penalty = -2.0 if free == 0 else 0.0
            score = gain + 0.15 * free + dead_end_penalty + bias * rng.random() + (1 - bias) * rng.random() * 0.4
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, i, rng, dimension):
    """Fast pivot move with early collision abort."""
    if i <= 1 or i >= len(chain) - 1:
        return None
    
    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]
    
    axis = rng.randint(0, dimension - 1)
    sign = rng.choice([-1, 1])
    
    new_tail = []
    seen_in_tail = set()
    head_set = set(head)
    
    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        if axis == 0:
            new_vec = [vec[0], -sign * vec[2], sign * vec[1]]
        elif axis == 1:
            new_vec = [sign * vec[2], vec[1], -sign * vec[0]]
        else:
            new_vec = [-sign * vec[1], sign * vec[0], vec[2]]
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))
        
        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)
        
    return head + new_tail


def _kink_move(chain, i, rng, dimension):
    """Try to move residue i to an adjacent free site (kink move)."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    # Check that moving residue i doesn't break connectivity
    # Residue i must remain adjacent to i-1 and i+1 after move
    # Actually, a kink move moves residue i to a neighbor site, 
    # but we need i-1 and i+1 to still be adjacent to the new position
    # This is tricky; simpler: try all 6 neighbors of chain[i]
    occupied = set(chain)
    for q in _neighbours(chain[i]):
        if q in occupied:
            continue
        # Check that q is adjacent to chain[i-1] and chain[i+1]
        # Actually no - in a kink move, we move residue i, so i-1 and i+1 
        # must be adjacent to the new position of i
        # But i-1 is adjacent to old i, and i+1 is adjacent to old i.
        # After moving i to q, we need i-1 adjacent to q AND i+1 adjacent to q
        # This is very restrictive. Let's just try simple single-residue moves
        # where we remove i and reinsert at a free neighbor of i-1 or i+1
        pass
    
    # Alternative: try moving the last few residues
    # For simplicity, let's do a "retraction" move: move the tail start
    # Actually, let's just return None for now and rely on pivot moves
    return None


def _local_search(sequence, chain, dimension, rng, max_iters=300):
    """Fast local search: accept only improving moves."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        i = rng.randint(2, len(current) - 2)
        new_chain = _pivot_move_fast(current, i, rng, dimension)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        # else: reject
    
    return best, best_energy


def _anneal(sequence, chain, dimension, rng, max_iters=500, T_start=3.0, T_end=0.01):
    """Simulated annealing refinement of a chain with adaptive cooling."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    if max_iters > 1:
        alpha = (T_end / T_start) ** (1.0 / max_iters)
    else:
        alpha = 1.0
    T = T_start
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        i = rng.randint(2, len(current) - 2)
        new_chain = _pivot_move_fast(current, i, rng, dimension)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or (T > 0.001 and rng.random() < math.exp(-delta / T)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        T *= alpha
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    
    n = len(sequence)
    
    # Determine search effort based on chain length
    # Shorter chains: more restarts, shorter refinement
    # Longer chains: fewer restarts, longer refinement
    if n <= 48:
        n_restarts = 300
        ls_iters = 150
        anneal_iters = 400
    elif n <= 80:
        n_restarts = 180
        ls_iters = 200
        anneal_iters = 500
    else:
        n_restarts = 100
        ls_iters = 250
        anneal_iters = 600
    
    # Phase 1: Quick local search on multiple restarts with varying bias
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 80:
            break
        # Vary bias to explore different growth strategies
        # Early restarts: more greedy; later: more random
        bias = 0.1 + 0.8 * (r / max(1, n_restarts - 1))
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=ls_iters)
        
        if best is None or refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    if best is None:
        # Fallback: try harder with pure greedy
        for r in range(20):
            chain = _grow(sequence, dimension, rng, bias=0.0)
            if chain is not None:
                best, best_energy = chain, _energy(sequence, chain)
                break
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
        return [list(p) for p in best]
    
    # Phase 2: Deep annealing on the best solution
    if time.time() < _DEADLINE - 40:
        refined, refined_energy = _anneal(sequence, best, dimension, rng, 
                                           max_iters=anneal_iters, T_start=4.0, T_end=0.005)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    # Phase 3: Try a few fresh starts with annealing if time permits
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 30:
        n_extra = min(4, int(time_remaining / 15))
        for extra in range(n_extra):
            if time.time() > _DEADLINE - 15:
                break
            # Use different bias for diversity
            bias = 0.3 + 0.4 * rng.random()
            chain = _grow(sequence, dimension, rng, bias=bias)
            if chain is None:
                continue
            # Quick local search first
            refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=100)
            # Then anneal if still improving
            if time.time() < _DEADLINE - 10:
                refined, refined_energy = _anneal(sequence, refined, dimension, rng,
                                                   max_iters=300, T_start=3.0, T_end=0.01)
            if refined_energy < best_energy:
                best, best_energy = refined, refined_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

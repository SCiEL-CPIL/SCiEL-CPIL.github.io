"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_DEADLINE = time.time() + 315  # safety margin before 330s kill


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


def _reversal_move(chain, i, j, rng):
    """Reverse a segment of the chain from i to j (inclusive)."""
    if i < 2 or j >= len(chain) - 1 or j <= i + 1:
        return None
    
    if i > 0 and j < len(chain) - 1:
        d1 = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(len(chain[i])))
        d2 = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(len(chain[i])))
        if d1 != 1 or d2 != 1:
            return None
    
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    
    if len(set(new_chain)) != len(new_chain):
        return None
    
    return new_chain


def _kink_move(chain, i, rng, dimension):
    """Try to move residue i to an adjacent free site (kink move)."""
    if i <= 0 or i >= len(chain) - 1:
        return None
    
    occupied = set(chain)
    occupied.discard(chain[i])
    
    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    
    for site in _neighbours(chain[i]):
        if site in occupied:
            continue
        
        # Check if site is adjacent to both neighbors to maintain chain connectivity
        dist_prev = sum(abs(site[k] - prev_pos[k]) for k in range(dimension))
        dist_next = sum(abs(site[k] - next_pos[k]) for k in range(dimension))
        
        if dist_prev == 1 and dist_next == 1:
            new_chain = list(chain)
            new_chain[i] = site
            return new_chain
            
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
        move_type = rng.random()
        if move_type < 0.5:
            i = rng.randint(2, len(current) - 2)
            new_chain = _pivot_move_fast(current, i, rng, dimension)
        elif move_type < 0.8:
            i = rng.randint(2, len(current) - 3)
            j = rng.randint(i + 1, min(i + 6, len(current) - 2))
            new_chain = _reversal_move(current, i, j, rng)
        else:
            i = rng.randint(1, len(current) - 2)
            new_chain = _kink_move(current, i, rng, dimension)
        
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
    
    return best, best_energy


def _anneal(sequence, chain, dimension, rng, max_iters=500, T_start=4.0, T_end=0.005):
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
        move_type = rng.random()
        if move_type < 0.55:
            i = rng.randint(2, len(current) - 2)
            new_chain = _pivot_move_fast(current, i, rng, dimension)
        elif move_type < 0.85:
            i = rng.randint(2, len(current) - 3)
            j = rng.randint(i + 1, min(i + 8, len(current) - 2))
            new_chain = _reversal_move(current, i, j, rng)
        else:
            i = rng.randint(1, len(current) - 2)
            new_chain = _kink_move(current, i, rng, dimension)
        
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


def _beam_search(sequence, dimension, rng, beam_width=50, max_depth=60):
    """Beam search over partial chains with contact-aware scoring."""
    n = len(sequence)
    if n > max_depth:
        return None
    
    start = (0,) * dimension
    # State: (score, chain, occupied_set)
    # Higher score is better
    beam = [(0, [start], {start: 0})]
    
    for i in range(1, n):
        if time.time() > _DEADLINE - 30:
            return None
        candidates = []
        for score, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                new_score = score + gain * 2 + 0.1 * free
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_chain = chain + [site]
                candidates.append((new_score, new_chain, new_occupied))
        
        if not candidates:
            return None
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: -x[0])
        beam = candidates[:beam_width]
    
    if not beam:
        return None
    
    # Return the best complete chain
    return beam[0][1]


def fold(sequence, dimension):
    """Best conformation using beam search, greedy growth, and simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    
    n = len(sequence)
    
    # Determine search effort based on chain length
    if n <= 48:
        n_restarts = 300
        ls_iters = 200
        anneal_iters = 600
        use_beam = True
    elif n <= 80:
        n_restarts = 180
        ls_iters = 250
        anneal_iters = 700
        use_beam = True
    else:
        n_restarts = 100
        ls_iters = 300
        anneal_iters = 800
        use_beam = False
    
    # Phase 0: Beam search for shorter chains
    if use_beam and n <= 64:
        beam_chain = _beam_search(sequence, dimension, rng, beam_width=80, max_depth=n)
        if beam_chain is not None:
            refined, refined_energy = _local_search(sequence, beam_chain, dimension, rng, max_iters=ls_iters)
            best, best_energy = refined, refined_energy
    
    # Phase 1: Quick local search on multiple restarts with varying bias
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 90:
            break
        progress = r / max(1, n_restarts - 1)
        bias = 0.05 + 0.85 * progress
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=ls_iters)
        
        if best is None or refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    if best is None:
        for r in range(30):
            chain = _grow(sequence, dimension, rng, bias=0.0)
            if chain is not None:
                best, best_energy = chain, _energy(sequence, chain)
                break
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
        return [list(p) for p in best]
    
    # Phase 2: Deep annealing on the best solution
    if time.time() < _DEADLINE - 50:
        refined, refined_energy = _anneal(sequence, best, dimension, rng,
                                           max_iters=anneal_iters, T_start=5.0, T_end=0.003)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    # Phase 3: Try fresh starts with annealing if time permits
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 40:
        n_extra = min(8, int(time_remaining / 10))
        for extra in range(n_extra):
            if time.time() > _DEADLINE - 20:
                break
            bias = 0.2 + 0.5 * rng.random()
            chain = _grow(sequence, dimension, rng, bias=bias)
            if chain is None:
                continue
            refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=150)
            if time.time() < _DEADLINE - 15:
                refined, refined_energy = _anneal(sequence, refined, dimension, rng,
                                                   max_iters=500, T_start=4.0, T_end=0.01)
            if best is None or refined_energy < best_energy:
                best, best_energy = refined, refined_energy
    
    # Phase 4: Final polish with more aggressive local search
    if time.time() < _DEADLINE - 5:
        refined, refined_energy = _local_search(sequence, best, dimension, rng, max_iters=300)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

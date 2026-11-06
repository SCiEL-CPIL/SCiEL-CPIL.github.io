"""HP lattice protein folding on the cubic lattice."""
import random
import math
import time

def _neighbours(point):
    """Generator of 6-neighbours on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _neighbours_list(point):
    """Return list of 6-neighbor tuples for faster iteration."""
    result = []
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            result.append(tuple(q))
    return result

def _compute_energy(sequence, chain):
    """Compute the energy of a complete chain."""
    n = len(chain)
    if n < 2:
        return 0
    is_h = [1 if s == 'H' else 0 for s in sequence]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if not is_h[i]:
            continue
        for nb in _neighbours(chain[i]):
            j = pos_to_idx.get(nb, None)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return -contacts

def _apply_pivot_move(chain, k, axis, direction, dimension):
    """
    Apply a pivot move: rotate segment k..n-1 around chain[k-1].
    Returns new chain or None if invalid (self-intersection).
    k is 1..n-1, pivot is at index k-1.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return None
    
    pivot = chain[k-1]
    tail = chain[k:]
    
    head_set = set(chain[:k])
    
    new_tail = []
    for p in tail:
        v = [p[d] - pivot[d] for d in range(dimension)]
        if axis == 0:
            if direction == 1:
                v_new = [v[0], -v[2], v[1]]
            else:
                v_new = [v[0], v[2], -v[1]]
        elif axis == 1:
            if direction == 1:
                v_new = [v[2], v[1], -v[0]]
            else:
                v_new = [-v[2], v[1], v[0]]
        else:
            if direction == 1:
                v_new = [-v[1], v[0], v[2]]
            else:
                v_new = [v[1], -v[0], v[2]]
        new_p = tuple(pivot[d] + v_new[d] for d in range(dimension))
        new_tail.append(new_p)
    
    for p in new_tail:
        if p in head_set:
            return None
    
    return chain[:k] + new_tail

def _apply_corner_move(chain, k, axis, direction, dimension):
    """
    Apply a corner move: move residue k by one unit along axis by direction,
    if that site is free.
    Returns new chain or None if invalid.
    """
    n = len(chain)
    if k < 1 or k >= n - 1:
        return None
    
    new_pos = list(chain[k])
    new_pos[axis] += direction
    new_pos = tuple(new_pos)
    
    # Check if new position is free (not occupied by any other residue)
    for i, p in enumerate(chain):
        if i != k and p == new_pos:
            return None
    
    # Check unit step constraints with neighbors
    if k > 0:
        prev = chain[k-1]
        dist = sum(abs(new_pos[d] - prev[d]) for d in range(dimension))
        if dist != 1:
            return None
    if k < n - 1:
        nxt = chain[k+1]
        dist = sum(abs(new_pos[d] - nxt[d]) for d in range(dimension))
        if dist != 1:
            return None
    
    new_chain = chain[:]
    new_chain[k] = new_pos
    return new_chain

def _compute_delta_energy_pivot(sequence, chain, k, new_chain, is_h):
    """
    Compute energy difference for a pivot move at position k.
    Only cross-contacts between head (0..k-1) and tail (k..n-1) change.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return 0
    
    head = chain[:k]
    old_tail = chain[k:]
    new_tail = new_chain[k:]
    
    old_tail_pos_set = set(old_tail)
    new_tail_pos_set = set(new_tail)
    
    old_tail_idx = {p: i for i, p in enumerate(old_tail)}
    new_tail_idx = {p: i for i, p in enumerate(new_tail)}
    
    old_cross = 0
    for i, p in enumerate(head):
        if not is_h[i]:
            continue
        for nb in _neighbours(p):
            if nb in old_tail_pos_set:
                j = old_tail_idx[nb]
                if is_h[k + j]:
                    old_cross += 1
    
    new_cross = 0
    for i, p in enumerate(head):
        if not is_h[i]:
            continue
        for nb in _neighbours(p):
            if nb in new_tail_pos_set:
                j = new_tail_idx[nb]
                if is_h[k + j]:
                    new_cross += 1
    
    return old_cross - new_cross

def _compute_delta_energy_corner(sequence, chain, k, new_chain, is_h):
    """
    Compute energy difference for a corner move at position k.
    Only contacts involving residue k change.
    """
    n = len(chain)
    if k < 1 or k >= n - 1:
        return 0
    
    old_pos = chain[k]
    new_pos = new_chain[k]
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Old contacts involving k
    old_contacts = 0
    if is_h[k]:
        for nb in _neighbours(old_pos):
            j = pos_to_idx.get(nb, None)
            if j is not None and abs(j - k) > 1 and is_h[j]:
                old_contacts += 1
    
    # New contacts involving k
    new_contacts = 0
    if is_h[k]:
        for nb in _neighbours(new_pos):
            j = pos_to_idx.get(nb, None)
            if j is not None and abs(j - k) > 1 and is_h[j]:
                new_contacts += 1
    
    # Also check if any other residue's contacts change (unlikely for single residue move)
    return old_contacts - new_contacts

def _beam_search_growth(sequence, dimension, rng, beam_width=200):
    """
    Beam search for chain growth.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[(0,) * dimension]]
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    start = (0,) * dimension
    initial_occupied = {start: 0}
    beams = [([start], initial_occupied, 0)]
    
    for i in range(1, n):
        if not beams:
            break
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_energy = energy
                if is_h[i]:
                    for nb in _neighbours(site):
                        j = occupied.get(nb, None)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_energy -= 1
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_chain = chain + [site]
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = min(beams, key=lambda x: x[2])
    return best[0]

def _local_search(sequence, chain, dimension, rng, max_time, is_h):
    """
    Local search with pivot and corner moves, simulated annealing.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = chain[:]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T_max = 8.0
    T_min = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Choose move type: 70% pivot, 30% corner
        if rng.random() < 0.7:
            k = rng.randint(1, n - 1)
            axis = rng.randint(0, dimension - 1)
            direction = rng.choice([-1, 1])
            new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
            if new_chain is None:
                continue
            delta = _compute_delta_energy_pivot(sequence, current_chain, k, new_chain, is_h)
        else:
            k = rng.randint(1, n - 2)
            axis = rng.randint(0, dimension - 1)
            direction = rng.choice([-1, 1])
            new_chain = _apply_corner_move(current_chain, k, axis, direction, dimension)
            if new_chain is None:
                continue
            delta = _compute_delta_energy_corner(sequence, current_chain, k, new_chain, is_h)
        
        if delta < 0:
            accept = True
        else:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / max_time)
            T = T_max - (T_max - T_min) * progress
            if T < T_min:
                T = T_min
            prob = math.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            current_chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
    
    return best_chain

def _refined_local_search(sequence, chain, dimension, rng, max_time, is_h):
    """
    Refined local search with multiple cooling schedules and move types.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = chain[:]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    schedules = [
        (10.0, 0.05, 0.0),
        (8.0, 0.03, 0.3),
        (6.0, 0.01, 0.5),
    ]
    
    for T_max, T_min, plateau in schedules:
        remaining = max_time - (time.time() - start_time)
        if remaining <= 0.1:
            break
        
        sa_time = remaining * 0.45
        if sa_time < 0.1:
            break
        
        sa_start = time.time()
        
        while time.time() - sa_start < sa_time:
            if rng.random() < 0.7:
                k = rng.randint(1, n - 1)
                axis = rng.randint(0, dimension - 1)
                direction = rng.choice([-1, 1])
                new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
                if new_chain is None:
                    continue
                delta = _compute_delta_energy_pivot(sequence, current_chain, k, new_chain, is_h)
            else:
                k = rng.randint(1, n - 2)
                axis = rng.randint(0, dimension - 1)
                direction = rng.choice([-1, 1])
                new_chain = _apply_corner_move(current_chain, k, axis, direction, dimension)
                if new_chain is None:
                    continue
                delta = _compute_delta_energy_corner(sequence, current_chain, k, new_chain, is_h)
            
            if delta < 0:
                accept = True
            else:
                elapsed = time.time() - sa_start
                progress = min(1.0, elapsed / sa_time)
                if progress < plateau:
                    T = T_max
                else:
                    T = T_max - (T_max - T_min) * ((progress - plateau) / (1 - plateau))
                if T < T_min:
                    T = T_min
                prob = math.exp(-delta / T)
                accept = rng.random() < prob
            
            if accept:
                current_chain = new_chain
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
    
    return best_chain

def _polish_pass(sequence, chain, dimension, rng, max_time, is_h):
    """
    High-precision polish pass with very low temperature and many iterations.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = chain[:]
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T_max = 2.0
    T_min = 0.01
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Focus on small moves
        if rng.random() < 0.5:
            k = rng.randint(1, n - 1)
            axis = rng.randint(0, dimension - 1)
            direction = rng.choice([-1, 1])
            new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
            if new_chain is None:
                continue
            delta = _compute_delta_energy_pivot(sequence, current_chain, k, new_chain, is_h)
        else:
            k = rng.randint(1, n - 2)
            axis = rng.randint(0, dimension - 1)
            direction = rng.choice([-1, 1])
            new_chain = _apply_corner_move(current_chain, k, axis, direction, dimension)
            if new_chain is None:
                continue
            delta = _compute_delta_energy_corner(sequence, current_chain, k, new_chain, is_h)
        
        if delta < 0:
            accept = True
        else:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / max_time)
            T = T_max - (T_max - T_min) * progress
            if T < T_min:
                T = T_min
            prob = math.exp(-delta / T)
            accept = rng.random() < prob
        
        if accept:
            current_chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold the protein chain using multi-start beam search initialization and refined local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
    
    elapsed_global = time.time() - fold._start_time
    total_budget = 300.0
    remaining = total_budget - elapsed_global
    if remaining <= 0:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Allocate time per call
    per_call_budget = min(remaining * 0.25, 12.0)
    if per_call_budget < 0.5:
        per_call_budget = 0.5
    
    rng = random.Random(12345 + len(sequence) * 7 + n)
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Phase 1: Multi-start beam search initialization
    init_budget = per_call_budget * 0.35
    num_starts = max(1, min(4, int(init_budget / 1.5)))
    
    if n <= 50:
        beam_width = 500
    elif n <= 100:
        beam_width = 350
    else:
        beam_width = 200
    
    best_init_chain = None
    best_init_energy = float('inf')
    
    for s in range(num_starts):
        if time.time() - fold._start_time >= total_budget - 10:
            break
        
        bw = max(80, beam_width + rng.randint(-80, 80))
        chain = _beam_search_growth(sequence, dimension, rng, bw)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_init_energy:
                best_init_energy = energy
                best_init_chain = chain
    
    # Fallback: greedy growth
    if best_init_chain is None:
        start = (0,) * dimension
        chain, occupied = [start], {start: 0}
        success = True
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if is_h[i]:
                    for nb in _neighbours(site):
                        j = occupied.get(nb, None)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                options.append((gain + 0.1 * free + rng.random(), site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        if success:
            energy = _compute_energy(sequence, chain)
            if energy < best_init_energy:
                best_init_energy = energy
                best_init_chain = chain
    
    if best_init_chain is None:
        best_init_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_init_energy = 0
    
    # Phase 2: Refined local search
    elapsed_so_far = time.time() - fold._start_time
    remaining_after_init = total_budget - elapsed_so_far
    ls_budget = min(remaining_after_init * 0.7, per_call_budget * 0.5)
    
    if ls_budget > 0.5:
        best_chain = _refined_local_search(sequence, best_init_chain, dimension, rng, ls_budget, is_h)
    else:
        best_chain = best_init_chain
    
    # Phase 3: Polish pass
    elapsed_so_far = time.time() - fold._start_time
    remaining_after_ls = total_budget - elapsed_so_far
    polish_budget = min(remaining_after_ls * 0.3, per_call_budget * 0.15)
    
    if polish_budget > 0.3:
        best_chain = _polish_pass(sequence, best_chain, dimension, rng, polish_budget, is_h)
    
    return [list(p) for p in best_chain]

"""HP lattice protein folding on the cubic lattice."""
import random
import math
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _neighbours_list(point):
    """Return list of neighbor tuples for faster iteration."""
    result = []
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            result.append(tuple(q))
    return result

def _beam_search_growth(sequence, dimension, rng, beam_width=200):
    """
    Beam search for chain growth.
    Each beam entry: (chain, occupied_dict, score)
    Score = negative contact count (we want to minimize = maximize contacts)
    Actually let's use: score = -contacts_so_far (lower is better)
    We keep the top beam_width entries.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[(0,) * dimension]]
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Initialize: first residue at origin
    start = (0,) * dimension
    initial_occupied = {start: 0}
    beams = [( [start], initial_occupied, 0 )]  # chain, occupied, energy_so_far
    
    for i in range(1, n):
        if not beams:
            break
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute new energy contribution
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
        
        # Sort by energy (ascending = most negative = best) and keep top beam_width
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    # Return the best complete chain
    best = min(beams, key=lambda x: x[2])
    return best[0]

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
    
    # Build set of head positions for collision check
    head_set = set(chain[:k])
    
    # Rotate tail around pivot
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
    
    # Check for self-intersection with head
    for p in new_tail:
        if p in head_set:
            return None
    
    return chain[:k] + new_tail

def _compute_delta_energy(sequence, chain, k, new_chain):
    """
    Compute energy difference when replacing tail (k..n-1) with new_tail.
    Only cross-contacts between head (0..k-1) and tail change.
    """
    n = len(chain)
    if k < 1 or k >= n:
        return 0
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Build position maps
    head = chain[:k]
    old_tail = chain[k:]
    new_tail = new_chain[k:]
    
    head_pos_set = set(head)
    old_tail_pos_set = set(old_tail)
    new_tail_pos_set = set(new_tail)
    
    # Map positions to indices in head and tails
    head_idx = {p: i for i, p in enumerate(head)}
    old_tail_idx = {p: i for i, p in enumerate(old_tail)}
    new_tail_idx = {p: i for i, p in enumerate(new_tail)}
    
    # Count old cross-contacts: H in head adjacent to H in old_tail
    old_cross = 0
    for i, p in enumerate(head):
        if not is_h[i]:
            continue
        for nb in _neighbours(p):
            if nb in old_tail_pos_set:
                j = old_tail_idx[nb]
                if is_h[k + j]:
                    old_cross += 1
    
    # Count new cross-contacts
    new_cross = 0
    for i, p in enumerate(head):
        if not is_h[i]:
            continue
        for nb in _neighbours(p):
            if nb in new_tail_pos_set:
                j = new_tail_idx[nb]
                if is_h[k + j]:
                    new_cross += 1
    
    # Energy = -contacts, so delta = -(new_cross - old_cross) = old_cross - new_cross
    return old_cross - new_cross

def _local_search(sequence, chain, dimension, rng, max_time, is_h):
    """
    Pivot-move local search with simulated annealing.
    Returns best chain found.
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
        # Choose pivot point k (1 to n-1)
        # Bias toward shorter tails for efficiency? Or uniform?
        # Let's use uniform for now, but maybe bias toward middle
        k = rng.randint(1, n - 1)
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])
        
        new_chain = _apply_pivot_move(current_chain, k, axis, direction, dimension)
        if new_chain is None:
            continue
        
        delta = _compute_delta_energy(sequence, current_chain, k, new_chain)
        
        # Metropolis acceptance
        if delta < 0:
            accept = True
        else:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / max_time)
            T = T_max - (T_max - T_min) * progress
            if T < T_min:
                T = T_min
            # Use math.exp for accuracy
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
    Fold the protein chain using beam search initialization and pivot-move local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Global deadline: 330s for all sequences. Assume worst case we're the only one,
    # but to be safe, use a per-call budget with margin.
    # The evaluator kills after 330s wall-clock. We don't know how many calls happen,
    # but the task says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # Wait, re-reading: "fold is called once per benchmark sequence in a sandboxed subprocess with a 240 s CPU budget for all sequences together."
    # And "Each evaluation must finish within 300 s."
    # So the entire process has 300s. If fold is called 26 times in the same process, we need to share time.
    # But it says "sandboxed subprocess" - does that mean one subprocess per sequence? Or one subprocess for all?
    # "240 s CPU budget for all sequences together" suggests one process.
    # Let's use a global time tracker.
    
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
    
    elapsed_global = time.time() - fold._start_time
    # Total budget: 300s, leave 10s safety margin
    total_budget = 290.0
    remaining = total_budget - elapsed_global
    if remaining <= 0:
        # Out of time, return a straight line or best effort
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Allocate time: use a fraction of remaining, but at least 1s, at most 12s per call
    # If we have lots of time left, spend more. If little, spend less.
    # Simple heuristic: spend min(remaining * 0.3, 12.0) seconds, but at least 0.5s
    per_call_budget = min(remaining * 0.3, 12.0)
    if per_call_budget < 0.5:
        per_call_budget = 0.5
    
    rng = random.Random(12345 + len(sequence) * 7 + n)
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Phase 1: Beam search initialization
    # Use a fraction of the budget for initialization
    init_budget = per_call_budget * 0.3
    best_init_chain = None
    best_init_energy = 0
    
    # Try multiple beam widths or restarts?
    # For now, single beam search with adaptive width
    # Smaller sequences can afford larger beams
    if n <= 50:
        beam_width = 500
    elif n <= 100:
        beam_width = 300
    else:
        beam_width = 200
    
    init_start = time.time()
    chain = _beam_search_growth(sequence, dimension, rng, beam_width)
    if chain is not None:
        best_init_chain = chain
        best_init_energy = _compute_energy(sequence, chain)
    
    # If beam search failed or took too long, try greedy as fallback
    if best_init_chain is None or time.time() - init_start > init_budget:
        # Simple greedy fallback
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
            if best_init_chain is None or energy < best_init_energy:
                best_init_chain = chain
                best_init_energy = energy
    
    if best_init_chain is None:
        # Last resort: straight line
        best_init_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_init_energy = 0
    
    # Phase 2: Local search with remaining budget
    ls_budget = per_call_budget - (time.time() - init_start)
    if ls_budget > 0.1:
        best_chain = _local_search(sequence, best_init_chain, dimension, rng, ls_budget, is_h)
    else:
        best_chain = best_init_chain
    
    return [list(p) for p in best_chain]

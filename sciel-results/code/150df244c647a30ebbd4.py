"""Improved HP lattice folding: fast growth + local search with time guard."""
# EVOLVE-BLOCK-START
import random
import time

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy from scratch using position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy(sequence, chain, where, i, new_pos):
    """
    Compute energy delta when moving residue i from chain[i] to new_pos.
    Assumes new_pos is not occupied by any other residue.
    Returns the change in energy (new_energy - old_energy).
    """
    old_pos = chain[i]
    if sequence[i] != "H":
        # If residue i is not H, moving it only affects contacts involving i
        # Check contacts that are lost (i was H? no, so no loss from i's perspective)
        # But other H residues might have been contacting i
        delta = 0
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and j != i and j != i - 1 and j != i + 1:
                if sequence[j] == "H":
                    delta += 1  # losing a contact
        for q in _neighbours(new_pos):
            j = where.get(q)
            if j is not None and j != i and j != i - 1 and j != i + 1:
                if sequence[j] == "H":
                    delta -= 1  # gaining a contact
        return -delta  # energy is -contacts, so delta_energy = -(new_contacts - old_contacts)
    
    # sequence[i] is H
    delta = 0
    # Remove contacts involving old position
    for q in _neighbours(old_pos):
        j = where.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == "H":
                delta += 1  # losing a contact (old had it)
    # Add contacts at new position
    for q in _neighbours(new_pos):
        j = where.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == "H":
                delta -= 1  # gaining a contact
    return -delta  # energy = -contacts


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


def _local_search(sequence, chain, rng, time_budget):
    """
    Local search with simulated annealing using kink moves.
    Returns improved chain.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_time = time.time()
    max_time = time_budget
    
    T = 5.0  # initial temperature
    T_min = 0.1
    cooling = 0.999
    
    # Precompute neighbors for each position for faster lookup
    # We'll use where dict for O(1) lookups
    
    steps = 0
    while T > T_min and (time.time() - start_time) < max_time and _remaining() > 2.0:
        # Try a random kink move: pick a residue i (not endpoints) and try to move it
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Get candidate positions: neighbors of old_pos that are not occupied
        candidates = []
        for q in _neighbours(old_pos):
            if q != current_chain[i - 1] and q != current_chain[i + 1]:
                # Check if q is occupied by any other residue
                if q not in current_chain or current_chain.index(q) == i:
                    # Need more efficient check
                    pass
        
        # Build occupied set for efficiency
        occupied_set = set(current_chain)
        
        for q in _neighbours(old_pos):
            if q == current_chain[i - 1] or q == current_chain[i + 1]:
                continue
            if q not in occupied_set or q == old_pos:
                candidates.append(q)
        
        if not candidates:
            T *= cooling
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute delta energy
        where = {p: idx for idx, p in enumerate(current_chain)}
        delta_e = _delta_energy(sequence, current_chain, where, i, new_pos)
        
        # Accept or reject
        if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
            # Accept move
            current_chain[i] = new_pos
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        steps += 1
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using growth + local search with time management."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n <= 1:
        return [[0]]
    
    best = None
    best_energy = 0  # worst possible (no contacts)
    
    # Phase 1: Generate initial conformations via growth
    num_growth = min(50, max(10, int(2.0 * n)))  # scale with sequence length
    growth_time_budget = min(5.0, max(1.0, 0.05 * n))
    
    growth_deadline = time.time() + growth_time_budget
    for _ in range(num_growth):
        if time.time() > growth_deadline or _remaining() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best = chain
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best)
    
    # Phase 2: Local search on best and some other good solutions
    # Keep top few solutions for local search
    solutions = []
    for _ in range(min(20, num_growth)):
        if _remaining() < 3.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        solutions.append((value, chain))
    
    solutions.sort(key=lambda x: x[0])
    
    # Local search on top solutions
    top_k = min(5, len(solutions))
    for k in range(top_k):
        if _remaining() < 2.0:
            break
        _, chain = solutions[k]
        # If this is already the best, skip duplicate work
        ls_time = min(3.0, max(0.5, _remaining() * 0.3 / (top_k - k)))
        improved = _local_search(sequence, chain, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved
    
    # Also do local search on the initial best
    if _remaining() > 1.0:
        ls_time = min(2.0, _remaining() * 0.5)
        improved = _local_search(sequence, best, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

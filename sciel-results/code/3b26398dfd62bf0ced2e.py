"""HP lattice protein folding: growth init + fragment-move simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 205.0  # wall-clock limit with safety margin


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


def _delta_energy_single(sequence, chain, where, i, new_pos):
    """
    Compute energy delta when moving residue i from chain[i] to new_pos.
    Assumes new_pos is not occupied by any other residue.
    Returns the change in energy (new_energy - old_energy).
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0

    delta = 0
    # Remove contacts involving old position
    for q in _neighbours(old_pos):
        j = where.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == "H" and sequence[i] == "H":
                delta += 1  # losing a contact
    # Add contacts at new position
    for q in _neighbours(new_pos):
        j = where.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == "H" and sequence[i] == "H":
                delta -= 1  # gaining a contact
    return -delta


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


def _fragment_move(sequence, chain, rng):
    """
    Try a fragment move: pick a random segment [i, j] and attempt to shift it
    by one unit in a random direction. Returns (new_chain, delta_energy) or (None, 0).
    """
    n = len(chain)
    if n < 3:
        return None, 0

    # Pick fragment length between 2 and min(10, n-2)
    max_len = min(10, n - 2)
    frag_len = rng.randint(2, max_len)
    i = rng.randint(1, n - frag_len - 1)
    j = i + frag_len - 1

    # Pick a random direction
    axis = rng.randint(0, 1)
    step = rng.choice([-1, 1])

    # Check if the move is valid
    # The fragment must not collide with the rest of the chain
    # Build set of positions occupied by non-fragment residues
    occupied_set = set()
    for k in range(n):
        if k < i or k > j:
            occupied_set.add(chain[k])

    # Check each residue in the fragment
    for k in range(i, j + 1):
        new_pos = list(chain[k])
        new_pos[axis] += step
        new_pos = tuple(new_pos)

        # Check if new_pos is occupied by a non-fragment residue
        if new_pos in occupied_set:
            return None, 0

        # Check if new_pos is adjacent to the previous/next residue in a way that breaks the chain
        # Actually, we need to ensure the chain remains connected
        # The key constraint: the new position of residue k must be adjacent to the new position of residue k-1 (if k > i)
        # and the new position of residue k must be adjacent to the new position of residue k+1 (if k < j)
        # But since we're shifting the whole fragment, the internal adjacencies are preserved.
        # We only need to check:
        # 1. The new position of residue i is adjacent to residue i-1 (if i > 0)
        # 2. The new position of residue j is adjacent to residue j+1 (if j < n-1)

    # Check connectivity at the boundaries
    # Left boundary: new position of i must be adjacent to chain[i-1]
    new_i_pos = list(chain[i])
    new_i_pos[axis] += step
    new_i_pos = tuple(new_i_pos)
    if i > 0:
        prev_pos = chain[i - 1]
        diff = sum(abs(a - b) for a, b in zip(new_i_pos, prev_pos))
        if diff != 1:
            return None, 0

    # Right boundary: new position of j must be adjacent to chain[j+1]
    new_j_pos = list(chain[j])
    new_j_pos[axis] += step
    new_j_pos = tuple(new_j_pos)
    if j < n - 1:
        next_pos = chain[j + 1]
        diff = sum(abs(a - b) for a, b in zip(new_j_pos, next_pos))
        if diff != 1:
            return None, 0

    # Also check that the new positions don't collide with each other (they shouldn't since it's a rigid shift)
    # But let's verify the new positions are all distinct
    new_positions = set()
    for k in range(i, j + 1):
        np = list(chain[k])
        np[axis] += step
        np = tuple(np)
        if np in new_positions:
            return None, 0
        new_positions.add(np)

    # Compute delta energy
    where = {p: idx for idx, p in enumerate(chain)}
    delta = 0

    # For each residue in the fragment, compute the change in contacts
    for k in range(i, j + 1):
        old_pos = chain[k]
        new_pos = list(old_pos)
        new_pos[axis] += step
        new_pos = tuple(new_pos)

        if sequence[k] != "H":
            continue

        # Count contacts lost at old position
        for q in _neighbours(old_pos):
            j_idx = where.get(q)
            if j_idx is not None and j_idx != k and j_idx != k - 1 and j_idx != k + 1:
                if sequence[j_idx] == "H":
                    # Check if j_idx is in the fragment (in which case the contact is preserved)
                    if not (i <= j_idx <= j):
                        delta += 1  # losing a contact

        # Count contacts gained at new position
        for q in _neighbours(new_pos):
            j_idx = where.get(q)
            if j_idx is not None and j_idx != k and j_idx != k - 1 and j_idx != k + 1:
                if sequence[j_idx] == "H":
                    # Check if j_idx is in the fragment (in which case the contact is preserved)
                    if not (i <= j_idx <= j):
                        delta -= 1  # gaining a contact

    # Also need to account for contacts between fragment residues and non-fragment residues
    # The above loop already handles this by checking if j_idx is in the fragment

    return None, -delta  # We'll construct the new chain if accepted


def _apply_fragment_move(chain, i, j, axis, step):
    """Apply a fragment move and return the new chain."""
    new_chain = list(chain)
    for k in range(i, j + 1):
        new_pos = list(chain[k])
        new_pos[axis] += step
        new_chain[k] = tuple(new_pos)
    return new_chain


def _local_search(sequence, chain, rng, time_budget):
    """
    Local search with simulated annealing using fragment moves.
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
    cooling = 0.9995

    steps = 0
    while T > T_min and (time.time() - start_time) < max_time and _remaining() > 2.0:
        # Try a random fragment move
        n = len(current_chain)
        if n < 3:
            break

        max_len = min(10, n - 2)
        frag_len = rng.randint(2, max_len)
        i = rng.randint(1, n - frag_len - 1)
        j = i + frag_len - 1
        axis = rng.randint(0, 1)
        step = rng.choice([-1, 1])

        # Check validity
        occupied_set = set()
        for k in range(n):
            if k < i or k > j:
                occupied_set.add(current_chain[k])

        valid = True
        new_positions = {}
        for k in range(i, j + 1):
            new_pos = list(current_chain[k])
            new_pos[axis] += step
            new_pos = tuple(new_pos)

            if new_pos in occupied_set:
                valid = False
                break
            if new_pos in new_positions.values():
                valid = False
                break
            new_positions[k] = new_pos

        if not valid:
            T *= cooling
            continue

        # Check boundary connectivity
        if i > 0:
            prev_pos = current_chain[i - 1]
            new_i_pos = new_positions[i]
            diff = sum(abs(a - b) for a, b in zip(new_i_pos, prev_pos))
            if diff != 1:
                T *= cooling
                continue

        if j < n - 1:
            next_pos = current_chain[j + 1]
            new_j_pos = new_positions[j]
            diff = sum(abs(a - b) for a, b in zip(new_j_pos, next_pos))
            if diff != 1:
                T *= cooling
                continue

        # Compute delta energy
        where = {p: idx for idx, p in enumerate(current_chain)}
        delta = 0

        for k in range(i, j + 1):
            old_pos = current_chain[k]
            new_pos = new_positions[k]

            if sequence[k] != "H":
                continue

            # Count contacts lost at old position
            for q in _neighbours(old_pos):
                j_idx = where.get(q)
                if j_idx is not None and j_idx != k and j_idx != k - 1 and j_idx != k + 1:
                    if sequence[j_idx] == "H":
                        if not (i <= j_idx <= j):
                            delta += 1

            # Count contacts gained at new position
            for q in _neighbours(new_pos):
                j_idx = where.get(q)
                if j_idx is not None and j_idx != k and j_idx != k - 1 and j_idx != k + 1:
                    if sequence[j_idx] == "H":
                        if not (i <= j_idx <= j):
                            delta -= 1

        delta_e = -delta

        # Accept or reject
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            # Accept move
            current_chain = _apply_fragment_move(current_chain, i, j, axis, step)
            current_energy += delta_e

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        T *= cooling
        steps += 1

    return best_chain


def fold(sequence, dimension):
    """Best conformation using growth + fragment-move local search with time management."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)

    if n <= 1:
        return [[0]]

    best = None
    best_energy = 0  # worst possible (no contacts)

    # Phase 1: Generate initial conformations via growth
    num_growth = min(50, max(10, int(2.0 * n)))
    growth_time_budget = min(5.0, max(1.0, 0.05 * n))

    growth_deadline = time.time() + growth_time_budget
    solutions = []
    for _ in range(num_growth):
        if time.time() > growth_deadline or _remaining() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _compute_energy(sequence, chain)
        solutions.append((value, chain))
        if value < best_energy:
            best_energy = value
            best = chain

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best)
        solutions.append((best_energy, best))

    solutions.sort(key=lambda x: x[0])

    # Phase 2: Local search on top solutions using fragment moves
    top_k = min(5, len(solutions))
    for k in range(top_k):
        if _remaining() < 2.0:
            break
        _, chain = solutions[k]
        ls_time = min(3.0, max(0.5, _remaining() * 0.3 / max(1, top_k - k)))
        improved = _local_search(sequence, chain, rng, ls_time)
        value = _compute_energy(sequence, improved)
        if value < best_energy:
            best_energy = value
            best = improved

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

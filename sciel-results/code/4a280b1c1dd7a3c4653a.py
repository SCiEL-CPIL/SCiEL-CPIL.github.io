"""Rewritten program: deterministic multi-start greedy growth + local-search polish."""
# EVOLVE-BLOCK-START
import time
import random

_START_TIME = time.time()
DEADLINE = 300.0  # safety margin before 330s kill


def _check_deadline():
    return (time.time() - _START_TIME) < DEADLINE


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def _grow(sequence, dimension, weights, rng, deadline_check):
    """Deterministic greedy growth with look-ahead penalty.
    weights: (contact_weight, free_weight, look_ahead_weight)
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    h_count = sequence.count("H")
    h_placed = 1 if sequence[0] == "H" else 0

    for i in range(1, len(sequence)):
        if not deadline_check():
            return None
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Contact gain
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Look-ahead: penalize if this move leaves no room for remaining chain
            # Simple heuristic: if free is very low, penalize
            remaining = len(sequence) - i - 1
            if remaining > 0 and free <= 2:
                look_penalty = 0.5 * (3 - free)
            else:
                look_penalty = 0.0
            # Penalize if we're moving away from unplaced H residues (rough)
            # Count H residues not yet placed
            h_remaining = h_count - h_placed
            score = weights[0] * gain + weights[1] * free - weights[2] * look_penalty + rng.random() * 0.01
            options.append((score, site))
        if not options:
            return None
        options.sort(key=lambda x: -x[0])
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        if sequence[i] == "H":
            h_placed += 1
    return chain


def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, max_iter=200, deadline_check=None):
    """Improve conformation via single-residue moves and segment reversals."""
    if deadline_check is None:
        deadline_check = lambda: True
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best = list(chain)
    best_energy = _compute_energy(sequence, best)
    improved = True
    iteration = 0
    while improved and iteration < max_iter and deadline_check():
        improved = False
        iteration += 1
        # Try moving each residue to a neighbouring free site
        for i in range(1, n - 1):
            if not deadline_check():
                break
            for site in _neighbours(chain[i]):
                if site in occupied and occupied[site] != i:
                    continue
                # Check self-avoidance after move
                new_chain = list(chain)
                new_chain[i] = site
                # Verify no collisions
                new_occ = set(new_chain)
                if len(new_occ) != n:
                    continue
                # Compute energy delta quickly
                # Remove old contacts involving i, add new ones
                old_e = _compute_energy(sequence, chain)
                new_e = _compute_energy(sequence, new_chain)
                if new_e < best_energy:
                    best = new_chain
                    best_energy = new_e
                    improved = True
                    chain = new_chain
                    occupied = {p: j for j, p in enumerate(chain)}
                    break
        # Try segment reversal (2-opt)
        for a in range(0, n - 3, 2):
            if not deadline_check():
                break
            for b in range(a + 2, n - 1, 2):
                if not deadline_check():
                    break
                # Reverse segment [a+1..b]
                new_chain = list(chain)
                seg = new_chain[a + 1:b + 1][::-1]
                new_chain[a + 1:b + 1] = seg
                # Check self-avoidance
                new_occ = set(new_chain)
                if len(new_occ) != n:
                    continue
                new_e = _compute_energy(sequence, new_chain)
                if new_e < best_energy:
                    best = new_chain
                    best_energy = new_e
                    improved = True
                    chain = new_chain
                    occupied = {p: j for j, p in enumerate(chain)}
                    break
            if improved:
                break
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation via deterministic multi-start greedy + local search polish."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    # Multiple greedy variants with different weightings
    weight_variants = [
        (1.0, 0.1, 0.5),
        (1.0, 0.2, 0.3),
        (1.5, 0.1, 0.4),
        (1.0, 0.05, 0.6),
        (2.0, 0.1, 0.3),
        (1.0, 0.15, 0.5),
    ]

    for w in weight_variants:
        if not _check_deadline():
            break
        chain = _grow(sequence, dimension, w, rng, _check_deadline)
        if chain is None:
            continue
        # Polish with local search
        polished, energy = _local_search(sequence, chain, max_iter=50, deadline_check=_check_deadline)
        if energy < best_energy:
            best = polished
            best_energy = energy

    # Additional stochastic restarts if time permits
    extra_restarts = 50
    for _ in range(extra_restarts):
        if not _check_deadline():
            break
        w = weight_variants[_ % len(weight_variants)]
        chain = _grow(sequence, dimension, w, rng, _check_deadline)
        if chain is None:
            continue
        polished, energy = _local_search(sequence, chain, max_iter=30, deadline_check=_check_deadline)
        if energy < best_energy:
            best = polished
            best_energy = energy

    if best is None:
        # Straight line fallback
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

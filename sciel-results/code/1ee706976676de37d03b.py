"""HP lattice protein folding on the cubic lattice.

Design: represent a conformation as a discrete bond-vector sequence (a walk on the
6-neighbourhood of the simple cubic lattice). Optimise the sequence with a genetic
algorithm whose crossover and mutation operators preserve unit-step connectivity and
self-avoidance, then finish with a time-guarded local-search polish pass.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

import numpy as np

# The six unit bond vectors (directions) on the simple cubic lattice.
_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int32)
# Opposite direction index for each direction (used to avoid immediate reversal).
_OPPOSITE = [1, 0, 3, 2, 5, 4]
# For each direction, the two perpendicular directions that can be used to rotate
# the endpoint (rotate the last bond into an adjacent direction).
_PERP = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1], 4: [0, 1], 5: [0, 1]}
# Direction index for each axis sign (axis, +1/-1).
_AXIS_SIGN = {0: 0, 1: 1, 2: 2, -1: 3, -2: 4, -3: 5}
# Precompute the neighbour offsets (unit vectors) for contact checks.
_NEIGH = _DIRS


def _bond_to_positions(bonds):
    """Convert a bond-vector sequence to cumulative lattice positions."""
    return np.cumsum(bonds, axis=0)


def _is_self_avoiding(positions):
    """Check that all positions are distinct (self-avoiding walk)."""
    key = positions[:, 0] * 500003 + positions[:, 1] * 701 + positions[:, 2]
    return len(np.unique(key)) == len(key)


def _energy_from_positions(sequence, positions):
    """Energy = minus the number of non-consecutive H-H contacts on neighbouring sites."""
    n = len(sequence)
    hpos = [tuple(positions[i]) for i in range(n) if sequence[i] == "H"]
    hset = set(hpos)
    contacts = 0
    for i in range(n):
        if sequence[i] != "H":
            continue
        p = positions[i]
        for d in _NEIGH:
            q = tuple(p + d)
            if q in hset:
                # Count each pair once: only when the other H is further along the chain.
                pass
    # Count H-H pairs on neighbouring sites that are not adjacent along the chain.
    # Use a set of (position, index) to find each H's neighbours.
    pos_to_idx = {}
    for i in range(n):
        pos_to_idx[tuple(positions[i])] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != "H":
            continue
        p = positions[i]
        for d in _NEIGH:
            q = tuple(p + d)
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _energy(sequence, bonds):
    """Energy of a bond-vector walk (returns +1 if not self-avoiding)."""
    positions = _bond_to_positions(bonds)
    if not _is_self_avoiding(positions):
        return 1
    return _energy_from_positions(sequence, positions)


def _initial_bonds(n, rng):
    """Random self-avoiding bond-vector walk of length n-1 (or None)."""
    for _ in range(2000):
        bonds = np.zeros((n - 1, 3), dtype=np.int32)
        pos = np.zeros(3, dtype=np.int32)
        occupied = {tuple(pos)}
        ok = True
        prev_dir = -1
        for i in range(n - 1):
            # Choose a direction that is not the immediate reverse and stays self-avoiding.
            candidates = [d for d in range(6) if d != _OPPOSITE[prev_dir]]
            rng.shuffle(candidates)
            placed = False
            for d in candidates:
                np_ = pos + _DIRS[d]
                key = tuple(np_)
                if key not in occupied:
                    bonds[i] = _DIRS[d]
                    pos = np_
                    occupied.add(key)
                    prev_dir = d
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return bonds
    return None


def _mutate_bonds(bonds, sequence, rng):
    """Point mutation: try to change a single bond direction, keep if self-avoiding."""
    n = len(bonds)
    if n == 0:
        return bonds
    new_bonds = bonds.copy()
    # Try several random single-bond changes; accept the first that stays self-avoiding.
    for _ in range(30):
        i = rng.randrange(n)
        old = new_bonds[i].copy()
        # Pick a new direction that is not the immediate reverse of neighbours.
        prev_dir = -1
        if i > 0:
            for d in range(6):
                if np.array_equal(bonds[i - 1], _DIRS[d]):
                    prev_dir = d
                    break
        next_dir = -1
        if i < n - 1:
            for d in range(6):
                if np.array_equal(bonds[i + 1], _DIRS[d]):
                    next_dir = d
                    break
        cands = [d for d in range(6) if d != _OPPOSITE[prev_dir] if d != next_dir]
        if not cands:
            continue
        d = rng.choice(cands)
        new_bonds[i] = _DIRS[d]
        positions = _bond_to_positions(new_bonds)
        if _is_self_avoiding(positions):
            return new_bonds
        new_bonds[i] = old
    return bonds


def _crossover(parent1, parent2, sequence, rng):
    """Single-point crossover that preserves self-avoidance."""
    n = len(parent1)
    if n == 0:
        return parent1.copy()
    cut = rng.randrange(1, n)
    child = np.concatenate([parent1[:cut], parent2[cut:]], axis=0)
    positions = _bond_to_positions(child)
    if _is_self_avoiding(positions):
        return child
    # Fallback: try a few other cut points.
    for _ in range(10):
        cut2 = rng.randrange(1, n)
        child2 = np.concatenate([parent1[:cut2], parent2[cut2:]], axis=0)
        if _is_self_avoiding(_bond_to_positions(child2)):
            return child2
    return parent1.copy()


def _local_search(sequence, bonds, rng, deadline):
    """First-improvement local search over single-bond direction changes."""
    n = len(bonds)
    if n == 0:
        return bonds
    current = bonds.copy()
    current_e = _energy(sequence, current)
    positions = _bond_to_positions(current)
    occupied = set(map(tuple, positions))
    # Precompute, for each bond index, the set of direction indices that are geometrically
    # forbidden by the immediate neighbours (reverse of previous, same as next).
    while time.time() < deadline:
        improved = False
        idx_order = list(range(n))
        rng.shuffle(idx_order)
        for i in idx_order:
            if time.time() >= deadline:
                break
            prev_dir = -1
            if i > 0:
                for d in range(6):
                    if np.array_equal(current[i - 1], _DIRS[d]):
                        prev_dir = d
                        break
            next_dir = -1
            if i < n - 1:
                for d in range(6):
                    if np.array_equal(current[i + 1], _DIRS[d]):
                        next_dir = d
                        break
            cands = [d for d in range(6) if d != _OPPOSITE[prev_dir] if d != next_dir]
            if not cands:
                continue
            # The old position contributed to occupied; removing it frees the site.
            old_pos = positions[i]
            # New position if we change bond i: shift all subsequent positions.
            # Cheap test: the new endpoint of bond i must not collide with any occupied
            # site other than the old position of i (which is being vacated) and the
            # subsequent sites that will move.
            # Full check is expensive; do a conservative check: new bond endpoint not in
            # occupied minus {old_pos} and not equal to any subsequent position.
            new_bond = None
            for d in cands:
                new_bond = _DIRS[d]
                new_pos_i = positions[i - 1] + new_bond if i > 0 else new_bond
                # Check collision with sites that will NOT move (indices < i) and the
                # vacated site is fine.
                collides = False
                for k in range(i):
                    if tuple(new_pos_i) == tuple(positions[k]):
                        collides = True
                        break
                if collides:
                    continue
                # Tentatively build full positions to verify self-avoidance.
                new_positions = positions.copy()
                delta = new_pos_i - old_pos
                if i + 1 < len(positions):
                    new_positions[i + 1:] = positions[i + 1:] + delta
                new_positions[i] = new_pos_i
                if _is_self_avoiding(new_positions):
                    new_e = _energy_from_positions(sequence, new_positions)
                    if new_e < current_e:
                        current[i] = new_bond
                        positions = new_positions
                        current_e = new_e
                        improved = True
                        break
        if not improved:
            break
    return current


def _genetic_algorithm(sequence, n, rng, deadline):
    """Genetic algorithm over bond-vector sequences preserving self-avoidance."""
    pop = []
    # Seed the population with valid self-avoiding walks.
    while len(pop) < 8 and time.time() < deadline:
        b = _initial_bonds(n, rng)
        if b is not None:
            pop.append(b)
    if not pop:
        return None
    # Evaluate initial population.
    energies = [_energy(sequence, b) for b in pop]
    best_idx = int(np.argmin(energies))
    best_bonds = pop[best_idx].copy()
    best_e = energies[best_idx]

    gen = 0
    while time.time() < deadline:
        gen += 1
        # Selection: keep the top half (elitism).
        order = np.argsort(energies)
        elites = [pop[i].copy() for i in order[:max(2, len(pop) // 2)]]
        new_pop = list(elites)
        # Fill the rest with crossover + mutation offspring.
        while len(new_pop) < len(pop) and time.time() < deadline:
            p1 = pop[rng.randrange(len(pop))]
            p2 = pop[rng.randrange(len(pop))]
            child = _crossover(p1, p2, sequence, rng)
            child = _mutate_bonds(child, sequence, rng)
            new_pop.append(child)
        pop = new_pop
        energies = [_energy(sequence, b) for b in pop]
        bi = int(np.argmin(energies))
        if energies[bi] < best_e:
            best_e = energies[bi]
            best_bonds = pop[bi].copy()
        # Periodically inject a fresh random walk to maintain diversity.
        if gen % 5 == 0:
            b = _initial_bonds(n, rng)
            if b is not None:
                pop[rng.randrange(len(pop))] = b
                energies[rng.randrange(len(pop))] = _energy(sequence, b)
    return best_bonds


def fold(sequence, dimension):
    """Best conformation via genetic search over bond-vector sequences plus polish."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    start_time = time.time()
    # Leave a safety margin before the 300 s wall-clock limit.
    deadline = start_time + 270.0
    rng = random.Random(12345 + len(sequence))

    best_bonds = None
    best_e = 1
    # Phase 1: genetic algorithm over bond-vector sequences.
    ga_deadline = start_time + 150.0
    result = _genetic_algorithm(sequence, n, rng, ga_deadline)
    if result is not None:
        best_bonds = result
        best_e = _energy(sequence, result)

    # Phase 2: local-search polish pass on the best bond sequence.
    if best_bonds is not None and time.time() < deadline:
        polished = _local_search(sequence, best_bonds, rng, deadline)
        pe = _energy(sequence, polished)
        if pe < best_e:
            best_bonds = polished
            best_e = pe

    if best_bonds is None:
        # Fallback: a straight line is always self-avoiding.
        best_bonds = np.array([[1, 0, 0]] * (n - 1), dtype=np.int32)
        best_e = _energy(sequence, best_bonds)

    positions = _bond_to_positions(best_bonds)
    # Prepend the origin.
    full = np.vstack([np.zeros(3, dtype=np.int32), positions])
    return [list(p) for p in full]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

# Deadline guard: keep a safety margin before the 330s wall-clock kill
_DEADLINE = 320.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _neighbours(point):
    """Yield the 6 neighbours of a point on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_h_set(sequence):
    """Set of indices that are H residues."""
    return set(i for i, c in enumerate(sequence) if c == 'H')


def _initial_grow(sequence, dimension, rng):
    """Greedy growth to get a valid self-avoiding walk, preferring H-H contacts."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    h_set = _build_h_set(sequence)
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and j in h_set:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    """Compute HP energy: minus the number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_move(sequence, chain, occupied, i, new_site, h_set):
    """Compute the change in energy if residue i is moved to new_site."""
    old_site = chain[i]
    if new_site == old_site:
        return 0
    if new_site in occupied:
        return None  # invalid move

    # Remove contacts involving old_site
    delta = 0
    if i in h_set:
        for q in _neighbours(old_site):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and j in h_set:
                delta += 1  # losing a contact

    # Add contacts involving new_site
    if i in h_set:
        for q in _neighbours(new_site):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and j in h_set:
                delta -= 1  # gaining a contact

    return delta


def _anneal(sequence, dimension, chain, rng, max_iters=None):
    """Simulated annealing with contact-aware moves."""
    n = len(chain)
    h_set = _build_h_set(sequence)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    if max_iters is None:
        max_iters = max(2000, n * 50)

    # Temperature schedule
    t_start = max(1.0, abs(current_energy) * 0.5)
    t_end = 0.01
    cool_rate = (t_end / t_start) ** (1.0 / max(1, max_iters - 1))

    temp = t_start
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 5.0:
            break

        # Pick a random residue to move
        i = rng.randint(0, n - 1)
        old_site = chain[i]

        # Try all 6 neighbours
        candidates = []
        for ns in _neighbours(old_site):
            if ns in occupied:
                continue
            # Check that moving i to ns doesn't create adjacency conflict
            # (i.e., ns is not adjacent to i-1 or i+1 in the chain unless those are the same)
            valid = True
            if i > 0 and ns == chain[i - 1]:
                valid = False
            if i < n - 1 and ns == chain[i + 1]:
                valid = False
            if not valid:
                continue

            de = _delta_move(sequence, chain, occupied, i, ns, h_set)
            if de is not None:
                candidates.append((de, ns))

        if not candidates:
            temp *= cool_rate
            continue

        # Choose move: prefer negative delta (energy improvement)
        # Use Metropolis acceptance
        best_de = min(c[0] for c in candidates)
        if best_de <= 0:
            # Always accept improvements; pick the best one
            de, ns = min(candidates, key=lambda x: x[0])
        else:
            # Pick a random candidate and apply Metropolis
            de, ns = rng.choice(candidates)
            if rng.random() > math.exp(-de / temp):
                temp *= cool_rate
                continue

        # Execute the move
        old_site = chain[i]
        del occupied[old_site]
        chain[i] = ns
        occupied[ns] = i
        current_energy += de

        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)

        temp *= cool_rate

    return best_chain, best_energy


def _local_search_contacts(sequence, dimension, chain, rng, max_iters=500):
    """Greedy local search: repeatedly move H residues to create new H-H contacts."""
    n = len(chain)
    h_set = _build_h_set(sequence)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    improved = True

    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 5.0:
            break
        if not improved:
            break
        improved = False

        # Try moving each H residue to a neighbour that creates a new H-H contact
        for i in sorted(h_set, key=lambda x: rng.random()):
            old_site = chain[i]
            for ns in _neighbours(old_site):
                if ns in occupied:
                    continue
                if i > 0 and ns == chain[i - 1]:
                    continue
                if i < n - 1 and ns == chain[i + 1]:
                    continue

                de = _delta_move(sequence, chain, occupied, i, ns, h_set)
                if de is not None and de < 0:
                    # Execute move
                    del occupied[old_site]
                    chain[i] = ns
                    occupied[ns] = i
                    current_energy += de
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break  # Move to next H residue

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using parallel restarts with growth + anneal + local search."""
    n = len(sequence)
    h_set = _build_h_set(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # worst possible (unfolded)

    # Determine number of restarts based on time budget
    # Each restart: grow (~O(n^2)) + anneal (~O(n * iters)) + local search
    # For n=48, one full cycle ~ 0.05s; for n=136, ~ 0.2s
    # Budget: ~300s total for all 26 sequences, so ~11.5s per sequence
    # But we need to be safe: use elapsed time guard

    max_restarts = 200 if n <= 64 else 100 if n <= 96 else 50

    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - 10.0:
            break

        # Phase 1: Greedy growth
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Phase 2: Simulated annealing
        chain, energy = _anneal(sequence, dimension, chain, rng)

        # Phase 3: Contact local search
        chain, energy = _local_search_contacts(sequence, dimension, chain, rng)

        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

    if best_chain is None:
        # Fallback: straight line (always self-avoiding)
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

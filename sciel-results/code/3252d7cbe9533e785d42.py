"""HP lattice protein folding on the square lattice – scratch rewrite."""
# EVOLVE-BLOCK-START
import random
from collections import deque


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each non-consecutive H-H pair on adjacent sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.5):
    """
    Randomized chain growth with simulated-annealing-like acceptance.
    Prefers sites that create H-H contacts, but occasionally accepts sub-optimal
    moves to escape local minima.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        current_pos = chain[-1]
        candidates = []
        for site in _neighbours(current_pos):
            if site in occupied:
                continue
            # Score: number of new H-H contacts this placement would create
            gain = 0
            if sequence[i] == 'H':
                for nb in _neighbours(site):
                    j = occupied.get(nb)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Bonus for having free space (avoid trapping)
            free_count = sum(1 for nb in _neighbours(site) if nb not in occupied)
            score = gain * 10.0 + free_count * 0.1 + rng.random() * temperature
            candidates.append((score, site))

        if not candidates:
            return None

        # Sort by score descending; pick top with some randomness
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Use a softmax-like selection among top few
        top_k = min(3, len(candidates))
        if rng.random() < 0.7:
            # Mostly pick the best
            site = candidates[0][1]
        elif top_k > 1 and rng.random() < 0.5:
            site = candidates[1][1]
        else:
            site = candidates[rng.randint(0, top_k - 1)][1]

        chain.append(site)
        occupied[site] = i

    return chain


def _local_search(sequence, chain, rng, max_moves=200):
    """
    1-step local search: try moving one non-terminal residue to an adjacent free site
    if it improves (or doesn't worsen) the energy.
    """
    n = len(chain)
    if n < 3:
        return chain

    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    for _ in range(max_moves):
        # Pick a random internal position to try moving
        pos = rng.randint(1, n - 2)
        current_site = best_chain[pos]
        moved = False

        for new_site in _neighbours(current_site):
            # Check self-avoidance: new_site must not be occupied by any other residue
            occupied_set = set(best_chain)
            occupied_set.discard(current_site)
            if new_site in occupied_set:
                continue
            # Check that the chain remains a valid walk: consecutive residues must be adjacent
            # Moving pos affects edges (pos-1, pos) and (pos, pos+1)
            if pos > 0:
                prev = best_chain[pos - 1]
                if abs(new_site[0] - prev[0]) + abs(new_site[1] - prev[1]) != 1:
                    continue
            if pos < n - 1:
                nxt = best_chain[pos + 1]
                if abs(new_site[0] - nxt[0]) + abs(new_site[1] - nxt[1]) != 1:
                    continue

            # Try the move
            trial = list(best_chain)
            trial[pos] = new_site
            e = _energy(sequence, trial)
            if e < best_energy or (e == best_energy and rng.random() < 0.1):
                best_chain = trial
                best_energy = e
                moved = True
                break

        if not moved and rng.random() < 0.3:
            # Occasionally accept a neutral move to explore
            pos = rng.randint(1, n - 2)
            current_site = best_chain[pos]
            for new_site in _neighbours(current_site):
                occupied_set = set(best_chain)
                occupied_set.discard(current_site)
                if new_site in occupied_set:
                    continue
                if pos > 0:
                    prev = best_chain[pos - 1]
                    if abs(new_site[0] - prev[0]) + abs(new_site[1] - prev[1]) != 1:
                        continue
                if pos < n - 1:
                    nxt = best_chain[pos + 1]
                    if abs(new_site[0] - nxt[0]) + abs(new_site[1] - nxt[1]) != 1:
                        continue
                trial = list(best_chain)
                trial[pos] = new_site
                e = _energy(sequence, trial)
                if e == best_energy:
                    best_chain = trial
                    break

    return best_chain


def _hill_climb_refine(sequence, chain, rng, iterations=50):
    """
    Iterative improvement: repeatedly try single-residue moves that improve energy.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)

    for _ in range(iterations):
        improved = False
        for pos in range(1, n - 1):
            current_site = best_chain[pos]
            for new_site in _neighbours(current_site):
                # Check self-avoidance
                occupied_set = set(best_chain)
                occupied_set.discard(current_site)
                if new_site in occupied_set:
                    continue
                # Check walk validity
                if pos > 0:
                    prev = best_chain[pos - 1]
                    if abs(new_site[0] - prev[0]) + abs(new_site[1] - prev[1]) != 1:
                        continue
                if pos < n - 1:
                    nxt = best_chain[pos + 1]
                    if abs(new_site[0] - nxt[0]) + abs(new_site[1] - nxt[1]) != 1:
                        continue

                trial = list(best_chain)
                trial[pos] = new_site
                e = _energy(sequence, trial)
                if e < best_energy:
                    best_chain = trial
                    best_energy = e
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_chain


def fold(sequence, dimension):
    """
    Fold an HP sequence on a 2D square lattice.
    Uses multiple randomized greedy growths followed by local search refinement.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    # Number of trials scaled by sequence length, capped for time budget
    if n <= 30:
        num_trials = 300
    elif n <= 60:
        num_trials = 150
    else:
        num_trials = 80

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0  # We want most negative (lowest) energy

    for trial in range(num_trials):
        # Vary temperature for diversity
        temp = 0.3 + 0.4 * (trial / num_trials)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        # Refine with local search
        refined = _local_search(sequence, chain, rng, max_moves=min(100, n * 2))
        energy = _energy(sequence, refined)

        if best_chain is None or energy < best_energy:
            best_chain = refined
            best_energy = energy

            # If we've found a very good solution, do additional hill climbing
            if energy < -n // 3:
                further = _hill_climb_refine(sequence, best_chain, rng, iterations=30)
                e2 = _energy(sequence, further)
                if e2 < best_energy:
                    best_chain = further
                    best_energy = e2

    if best_chain is None:
        # Fallback: straight line (always valid, energy = 0)
        best_chain = [[i, 0] for i in range(n)]

    return best_chain
# EVOLVE-BLOCK-END

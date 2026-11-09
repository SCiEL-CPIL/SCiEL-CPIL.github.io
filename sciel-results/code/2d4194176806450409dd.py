"""Rewritten HP lattice protein folding program."""
# EVOLVE-BLOCK-START
import time
import math

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta_local(sequence, chain, where, idx, new_site):
    """
    Compute energy delta for moving residue idx from chain[idx] to new_site.
    Only contacts involving residue idx can change.
    Returns delta_e (new_energy - old_energy).
    """
    old_site = chain[idx]
    if new_site == old_site:
        return 0

    # Check if new_site is occupied by another residue
    if new_site in where and where[new_site] != idx:
        return None  # invalid

    # Contacts that will be broken: H-H contacts between idx and other residues
    # that are adjacent to old_site
    broken = 0
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                broken += 1

    # Contacts that will be formed: H-H contacts between idx and other residues
    # that are adjacent to new_site
    formed = 0
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                # Make sure j is not at old_site (we're removing idx from old_site)
                # Actually j is at q, and we're moving idx to new_site.
                # If q == old_site, that would mean j is at old_site, but we're
                # removing idx from there. But j != idx, so j is a different residue
                # at old_site? No, old_site was occupied by idx. So no other
                # residue is at old_site.
                formed += 1

    return -(formed - broken)


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _anneal(sequence, chain, max_time, seed):
    """
    Simulated annealing with local moves (pivot), reversals, and rotations.
    Uses incremental energy computation for speed.
    """
    n = len(chain)
    if n <= 2:
        return list(chain), _energy(sequence, chain)

    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 4.0
    T_min = 0.05
    cooling = 0.9998
    rng = __import__('random').Random(seed)

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.5:
            # Local move (pivot)
            idx = rng.randint(1, n - 2)
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]

            candidates = []
            for site in _neighbours(old_site):
                if site == old_site:
                    continue
                if _manhattan(site, prev_site) != 1:
                    continue
                if _manhattan(site, next_site) != 1:
                    continue
                if site in where and where[site] != idx:
                    continue
                candidates.append(site)

            if candidates:
                new_site = rng.choice(candidates)
                if new_site != old_site:
                    delta_e = _energy_delta_local(sequence, chain, where, idx, new_site)
                    if delta_e is not None:
                        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                            del where[old_site]
                            where[new_site] = idx
                            chain[idx] = new_site
                            current_energy += delta_e
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = list(chain)
        elif r < 0.8:
            # Reversal move
            if n < 4:
                T *= cooling
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)

            # Check boundary connectivity
            valid = True
            if i > 0:
                if _manhattan(chain[i - 1], chain[j]) != 1:
                    valid = False
            if valid and j < n - 1:
                if _manhattan(chain[i], chain[j + 1]) != 1:
                    valid = False

            if valid:
                # Perform reversal
                chain[i:j + 1] = chain[i:j + 1][::-1]
                for k in range(i, j + 1):
                    where[chain[k]] = k

                new_energy = _energy(sequence, chain)
                delta_e = new_energy - current_energy

                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    chain[i:j + 1] = chain[i:j + 1][::-1]
                    for k in range(i, j + 1):
                        where[chain[k]] = k
        else:
            # Rotation move
            if n < 4:
                T *= cooling
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)

            pivot = chain[i]
            seg = chain[i:j + 1]

            # Rotate 90 degrees clockwise around pivot
            rotated_seg = [pivot]
            for pt in seg[1:]:
                dx = pt[0] - pivot[0]
                dy = pt[1] - pivot[1]
                new_pt = (pivot[0] + dy, pivot[1] - dx)
                rotated_seg.append(new_pt)

            # Check self-avoidance
            new_where = dict(where)
            for pt in seg[1:]:
                if pt in new_where:
                    del new_where[pt]

            valid = True
            for pt in rotated_seg[1:]:
                if pt in new_where:
                    valid = False
                    break

            if valid and j < n - 1:
                if _manhattan(rotated_seg[-1], chain[j + 1]) != 1:
                    valid = False

            if valid:
                chain[i:j + 1] = rotated_seg
                for k in range(i, j + 1):
                    where[chain[k]] = k

                new_energy = _energy(sequence, chain)
                delta_e = new_energy - current_energy

                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    chain[i:j + 1] = seg
                    for k in range(i, j + 1):
                        where[chain[k]] = k

        T *= cooling

    return best_chain, best_energy


def _polish(sequence, chain, max_time, seed):
    """
    High-precision polish: only accept moves that improve energy.
    """
    n = len(chain)
    if n <= 2:
        return list(chain), _energy(sequence, chain)

    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = __import__('random').Random(seed)
    start = time.time()

    while (time.time() - start) < max_time:
        improved = False
        r = rng.random()
        if r < 0.6:
            idx = rng.randint(1, n - 2)
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]

            for site in _neighbours(old_site):
                if site == old_site:
                    continue
                if _manhattan(site, prev_site) != 1:
                    continue
                if _manhattan(site, next_site) != 1:
                    continue
                if site in where and where[site] != idx:
                    continue

                delta_e = _energy_delta_local(sequence, chain, where, idx, site)
                if delta_e is not None and delta_e < 0:
                    del where[old_site]
                    where[site] = idx
                    chain[idx] = site
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
                    break
        else:
            if n < 4:
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)

            valid = True
            if i > 0 and _manhattan(chain[i - 1], chain[j]) != 1:
                valid = False
            if valid and j < n - 1 and _manhattan(chain[i], chain[j + 1]) != 1:
                valid = False

            if valid:
                chain[i:j + 1] = chain[i:j + 1][::-1]
                for k in range(i, j + 1):
                    where[chain[k]] = k
                new_energy = _energy(sequence, chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
                else:
                    chain[i:j + 1] = chain[i:j + 1][::-1]
                    for k in range(i, j + 1):
                        where[chain[k]] = k

        if not improved and rng.random() < 0.1:
            # Try a few more iterations before giving up
            pass

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies and annealing.
    """
    n = len(sequence)
    rng = __import__('random').Random(12345 + n)
    best = None
    best_energy = 0

    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 50
    biases = [1.0, 1.5, 2.0, 0.5, 0.8, 1.2, 1.8, 0.7, 1.3, 0.6]
    for trial in range(growth_budget):
        if _remaining_time() < 20:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: simple linear chain
        best = [[i, 0] for i in range(n)]
        return best

    # Sort by initial energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))

    # Phase 2: Anneal each initial conformation
    # Allocate time based on remaining budget
    num_to_anneal = min(len(initial_chains), 10)
    time_per_anneal = max(0.5, (_remaining_time() - 5) / max(1, num_to_anneal))

    for idx in range(num_to_anneal):
        if _remaining_time() < 3:
            break
        chain = initial_chains[idx]
        # Give more time to better initial conformations
        if idx < 3:
            t = time_per_anneal * 1.5
        else:
            t = time_per_anneal
        t = min(t, _remaining_time() - 2)
        if t <= 0:
            break

        annealed_chain, annealed_energy = _anneal(sequence, chain, max_time=t, seed=12345 + idx)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    # Phase 3: Polish the best result
    if best is not None and _remaining_time() > 2:
        polish_time = min(5.0, _remaining_time() - 1)
        polished_chain, polished_energy = _polish(sequence, best, max_time=polish_time, seed=99999)
        if polished_energy < best_energy:
            best, best_energy = polished_chain, polished_energy

    if best is None:
        best = [[i, 0] for i in range(n)]

    return [list(p) if isinstance(p, tuple) else p for p in best]
# EVOLVE-BLOCK-END

"""HP lattice protein folding on the square lattice – path-based SA strategy."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _random_saw(n, rng):
    """Generate a random self-avoiding walk of length n. Returns None if stuck."""
    if n <= 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    for _ in range(1, n):
        last = chain[-1]
        options = [s for s in _neighbours(last) if s not in occupied]
        if not options:
            return None
        choice = rng.choice(options)
        chain.append(choice)
        occupied.add(choice)
    return chain


def _regrow_segment(sequence, chain, split, rng, temperature=1.0):
    """
    Regrow the suffix starting at index `split` (0-based) of the chain.
    The prefix chain[:split] is kept fixed. We regrow from chain[split-1]
    to place residues split..n-1.
    Returns the new suffix (list of points) or None if stuck.
    """
    n = len(sequence)
    if split >= n:
        return []
    if split == 0:
        # Regrow entire chain
        start = (0, 0)
        occupied = {start: 0}
        new_chain = [start]
        for i in range(1, n):
            last = new_chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 10.0 + free * 0.1 + rng.random() * temperature
                options.append((score, site))
            if not options:
                return None
            best_site = max(options, key=lambda x: x[0])[1]
            occupied[best_site] = i
            new_chain.append(best_site)
        return new_chain[1:]  # exclude start (0,0) which is index 0

    # split > 0: prefix is chain[:split], last prefix point is chain[split-1]
    prefix = chain[:split]
    prefix_occupied = {p: i for i, p in enumerate(prefix)}
    last_prefix = prefix[-1]

    new_suffix = []
    occupied = dict(prefix_occupied)
    current_pos = last_prefix

    for i in range(split, n):
        options = []
        for site in _neighbours(current_pos):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        new_suffix.append(best_site)
        current_pos = best_site

    return new_suffix


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Local search using single-residue moves and regrow moves.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 3:
        return list(chain), _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for move_iter in range(max_moves):
        if time.time() > _DEADLINE - 10:
            break

        if rng.random() < 0.7:
            # Single-residue move
            i = rng.randint(1, n - 2)
            p = chain[i]

            for np_ in _neighbours(p):
                if np_ in occupied and occupied[np_] != i:
                    continue
                dist1 = abs(np_[0] - chain[i - 1][0]) + abs(np_[1] - chain[i - 1][1])
                dist2 = abs(np_[0] - chain[i + 1][0]) + abs(np_[1] - chain[i + 1][1])
                if dist1 != 1 or dist2 != 1:
                    continue

                del occupied[p]
                chain[i] = np_
                occupied[np_] = i
                new_energy = _compute_energy(sequence, chain)

                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                else:
                    del occupied[np_]
                    chain[i] = p
                    occupied[p] = i
                    no_improve_count += 1
                break
        else:
            # Regrow move
            split = rng.randint(2, max(2, n - 3))
            prefix = chain[:split]
            prefix_occupied = {p: i for i, p in enumerate(prefix)}

            new_suffix = _regrow_segment(sequence, chain, split, rng, temperature=1.0)

            if new_suffix is not None:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)

                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    occupied = {p: i for i, p in enumerate(chain)}
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        if no_improve_count > 500:
            break

    return best_chain, best_energy


def _path_sa(sequence, rng, time_budget):
    """
    Path-based simulated annealing:
    1. Start from a random SAW.
    2. Use regrow-segment moves with Metropolis acceptance.
    3. Adaptive cooling schedule.
    """
    n = len(sequence)
    if n < 2:
        return _random_saw(n, rng) or [(0, 0)]

    # Initial random SAW
    chain = _random_saw(n, rng)
    if chain is None:
        # Fallback: greedy
        chain = _greedy_grow_simple(sequence, rng)
        if chain is None:
            return [(i, 0) for i in range(n)]

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    # SA parameters
    t_start = 2.0
    t_end = 0.01
    total_steps = 0
    # Estimate steps per second based on n
    steps_per_sec = max(500, 50000 // n)
    max_steps = int(time_budget * steps_per_sec)
    max_steps = min(max_steps, 200000)  # cap

    for step in range(max_steps):
        if time.time() > _DEADLINE - 15:
            break

        # Adaptive temperature: geometric cooling with occasional reheating
        progress = step / max(1, max_steps)
        temperature = t_start * (t_end / t_start) ** progress
        # Add small noise to temperature
        temperature *= (1.0 + 0.1 * rng.random())

        # Choose split point for regrow
        # Bias toward middle and high-impact regions
        if n <= 4:
            split = 1
        else:
            # Center-biased: prefer splits in the middle third
            low = max(2, n // 4)
            high = min(n - 2, 3 * n // 4)
            if rng.random() < 0.6:
                split = rng.randint(low, high)
            else:
                split = rng.randint(2, max(2, n - 2))

        new_suffix = _regrow_segment(sequence, chain, split, rng, temperature=temperature)

        if new_suffix is None:
            continue

        prefix = chain[:split]
        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)

        delta_e = new_energy - current_energy

        # Metropolis acceptance
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < math.exp(-delta_e / max(temperature, 1e-6))

        if accept:
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Small chance to do a local improvement kick
            if rng.random() < 0.05:
                improved, imp_e = _local_improve(sequence, chain, rng, max_moves=200)
                if imp_e < best_energy:
                    best_chain = improved
                    best_energy = imp_e
                    chain = improved
                    current_energy = imp_e

    return best_chain


def _greedy_grow_simple(sequence, rng, temperature=0.0):
    """Simple greedy growth fallback."""
    n = len(sequence)
    if n <= 0:
        return []
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def fold(sequence, dimension):
    """
    Fold the sequence using path-based simulated annealing with regrow moves.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n <= 1:
        return [[0, 0]] if n == 1 else []

    # Allocate time budget
    time_left = _DEADLINE - time.time()
    if time_left <= 5:
        # Emergency: quick greedy
        chain = _greedy_grow_simple(sequence, rng)
        if chain is None:
            chain = [(i, 0) for i in range(n)]
        return [list(p) for p in chain]

    # Phase 1: Path-based SA with multiple restarts
    best_chain = None
    best_energy = 0

    # Number of SA restarts based on time and sequence length
    # Shorter sequences can afford more restarts
    num_sa = min(50, max(5, int(200 // n)))

    for attempt in range(num_sa):
        if time.time() > _DEADLINE - 20:
            break

        # Allocate time per SA run
        remaining = _DEADLINE - time.time() - 20
        per_run = remaining / (num_sa - attempt)
        per_run = max(0.5, min(per_run, 15.0))

        chain = _path_sa(sequence, rng, per_run)
        if chain is None:
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local improvement on the best
    if best_chain is not None:
        remaining = _DEADLINE - time.time() - 10
        if remaining > 2:
            improved_chain, improved_energy = _local_improve(
                sequence, best_chain, rng, max_moves=min(10000, int(remaining * 1000)))
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy

    # Phase 3: Second local improvement if time permits
    if best_chain is not None and time.time() < _DEADLINE - 15:
        remaining = _DEADLINE - time.time() - 15
        if remaining > 1:
            improved_chain2, improved_energy2 = _local_improve(
                sequence, best_chain, rng, max_moves=min(5000, int(remaining * 800)))
            if improved_energy2 < best_energy:
                best_chain = improved_chain2
                best_energy = improved_energy2

    # Fallback
    if best_chain is None:
        best_chain = _greedy_grow_simple(sequence, rng)
        if best_chain is None:
            best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

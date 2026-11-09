"""HP lattice protein folding: diverse growth + SA with reversals/pivots + greedy polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _energy_delta(sequence, chain, idx, old_site, new_site, where):
    """
    Compute energy delta for moving residue idx from old_site to new_site.
    Returns delta_e (new_energy - old_energy).
    """
    delta = 0
    # Remove contacts involving idx at old_site
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta += 1
    # Add contacts involving idx at new_site
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta -= 1
    return delta


def _grow(sequence, dimension, rng, bias_hh=1.0, bias_free=0.1):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias_hh + free * bias_free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move (pivot). Returns (new_chain, new_where, new_energy, accepted).
    Uses incremental energy delta for speed.
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    # Compute delta before modifying state
    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)

    if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
        del where[old_site]
        where[new_site] = idx
        chain[idx] = new_site
        new_energy = current_energy + delta_e
        return chain, where, new_energy, True
    else:
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Check self-avoidance before applying
    seg = chain[i:j + 1]
    new_seg = seg[::-1]
    
    # Temporarily remove segment from where
    for pt in seg:
        if pt in where:
            del where[pt]
    
    # Check for collisions
    collision = False
    for pt in new_seg:
        if pt in where:
            collision = True
            break
    
    if collision:
        # Restore
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False
    
    # Apply reversal
    chain[i:j + 1] = new_seg
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        chain[i:j + 1] = seg
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _pivot_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a pivot move: rotate the tail of the chain around a pivot point.
    Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 5:
        return chain, where, current_energy, False

    # Choose a pivot position (not at the ends)
    i = rng.randint(1, n - 2)
    pivot = chain[i]
    tail = chain[i + 1:]
    prefix_points = set(chain[:i + 1])

    # Try 3 rotations (90, 180, 270 degrees)
    for k in (1, 2, 3):
        rotated = []
        valid = True
        for p in tail:
            x = p[0] - pivot[0]
            y = p[1] - pivot[1]
            for _ in range(k):
                x, y = -y, x
            new_p = (pivot[0] + x, pivot[1] + y)
            rotated.append(new_p)
            if new_p in prefix_points:
                valid = False
                break

        if not valid:
            continue

        # Check that the first rotated point is not the same as the previous point
        if rotated[0] == chain[i - 1]:
            continue

        # Apply the rotation
        old_tail = tail[:]
        chain[i + 1:] = rotated
        for j, p in enumerate(rotated, start=i + 1):
            where[p] = j
        # Remove old tail positions from where
        for p in old_tail:
            if p in where and where[p] > i:
                del where[p]

        new_energy = _energy(sequence, chain)
        delta_e = new_energy - current_energy

        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            return chain, where, new_energy, True
        else:
            # Revert
            chain[i + 1:] = old_tail
            for j, p in enumerate(old_tail, start=i + 1):
                where[p] = j
            for p in rotated:
                if p in where and where[p] > i:
                    del where[p]

    return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0, seed=12345):
    """Simulated annealing with local moves, reversals, and pivots."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 3.0
    T_min = 0.005
    cooling = 0.999
    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 5000

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.40:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.70:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _pivot_move(sequence, chain, where, current_energy, T, rng)

        if accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Restart if stuck for too long
        if no_improve_count > max_no_improve:
            T *= 0.95
            no_improve_count = 0

        T *= cooling

    return best_chain, best_energy


def _polish(sequence, chain, max_time=3.0, seed=77777):
    """
    High-precision polish pass: greedy local search at very low temperature
    to find the exact local optimum around the best conformation.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = random.Random(seed)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        # Try all possible single-residue moves greedily
        for idx in range(1, n - 1):
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]
            for site in _neighbours(old_site):
                if site == old_site:
                    continue
                if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
                    continue
                if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
                    continue
                if site in where:
                    occ_idx = where[site]
                    if occ_idx != idx:
                        continue
                delta_e = _energy_delta(sequence, chain, idx, old_site, site, where)
                if delta_e < 0:
                    del where[old_site]
                    where[site] = idx
                    chain[idx] = site
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break  # move to next idx after a successful move
        # Also try small segment reversals greedily
        for i in range(n - 2):
            for j in range(i + 2, n):
                if j - i > 5:  # limit segment length
                    break
                # Check boundary connectivity
                if i > 0:
                    if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
                        continue
                if j < n - 1:
                    if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
                        continue
                # Check self-avoidance of reversed segment
                seg = chain[i:j + 1]
                new_seg = seg[::-1]
                # Remove old segment from where
                for pt in seg:
                    if where.get(pt) is not None and i <= where[pt] <= j:
                        del where[pt]
                # Check for collisions
                collision = False
                for pt in new_seg:
                    if pt in where:
                        collision = True
                        break
                if collision:
                    # Restore
                    for k in range(i, j + 1):
                        where[chain[k]] = k
                    continue
                # Compute energy delta
                old_e = 0
                for k in range(i, j + 1):
                    if sequence[k] == 'H':
                        for q in _neighbours(chain[k]):
                            jj = where.get(q)
                            if jj is not None and abs(jj - k) > 1 and sequence[jj] == 'H' and jj > k:
                                old_e += 1
                # Temporarily apply
                for k in range(i, j + 1):
                    where[new_seg[k - i]] = k
                new_e = 0
                for k in range(i, j + 1):
                    if sequence[k] == 'H':
                        for q in _neighbours(new_seg[k - i]):
                            jj = where.get(q)
                            if jj is not None and abs(jj - k) > 1 and sequence[jj] == 'H' and jj > k:
                                new_e += 1
                delta_e = new_e - old_e
                if delta_e < 0:
                    for k in range(i, j + 1):
                        chain[k] = new_seg[k - i]
                        where[chain[k]] = k
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    # Restore where
                    for k in range(i, j + 1):
                        where[chain[k]] = k
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies and annealing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 500
    biases = [(1.0, 0.1), (1.5, 0.1), (2.0, 0.1), (0.5, 0.2), (0.8, 0.15),
              (1.2, 0.1), (1.8, 0.05), (0.6, 0.25), (1.4, 0.1), (0.9, 0.15)]
    for trial in range(growth_budget):
        if _remaining_time() < 20:
            break
        bias_hh, bias_free = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias_hh=bias_hh, bias_free=bias_free)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation
    # Allocate time proportionally, but give more to the first few
    time_per_anneal = max(0.3, _remaining_time() / max(1, len(initial_chains)) * 0.8)

    # Sort by initial energy to prioritize better starts
    initial_chains.sort(key=lambda c: _energy(sequence, c))

    for idx, chain in enumerate(initial_chains):
        if _remaining_time() < 10:
            break
        # Give more time to earlier (better) chains
        if idx < 3:
            t = time_per_anneal * 2.0
        elif idx < 10:
            t = time_per_anneal * 1.5
        else:
            t = time_per_anneal
        t = min(t, _remaining_time() - 5)
        if t <= 0:
            break
        annealed_chain, annealed_energy = _anneal(sequence, dimension, chain, max_time=t, seed=12345 + idx)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Phase 3: High-precision polish - greedy local search for exact local optimum
    if _remaining_time() > 5 and best is not None:
        polish_time = min(8.0, _remaining_time() - 3)
        best, best_energy = _polish(sequence, best, max_time=polish_time, seed=99999)

    # Phase 4: If time remains, do one more anneal pass on the polished result
    if _remaining_time() > 10 and best is not None:
        extra_time = min(5.0, _remaining_time() - 5)
        best, best_energy = _anneal(sequence, dimension, best, max_time=extra_time, seed=55555)
        # Final polish again
        if _remaining_time() > 5:
            final_polish_time = min(3.0, _remaining_time() - 2)
            best, best_energy = _polish(sequence, best, max_time=final_polish_time, seed=88888)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

"""HP lattice protein folding using correlated pivots, greedy initialization, and memetic SA."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 198.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, DEADLINE - (time.time() - _START_TIME))


def _neighbours(x, y):
    """Return 4 neighbors as tuples."""
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy_full(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if sequence[i] == 'H':
            for nx, ny in _neighbours(x, y):
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _energy_delta_move(sequence, chain, where, idx, old_site, new_site):
    """
    Compute energy delta for moving residue idx from old_site to new_site.
    Assumes new_site is not occupied by any other residue.
    Returns delta_e = new_energy - old_energy.
    """
    delta = 0
    ox, oy = old_site
    nx, ny = new_site
    if sequence[idx] == 'H':
        # Remove old contacts
        for qx, qy in _neighbours(ox, oy):
            j = where.get((qx, qy))
            if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                delta += 1
        # Add new contacts
        for qx, qy in _neighbours(nx, ny):
            j = where.get((qx, qy))
            if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                delta -= 1
    return delta


def _grow_greedy(sequence, rng, temp=0.0):
    """
    Greedy chain growth with contact-aware scoring.
    temp controls randomness: 0 = fully greedy, higher = more random.
    Returns chain as list of (x,y) tuples or None if stuck.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        last_x, last_y = chain[-1]
        options = []
        for sx, sy in _neighbours(last_x, last_y):
            if (sx, sy) in occupied:
                continue
            # Score: contacts gained + free space
            gain = 0
            if sequence[i] == 'H':
                for qx, qy in _neighbours(sx, sy):
                    j = occupied.get((qx, qy))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for qx, qy in _neighbours(sx, sy):
                if (qx, qy) not in occupied:
                    free += 1
            score = gain * 2.0 + 0.05 * free + rng.random() * temp
            options.append((score, sx, sy))
        if not options:
            return None
        # Pick best (or random among top for diversity)
        options.sort(key=lambda t: -t[0])
        if temp > 0 and rng.random() < 0.3:
            # Pick randomly from top 3
            k = min(3, len(options))
            _, sx, sy = options[rng.randint(0, k - 1)]
        else:
            _, sx, sy = options[0]
        chain.append((sx, sy))
        occupied[(sx, sy)] = i

    return chain


def _correlated_pivot(sequence, chain, where, current_energy, T, rng):
    """
    Correlated pivot: pick a segment [i, j], reverse it, check connectivity and self-avoidance.
    This is a single move that can reconfigure many residues at once.
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    # Choose segment length: short segments more likely
    max_len = min(n - 1, 20)
    seg_len = 1 + rng.randint(0, max_len - 1)
    i = rng.randint(0, n - 1 - seg_len)
    j = i + seg_len

    # Check boundary connectivity after reversal
    if i > 0:
        px, py = chain[i - 1]
        rx, ry = chain[j]
        if abs(px - rx) + abs(py - ry) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        px, py = chain[i]
        rx, ry = chain[j + 1]
        if abs(px - rx) + abs(py - ry) != 1:
            return chain, where, current_energy, False

    # Extract segment
    old_segment = chain[i:j + 1]
    new_segment = old_segment[::-1]

    # Check self-avoidance: new_segment points must not collide with outside
    outside = set()
    for k in range(n):
        if k < i or k > j:
            outside.add(chain[k])

    # Also need to check that new_segment doesn't self-collide (it can't since it's a reversal of a SAW)
    # But it could collide with outside
    valid = True
    for pt in new_segment:
        if pt in outside:
            valid = False
            break
    if not valid:
        return chain, where, current_energy, False

    # Apply
    chain[i:j + 1] = new_segment
    for k in range(i, j + 1):
        where[chain[k]] = k

    # Compute energy delta
    # For a reversal, the set of positions is the same, so contacts within the segment are preserved.
    # Only contacts between the segment and outside change.
    # Actually, the positions are the same set, so the energy is the same!
    # Wait: the positions are the same set, but the chain order changes.
    # H-H contacts depend on lattice adjacency, not chain adjacency (except we exclude chain-adjacent).
    # Since the set of occupied positions is the same, the lattice adjacencies are the same.
    # But the "chain-adjacent" exclusion changes! Two residues that were chain-adjacent in the old
    # segment might not be in the new, and vice versa.
    # So energy CAN change. Let's just recompute for safety (segment is short, so it's cheap).
    new_energy = _energy_full(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
        return chain, where, new_energy, True
    else:
        # Revert
        chain[i:j + 1] = old_segment
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _single_pivot(sequence, chain, where, current_energy, T, rng):
    """Single-residue pivot move."""
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]
    ox, oy = old_site
    px, py = prev_site
    qx, qy = next_site

    candidates = []
    for sx, sy in _neighbours(ox, oy):
        if (sx, sy) == old_site:
            continue
        if abs(sx - px) + abs(sy - py) != 1:
            continue
        if abs(sx - qx) + abs(sy - qy) != 1:
            continue
        occ = where.get((sx, sy))
        if occ is not None and occ != idx:
            continue
        candidates.append((sx, sy))

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    delta_e = _energy_delta_move(sequence, chain, where, idx, old_site, new_site)

    if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
        del where[old_site]
        where[new_site] = idx
        chain[idx] = new_site
        return chain, where, current_energy + delta_e, True
    else:
        return chain, where, current_energy, False


def _local_search(sequence, chain, where, current_energy, rng, max_iter=200):
    """Greedy local search: repeatedly apply improving single-residue moves."""
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        n = len(chain)
        # Try a random subset of positions
        for _ in range(min(n, 20)):
            idx = rng.randint(1, n - 2)
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]
            ox, oy = old_site
            px, py = prev_site
            qx, qy = next_site

            best_delta = 0
            best_site = None
            for sx, sy in _neighbours(ox, oy):
                if (sx, sy) == old_site:
                    continue
                if abs(sx - px) + abs(sy - py) != 1:
                    continue
                if abs(sx - qx) + abs(sy - qy) != 1:
                    continue
                occ = where.get((sx, sy))
                if occ is not None and occ != idx:
                    continue
                delta = _energy_delta_move(sequence, chain, where, idx, old_site, (sx, sy))
                if delta < best_delta:
                    best_delta = delta
                    best_site = (sx, sy)

            if best_site is not None:
                del where[old_site]
                where[best_site] = idx
                chain[idx] = best_site
                current_energy += best_delta
                improved = True

    return chain, where, current_energy


def _anneal(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing with correlated pivots and single pivots, plus periodic local search."""
    start_time = time.time()
    where = {}
    for i, pt in enumerate(chain):
        where[pt] = i
    current_energy = _energy_full(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995

    local_search_counter = 0
    local_search_interval = 500

    while T > T_min and (time.time() - start_time) < time_budget and _remaining_time() > 3:
        # Choose move type
        r = rng.random()
        if r < 0.5:
            chain, where, current_energy, _ = _correlated_pivot(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, _ = _single_pivot(sequence, chain, where, current_energy, T, rng)

        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)

        T *= cooling_rate

        # Periodic local search
        local_search_counter += 1
        if local_search_counter >= local_search_interval:
            local_search_counter = 0
            chain, where, current_energy = _local_search(sequence, chain, where, current_energy, rng, max_iter=50)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)

    # Final local search
    chain, where, current_energy = _local_search(sequence, chain, where, current_energy, rng, max_iter=100)
    if current_energy < best_energy:
        best_energy = current_energy
        best_chain = list(chain)

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy init + memetic SA with correlated pivots."""
    n = len(sequence)
    rng = random.Random(42 + n)

    # Generate diverse initial chains
    initial_chains = []
    temps = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    for t in temps:
        chain = _grow_greedy(sequence, rng, temp=t)
        if chain is not None:
            initial_chains.append(chain)

    # Add a few more random growths
    for _ in range(3):
        chain = _grow_greedy(sequence, rng, temp=rng.uniform(0.1, 1.5))
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]

    # Sort by energy
    initial_chains.sort(key=lambda c: _energy_full(sequence, c))

    best_chain = initial_chains[0]
    best_energy = _energy_full(sequence, best_chain)

    # Time allocation
    total_budget = max(1.0, _remaining_time() - 4.0)

    # Run SA on top initial chains
    num_runs = min(len(initial_chains), 5)

    for i in range(num_runs):
        if _remaining_time() < 4:
            break

        # Allocate time: first run gets more
        frac = 0.4 / (i + 1)
        time_for_this = total_budget * frac
        time_for_this = max(0.5, min(time_for_this, _remaining_time() - 4.0))

        current_chain = list(initial_chains[i])
        improved_chain, improved_energy = _anneal(sequence, current_chain, dimension, rng, time_for_this)

        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    # If we still have time, do another round on the best chain
    if _remaining_time() > 10:
        current_chain = list(best_chain)
        time_left = _remaining_time() - 4.0
        improved_chain, improved_energy = _anneal(sequence, current_chain, dimension, rng, time_left)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

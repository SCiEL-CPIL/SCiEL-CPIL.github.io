"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 165.0


def _neighbours2d(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with lookahead for future H-H contacts."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours2d(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                x, y = site
                for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Lookahead: if next residue is H, check if placing current P near existing H helps
            future_gain = 0.0
            if i + 1 < n and sequence[i + 1] == 'H':
                x, y = site
                for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    if q == last:
                        continue
                    if q not in occupied:
                        qx, qy = q
                        for r in ((qx+1, qy), (qx-1, qy), (qx, qy+1), (qx, qy-1)):
                            j = occupied.get(r)
                            if j is not None and sequence[j] == 'H' and j < i:
                                future_gain += 0.3
                                break
            free = 0
            x, y = site
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if q not in occupied and q != last:
                    free += 1
            score = gain * 2.5 + future_gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _compute_energy_delta(sequence, chain, occupied, i, new_site):
    """Compute energy change when moving residue i to new_site.
    Returns delta_energy (negative = improvement) or None if invalid."""
    n = len(chain)
    old_site = chain[i]
    if new_site == old_site:
        return 0

    # Check self-avoidance
    if new_site in occupied and occupied[new_site] != i:
        return None

    # Check connectivity
    if i > 0:
        px, py = chain[i-1]
        nx, ny = new_site
        if abs(nx - px) + abs(ny - py) != 1:
            return None
    if i < n - 1:
        px, py = chain[i+1]
        nx, ny = new_site
        if abs(nx - px) + abs(ny - py) != 1:
            return None

    # Compute energy delta
    delta = 0
    seq_i = sequence[i]
    ox, oy = old_site
    nx, ny = new_site

    if seq_i == 'H':
        # Remove old contacts
        for q in ((ox+1, oy), (ox-1, oy), (ox, oy+1), (ox, oy-1)):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == 'H':
                delta += 1  # removing contact increases energy

        # Add new contacts
        for q in ((nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == 'H':
                delta -= 1  # adding contact decreases energy

    return delta


def _local_optimize_sa(sequence, chain, occupied, time_budget=5.0):
    """Simulated annealing with single-residue and kink moves using efficient delta."""
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)

    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_energy = _energy(sequence, best_chain)
    current_energy = best_energy

    T = max(0.5, n * 0.05)
    T_min = 0.01
    cooling = 0.998

    start_time = time.time()
    iters = 0
    no_improve_count = 0

    while time.time() < _DEADLINE and (time.time() - start_time) < time_budget:
        iters += 1
        move_type = random.random()

        if move_type < 0.6:
            # Single residue move
            i = random.randint(1, n - 2)
            x, y = chain[i]
            candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            new_site = random.choice(candidates)
            delta = _compute_energy_delta(sequence, chain, occupied, i, new_site)
            if delta is None:
                continue
            if delta < 0 or random.random() < math.exp(-delta / T):
                old_site = chain[i]
                chain[i] = new_site
                del occupied[old_site]
                occupied[new_site] = i
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Kink move: rotate pair (i, i+1)
            i = random.randint(1, n - 3)
            p1, p2 = chain[i], chain[i+1]
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1

            # Generate perpendicular rotations
            if dx != 0:  # horizontal step
                new_p1s = [(x1, y1+1), (x1, y1-1)]
                new_p2s = [(x2, y2+1), (x2, y2-1)]
            else:  # vertical step
                new_p1s = [(x1+1, y1), (x1-1, y1)]
                new_p2s = [(x2+1, y2), (x2-1, y2)]

            for sign in (1, -1):
                if dx != 0:
                    np1 = (x1, y1 + sign)
                    np2 = (x2, y2 + sign)
                else:
                    np1 = (x1 + sign, y1)
                    np2 = (x2 + sign, y2)

                # Check validity
                valid = True
                # Check np1 not occupied by other than i
                if np1 in occupied and occupied[np1] != i:
                    valid = False
                # Check np2 not occupied by other than i+1
                if np2 in occupied and occupied[np2] != i + 1:
                    valid = False
                # Check np1 != np2
                if np1 == np2:
                    valid = False
                # Check connectivity with i-1
                if i > 0:
                    px, py = chain[i-1]
                    if abs(np1[0] - px) + abs(np1[1] - py) != 1:
                        valid = False
                # Check connectivity with i+2
                if i + 2 < n:
                    px, py = chain[i+2]
                    if abs(np2[0] - px) + abs(np2[1] - py) != 1:
                        valid = False
                if not valid:
                    continue

                # Compute delta for both moves
                delta1 = _compute_energy_delta(sequence, chain, occupied, i, np1)
                if delta1 is None:
                    continue
                # Temporarily apply first move to compute second delta
                old1 = chain[i]
                chain[i] = np1
                del occupied[old1]
                occupied[np1] = i
                delta2 = _compute_energy_delta(sequence, chain, occupied, i+1, np2)
                if delta2 is None:
                    # Revert
                    chain[i] = old1
                    del occupied[np1]
                    occupied[old1] = i
                    continue
                total_delta = delta1 + delta2

                if total_delta < 0 or random.random() < math.exp(-total_delta / T):
                    # Apply second move
                    old2 = chain[i+1]
                    chain[i+1] = np2
                    del occupied[old2]
                    occupied[np2] = i + 1
                    current_energy += total_delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_occupied = dict(occupied)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    break
                else:
                    # Revert
                    chain[i] = old1
                    del occupied[np1]
                    occupied[old1] = i

        T = max(T_min, T * cooling)

    return best_chain, best_occupied, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware multi-phase search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1

    # Phase 1: Multiple greedy growths with local optimization
    phase1_time = 40.0
    phase1_start = time.time()
    restarts = 0
    while time.time() < _DEADLINE and (time.time() - phase1_start) < phase1_time:
        restarts += 1
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        occupied = {p: i for i, p in enumerate(chain)}
        chain, occupied, value = _local_optimize_sa(sequence, chain, occupied, time_budget=3.0)
        if value < best_energy:
            best, best_energy = chain, value
        if restarts > 200:
            break

    # Phase 2: If time remains, do SA on the best chain found
    if best is not None and time.time() < _DEADLINE:
        remaining = _DEADLINE - time.time()
        if remaining > 5:
            occupied = {p: i for i, p in enumerate(best)}
            best, occupied, value = _local_optimize_sa(sequence, best, occupied, time_budget=min(remaining - 2, 30))
            if value < best_energy:
                best_energy = value

    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

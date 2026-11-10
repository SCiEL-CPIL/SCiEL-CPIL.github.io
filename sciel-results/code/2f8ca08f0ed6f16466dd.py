"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 170.0
_START = time.time()


def _neighbours2d(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _grow(sequence, dimension, rng, greedy_bias=1.0):
    """Randomized greedy chain growth with lookahead."""
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
                for q in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Look-ahead: if current is P, check if next is H and could benefit
            future_gain = 0
            if i + 1 < n and sequence[i + 1] == 'H':
                x, y = site
                for q in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if q == last:
                        continue
                    if q not in occupied:
                        # This site is free, next H could potentially contact
                        # Check if this free neighbor is near existing H
                        qx, qy = q
                        for r in ((qx+1,qy),(qx-1,qy),(qx,qy+1),(qx,qy-1)):
                            j = occupied.get(r)
                            if j is not None and sequence[j] == 'H' and j < i:
                                future_gain += 0.3
                                break
            free = 0
            x, y = site
            for q in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if q not in occupied and q != last:
                    free += 1
            score = gain * 2.5 + future_gain + 0.15 * free * greedy_bias + rng.random() * 0.5
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
            for q in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
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

    # Check self-avoidance: new_site not occupied by another residue
    if new_site in occupied and occupied[new_site] != i:
        return None

    # Check connectivity: must be unit step from neighbors
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
    # Remove contributions of residue i at old_site, add at new_site
    delta = 0
    seq_i = sequence[i]
    ox, oy = old_site
    nx, ny = new_site

    # Old contacts involving i at old_site
    if seq_i == 'H':
        for q in ((ox+1,oy),(ox-1,oy),(ox,oy+1),(ox,oy-1)):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == 'H':
                # This contact is being removed
                delta += 1  # energy was -contacts, removing a contact increases energy by 1

    # New contacts involving i at new_site
    if seq_i == 'H':
        for q in ((nx+1,ny),(nx-1,ny),(nx,ny+1),(nx,ny-1)):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == 'H':
                delta -= 1  # adding a contact decreases energy by 1

    return delta


def _local_optimize_sa(sequence, chain, occupied, max_iters=2000, time_budget=None):
    """Simulated annealing with kink and segment moves."""
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)

    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_energy = _energy(sequence, best_chain)

    # SA parameters
    T = max(0.5, n * 0.05)
    T_min = 0.01
    cooling = 0.995

    iters = 0
    start_time = time.time()

    while iters < max_iters and time.time() < _DEADLINE:
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        iters += 1

        # Choose move type
        move_type = rng_move = random.random()

        if move_type < 0.5:
            # Single residue move
            i = random.randint(1, n - 2)
            x, y = chain[i]
            candidates = []
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                candidates.append((nx, ny))
            if not candidates:
                continue
            new_site = random.choice(candidates)
            delta = _compute_energy_delta(sequence, chain, occupied, i, new_site)
            if delta is None:
                continue

            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Apply move
                old_site = chain[i]
                chain[i] = new_site
                del occupied[old_site]
                occupied[new_site] = i

                if delta < 0:
                    current_energy = best_energy + delta if 'current_energy' not in dir() else None
                # Track current energy
                if 'current_energy' not in locals():
                    current_energy = best_energy
                current_energy = best_energy + delta if not hasattr(_local_optimize_sa, '_ce') else None

                # Actually let's track properly
                if not hasattr(_local_optimize_sa, '_ce'):
                    _local_optimize_sa._ce = best_energy
                _local_optimize_sa._ce += delta
                current_energy = _local_optimize_sa._ce

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
        else:
            # Kink move: move residues i and i+1 together
            i = random.randint(1, n - 3)
            p1, p2 = chain[i], chain[i+1]
            x1, y1 = p1
            x2, y2 = p2

            # Generate kink moves: rotate the pair
            # The vector from p1 to p2 is (dx, dy)
            dx, dy = x2 - x1, y2 - y1
            # Perpendicular directions
            if dx == 1:  # right
                new_p1s = [(x1, y1+1), (x1, y1-1)]
                new_p2s = [(x2, y2+1), (x2, y2-1)]
            elif dx == -1:  # left
                new_p1s = [(x1, y1+1), (x1, y1-1)]
                new_p2s = [(x2, y2+1), (x2, y2-1)]
            elif dy == 1:  # up
                new_p1s = [(x1+1, y1), (x1-1, y1)]
                new_p2s = [(x2+1, y2), (x2-1, y2)]
            else:  # down
                new_p1s = [(x1+1, y1), (x1-1, y1)]
                new_p2s = [(x2+1, y2), (x2-1, y2)]

            # Try rotating the kink
            for sign in (1, -1):
                if dx == 1 or dx == -1:
                    np1 = (x1, y1 + sign)
                    np2 = (x2, y2 + sign)
                else:
                    np1 = (x1 + sign, y1)
                    np2 = (x2 + sign, y2)

                # Check validity
                if np1 in occupied and occupied[np1] not in (i, i+1):
                    continue
                if np2 in occupied and occupied[np2] not in (i, i+1):
                    continue
                # Check connectivity
                if i > 0:
                    px, py = chain[i-1]
                    if abs(np1[0]-px) + abs(np1[1]-py) != 1:
                        continue
                if i+1 < n-1:
                    px, py = chain[i+2]
                    if abs(np2[0]-px) + abs(np2[1]-py) != 1:
                        continue

                # Compute energy delta for both moves
                # Temporarily remove both
                old_occ = dict(occupied)
                del old_occ[p1]
                del old_occ[p2]

                delta = 0
                for idx, new_pos in [(i, np1), (i+1, np2)]:
                    seq_r = sequence[idx]
                    if seq_r == 'H':
                        ox, oy = (p1 if idx == i else p2)
                        nx, ny = new_pos
                        # Old contacts
                        for q in ((ox+1,oy),(ox-1,oy),(ox,oy+1),(ox,oy-1)):
                            j = old_occ.get(q)
                            if j is not None and j not in (i, i+1) and sequence[j] == 'H':
                                delta += 1
                        # New contacts
                        for q in ((nx+1,ny),(nx-1,ny),(nx,ny+1),(nx,ny-1)):
                            j = old_occ.get(q)
                            if j is not None and j not in (i, i+1) and sequence[j] == 'H':
                                delta -= 1

                if not hasattr(_local_optimize_sa, '_ce'):
                    _local_optimize_sa._ce = best_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Apply
                    chain[i] = np1
                    chain[i+1] = np2
                    del occupied[p1]
                    del occupied[p2]
                    occupied[np1] = i
                    occupied[np2] = i + 1
                    _local_optimize_sa._ce += delta
                    if _local_optimize_sa._ce < best_energy:
                        best_energy = _local_optimize_sa._ce
                        best_chain = list(chain)
                        best_occupied = dict(occupied)
                break  # Only try one sign per iteration

        T *= cooling
        if T < T_min:
            T = T_min

    # Clean up
    if hasattr(_local_optimize_sa, '_ce'):
        del _local_optimize_sa._ce

    return best_chain, best_occupied, best_energy


def _local_optimize_simple(sequence, chain, occupied, max_iters=1000):
    """Fast greedy local optimization with single and kink moves."""
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)

    best_energy = _energy(sequence, chain)
    improved = True
    iters = 0

    while improved and iters < max_iters and time.time() < _DEADLINE:
        improved = False
        iters += 1

        # Single residue moves
        for i in range(1, n - 1):
            if time.time() >= _DEADLINE:
                break
            x, y = chain[i]
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                site = (nx, ny)
                if site in occupied and occupied[site] != i:
                    continue
                # Check connectivity
                if i > 0:
                    px, py = chain[i-1]
                    if abs(nx-px) + abs(ny-py) != 1:
                        continue
                if i < n-1:
                    px, py = chain[i+1]
                    if abs(nx-px) + abs(ny-py) != 1:
                        continue

                # Compute delta
                delta = _compute_energy_delta(sequence, chain, occupied, i, site)
                if delta is None or delta >= 0:
                    continue

                # Apply
                old_site = chain[i]
                chain[i] = site
                del occupied[old_site]
                occupied[site] = i
                best_energy += delta
                improved = True
                break

        if not improved:
            # Try kink moves
            for i in range(1, n - 2):
                if time.time() >= _DEADLINE:
                    break
                p1, p2 = chain[i], chain[i+1]
                x1, y1 = p1
                x2, y2 = p2
                dx, dy = x2-x1, y2-y1

                for sign in (1, -1):
                    if dx != 0:
                        np1 = (x1, y1+sign)
                        np2 = (x2, y2+sign)
                    else:
                        np1 = (x1+sign, y1)
                        np2 = (x2+sign, y2)

                    if np1 in occupied and occupied[np1] not in (i, i+1):
                        continue
                    if np2 in occupied and occupied[np2] not in (i, i+1):
                        continue
                    if i > 0:
                        px, py = chain[i-1]
                        if abs(np1[0]-px) + abs(np1[1]-py) != 1:
                            continue
                    if i+1 < n-1:
                        px, py = chain[i+2]
                        if abs(np2[0]-px) + abs(np2[1]-py) != 1:
                            continue

                    # Compute delta
                    old_occ = dict(occupied)
                    del old_occ[p1]
                    del old_occ[p2]
                    delta = 0
                    for idx, old_pos, new_pos in [(i, p1, np1), (i+1, p2, np2)]:
                        if sequence[idx] == 'H':
                            ox, oy = old_pos
                            nx, ny = new_pos
                            for q in ((ox+1,oy),(ox-1,oy),(ox,oy+1),(ox,oy-1)):
                                j = old_occ.get(q)
                                if j is not None and j not in (i, i+1) and sequence[j] == 'H':
                                    delta += 1
                            for q in ((nx+1,ny),(nx-1,ny),(nx,ny+1),(nx,ny-1)):
                                j = old_occ.get(q)
                                if j is not None and j not in (i, i+1) and sequence[j] == 'H':
                                    delta -= 1

                    if delta < 0:
                        chain[i] = np1
                        chain[i+1] = np2
                        del occupied[p1]
                        del occupied[p2]
                        occupied[np1] = i
                        occupied[np2] = i+1
                        best_energy += delta
                        improved = True
                        break
                if improved:
                    break

    return chain, occupied, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search using SA and multiple strategies."""
    n = len(sequence)
    best, best_energy = None, 0

    # Time budget allocation: longer sequences get more time
    # Total budget ~170s, 10 sequences
    # Shorter sequences (20) need less, longer (100) need more
    if n <= 30:
        seq_time = 8.0
    elif n <= 50:
        seq_time = 15.0
    elif n <= 70:
        seq_time = 25.0
    else:
        seq_time = 35.0

    seq_deadline = time.time() + seq_time

    # Multiple growth strategies
    strategies = [
        (1.0, 12345),   # balanced
        (2.0, 67890),   # more greedy
        (0.5, 11111),   # more random
        (1.5, 22222),   # slightly greedy
        (0.8, 33333),   # slightly random
    ]

    for bias, seed in strategies:
        if time.time() >= seq_deadline or time.time() >= _DEADLINE:
            break

        rng = random.Random(seed + n)
        chain = _grow(sequence, dimension, rng, greedy_bias=bias)
        if chain is None:
            continue

        occupied = {p: i for i, p in enumerate(chain)}
        energy = _energy(sequence, chain)

        # Quick greedy optimization first
        chain, occupied, energy = _local_optimize_simple(sequence, chain, occupied, max_iters=500)

        if best is None or energy < best_energy:
            best, best_energy = list(chain), energy

        # If we have time, do SA
        if time.time() < seq_deadline - 1.0 and time.time() < _DEADLINE - 2.0:
            sa_budget = min(seq_deadline - time.time() - 0.5, 10.0)
            if sa_budget > 1.0:
                chain2, occupied2, energy2 = _local_optimize_sa(
                    sequence, list(chain), dict(occupied),
                    max_iters=3000, time_budget=sa_budget
                )
                if energy2 < best_energy:
                    best, best_energy = list(chain2), energy2

    if best is None:
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 170.0


def _neighbours2d(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied:
                    free += 1
            score = gain * 2.0 + 0.1 * free + rng.random()
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
        if sequence[i] == "H":
            x, y = p
            for q in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_optimize(sequence, chain, occupied, max_iters=500):
    """Try single-residue moves to reduce energy. Returns improved chain and occupied."""
    n = len(chain)
    best_energy = _energy(sequence, chain)
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < _DEADLINE:
        improved = False
        iters += 1
        for i in range(1, n - 1):
            if time.time() >= _DEADLINE:
                break
            x, y = chain[i]
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                site = (nx, ny)
                if site in occupied and occupied[site] != i:
                    continue
                # Check if move is valid: new position not occupied by another residue
                if site in occupied and occupied[site] != i:
                    continue
                # Temporarily remove i from occupied
                old_occupied = dict(occupied)
                del old_occupied[chain[i]]
                if site in old_occupied:
                    continue
                old_occupied[site] = i
                # Check self-avoidance: consecutive residues must be unit steps
                if i > 0:
                    px, py = chain[i-1]
                    if abs(nx - px) + abs(ny - py) != 1:
                        continue
                if i < n - 1:
                    px, py = chain[i+1]
                    if abs(nx - px) + abs(ny - py) != 1:
                        continue
                # Build new chain
                new_chain = chain[:i] + [site] + chain[i+1:]
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    chain[i] = site
                    occupied = old_occupied
                    best_energy = new_energy
                    improved = True
                    break
    return chain, occupied, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    restarts = 0
    while time.time() < _DEADLINE:
        restarts += 1
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Build occupied set
        occupied = {p: i for i, p in enumerate(chain)}
        # Local optimization
        chain, occupied, value = _local_optimize(sequence, chain, occupied)
        if value < best_energy:
            best, best_energy = chain, value
        # If we've done enough restarts and no improvement recently, stop
        if restarts > 500:
            break
    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END

"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_deadline_global = 0.0

# Fast 4-neighbour offsets
_DX = (1, -1, 0, 0)
_DY = (0, 0, 1, -1)


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Total energy = -# H-H contacts (non-consecutive neighbours)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DX, _DY:
                pass
            # unrolled
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _h_neighbours(sequence):
    """Return list H[i] = list of residue indices j (j>i+1) that could be H-neighbours
    if they end up on adjacent lattice sites. We build it from the chain at search time
    instead; this helper is a placeholder for clarity."""
    return None


def _grow(sequence, rng):
    """Greedy chain growth: pick the unoccupied neighbour with the best (contacts, free) score."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        cx, cy = chain[-1]
        is_h = sequence[i] == "H"
        best_score = None
        best_site = None
        for dx, dy in _DX, _DY:
            pass
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (cx + dx, cy + dy)
            if site in occupied:
                continue
            sx, sy = site
            gain = 0
            if is_h:
                for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + dx2, sy + dy2)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + dx2, sy + dy2) not in occupied:
                    free += 1
            score = gain * 100 + free + rng.random()
            if best_score is None or score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            return None
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def _local_search(sequence, chain, max_iter=20000, rng=None, deadline=None):
    """Steepest-descent single-residue moves with incremental O(1) energy updates."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)

    if rng is None:
        rng = random.Random(12345)

    # Precompute: for each residue i, the set of residue indices j that are
    # "contact-eligible" (|j-i|>1 and both H). We track which of these are
    # actually in contact via a per-residue counter.
    is_h = [c == "H" for c in sequence]
    eligible = [[] for _ in range(n)]
    for i in range(n):
        if not is_h[i]:
            continue
        for j in range(i + 2, n):
            if is_h[j]:
                eligible[i].append(j)
                eligible[j].append(i)

    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}

    # contact_count[i] = number of H-H contacts residue i currently participates in
    contact_count = [0] * n
    # Build initial contact_count
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = occupied.get((x + dx, y + dy))
            if j is not None and j > i + 1 and is_h[j]:
                contact_count[i] += 1
                contact_count[j] += 1

    energy = -sum(contact_count) // 2

    # For each residue, cache the best valid move (delta, new_pos) found last sweep.
    # We recompute on the fly each sweep for correctness.

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        best_delta = 0
        best_i = -1
        best_old = None
        best_new = None

        for i in range(1, n - 1):
            if not is_h[i]:
                # P residue: only matters for validity (self-avoidance), no energy
                # But moving it could free space for H moves. Still, to save time,
                # we still consider it since it can unblock.
                pass
            x, y = chain[i]
            # Try the 4 neighbours
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                if new_pos in occupied:
                    continue
                # Validity: new_pos must not be adjacent to any non-consecutive residue
                valid = True
                nx, ny = new_pos
                for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((nx + dx2, ny + dy2))
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Energy delta
                delta = 0
                if is_h[i]:
                    # Remove old contacts
                    old_x, old_y = x, y
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = occupied.get((old_x + dx2, old_y + dy2))
                        if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                            delta += 1
                    # Add new contacts
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = occupied.get((nx + dx2, ny + dy2))
                        if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                            delta -= 1
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_old = (x, y)
                    best_new = new_pos
                    if delta == -2:  # can't do better than -2 for single move? Actually -2 is min
                        # Could break early but keep scanning for consistency
                        pass

        if best_i >= 0:
            # Apply the move
            i = best_i
            chain[i] = best_new
            occupied.pop(best_old)
            occupied[best_new] = i
            # Update contact_count
            if is_h[i]:
                # Remove old contacts
                ox, oy = best_old
                for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((ox + dx2, oy + dy2))
                    if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                        contact_count[i] -= 1
                        contact_count[j] -= 1
                # Add new contacts
                nx, ny = best_new
                for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((nx + dx2, ny + dy2))
                    if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                        contact_count[i] += 1
                        contact_count[j] += 1
            energy += best_delta
            improved = True

        if not improved:
            break

    return chain, energy


def fold(sequence, dimension):
    """Best conformation via multi-restart greedy growth + steepest-descent local search."""
    global _deadline_global
    start_time = time.time()
    _deadline_global = start_time + 200.0  # safety margin before 210s kill

    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best = None
    best_energy = 0  # energy is negative or zero; 0 is "no improvement yet"

    # Number of restarts: scale with time budget and sequence length
    # Each restart is roughly O(n * 4 * n) for grow + O(n^2 * sweeps) for local search.
    # For n=100, local search is the bottleneck. Budget per restart ~ 0.05-0.2s.
    if n <= 30:
        num_restarts = 400
    elif n <= 60:
        num_restarts = 200
    else:
        num_restarts = 100

    for attempt in range(num_restarts):
        if time.time() > _deadline_global:
            break

        chain = _grow(sequence, rng)
        if chain is None:
            continue

        # Local search with a per-restart iteration cap
        max_iter_ls = max(5000, min(20000, 300000 // n))
        chain, energy = _local_search(sequence, chain, max_iter=max_iter_ls,
                                       rng=rng, deadline=_deadline_global)

        if best is None or energy < best_energy:
            best = chain
            best_energy = energy

    # If time remains, run a final longer local search on the best solution
    if best is not None and time.time() < _deadline_global - 2.0:
        remaining = _deadline_global - time.time() - 2.0
        extra_iter = int(min(50000, max(5000, remaining * 1000)))
        best, best_energy = _local_search(sequence, best, max_iter=extra_iter,
                                           rng=rng, deadline=_deadline_global)

    if best is None:
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

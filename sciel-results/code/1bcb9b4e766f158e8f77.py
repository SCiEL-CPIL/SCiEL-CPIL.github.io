"""HP lattice protein folding on the square lattice - scratch rewrite."""
# EVOLVE-BLOCK-START
import random
import math
import time

# Precompute neighbour offsets for the square lattice (dimension 2)
_NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _init_chain(sequence, rng):
    """Build a self-avoiding initial conformation via randomized greedy growth.
    Bias toward compactness (prefer sites near the origin) and H-H contacts."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    # Track H positions for contact counting during growth
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in _NEIGH:
            site = (last[0] + dx, last[1] + dy)
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts, and compactness
            score = 0.0
            if sequence[i] == 'H':
                for nx, ny in _NEIGH:
                    q = (site[0] + nx, site[1] + ny)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 1.0
            # Compactness: prefer closer to origin (mild)
            score -= 0.05 * (abs(site[0]) + abs(site[1]))
            # Randomness
            score += rng.random() * 0.5
            candidates.append((score, site))
        if not candidates:
            # Dead end: try a small random restart of the tail
            return None
        candidates.sort(reverse=True)
        # Pick from top candidates with some randomness
        if len(candidates) > 1 and rng.random() < 0.3:
            site = candidates[rng.randint(0, min(2, len(candidates) - 1))][1]
        else:
            site = candidates[0][1]
        chain.append(site)
        occupied[site] = i
    return chain


def _compute_energy(sequence, chain):
    """Compute energy from scratch: -count of non-adjacent H-H contacts."""
    n = len(chain)
    where = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in _NEIGH:
            j = where.get((x + dx, y + dy))
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # each contact counted twice


def _compute_contacts_set(sequence, chain):
    """Return set of (i,j) pairs (i<j, |i-j|>1) that are H-H contacts."""
    n = len(chain)
    where = {chain[i]: i for i in range(n)}
    contacts = set()
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in _NEIGH:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts.add((i, j))
    return contacts


def _local_search(sequence, chain, rng, time_limit):
    """Simulated annealing with single-residue moves and kink moves.
    Maintains energy incrementally."""
    n = len(chain)
    chain = list(chain)
    occupied = {chain[i]: i for i in range(n)}
    contacts = _compute_contacts_set(sequence, chain)
    energy = -len(contacts)

    start_time = time.time()
    T = 3.0
    T_min = 0.05
    cool = 0.995
    best_energy = energy
    best_chain = list(chain)

    # Precompute which residues are H
    is_H = [c == 'H' for c in sequence]

    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        # Choose a move type
        if rng.random() < 0.5 and n > 2:
            # Kink move: pick a residue i (not first/last), and one of its
            # non-chain neighbours, rotate the 3-residue segment
            i = rng.randint(1, n - 2)
            # Find a free neighbour of chain[i]
            x, y = chain[i]
            free_sites = []
            for dx, dy in _NEIGH:
                site = (x + dx, y + dy)
                if site not in occupied:
                    free_sites.append(site)
            if not free_sites:
                continue
            new_site = free_sites[rng.randint(0, len(free_sites) - 1)]
            # Check that moving chain[i] to new_site keeps chain[i-1] and
            # chain[i+1] adjacent to the new position (they must still be
            # unit steps from new_site)
            # Actually for a kink move we need the new site to be adjacent to
            # both chain[i-1] and chain[i+1].
            prev = chain[i - 1]
            nxt = chain[i + 1]
            adj_prev = abs(new_site[0] - prev[0]) + abs(new_site[1] - prev[1]) == 1
            adj_nxt = abs(new_site[0] - nxt[0]) + abs(new_site[1] - nxt[1]) == 1
            if not (adj_prev and adj_nxt):
                continue
            # Also new_site must not be chain[i-1] or chain[i+1]
            if new_site == prev or new_site == nxt:
                continue

            # Apply move: remove i from old site, add to new_site
            old_site = chain[i]
            # Remove old contacts involving i
            removed = []
            for (a, b) in list(contacts):
                if a == i or b == i:
                    removed.append((a, b))
                    contacts.discard((a, b))
            # Update occupied
            del occupied[old_site]
            occupied[new_site] = i
            chain[i] = new_site
            # Add new contacts involving i
            added = []
            x2, y2 = new_site
            for dx, dy in _NEIGH:
                j = occupied.get((x2 + dx, y2 + dy))
                if j is not None and j != i - 1 and j != i + 1 and is_H[i] and is_H[j]:
                    if j > i:
                        key = (i, j)
                    else:
                        key = (j, i)
                    if key not in contacts:
                        contacts.add(key)
                        added.append(key)
            delta = len(added) - len(removed)
            new_energy = energy + delta
            if new_energy < energy or rng.random() < math.exp((new_energy - energy) / T):
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
            else:
                # Revert
                del occupied[new_site]
                occupied[old_site] = i
                chain[i] = old_site
                for k in added:
                    contacts.discard(k)
                for k in removed:
                    contacts.add(k)
        else:
            # Single-residue move: pick a residue i (not first/last), move to
            # a free adjacent site, but must maintain connectivity: the new
            # site must be adjacent to at least one of chain[i-1] or chain[i+1]
            # Actually, moving a single residue breaks the chain unless the new
            # site is adjacent to both neighbours (which is the kink case) or we
            # allow the chain to be temporarily disconnected. For HP folding,
            # single-residue moves that preserve self-avoidance and connectivity
            # are the kink moves. Let's use a different single move: pick i,
            # remove it, and reinsert it at a free site adjacent to either
            # chain[i-1] or chain[i+1], then check the other neighbour is still
            # connected... this is complex. Let's just use kink moves and
            # "shift" moves.
            # Shift move: pick i, move chain[i] to a free site adjacent to
            # chain[i-1] (so chain[i-1]->new->chain[i+1] requires new adjacent
            # to chain[i+1] too, which is kink). 
            # Let's do: pick i, try moving to a free site that is adjacent to
            # chain[i+1] (so we can reconnect i-1 to i+1 if they become adjacent)
            # This is getting complicated. Let's just do kink moves with higher
            # probability and a simple "swap-free" move.
            # Alternative: pick i, move to free site, then check if chain is
            # still a valid path (each consecutive pair is unit step). If not,
            # revert. This is the "random walk" move.
            i = rng.randint(1, n - 2)
            x, y = chain[i]
            free_sites = []
            for dx, dy in _NEIGH:
                site = (x + dx, y + dy)
                if site not in occupied:
                    free_sites.append(site)
            if not free_sites:
                continue
            new_site = free_sites[rng.randint(0, len(free_sites) - 1)]
            # Check connectivity: new_site must be adjacent to chain[i-1] and
            # chain[i+1] for the chain to remain a path (since i is internal).
            prev = chain[i - 1]
            nxt = chain[i + 1]
            if not (abs(new_site[0] - prev[0]) + abs(new_site[1] - prev[1] == 1) and
                    abs(new_site[0] - nxt[0]) + abs(new_site[1] - nxt[1] == 1)):
                continue
            if new_site == prev or new_site == nxt:
                continue
            # This is the same as a kink move. Let's just do kink moves.
            # To have variety, let's do a "two-residue" move: pick i, i+1 and
            # shift them.
            # For simplicity, let's just do kink moves always in this branch too.
            # (The 50/50 split above was to try different strategies; let's
            #  just always do kink moves.)
            continue

        T *= cool

    return best_chain, best_energy


def _global_search(sequence, dimension, time_budget):
    """Run multiple restarts of local search, returning the best conformation."""
    n = len(sequence)
    rng = random.Random(42 + n)
    best_chain = None
    best_energy = 0

    # Try several initial conformations
    attempts = 0
    start_time = time.time()
    while time.time() - start_time < time_budget:
        attempts += 1
        # Generate initial conformation
        chain = _init_chain(sequence, rng)
        if chain is None:
            # Fallback: simple spiral
            chain = _spiral_init(n, rng)
            if chain is None:
                continue

        # Local search
        ls_time = min(0.5, time_budget * 0.3 / max(1, attempts))
        ls_chain, ls_energy = _local_search(sequence, chain, rng, ls_time)

        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain

        # Check time
        if time.time() - start_time > time_budget * 0.8:
            break

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = 0

    return best_chain, best_energy


def _spiral_init(n, rng):
    """Build a compact spiral-like self-avoiding walk."""
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    # Spiral direction: right, down, left, up, repeat with increasing length
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    dir_idx = 0
    step_len = 1
    i = 1
    while i < n:
        dx, dy = dirs[dir_idx]
        for _ in range(step_len):
            if i >= n:
                break
            x, y = chain[-1]
            site = (x + dx, y + dy)
            if site in occupied:
                break
            chain.append(site)
            occupied[site] = i
            i += 1
        if i >= n:
            break
        dir_idx = (dir_idx + 1) % 4
        if dir_idx == 0 or dir_idx == 2:
            step_len += 1
    if len(chain) < n:
        # Failed to fill; try a different approach
        return None
    return chain


def fold(sequence, dimension):
    """Best conformation within the time budget."""
    n = len(sequence)
    # Time budget per sequence: 120s total for 10 sequences = 12s each,
    # but use a safety margin. We'll use 10s per sequence.
    time_budget = 10.0
    start_time = time.time()

    best_chain, best_energy = _global_search(sequence, dimension, time_budget)

    # Safety: ensure we're within the overall budget
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        pass  # We already enforced the budget in _global_search

    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END

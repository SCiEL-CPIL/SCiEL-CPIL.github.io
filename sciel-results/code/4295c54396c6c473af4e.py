"""HP lattice protein folding: beam search growth + fragment re-insertion local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map and precomputed H indices."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth. Keeps top beam_width partial chains scored by
    immediate H-H contacts plus a small free-space heuristic.
    """
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    # beam entries: (chain, occupied_dict, score)
    # lower score is better (more contacts)
    beam = [([start], {start: 0}, 0.0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            cur = chain[-1]
            x, y = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    sx, sy = site
                    for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (sx + adx, sy + ady)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # free space heuristic: prefer sites with more open neighbors
                free = 0
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (sx + adx, sy + ady) not in occupied:
                        free += 1
                new_score = score - 2.0 * gain - 0.1 * free
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                next_beam.append((new_chain, new_occupied, new_score))
        
        if not next_beam:
            return None
        
        # Sort by score (lower = better), tie-break randomly for diversity
        next_beam.sort(key=lambda x: (x[2], rng.random()))
        beam = next_beam[:beam_width]
        
        if _remaining() < 2.0:
            break
    
    if not beam:
        return None
    return beam[0][0]


def _grow_greedy(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            score = 2.0 * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_reinsert_fragments(chain, sequence, rng, max_frag_len=5):
    """
    Attempt to re-insert a fragment: cut out chain[a:b], then try to regrow it
    from chain[a-1] (if a>0) connecting to chain[b] (if b<n).
    Returns (new_chain, energy_delta) or (None, 0).
    """
    n = len(chain)
    if n < 5:
        return None, 0

    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)
    b = a + L

    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    rest = chain[:a] + chain[b:]
    rest_occupied = set(rest)

    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None, 0

    def _find_path(start, end, length, occupied_set):
        if length == 0:
            return [start] if start == end else None

        def _dfs(current, remaining, path, occupied):
            if remaining == 0:
                if current == end:
                    return list(path)
                return None
            x, y = current
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                dist_to_end = abs(end[0] - site[0]) + abs(end[1] - site[1])
                if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                    continue
                path.append(site)
                occupied.add(site)
                result = _dfs(site, remaining - 1, path, occupied)
                if result is not None:
                    return result
                path.pop()
                occupied.discard(site)
            return None

        return _dfs(start, length, [start], {start} | occupied_set)

    path = _find_path(left_neighbor, right_neighbor, L, rest_occupied)
    if path is None:
        return None, 0
    if path[-1] != right_neighbor:
        return None, 0

    new_frag = path[1:]
    if len(new_frag) != L:
        return None, 0

    new_chain = chain[:a] + new_frag + chain[b:]
    if not _is_valid_chain(new_chain):
        return None, 0
    return new_chain, 0


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """Kink move: pick a residue i (1..n-2) and try to move it to an adjacent free position."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in chain and site != chain[i]:
            continue
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        nx, ny = chain[i + 1]
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        options.append(site)
    if not options:
        return None
    site = options[rng.randint(0, len(options) - 1)]
    new_chain = chain[:]
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _local_search(sequence, chain, rng, max_iters=500):
    """Local search with fragment re-insertion, reversals, and kinks."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    no_improve_count = 0
    max_no_improve = 30

    for it in range(max_iters):
        if _remaining() < 5.0:
            break

        if no_improve_count >= max_no_improve:
            perturbed = _reversal_move(best_chain, rng, max_len=min(10, n - 2))
            if perturbed is not None:
                best_chain = perturbed
                best_energy = _energy_fast(sequence, best_chain)
            no_improve_count = 0
            max_no_improve = min(60, max_no_improve * 1.5)

        r = rng.random()
        if r < 0.30:
            new_chain, _ = _try_reinsert_fragments(best_chain, sequence, rng)
        elif r < 0.65:
            new_chain = _reversal_move(best_chain, rng)
        else:
            new_chain = _kink_move(best_chain, rng)

        if new_chain is None:
            no_improve_count += 1
            continue

        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
            no_improve_count = 0
        elif rng.random() < 0.05:
            best_chain = new_chain
            no_improve_count += 1
        else:
            no_improve_count += 1

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search + greedy restarts and local search."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    n = len(sequence)

    num_restarts = 500
    for r in range(num_restarts):
        if _remaining() < 3.0:
            break

        # Alternate between beam search and greedy growth
        if r % 2 == 0:
            chain = _grow_beam(sequence, dimension, rng, beam_width=8)
        else:
            chain = _grow_greedy(sequence, dimension, rng)
        if chain is None:
            continue

        e = _energy_fast(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain

        # Apply local search periodically
        if r % 3 == 0 and best is not None and _remaining() > 8.0:
            iters = 200 if _remaining() > 30 else 100
            improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved

    # Final polish
    if best is not None and _remaining() > 5.0:
        iters = 500 if _remaining() > 20 else 200
        improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    # Second polish if time allows
    if best is not None and _remaining() > 3.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=200)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    if best is None:
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

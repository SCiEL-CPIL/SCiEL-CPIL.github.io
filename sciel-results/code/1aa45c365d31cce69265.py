"""HP lattice protein folding: beam search + simulated annealing with fragment moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # safety margin before 210s limit


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy(seq, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {tuple(p): i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == 'H')
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _is_valid(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        tp = tuple(p)
        if tp in seen:
            return False
        seen.add(tp)
        if i > 0:
            if abs(p[0] - chain[i - 1][0]) + abs(p[1] - chain[i - 1][1]) != 1:
                return False
    return True


def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth: maintain a beam of partial chains, expanding each
    with scored neighbors. Returns the best complete chain found.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    # Each beam entry: (chain, occupied_set, score)
    # score = energy_so_far (negative is better) + heuristic
    beam = [([(0, 0)], {(0, 0)}, 0.0)]

    for i in range(1, n):
        new_beam = []
        for chain, occ, score in beam:
            x, y = chain[-1]
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occ:
                    continue
                gain = 0
                if i in h_set:
                    sx, sy = site
                    for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (sx + adx, sy + ady)
                        if q in occ:
                            idx = chain.index(q) if q in [tuple(c) for c in chain] else -1
                            if idx >= 0 and idx < i - 1 and idx in h_set:
                                gain += 1
                # Count free neighbors for diversity
                free = 0
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (sx + adx, sy + ady) not in occ:
                        free += 1
                s = 3.0 * gain + 0.1 * free + rng.random() * 0.5
                options.append((s, site))
            if options:
                options.sort(key=lambda t: -t[0])
                for s, site in options[:2]:  # keep top 2 per parent
                    new_chain = chain + [site]
                    new_occ = occ | {site}
                    # Compute incremental energy
                    e_inc = 0
                    if i in h_set:
                        sx, sy = site
                        for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            q = (sx + adx, sy + ady)
                            if q in occ:
                                # find index
                                for j in range(len(chain)):
                                    if tuple(chain[j]) == q:
                                        if j < i - 1 and j in h_set:
                                            e_inc += 1
                                        break
                    new_score = score - e_inc + s
                    new_beam.append((new_chain, new_occ, new_score))

        if not new_beam:
            return None

        # Sort by score (higher is better for our heuristic) and keep beam_width
        new_beam.sort(key=lambda t: -t[2])
        beam = new_beam[:beam_width]

    # Return the best chain
    best = max(beam, key=lambda t: t[2])
    return best[0]


def _fragment_move(chain, sequence, rng, max_frag=6):
    """
    Remove a fragment of length L from positions [a, b) and regrow it
    via DFS to connect chain[a-1] to chain[b].
    """
    n = len(chain)
    if n < 5:
        return None
    L = rng.randint(2, min(max_frag, n - 3))
    a = rng.randint(1, n - L - 1)
    b = a + L

    left = tuple(chain[a - 1])
    right = tuple(chain[b])
    lx, ly = left
    rx, ry = right
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None

    rest = chain[:a] + chain[b:]
    rest_set = set(tuple(p) for p in rest)

    def _dfs(current, remaining, path, occupied):
        if remaining == 0:
            return list(path) if current == right else None
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            dist = abs(right[0] - site[0]) + abs(right[1] - site[1])
            if dist > remaining or (remaining - dist) % 2 != 0:
                continue
            path.append(site)
            occupied.add(site)
            res = _dfs(site, remaining - 1, path, occupied)
            if res is not None:
                return res
            path.pop()
            occupied.discard(site)
        return None

    path = _dfs(left, L, [left], {left} | rest_set)
    if path is None or path[-1] != right:
        return None

    new_frag = path[1:]
    new_chain = chain[:a] + new_frag + chain[b:]
    if _is_valid(new_chain):
        return new_chain
    return None


def _reversal(chain, rng, max_len=None):
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid(new_chain):
        return new_chain
    return None


def _kink(chain, rng):
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    opts = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in [tuple(p) for p in chain] and site != (x, y):
            continue
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        nx, ny = chain[i + 1]
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        opts.append(site)
    if not opts:
        return None
    site = opts[rng.randint(0, len(opts) - 1)]
    new_chain = [list(p) for p in chain]
    new_chain[i] = list(site)
    if _is_valid(new_chain):
        return new_chain
    return None


def _anneal(sequence, chain, rng, max_iters=800):
    """Simulated annealing with fragment, reversal, and kink moves."""
    best_chain = [list(p) for p in chain]
    best_e = _energy(sequence, best_chain)
    cur_chain = [list(p) for p in chain]
    cur_e = best_e
    n = len(cur_chain)
    T = 2.0
    T_min = 0.01
    cool = (T_min / T) ** (1.0 / max(1, max_iters))

    for it in range(max_iters):
        if _remaining() < 3.0:
            break
        r = rng.random()
        if r < 0.3:
            new = _fragment_move(cur_chain, sequence, rng)
        elif r < 0.65:
            new = _reversal(cur_chain, rng)
        else:
            new = _kink(cur_chain, rng)

        if new is None:
            continue

        e = _energy(sequence, new)
        delta = e - cur_e
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cur_chain = new
            cur_e = e
            if e < best_e:
                best_chain = [list(p) for p in new]
                best_e = e
        T *= cool

    return best_chain, best_e


def fold(sequence, dimension):
    """Fold a sequence using beam search + simulated annealing."""
    rng = random.Random(42 + len(sequence))
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    best = None
    best_energy = 0

    # Phase 1: Multiple beam-search restarts
    num_restarts = 300
    for r in range(num_restarts):
        if _remaining() < 5.0:
            break
        chain = _grow_beam(sequence, dimension, rng, beam_width=6)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = [list(p) for p in chain]

        # Periodic annealing on best
        if r % 5 == 0 and best is not None and _remaining() > 10.0:
            iters = 300 if _remaining() > 40 else 150
            improved, imp_e = _anneal(sequence, best, rng, max_iters=iters)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved

    # Phase 2: Intensive annealing on best
    if best is not None and _remaining() > 5.0:
        iters = 600 if _remaining() > 25 else 200
        improved, imp_e = _anneal(sequence, best, rng, max_iters=iters)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    # Phase 3: Final polish
    if best is not None and _remaining() > 3.0:
        improved, imp_e = _anneal(sequence, best, rng, max_iters=200)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    if best is None:
        best = [[i, 0] for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END

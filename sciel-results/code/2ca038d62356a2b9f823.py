"""Structural rewrite: maximum-clique search over difference-disjoint rows."""
import time
from itertools import combinations

_DEADLINE = 150.0
_START = None
_STOP = None
_MAX_CANDIDATES = 30000


def _row_diffs(row):
    """Return the frozenset of positive differences in a strictly-increasing row starting at 0."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return frozenset(d)


def _greedy_rows(n_rows, k):
    """Original greedy (safe fallback): each row takes smallest next mark with fresh differences."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _lower_bound(n_rows, k):
    """Simple counting bound: each row uses k(k+1)/2 differences, all distinct, each <= M."""
    per = k * (k + 1) // 2
    need = n_rows * per
    M = 0
    while M * (M + 1) // 2 < need:
        M += 1
    return M


def _generate_candidates(M, k, limit):
    """Generate candidate rows (size k+1, start 0, in [0,M]) with a differential-constraint filter."""
    size = k + 1
    tail = list(range(1, M + 1))
    n_tail = len(tail)
    k_tail = size - 1
    cand = []
    # Precompute a quick filter: for a partial tail, the set of diffs among chosen tail
    # elements plus their offsets from 0 must be small & non-conflicting internally (always true
    # within one row since a row's own diffs are distinct by construction of strictly increasing).
    # The real filter is size; we just bound by M.
    if n_tail < k_tail:
        return cand
    for combo in combinations(tail, k_tail):
        if len(cand) >= limit:
            break
        # combo already strictly increasing
        cand.append((0,) + combo)
    return cand


def _build_graph(cands, k):
    """Build adjacency as bitmasks (int) for clique search. Returns list of int masks."""
    diffs = [_row_diffs(r) for r in cands]
    n = len(cands)
    adj = [0] * n
    for i in range(n):
        di = diffs[i]
        for j in range(i + 1, n):
            if not (di & diffs[j]):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _color_sort_order(adj, n):
    """Greedy coloring to produce an ordering and per-position color numbers (for upper bound).
    Returns (order, colors) where colors[i] is the color of node order[i]."""
    # We color the graph; use order by descending degree for better bounds.
    deg = [bin(adj[i]).count('1') for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: deg[x], reverse=True)
    color_of = [0] * n
    used_colors = 0
    for idx, v in enumerate(order):
        # find smallest color not used by already-colored neighbors (that appear later in order)
        # For Tomita, we need colors such that each color class is independent.
        forbidden = set()
        av = adj[v]
        for pos in range(idx + 1, n):
            u = order[pos]
            if (av >> u) & 1:
                if color_of[u]:
                    forbidden.add(color_of[u])
        c = 1
        while c in forbidden:
            c += 1
        color_of[v] = c
        if c > used_colors:
            used_colors = c
    colors = [color_of[v] for v in order]
    return order, colors, used_colors


def _max_clique(adj, n, order, colors, target, deadline):
    """Branch and bound to find a clique of size >= target. Returns the clique (list of node idx) or None."""
    global _STOP
    best = []
    best_size = 0

    # R = current clique (node indices), P = candidates (indices into order positions)
    def search(clique, P):
        nonlocal best, best_size
        if _STOP is not None and time.time() >= _STOP:
            return
        if not P:
            if len(clique) > best_size:
                best_size = len(clique)
                best = list(clique)
            return
        if len(clique) + len(P) <= best_size:
            return
        # color upper bound: colors are aligned with P's order
        # P is a list of (position, node) where position index maps to colors
        # Use the last color in the color sequence restricted to P as upper bound.
        # We store P as list of node indices in color order.
        # Upper bound = len(clique) + max color among P.
        ub = len(clique) + max(colors_map.get(v, 0) for v in P) if P else len(clique)
        if ub <= best_size:
            return
        # If even reaching target is possible, try to find target clique.
        for pos in range(len(P) - 1, -1, -1):
            if _STOP is not None and time.time() >= _STOP:
                return
            v = P[pos]
            if len(clique) + (pos + 1) <= best_size:
                break
            # v can join if adjacent to all in clique
            ok = True
            av = adj[v]
            for c in clique:
                if not ((av >> c) & 1):
                    ok = False
                    break
            if not ok:
                continue
            new_clique = clique + [v]
            if len(new_clique) >= target:
                best_size = len(new_clique)
                best = list(new_clique)
                # try to keep searching for larger, but we already have target; continue to grow
            # P' = P[:pos] intersected with neighbors of v
            newP = []
            for u in P[:pos]:
                if (av >> u) & 1:
                    newP.append(u)
            search(new_clique, newP)
            if best_size >= target and len(clique) + len(P) <= best_size:
                return

    # colors_map: node -> its color in the ordering
    colors_map = {}
    for i, v in enumerate(order):
        colors_map[v] = colors[i]

    # Initial P = all nodes in color order
    P0 = list(order)
    search([], P0)
    return best if len(best) >= target else (best if best else None)


def construct(n_rows, k):
    global _START, _STOP
    _START = time.time()
    _STOP = _START + _DEADLINE - 5.0  # safety margin

    # Fallback greedy immediately (cheap) to always have a valid answer.
    try:
        greedy = _greedy_rows(n_rows, k)
    except Exception:
        greedy = None

    # Try to improve via clique search over M.
    M0 = _lower_bound(n_rows, k)
    best_rows = greedy
    best_M = greedy[-1][-1] if greedy else 0

    # We'll try a range of M values. Keep candidate set bounded.
    # Choose a modest M window to keep graph small enough.
    M_hi = max(M0, best_M) + 2
    for M in range(M0, M_hi + 1):
        if _STOP is not None and time.time() >= _STOP:
            break
        try:
            cands = _generate_candidates(M, k, _MAX_CANDIDATES)
        except Exception:
            continue
        if len(cands) < n_rows:
            continue
        if len(cands) > _MAX_CANDIDATES:
            continue
        try:
            adj = _build_graph(cands, k)
            n = len(cands)
            order, colors, _ = _color_sort_order(adj, n)
            clique = _max_clique(adj, n, order, colors, n_rows, _STOP)
        except Exception:
            continue
        if clique and len(clique) >= n_rows:
            # Take first n_rows nodes of the clique
            chosen = clique[:n_rows]
            rows = [list(cands[i]) for i in chosen]
            rows.sort(key=lambda r: r[-1])
            scope = max(r[-1] for r in rows)
            if scope < best_M:
                best_M = scope
                best_rows = rows
            if M == M0:
                break  # can't do better than lower bound

    return best_rows

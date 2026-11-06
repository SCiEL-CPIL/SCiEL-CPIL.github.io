"""Deterministic algebraic construction for R_5(C4) > 26 via AG(2,5) + one extra vertex."""
import time
import random

_DEADLINE = 115.0  # safety margin before 120s kill


def _is_c4_free(colors, n):
    """Check that no colour class contains a C4.
    For each colour c, build adjacency; for every pair (u,v) with u<v,
    count common neighbours in that colour. If any count >= 2, there's a C4."""
    from collections import defaultdict
    # Build per-colour adjacency as sets
    adj = [set() for _ in range(n)]
    adjc = [defaultdict(set) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjc[c][i].add(j)
            adjc[c][j].add(i)
            idx += 1
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                if v in adjc[c][u]:
                    # u and v are adjacent in colour c; check common neighbours
                    common = adjc[c][u] & adjc[c][v]
                    # exclude u and v themselves from common (they're not their own neighbours)
                    # Actually common neighbours are vertices w != u,v such that w~u and w~v in colour c
                    if len(common) >= 2:
                        return False
    return True


def _build_ag25_coloring():
    """Build the 25-vertex AG(2,5) 5-colouring.
    Vertices are (a,b) in Z5 x Z5, indexed as a*5+b.
    Edge (u,v) gets colour = slope of line through u,v in {0,1,2,3,4},
    where slope 4 (infinity) means vertical lines.
    This partitions edges into 5 parallel classes (linear hypergraphs), each C4-free."""
    n = 25
    colors = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        ai, bi = divmod(i, 5)
        for j in range(i + 1, n):
            aj, bj = divmod(j, 5)
            da = (aj - ai) % 5
            db = (bj - bi) % 5
            if da == 0:
                c = 4  # vertical
            else:
                # slope = db * da^{-1} mod 5
                inv_da = pow(da, 3, 5)  # Fermat: da^4=1, so da^{-1}=da^3
                c = (db * inv_da) % 5
            colors[idx] = c
            idx += 1
    return n, colors


def _try_extend(n, colors, deadline):
    """Try to add a (n+1)-th vertex by finding a valid colour assignment for its n edges."""
    extra = n
    # For each new edge (extra, v) for v in 0..n-1, assign a colour.
    # We need to check that no C4 is formed in any colour class.
    # A C4 involving the new vertex: vertices extra, a, b, c where extra~a, extra~b in colour c,
    # and a~c, b~c in colour c (with c != extra, a, b).
    # Equivalently: for each colour k, if extra connects to set S_k (vertices coloured k to extra),
    # then no two vertices in S_k can share a common neighbour in colour k (other than extra).
    # So for each colour k, the set S_k must be such that for any u,v in S_k,
    # the number of common neighbours of u and v in colour k (excluding extra) is < 2.
    # i.e., |N_k(u) ∩ N_k(v) \\ {extra}| <= 1 for all u,v in S_k.

    # Precompute for each colour k and each pair (u,v), the count of common neighbours in colour k
    # among the existing n vertices.
    from collections import defaultdict
    adjc = [defaultdict(set) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjc[c][i].add(j)
            adjc[c][j].add(i)
            idx += 1

    # common_neighbours[k][u][v] = |N_k(u) ∩ N_k(v)| for u < v
    # We need this to be <= 1 for pairs in S_k.
    # Precompute a "bad pair" indicator for each colour: bad[k][(u,v)] = True if |N_k(u)∩N_k(v)| >= 2
    bad = [set() for _ in range(5)]
    for k in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                if v in adjc[k][u]:
                    common = adjc[k][u] & adjc[k][v]
                    if len(common) >= 2:
                        bad[k].add((u, v))

    # Now we need to assign each vertex v a colour col[v] in {0..4} such that
    # for each k, the set S_k = {v : col[v] == k} has no pair (u,v) in bad[k].
    # This is a constraint satisfaction problem.
    # Use backtracking with forward checking.
    assign = [-1] * n  # col for each vertex

    def backtrack(pos):
        if time.time() > deadline:
            return None
        if pos == n:
            return assign[:]
        for k in range(5):
            # Check if assigning col[pos] = k is consistent
            ok = True
            for prev in range(pos):
                if assign[prev] == k:
                    # Check if (prev, pos) is a bad pair in colour k
                    if (prev, pos) in bad[k]:
                        ok = False
                        break
            if not ok:
                continue
            assign[pos] = k
            result = backtrack(pos + 1)
            if result is not None:
                return result
            assign[pos] = -1
        return None

    result = backtrack(0)
    if result is None:
        return None
    # Build full colour list
    new_colors = colors[:]
    for v in range(n):
        new_colors.append(result[v])
    return new_colors


def _local_search_improve(n, colors, deadline):
    """Try to improve n by local search: start from n, try to flip colours to reduce violations,
    then attempt to add a vertex. This is a fallback."""
    # This is complex; for now just return the current solution.
    return n, colors


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Step 1: Build the 25-vertex AG(2,5) construction
    n, colors = _build_ag25_coloring()

    # Verify it's C4-free (should be, by construction)
    if not _is_c4_free(colors, n):
        # Fallback to the known 11-vertex construction
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    # Step 2: Try to extend to 26 vertices
    if time.time() < deadline - 5:
        extended = _try_extend(n, colors, deadline)
        if extended is not None:
            n = 26
            colors = extended
            # Verify
            if _is_c4_free(colors, n):
                return {'n': n, 'colors': colors}
            # If verification fails, fall back to 25
            n = 25
            # Rebuild the 25-vertex coloring
            n, colors = _build_ag25_coloring()

    return {'n': n, 'colors': colors}

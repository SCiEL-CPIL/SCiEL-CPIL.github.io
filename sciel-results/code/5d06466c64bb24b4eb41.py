"""4-colouring of K_n with no monochromatic triangle (R_4(3) > n).

Strategy (combines the seed's fast K32 construction with the inspiration's
vertex-addition + local-search, but vectorized with NumPy):
  1. Build the Greenwood-Gleason K32 (two Clebsch copies, inter-copy colour 3).
  2. Verify it with a fast NumPy triangle count.
  3. Add vertices 33..50 one at a time via a fast (NumPy-aided) backtracking
     solver for the new vertex's colours.
  4. If we stall below the target, run a randomized local search on the whole
     colouring (recolor edges that sit in monochromatic triangles) and retry
     vertex addition, to break through to 48-50.

Time budget: explicit deadline guard, return best valid result with margin.
"""

import time
import random
import numpy as np


def construct():
    t0 = time.time()
    DEADLINE = t0 + 142.0  # safety margin before the 150s kill
    random.seed(12345)

    # ---------- fast K32 (Greenwood-Gleason / Clebsch) ----------
    def _mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)

    def _build_colors(n):
        cols = np.empty(n * (n - 1) // 2, dtype=np.int8)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    cols[k] = log[(i % 16) ^ (j % 16)] % 3
                else:
                    cols[k] = 3
                k += 1
        return cols

    n = 32
    colors = _build_colors(n)

    # ---------- NumPy triangle counting ----------
    # For a fixed n, precompute the edge index of every triangle (i,j,k) and
    # the three edges per triangle. A mono triangle of colour c is one where
    # all three edges equal c.
    _tri_cache = {}

    def _triangles(n):
        if n in _tri_cache:
            return _tri_cache[n]
        m = n * (n - 1) // 2
        # edge index of (a,b), a<b
        # idx(a,b) = a*n - a*(a+1)//2 + (b-a-1)
        ii = np.arange(n, dtype=np.int64)
        base = ii * n - ii * (ii + 1) // 2  # base[i] = index of (i, i+1)
        # For triangle (i,j,k), i<j<k: edges (i,j),(i,k),(j,k)
        # (i,j): base[i] + (j - i - 1)
        # (i,k): base[i] + (k - i - 1)
        # (j,k): base[j] + (k - j - 1)
        # Build arrays of all triples.
        idx = np.array(
            [
                (i, j, k)
                for i in range(n)
                for j in range(i + 1, n)
                for k in range(j + 1, n)
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        if idx.size == 0:
            _tri_cache[n] = (idx, None, None)
            return idx, None, None
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        e_ij = base[i] + (j - i - 1)
        e_ik = base[i] + (k - i - 1)
        e_jk = base[j] + (k - j - 1)
        edges = np.stack([e_ij, e_ik, e_jk], axis=1)
        _tri_cache[n] = (idx, edges)
        return idx, edges

    def _count_mono_triangles(n, colors):
        idx, edges = _triangles(n)
        if edges is None:
            return 0, None
        c = colors[edges]  # (T,3)
        # mono triangle where all three equal
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2])
        return int(mono.sum()), (c[mono], mono)

    # verify seed
    cnt, _ = _count_mono_triangles(n, colors)
    assert cnt == 0, "seed K32 invalid"

    best_n = n
    best_colors = colors.copy()

    # ---------- fast new-vertex colour assignment ----------
    # For a new vertex v (index n), we assign colours g[0..n-1] for edges (i,v).
    # Constraint: for all i<j<n, NOT (g[i]==g[j]==c AND colors[e(i,j)]==c).
    # Equivalent: for each colour c, the set S_c = {i : g[i]==c} must be an
    # independent set in the subgraph of edges coloured c (among 0..n-1).
    # i.e. S_c must contain no pair (i,j) with colors[e(i,j)]==c.
    #
    # We do backtracking over vertices, using NumPy to precompute, for each
    # colour c, the adjacency of the c-coloured subgraph, then do a fast
    # independent-set backtracking with bitsets.

    def _assign_new_vertex(n, colors):
        # Build, for each colour c, a list of bitsets: adj_c[i] = bitmask of j>i
        # such that edge (i,j) has colour c.
        m = n * (n - 1) // 2
        ii = np.arange(n, dtype=np.int64)
        base = ii * n - ii * (ii + 1) // 2
        # For each edge index, recover (i,j)
        # We'll just build per-colour adjacency bitsets by iterating edges.
        adj = [ [0]*n for _ in range(4) ]  # adj[c][i] bitmask of j (any) with colour c
        # iterate over all pairs
        for i in range(n):
            bi = base[i]
            for j in range(i + 1, n):
                c = int(colors[bi + (j - i - 1)])
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

        # Backtracking: assign g[0..n-1]. Maintain for each colour c the set of
        # already-chosen vertices S_c (bitmask). When assigning g[v]=c, we need
        # v to have no neighbour in S_c within colour c's graph.
        # i.e. (adj[c][v] & S_c) == 0.
        # Also, after adding v to S_c, future vertices w must satisfy
        # (adj[c][w] & S_c') == 0 where S_c' includes v. That's handled as we go.
        #
        # To make it fast, order vertices by how constrained they are (most
        # constrained first). But keep it simple: use a recursive search with
        # forward checking and a time limit.

        deadline = DEADLINE - 2.0
        S = [0, 0, 0, 0]
        g = [None] * n

        # order: sort vertices by number of "forbidden colour" pressure.
        # A vertex v has, for each colour c, the set of neighbours in c-graph.
        # If we assign v colour c, it's forbidden if it has a neighbour already
        # in S_c. Hard to precompute ordering; use natural order but with
        # random tie-breaks for diversity.
        order = list(range(n))

        def _search(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            v = order[pos]
            for c in (0, 1, 2, 3):
                if (adj[c][v] & S[c]) == 0:
                    S[c] |= (1 << v)
                    g[v] = c
                    if _search(pos + 1):
                        return True
                    S[c] ^= (1 << v)
                    g[v] = None
            return False

        # Try a few random orderings if natural fails.
        if _search(0):
            return g
        # randomize order a few times
        for _ in range(30):
            if time.time() > deadline:
                break
            random.shuffle(order)
            S = [0, 0, 0, 0]
            g = [None] * n
            if _search(0):
                return g
        return None

    # ---------- vertex addition loop ----------
    def _try_add_vertex(n, colors):
        g = _assign_new_vertex(n, colors)
        if g is None:
            return False
        # append
        new = np.empty(n * (n + 1) // 2, dtype=np.int8)
        new[: len(colors)] = colors
        for i in range(n):
            new[len(colors) + i] = g[i]
        # verify
        cnt, _ = _count_mono_triangles(n + 1, new)
        if cnt == 0:
            colors[:] = new
            return True
        return False

    target = 50
    for _ in range(target - n):
        if time.time() > DEADLINE - 8:
            break
        if _try_add_vertex(n, colors):
            n += 1
            if n > best_n:
                best_n = n
                best_colors = colors.copy()
        else:
            break

    # ---------- local search to break through ----------
    # If we stalled, try to recolor edges that are in monochromatic triangles,
    # to free up room, then retry vertex addition.
    if n < target and time.time() < DEADLINE - 10:
        # Precompute triangle edges for current n
        idx, edges = _triangles(n)
        # We'll maintain a running count of mono triangles.
        # Approach: repeatedly pick a random mono triangle, recolor one of its
        # edges to a non-conflicting colour, if it reduces total mono triangles.
        # To keep it fast, recompute count after each accepted move (NumPy).
        # But full recompute is O(T) ~ n^3/6; for n=48 that's ~18k triangles, fine.

        # We want to allow adding vertices. So alternate: try to add a vertex;
        # if it fails, do a batch of local-search moves on the existing n, then
        # retry.
        improved = True
        while n < target and time.time() < DEADLINE - 6 and improved:
            improved = False
            # try to add a vertex
            if _try_add_vertex(n, colors):
                n += 1
                if n > best_n:
                    best_n = n
                    best_colors = colors.copy()
                improved = True
                continue
            # local search on current n
            moves = 0
            max_moves = 2000
            while moves < max_moves and time.time() < DEADLINE - 6:
                cnt, (cmono, mono) = _count_mono_triangles(n, colors)
                if cnt == 0:
                    break
                # pick a random mono triangle
                t = mono[random.randrange(int(mono.sum()))]
                e_ij, e_ik, e_jk = int(edges[t, 0]), int(edges[t, 1]), int(edges[t, 2])
                col = int(colors[e_ij])
                # try recoloring one of the three edges to a different colour
                # that doesn't create a new mono triangle.
                cand = None
                for e in (e_ij, e_ik, e_jk):
                    for newc in (0, 1, 2, 3):
                        if newc == col:
                            continue
                        old = colors[e]
                        colors[e] = newc
                        cnt2, _ = _count_mono_triangles(n, colors)
                        if cnt2 < cnt:
                            cand = (e, newc, cnt2)
                            break
                        colors[e] = old
                    if cand:
                        break
                if cand:
                    e, newc, _ = cand
                    colors[e] = newc
                    moves += 1
                    improved = True
                else:
                    # no improving single move; try a random recolor that keeps
                    # count the same (to escape local minima)
                    e = random.choice((e_ij, e_ik, e_jk))
                    old = colors[e]
                    newc = random.choice((0, 1, 2, 3))
                    if newc != old:
                        colors[e] = newc
                        cnt2, _ = _count_mono_triangles(n, colors)
                        if cnt2 > cnt:
                            colors[e] = old  # revert
                    moves += 1
            # after local search, loop will try to add a vertex again

    # ---------- final: return best ----------
    # Ensure best is valid
    if best_n >= n:
        return {'n': best_n, 'colors': best_colors.tolist()}
    return {'n': n, 'colors': colors.tolist()}

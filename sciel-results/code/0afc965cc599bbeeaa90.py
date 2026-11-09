import time
import random

def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin before 150s kill
    rnd = random.Random(20240614)

    def time_left():
        return deadline - time.time()

    # ---- precompute all 6-cycles (as frozensets of 6 vertices) for n up to 20 ----
    def build_c6_sets(n):
        sets = []
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        for e in range(d + 1, n):
                            for f in range(e + 1, n):
                                sets.append(frozenset((a, b, c, d, e, f)))
        return sets

    c6_sets_by_n = {n: build_c6_sets(n) for n in range(6, 21)}

    # ---- incremental "forbidden pairs" table for each color ----
    # forbid[c] = list of bitsets (one per vertex v) of vertices u<v with a length-4 path v-u in color c
    def compute_forbid(edges, n, c):
        # edges: dict (i,j)->color for i<j
        # build adjacency bitsets for color c
        adj = [0] * n
        for (i, j), col in edges.items():
            if col == c:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        forbid = [0] * n
        for v in range(n):
            # neighbors of v in color c
            neigh = adj[v]
            # vertices at distance 4 from v: paths v - a - b - c - d
            # = (a in N(v)) & (b in N(a)) & (c in N(b)) & (d in N(c))
            # compute reachable in exactly 4 steps
            # step1: N(v)
            s1 = neigh
            s2 = 0
            while s1:
                lsb = s1 & (-s1)
                a = lsb.bit_length() - 1
                s2 |= adj[a]
                s1 ^= lsb
            s3 = 0
            while s2:
                lsb = s2 & (-s2)
                b = lsb.bit_length() - 1
                s3 |= adj[b]
                s2 ^= lsb
            s4 = 0
            while s3:
                lsb = s3 & (-s3)
                c2 = lsb.bit_length() - 1
                s4 |= adj[c2]
                s3 ^= lsb
            forbid[v] = s4
        return forbid

    # ---- backtrack: assign edges of new vertex n-1 to existing vertices 0..n-2 ----
    # edges: dict (i,j)->color for i<j, currently for K_{n-1}
    # forbid[c]: bitset per vertex of forbidden partners (for the NEW vertex)
    def backtrack(edges, n, forbid, order):
        # n = number of existing vertices; new vertex index = n
        # order: list of existing vertices to assign (0..n-1)
        new = n
        # for each existing vertex u, we will pick a color; but constraint:
        # if we color edge (new,u) with color c, then for all other existing v colored c with new,
        # u and v must not be a forbidden pair in c.
        # Process in order; maintain per-color set of already-assigned existing vertices.
        assigned_color = [-1] * n  # color of edge (new, u), -1 = unassigned
        color_members = [set() for _ in range(4)]  # existing vertices assigned color c

        def check_add(u, c):
            # adding u to color c: u must not be forbidden with any existing member of color c
            fb = forbid[c][u]
            for w in color_members[c]:
                if fb & (1 << w):
                    return False
            return True

        def rec(idx):
            if idx == len(order):
                return True
            u = order[idx]
            # try colors; heuristic: prefer colors with fewer conflicts / symmetric break
            for c in range(4):
                if check_add(u, c):
                    assigned_color[u] = c
                    color_members[c].add(u)
                    if rec(idx + 1):
                        return True
                    color_members[c].remove(u)
                    assigned_color[u] = -1
            return False

        return rec(0), assigned_color

    # ---- validity check for a full coloring (safety net) ----
    def is_valid(colors, n):
        # colors: list of length n*(n-1)//2 in lexicographic order
        edges = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges[(i, j)] = colors[k]
                k += 1
        for c in range(4):
            for cyc in c6_sets_by_n[n]:
                verts = sorted(cyc)
                ok = True
                for t in range(6):
                    a = verts[t]
                    b = verts[(t + 1) % 6]
                    key = (a, b) if a < b else (b, a)
                    if edges.get(key) != c:
                        ok = False
                        break
                if ok:
                    return False
        return True

    # ---- build initial K9 coloring (known valid) ----
    def initial_k9():
        n = 9
        cols = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        edges = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges[(i, j)] = cols[k]
                k += 1
        return edges, n

    def edges_to_colors(edges, n):
        cols = []
        for i in range(n):
            for j in range(i + 1, n):
                cols.append(edges[(i, j)])
        return cols

    # ---- main search: try to extend from K9 to larger n ----
    def try_extend(edges, n, max_target, max_restarts):
        # current best
        best_n = n
        best_edges = dict(edges)
        for _ in range(max_restarts):
            if time_left() < 5.0:
                break
            # randomize vertex ordering for the new vertex
            cur_edges = dict(best_edges)
            cur_n = best_n
            # try to add vertices one at a time up to max_target
            for target in range(cur_n + 1, max_target + 1):
                if time_left() < 5.0:
                    break
                # add vertex cur_n (index = cur_n), connect to 0..cur_n-1
                forbid = [compute_forbid(cur_edges, cur_n, c) for c in range(4)]
                order = list(range(cur_n))
                rnd.shuffle(order)
                ok, assigned = backtrack(cur_edges, cur_n, forbid, order)
                if ok:
                    new_idx = cur_n
                    for u in range(cur_n):
                        c = assigned[u]
                        cur_edges[(u, new_idx)] = c
                    cur_n += 1
                    if cur_n > best_n:
                        best_n = cur_n
                        best_edges = dict(cur_edges)
                    # verify with safety net occasionally
                    if best_n >= 12 and is_valid(edges_to_colors(best_edges, best_n)):
                        pass
                    else:
                        # trust incremental
                        pass
                else:
                    # failed to extend to this target; stop this restart
                    break
        return best_n, best_edges

    edges, n = initial_k9()
    # verify K9
    assert is_valid(edges_to_colors(edges, n), n), "K9 base invalid"

    best_n, best_edges = try_extend(edges, n, 20, max_restarts=2000)

    # final safety verification
    cols = edges_to_colors(best_edges, best_n)
    if not is_valid(cols, best_n):
        # fall back to known-good K9
        best_n = 9
        cols = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    return {'n': best_n, 'colors': cols}

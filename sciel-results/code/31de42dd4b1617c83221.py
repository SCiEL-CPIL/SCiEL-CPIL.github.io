"""2-switch shrink search on top of the Lazebnik-Ustimenko-Woldar seed graph."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before the 150 s kill
    random.seed(12345)

    # ---- Build the L-U-W seed graph (4-regular, girth >= g) ----
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, seed_edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                seed_edges.append((index[vertex], index[other]))

    if time.time() > deadline:
        return len(order), seed_edges

    # ---- State as adjacency lists ----
    n = len(order)
    adj = [[] for _ in range(n)]
    for u, v in seed_edges:
        adj[u].append(v)
        adj[v].append(u)
    for v in range(n):
        adj[v].sort()

    def girth_of(adj, n, g):
        """Return True iff no cycle shorter than g (non-backtracking BFS from every vertex)."""
        for s in range(n):
            # dist[v] = length of shortest non-backtracking walk s -> v
            dist = {s: 0}
            prev = {s: -1}
            queue = [s]
            found = False
            head = 0
            while head < len(queue) and not found:
                u = queue[head]
                head += 1
                du = dist[u]
                pu = prev[u]
                for w in adj[u]:
                    if w == pu:
                        continue  # non-backtracking
                    if w in dist:
                        if du + 1 + dist[w] < g:
                            found = True
                            break
                    else:
                        dist[w] = du + 1
                        prev[w] = u
                        queue.append(w)
            if found:
                return False
        return True

    # ---- Verify the seed is valid (it should be) ----
    if not girth_of(adj, n, g):
        # fall back to returning the raw seed (still 4-regular, simple)
        return n, seed_edges

    best_n = n
    best_adj = [list(x) for x in adj]

    # ---- 2-switch shrink search ----
    # A 2-switch: pick vertices a, b with a common neighbour v, and an edge (c, d)
    # with no shared endpoints. Replace (v,a),(v,b),(c,d) with (v,c),(v,d),(a,b).
    # This removes vertex v (degree 2) and reduces n by 1, preserving 4-regularity.
    # We accept only if girth stays >= g.

    def build_edge_set(adj, n):
        es = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    es.add((u, v))
        return es

    def try_shrink_once(adj, n, g, deadline):
        """Attempt a single successful 2-switch that removes one vertex.
        Returns (new_adj, new_n) or None."""
        # Find candidate degree-2 vertices
        deg2 = [v for v in range(n) if len(adj[v]) == 2]
        if not deg2:
            return None
        random.shuffle(deg2)
        edge_set = build_edge_set(adj, n)
        # For each candidate v, its two neighbours a,b. Need an edge (c,d) disjoint
        # from {v,a,b} and with c,d not creating a loop / duplicate.
        for v in deg2:
            a, b = adj[v]
            # collect edges disjoint from {v, a, b}
            blocked = {v, a, b}
            candidates = []
            for (c, dd) in edge_set:
                if c in blocked or dd in blocked:
                    continue
                candidates.append((c, dd))
            if not candidates:
                continue
            random.shuffle(candidates)
            for (c, dd) in candidates:
                # Check that new edges (v,c),(v,d),(a,b) don't create loops/multiedges
                # and that the resulting graph stays simple.
                # (v,c): v != c (c not in blocked so c != v). Need (v,c) not in edge_set.
                if (min(v, c), max(v, c)) in edge_set:
                    continue
                if (min(v, dd), max(v, dd)) in edge_set:
                    continue
                if (min(a, b), max(a, b)) in edge_set:
                    continue
                # Also ensure a != b (true) and new edges distinct from each other.
                # Apply the switch in a copy.
                new_adj = [list(x) for x in adj]
                # remove (v,a),(v,b),(c,d)
                new_adj[v].remove(a)
                new_adj[a].remove(v)
                new_adj[v].remove(b)
                new_adj[b].remove(v)
                new_adj[c].remove(dd)
                new_adj[dd].remove(c)
                # add (v,c),(v,d),(a,b)
                new_adj[v].append(c)
                new_adj[c].append(v)
                new_adj[v].append(dd)
                new_adj[dd].append(v)
                new_adj[a].append(b)
                new_adj[b].append(a)
                # Now vertex v has degree 0 (it was 2, we removed 2 and added 2 -> wait)
                # Actually v loses a,b (2 removed) and gains c,dd (2 added) -> degree still 2.
                # That's wrong: a 2-switch should remove v. Let me reconsider.
                # To remove v: replace (v,a),(v,b) with (a,b) ONLY, and replace (c,d) with
                # (c,?),(d,?) ... no. Standard 2-switch keeps all vertices.
                # To reduce vertex count we must delete v and reconnect.
                # Correct vertex-removing move: v has neighbours a,b. Delete v and its two
                # edges, then add edge (a,b). This drops v and makes a,b lose a connection.
                # To keep 4-regular, a and b each lose 1, so we must give each a new neighbour.
                # Use an edge (c,d) and replace it with (a,c) and (b,d) (or (a,d),(b,c)).
                # That's a 2-switch that removes v. Let's implement that instead.
                break
            break
        return None

    # The above vertex-removing move is subtle; use the clean formulation:
    # v (deg 2, neighbours a,b) + edge (c,d)  ->  remove v, add (a,b), and rewire
    # (c,d) into (c, a?) ... To keep degrees: a loses v (deg 3), b loses v (deg 3).
    # c loses d (deg 3), d loses c (deg 3). Add edges (a,c) and (b,d): a->4,b->4,c->4,d->4.
    # And add (a,b)? No, that over-fills. So the move is:
    #   remove v, remove (c,d), add (a,b), (c,?) ...
    # Let's use: remove v; add (a,b); and to fix c,d (each now deg 3) we cannot without
    # another vertex. So the clean 4-regular vertex-removal is:
    #   v(a,b) + edge(c,d) -> remove v and (c,d); add (a,c),(b,d). Net: v gone; a,b,c,d
    #   each: a -v +c = same, b -v +d = same, c -d +a = same, d -c +b = same. 4-regular,
    #   n reduced by 1. This is the move. No (a,b) edge added.

    def try_shrink_once(adj, n, g, deadline):
        deg2 = [v for v in range(n) if len(adj[v]) == 2]
        if not deg2:
            return None
        random.shuffle(deg2)
        edge_set = build_edge_set(adj, n)
        for v in deg2:
            a, b = adj[v]
            blocked = {v, a, b}
            candidates = []
            for (c, dd) in edge_set:
                if c in blocked or dd in blocked:
                    continue
                candidates.append((c, dd))
            if not candidates:
                continue
            random.shuffle(candidates)
            for (c, dd) in candidates:
                # remove v, remove (c,d); add (a,c),(b,d). Check simplicity.
                if (min(a, c), max(a, c)) in edge_set:
                    continue
                if (min(b, dd), max(b, dd)) in edge_set:
                    continue
                # a != c, b != dd guaranteed (c,dd not in blocked). a != b true.
                # Also need (a,c) and (b,d) distinct: a!=b and c!=dd so fine.
                new_adj = [list(x) for x in adj]
                new_adj[v].remove(a); new_adj[a].remove(v)
                new_adj[v].remove(b); new_adj[b].remove(v)
                new_adj[c].remove(dd); new_adj[dd].remove(c)
                new_adj[a].append(c); new_adj[c].append(a)
                new_adj[b].append(dd); new_adj[dd].append(b)
                # drop vertex v
                new_adj.pop(v)
                # reindex
                m = n - 1
                remap = {x: (x if x < v else x - 1) for x in range(n) if x != v}
                fixed = []
                for u in range(m):
                    orig = u + (1 if u >= v else 0)
                    fixed.append([remap[w] for w in new_adj[orig]])
                # verify 4-regular and simple
                ok = True
                seen = set()
                for u in range(m):
                    if len(fixed[u]) != k:
                        ok = False; break
                    for w in fixed[u]:
                        if u == w:
                            ok = False; break
                        key = (min(u, w), max(u, w))
                        if key in seen:
                            ok = False; break
                        seen.add(key)
                    if not ok:
                        break
                if not ok:
                    continue
                if girth_of(fixed, m, g):
                    return fixed, m
        return None

    # Main loop: repeatedly shrink until stuck or deadline.
    cur_adj = [list(x) for x in adj]
    cur_n = n
    while time.time() < deadline:
        if cur_n < best_n:
            best_n = cur_n
            best_adj = [list(x) for x in cur_adj]
        res = try_shrink_once(cur_adj, cur_n, g, deadline)
        if res is None:
            # Stuck: no single shrinking move works. Try a random restart from a
            # slightly perturbed state, or stop.
            # Attempt a few random 2-switches (degree-preserving) to escape the local
            # minimum, accepting only girth-preserving ones, then retry shrinking.
            escaped = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                # random degree-preserving 2-switch
                edges = list(build_edge_set(cur_adj, cur_n))
                if len(edges) < 2:
                    break
                (u1, v1), (u2, v2) = random.sample(edges, 2)
                if len({u1, v1, u2, v2}) < 4:
                    continue
                # remove (u1,v1),(u2,v2); add (u1,u2),(v1,v2)
                if (min(u1, u2), max(u1, u2)) in build_edge_set(cur_adj, cur_n):
                    continue
                if (min(v1, v2), max(v1, v2)) in build_edge_set(cur_adj, cur_n):
                    continue
                cand = [list(x) for x in cur_adj]
                cand[u1].remove(v1); cand[v1].remove(u1)
                cand[u2].remove(v2); cand[v2].remove(u2)
                cand[u1].append(u2); cand[u2].append(u1)
                cand[v1].append(v2); cand[v2].append(v1)
                if girth_of(cand, cur_n, g):
                    cur_adj = cand
                    escaped = True
                    break
            if not escaped:
                break
        else:
            cur_adj, cur_n = res

    # Return the best (smallest) valid graph found.
    out_edges = []
    for u in range(best_n):
        for w in best_adj[u]:
            if u < w:
                out_edges.append((u, w))
    return best_n, out_edges
# EVOLVE-BLOCK-END

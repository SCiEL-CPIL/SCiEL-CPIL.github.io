import time
from random import Random

def _gen_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    import sys
    sys.setrecursionlimit(100000)
    N = 20
    trees = _gen_trees(N, 611)
    tree_info = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v); adj[v].append(u)
            deg[u] += 1; deg[v] += 1
        maxdeg = max(deg) if deg else 0
        tree_info.append(dict(k=k, edges=edges, adj=adj, deg=deg, maxdeg=maxdeg, nedges=len(edges)))

    # Deterministic 1-factorization of K20 into 19 perfect matchings
    # Using the standard round-robin tournament scheduling method
    matchings = []
    # Vertices 0..18 are in a circle, vertex 19 is the "center"
    # For round r in 0..18: vertex 19 pairs with vertex r, and others pair up
    for r in range(19):
        m = set()
        # Center vertex 19 pairs with vertex r
        m.add((min(19, r), max(19, r)))
        # Pair up the remaining 18 vertices
        # The remaining vertices are {0,...,18} \ {r}
        # Arrange them: for i in 1..9, pair (r+i mod 19, r-i mod 19)
        for i in range(1, 10):
            a = (r + i) % 19
            b = (r - i) % 19
            m.add((min(a, b), max(a, b)))
        matchings.append(m)

    # Verify: all 190 edges covered exactly once
    all_edges = set()
    for m in matchings:
        for e in m:
            assert e not in all_edges, f"Edge {e} appears in multiple matchings"
            all_edges.add(e)
    assert len(all_edges) == 190

    start = time.time()
    DEADLINE = start + 145.0

    def find_embedding_color(info, used, color_set, node_limit, tlimit):
        """Find embedding of tree using only edges from color_set (set of matching indices)."""
        k = info['k']; edges = info['edges']; adj = info['adj']; deg = info['deg']
        if len(edges) > len(all_edges) - len(used):
            return None
        if k <= 1:
            return [0]

        # Build available adjacency: only edges in color_set that are not used
        avail = set()
        for ci in color_set:
            for e in matchings[ci]:
                if e not in used:
                    avail.add(e)

        if len(avail) < info['nedges']:
            return None

        an = [set() for _ in range(N)]
        for (a, b) in avail:
            an[a].add(b); an[b].add(a)
        host_deg = [len(s) for s in an]

        # Hall-like necessary condition
        tree_deg_sorted = sorted(deg, reverse=True)
        host_deg_sorted = sorted(host_deg, reverse=True)
        maxdeg = info['maxdeg']
        for t in range(1, maxdeg + 1):
            ct = sum(1 for d in tree_deg_sorted if d >= t)
            ch = sum(1 for d in host_deg_sorted if d >= t)
            if ct > ch:
                return None

        t2h = [-1] * k
        h2t = [-1] * N
        nodes = [0]
        t0 = time.time()
        root = max(range(k), key=lambda v: deg[v])

        def dfs(frontier, assigned_count):
            if nodes[0] > node_limit or time.time() - t0 > tlimit:
                return False
            if assigned_count == k:
                return True
            bestv = None; bestc = None; bestscore = None
            for v in frontier:
                p = -1
                for nb in adj[v]:
                    if t2h[nb] != -1:
                        p = nb; break
                host = t2h[p]
                cands = []
                for h in an[host]:
                    if h2t[h] == -1:
                        cands.append(h)
                if not cands:
                    return False
                score = (len(cands), -deg[v])
                if bestscore is None or score < bestscore:
                    bestv = v; bestc = cands; bestscore = score
            v = bestv
            p = -1
            for nb in adj[v]:
                if t2h[nb] != -1:
                    p = nb; break
            host = t2h[p]
            for h in sorted(bestc, key=lambda x: len(an[x])):
                nodes[0] += 1
                t2h[v] = h; h2t[h] = v
                newfrontier = [x for x in frontier if x != v]
                for nb in adj[v]:
                    if t2h[nb] == -1:
                        newfrontier.append(nb)
                ok = True
                for x in newfrontier:
                    if x == v:
                        continue
                    pp = -1
                    for nb2 in adj[x]:
                        if t2h[nb2] != -1:
                            pp = nb2; break
                    hh = t2h[pp]
                    has_cand = False
                    for h2 in an[hh]:
                        if h2t[h2] == -1:
                            has_cand = True
                            break
                    if not has_cand:
                        ok = False
                        break
                if ok and dfs(newfrontier, assigned_count + 1):
                    return True
                t2h[v] = -1; h2t[h] = -1
            return False

        # Root placement with symmetry breaking
        seen = {}
        for h in range(N):
            key = frozenset(an[h])
            if key in seen:
                continue
            seen[key] = True
            nodes[0] += 1
            t2h[root] = h; h2t[h] = root
            frontier = [nb for nb in adj[root]]
            if dfs(frontier, 1):
                return [int(t2h[i]) for i in range(k)]
            t2h[root] = -1; h2t[h] = -1
        return None

    def value_of(result):
        return sum(tree_info[i]['nedges'] for i in range(N) if result[i] is not None)

    # Strategy: assign each tree to a specific matching (or small set of matchings)
    # Trees with more edges need more matchings; trees with maxdeg > 10 need multiple
    # A single perfect matching has 10 edges, so a tree with nedges <= 10 can use one matching
    # Trees with nedges > 10 need multiple matchings

    # Sort trees by difficulty (more edges, higher degree = harder)
    order = sorted(range(N), key=lambda i: (-tree_info[i]['nedges'], -tree_info[i]['maxdeg']))

    # For each tree, determine which matchings it can use
    # A tree with maxdeg d needs at least ceil(d/10) matchings (since each matching gives degree at most 10 to any vertex... actually degree at most 1 in each matching, so total degree across m matchings is at most m)
    # Wait: in a perfect matching, each vertex has degree exactly 1. So across m matchings, each vertex has degree at most m.
    # So we need m >= maxdeg for the tree.
    # Also need m*10 >= nedges (total edges available from m matchings is m*10, but we only use nedges of them)

    def get_min_matchings(info):
        """Minimum number of matchings needed for this tree."""
        needed_by_degree = info['maxdeg']
        needed_by_edges = (info['nedges'] + 9) // 10  # ceil(nedges/10)
        return max(needed_by_degree, needed_by_edges, 1)

    # Assign matchings greedily: give each tree the matchings it needs
    # We have 19 matchings total. Sum of min_matchings across all trees should be <= 19*10 = 190 edges total
    # Actually we just need to assign disjoint sets of matching indices to each tree

    # Greedy assignment: sort by number of matchings needed (descending), assign first available
    assignment = {}  # tree_index -> set of matching indices
    matching_used = [False] * 19

    # Try different assignment strategies
    def try_assignment(order_list):
        assignment = {}
        matching_used = [False] * 19
        for ti in order_list:
            info = tree_info[ti]
            if info['nedges'] == 0:
                assignment[ti] = set()
                continue
            min_m = get_min_matchings(info)
            # Find a set of min_m unused matchings
            avail_indices = [i for i in range(19) if not matching_used[i]]
            if len(avail_indices) < min_m:
                return None
            # Use the first min_m available
            chosen = set(avail_indices[:min_m])
            for ci in chosen:
                matching_used[ci] = True
            assignment[ti] = chosen
        return assignment

    # Try multiple orderings for assignment
    best_result = [None] * N
    best_value = 0

    def record(result):
        nonlocal best_value, best_result
        val = value_of(result)
        if val > best_value:
            best_value = val
            best_result = [x[:] if x is not None else None for x in result]

    # Attempt 1: Greedy by size (largest first)
    order1 = sorted(range(N), key=lambda i: (-tree_info[i]['nedges'], -tree_info[i]['maxdeg']))
    # Attempt 2: Greedy by degree
    order2 = sorted(range(N), key=lambda i: (-tree_info[i]['maxdeg'], -tree_info[i]['nedges']))
    # Attempt 3: Reverse (smallest first)
    order3 = sorted(range(N), key=lambda i: (tree_info[i]['nedges'], tree_info[i]['maxdeg']))
    # Attempt 4-7: Random permutations
    rr = Random(777)
    random_orders = []
    for _ in range(4):
        o = list(range(N))
        rr.shuffle(o)
        random_orders.append(o)

    all_orders = [order1, order2, order3] + random_orders

    for oi, order in enumerate(all_orders):
        if time.time() > DEADLINE - 8.0:
            break
        if best_value >= 190:
            break

        assignment = try_assignment(order)
        if assignment is None:
            continue

        # Now embed each tree using its assigned matchings
        used = set()
        result = [None] * N
        remaining_time = DEADLINE - time.time() - 5.0
        per_tree_time = max(0.5, remaining_time / (N + 1))

        for ti in order:
            if time.time() > DEADLINE - 5.0:
                break
            info = tree_info[ti]
            if info['nedges'] == 0:
                result[ti] = [0] if info['k'] == 1 else [0, 1]
                continue
            color_set = assignment[ti]
            tlimit = min(per_tree_time, 8.0)
            node_limit = 2000000
            emb = find_embedding_color(info, used, color_set, node_limit, tlimit)
            if emb is not None:
                for u, v in info['edges']:
                    a, b = emb[u], emb[v]
                    used.add((a, b) if a < b else (b, a))
                result[ti] = [int(x) for x in emb]

        record(result)
        if best_value >= 190:
            break

    # Polish: try to fill gaps with more flexible search (any available edges)
    if best_value < 190 and time.time() < DEADLINE - 2.0:
        used = set()
        for i in range(N):
            if best_result[i] is not None:
                for u, v in tree_info[i]['edges']:
                    a, b = best_result[i][u], best_result[i][v]
                    used.add((a, b) if a < b else (b, a))

        # For missing trees, try with all remaining edges (not restricted to specific matchings)
        for i in range(N):
            if time.time() > DEADLINE - 1.0:
                break
            if best_result[i] is None:
                info = tree_info[i]
                if info['nedges'] == 0:
                    best_result[i] = [0] if info['k'] == 1 else [0, 1]
                    continue
                # Use all 19 matchings as available colors
                emb = find_embedding_color(info, used, set(range(19)), 3000000, 8.0)
                if emb is not None:
                    for u, v in info['edges']:
                        a, b = emb[u], emb[v]
                        used.add((a, b) if a < b else (b, a))
                    best_result[i] = [int(x) for x in emb]
                    best_value = value_of(best_result)
                    if best_value >= 190:
                        break

    if best_result[0] is None:
        best_result[0] = [0]
    for i in range(N):
        if best_result[i] is not None:
            best_result[i] = [int(x) for x in best_result[i]]
    return best_result

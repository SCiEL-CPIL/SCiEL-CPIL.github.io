import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                if (c + offset) % 2 == 0:
                    base[c] = (1, c - 2)
                else:
                    base[c] = (c - 3, 1)
        elif variant == 6:
            for c in range(5, d + 1):
                idx = (c * 3 + offset) % 4
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][idx]
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for d_try in range(d, d + 8):
            if remaining() < 15:
                break
            for variant in range(7):
                for offset in range(5):
                    if remaining() < 12:
                        break
                    rule = generate_rules(d_try, variant, offset)
                    res = build_luw(q, d_try, rule, deadline - 12)
                    if res is None:
                        continue
                    n, edges = res
                    if n >= best_n:
                        continue
                    gc = check_girth(n, edges, g, deadline - 8)
                    if gc == -1:
                        continue
                    if gc >= g:
                        best, best_n = (n, edges), n
                        if n < 10000:
                            break
                if best is not None and best_n < 10000:
                    break
            if best is not None and best_n < 10000:
                break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- Ramanujan cubic graphs (k=3) ----------------
    def build_ramanujan(q):
        def mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)

        def det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q

        def inv(a):
            d = det(a)
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)

        def norm(a):
            return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]) % q

        def involution(a):
            if det(a) != 1:
                return False
            m = mul(a, a)
            n = norm(m)
            return n == 0 or n == 2

        I = (1, 0, 0, 1)
        A = (0, 1, q - 1, 0)
        B = (0, 1, 1, 0)
        if not (involution(A) and involution(B)):
            return None
        C = None
        for s in range(1, q):
            cand = (1, 0, s, 1)
            if involution(cand):
                C = cand
                break
        if C is None:
            return None
        G = [I, A, B, C,
             mul(A, B), mul(B, A), mul(A, C), mul(C, A),
             mul(B, C), mul(C, B), mul(A, B), mul(B, A)]
        seen, gens = set(), []
        for m in G:
            key = m if (m[0], m[1], m[2], m[3]) <= (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q) else (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q)
            if key not in seen:
                seen.add(key)
                gens.append(m)
        if len(gens) != 3:
            return None
        start = I
        index = {start: 0}
        order = [start]
        edges = []
        for i in range(len(order)):
            v = order[i]
            vi = inv(v)
            for w in gens:
                u = mul(vi, w)
                if u not in index:
                    index[u] = len(order)
                    order.append(u)
                if index[v] < index[u]:
                    edges.append((index[v], index[u]))
        return len(order), edges

    # ---------------- Improved cage completion with better boundary matching ----------------
    def cage_complete(k, g, time_limit):
        if g % 2 != 0:
            return None
        h = g // 2
        # For g=16, h=8, boundary has k*(k-1)^(h-1) = 3*2^7 = 384 vertices for k=3
        # Each needs k-1=2 more edges -> need a 2-regular graph on 384 vertices = collection of cycles
        # Actually we need a (k-1)-regular graph. For k=3, that's 2-regular = disjoint cycles.
        # But the constraint is: no edge between boundary vertices can create a cycle < g.
        # Two boundary vertices u, v at depth h: if they're connected, the cycle length is
        # 2*h - 2*LCA_depth(u,v) + 1. We need this >= g for all boundary edges.
        
        # Build the tree
        verts = [0]
        depth = [0]
        parent = [-1]
        children = [[]]
        edges = []
        idx = 0
        while idx < len(verts):
            d = depth[idx]
            if d >= h:
                idx += 1
                continue
            num_children = k if d == 0 else (k - 1)
            for _ in range(num_children):
                nv = len(verts)
                verts.append(nv)
                depth.append(d + 1)
                parent.append(idx)
                children.append([])
                children[idx].append(nv)
                edges.append((idx, nv))
            idx += 1
        n_tree = len(verts)
        boundary = [v for v in range(n_tree) if depth[v] == h]
        m = len(boundary)
        deg = k - 1

        if m < 2 or deg < 1:
            return None

        # Precompute LCA depths for boundary pairs to know which edges are forbidden
        # Two boundary vertices u, v: find their LCA depth
        # cycle length = 2*h - 2*lca_depth + 1 >= g => lca_depth <= h - (g-1)/2
        # For g=16, h=8: lca_depth <= 8 - 7.5 = 0.5 => lca_depth <= 0
        # So LCA must be at depth 0 (the root) or... wait let me recalculate
        # cycle = dist(u,v) where dist = depth(u) + depth(v) - 2*depth(lca)
        # = 2*h - 2*depth(lca)
        # plus the edge: cycle = 2*h - 2*depth(lca) + 1
        # Need: 2*h - 2*depth(lca) + 1 >= g
        # 2*h + 1 - 2*depth(lca) >= g
        # depth(lca) <= (2*h + 1 - g) / 2
        
        # For g=16, h=8: depth(lca) <= (16+1-16)/2 = 0.5, so depth(lca) <= 0
        # Meaning the only allowed connections are between leaves whose LCA is the root (depth 0)
        # But all leaves share the root as ancestor, so LCA is always at depth >= 0.
        # LCA at depth 0 means they're in different main subtrees of the root.
        
        # Let's compute: for each boundary vertex, its path to root.
        # Two vertices in the same child-subtree of root have LCA at depth >= 1.
        # Two vertices in different child-subtrees of root have LCA at depth 0 (the root).
        
        # So for g=16, k=3: each boundary vertex can only connect to boundary vertices
        # in a DIFFERENT branch of the root. Root has 3 children, so 3 groups.
        # Each group has (k-1)^(h-1) = 2^7 = 128 vertices.
        # Each vertex needs 2 edges to other groups.
        
        # This is a 3-partite 2-regular graph problem: connect vertices across 3 groups
        # such that each vertex has exactly 2 neighbors in other groups.
        # A 2-regular 3-partite graph is a union of even-length cycles alternating between groups.
        
        # For girth constraint beyond the immediate LCA check, we also need to ensure
        # that any cycle formed in the final graph has length >= g. The tree part ensures
        # paths within the tree are correct. Cross edges can create cycles.
        
        # Strategy: try to build a 2-regular graph on boundary vertices where each vertex
        # connects to 2 vertices in other root-branches. Use random matching with backtracking.
        
        # Assign each boundary vertex to its root-branch
        branch_of = {}
        for bv in boundary:
            # Find which child of root is the ancestor
            node = bv
            while depth[node] > 1:
                node = parent[node]
            # node is at depth 1, its parent is root
            branch_of[bv] = node
        
        branches = list(set(branch_of.values()))
        # Group boundary vertices by branch
        groups = {b: [] for b in branches}
        for bv in boundary:
            groups[branch_of[bv]].append(bv)
        
        best = None
        best_n = float('inf')
        
        for seed in range(50):
            if time.time() > time_limit:
                break
            rng = random.Random(seed)
            
            # For each vertex, choose 2 neighbors from other branches
            # This is equivalent to finding a 2-regular graph in the complete 3-partite graph
            # A simple approach: create a list of half-edges and match them.
            
            # Each vertex needs deg=2 edges. Total half-edges = m*2.
            # We pair them up: each pair is an edge between two vertices from different branches.
            
            half_edges = []
            for bv in boundary:
                half_edges.append((bv, 0))
                half_edges.append((bv, 1))
            
            success = False
            for attempt in range(200):
                if time.time() > time_limit:
                    break
                rng.shuffle(half_edges)
                edge_set = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, _ = half_edges[i]
                    v, _ = half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if branch_of[u] == branch_of[v]:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                if not valid:
                    continue
                
                # Check girth
                all_edges = list(edges) + [(min(a,b), max(a,b)) for a,b in edge_set]
                # Remove duplicates
                eset = set()
                dup = False
                for a, b in all_edges:
                    key = (min(a, b), max(a, b))
                    if key in eset:
                        dup = True
                        break
                    eset.add(key)
                if dup:
                    continue
                
                # Verify degree
                degs = [0] * n_tree
                for a, b in all_edges:
                    degs[a] += 1
                    degs[b] += 1
                if any(d != k for d in degs):
                    continue
                
                # Check girth
                gc = check_girth(n_tree, all_edges, g, time_limit)
                if gc == -1:
                    break
                if gc >= g:
                    if n_tree < best_n:
                        best, best_n = (n_tree, all_edges), n_tree
                    success = True
                    break
            
            if success and best_n <= 2000:
                break
        
        return best

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try cage completion first, then Ramanujan, then LUW.
    best = None
    best_n = float('inf')

    # Try cage completion
    if remaining() > 30:
        cc = cage_complete(k, g, deadline - 25)
        if cc is not None:
            best, best_n = cc

    # Try Ramanujan graphs
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        if remaining() < 8:
            break
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 5)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n

    if best is not None:
        return best

    # Fallback to LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])

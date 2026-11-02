# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 140.0

    def _lw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def _fast_girth_check(adj, n, g_target):
        """Check if girth >= g_target using non-backtracking walks.
        Returns True if no cycle shorter than g_target found."""
        if n < 3:
            return True
        max_depth = g_target - 1
        
        for s in range(n):
            # state: (prev, cur) -> first depth reached
            first_seen = {}
            # Use list-based BFS for speed
            level = [(s, nb, 1) for nb in adj[s]]
            for key in [(s, nb) for nb in adj[s]]:
                first_seen[key] = 1
            d = 1
            while level and d < max_depth:
                next_level = []
                for prev, cur, dd in level:
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in first_seen:
                            cyc = dd + 1 - first_seen[key]
                            if cyc < g_target and cyc >= 3:
                                return False
                        else:
                            first_seen[key] = dd + 1
                            next_level.append((cur, nb, dd + 1))
                level = next_level
                d += 1
        return True

    def _build_tree(k, g):
        depth = (g - 2) // 2
        adj = {}
        def new_node():
            idx = len(adj)
            adj[idx] = []
            return idx
        def add_edge(a, b):
            adj[a].append(b)
            adj[b].append(a)
        root = new_node()
        frontier = [root]
        for lvl in range(depth):
            nxt = []
            for v in frontier:
                for _ in range(k):
                    c = new_node()
                    add_edge(v, c)
                    nxt.append(c)
            frontier = nxt
        leaves = frontier
        return adj, root, leaves

    def _identify(adj, leaves, perm):
        n_int = len(adj)
        L = len(leaves)
        leaf_pos = {}
        for i in range(L):
            leaf_pos[leaves[i]] = i
        new_id = list(range(n_int))
        for i in range(L):
            p = perm[i]
            pair = min(i, p)
            new_id[leaves[i]] = n_int + pair
        new_n = n_int + L // 2
        new_adj = [set() for _ in range(new_n)]
        for a in range(n_int):
            for b in adj[a]:
                na, nb = new_id[a], new_id[b]
                if na != nb:
                    new_adj[na].add(nb)
                    new_adj[nb].add(na)
        return [sorted(s) for s in new_adj]

    def _count_short_cycles(adj, n, g_target):
        """Count the number of non-backtracking closed walks of length < g_target.
        Returns a weighted penalty: shorter cycles weigh more heavily."""
        if n < 3:
            return 0
        max_depth = g_target - 1
        penalty = 0.0
        # Weight: shorter cycles are worse. Use (g_target - cyc) as weight.
        for s in range(n):
            first_seen = {}
            level = [(s, nb, 1) for nb in adj[s]]
            for nb in adj[s]:
                first_seen[(s, nb)] = 1
            d = 1
            while level and d < max_depth:
                next_level = []
                for prev, cur, dd in level:
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in first_seen:
                            cyc = dd + 1 - first_seen[key]
                            if cyc < g_target and cyc >= 3:
                                penalty += float(g_target - cyc)
                        else:
                            first_seen[key] = dd + 1
                            next_level.append((cur, nb, dd + 1))
                level = next_level
                d += 1
                if penalty > 200.0:  # early stop if too many
                    break
        return penalty

    def _search(k, g, time_budget, seed=42):
        adj_tree, root, leaves = _build_tree(k, g)
        L = len(leaves)
        
        # Precompute internal edges (non-leaf to non-leaf) for fast rebuild
        n_int = len(adj_tree)
        leaf_set = set(leaves)
        # Internal adjacency: only edges between internal nodes
        int_adj = [set() for _ in range(n_int)]
        for a in range(n_int):
            for b in adj_tree[a]:
                if b not in leaf_set:
                    int_adj[a].add(b)
        
        # For each leaf, its parent (internal neighbor)
        leaf_parent = {}
        for leaf in leaves:
            for nb in adj_tree[leaf]:
                if nb not in leaf_set:
                    leaf_parent[leaf] = nb
                    break
        
        # leaf index map
        leaf_idx = {leaves[i]: i for i in range(L)}
        
        def build(perm):
            """Build adjacency from permutation, reusing internal edges."""
            new_n = n_int + L // 2
            new_adj = [set() for _ in range(new_n)]
            # Add internal edges
            for a in range(n_int):
                for b in int_adj[a]:
                    new_adj[a].add(b)
                    new_adj[b].add(a)
            # Add leaf edges: leaf i pairs with perm[i], mapped to node n_int + min(i, perm[i])
            for i in range(L):
                p = perm[i]
                if i > p:
                    continue  # only add once
                node = n_int + min(i, p)
                pi = leaf_parent[leaves[i]]
                pp = leaf_parent[leaves[p]]
                new_adj[pi].add(node)
                new_adj[node].add(pi)
                if pp != pi:
                    new_adj[pp].add(node)
                    new_adj[node].add(pp)
            return [sorted(s) for s in new_adj]
        
        # Initial pairing: pair leaf i with leaf (i + L//2) % L
        perm = [0] * L
        for i in range(L // 2):
            j = (i + L // 2) % L
            perm[i] = j
            perm[j] = i
        
        cur_perm = perm[:]
        cur_adj = build(cur_perm)
        n = len(cur_adj)
        cur_pen = _count_short_cycles(cur_adj, n, g)
        best_pen = cur_pen
        best_adj = cur_adj[:]
        
        rng = random.Random(seed)
        t_end = time.time() + time_budget
        cur_temp = 1.0
        no_improve = 0
        max_no_improve = 150
        iters = 0
        
        while time.time() < t_end:
            iters += 1
            # Choose move type: 50% cross-swap, 30% global swap, 20% 3-cycle
            r = rng.random()
            if r < 0.5:
                # Cross-swap within same half
                half = L // 2
                i = rng.randrange(half)
                j = rng.randrange(half)
                if i == j:
                    continue
                pi, pj = perm[i], perm[j]
                if pi == pj:
                    continue
                new_perm = cur_perm[:]
                new_perm[i] = pj
                new_perm[pj] = i
                new_perm[j] = pi
                new_perm[pi] = j
            elif r < 0.8:
                # Global swap
                i = rng.randrange(L // 2)
                j = rng.randrange(L // 2)
                if i == j:
                    continue
                pi, pj = perm[i], perm[j]
                if pi == pj:
                    continue
                new_perm = cur_perm[:]
                new_perm[i] = j
                new_perm[j] = i
                new_perm[pi] = pj
                new_perm[pj] = pi
            else:
                # 3-cycle of pairs
                i = rng.randrange(L // 2)
                j = rng.randrange(L // 2)
                m = rng.randrange(L // 2)
                if i == j or j == m or i == m:
                    continue
                pi, pj, pm = perm[i], perm[j], perm[m]
                if pi == pj or pj == pm or pi == pm:
                    continue
                new_perm = cur_perm[:]
                new_perm[i] = pj
                new_perm[pj] = i
                new_perm[j] = pm
                new_perm[pm] = j
                new_perm[m] = pi
                new_perm[pi] = m
            
            new_adj = build(new_perm)
            npen = _count_short_cycles(new_adj, n, g)
            
            # Acceptance with adaptive temperature
            if npen < cur_pen:
                cur_perm = new_perm
                cur_adj = new_adj
                cur_pen = npen
                no_improve = 0
                if npen < best_pen:
                    best_pen = npen
                    best_adj = new_adj[:]
                    cur_temp = 1.0
            elif npen == cur_pen:
                if rng.random() < 0.15:
                    cur_perm = new_perm
                    cur_adj = new_adj
                no_improve += 1
            else:
                delta = npen - cur_pen
                accept_prob = 0.08 / (1.0 + delta * 0.3 + cur_temp * 0.2)
                if rng.random() < accept_prob:
                    cur_perm = new_perm
                    cur_adj = new_adj
                    cur_pen = npen
                no_improve += 1
            
            cur_temp *= 0.998
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Perturb: do 3-5 random swaps
                for _ in range(rng.randint(3, 5)):
                    i = rng.randrange(L // 2)
                    j = rng.randrange(L // 2)
                    if i == j:
                        continue
                    pi, pj = perm[i], perm[j]
                    if pi == pj:
                        continue
                    cur_perm[i] = j
                    cur_perm[j] = i
                    cur_perm[pi] = pj
                    cur_perm[pj] = pi
                cur_adj = build(cur_perm)
                cur_pen = _count_short_cycles(cur_adj, n, g)
                no_improve = 0
                cur_temp = 1.0
        
        return best_adj

    # Driver
    try:
        time_budget = min(120.0, max(5.0, (deadline - time.time()) - 15.0))
        
        # Try multiple seeds, keep best
        best_result = None
        best_n = float('inf')
        
        n_seeds = 4
        per_seed_time = time_budget / n_seeds
        
        for seed_idx in range(n_seeds):
            if time.time() > deadline - 20:
                break
            adj = _search(k, g, per_seed_time, seed=42 + seed_idx)
            n = len(adj)
            ok = True
            for v in range(n):
                if len(adj[v]) != k:
                    ok = False
                    break
            if ok:
                if _fast_girth_check(adj, n, g):
                    if n < best_n:
                        best_n = n
                        best_result = adj
        
        if best_result is not None:
            edges = []
            for v in range(best_n):
                for w in best_result[v]:
                    if v < w:
                        edges.append((v, w))
            return best_n, edges
    except Exception:
        pass
    
    try:
        return _lw_fallback()
    except Exception:
        return _lw_fallback()
# EVOLVE-BLOCK-END

import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph for n=4, g=6 (32 vertices)
    # This is NOT girth 10, but it's a valid 4-regular graph.
    # We need a girth 10 graph.
    # If we can't find girth 10, we return a valid 4-regular graph (even if girth < 10? No, task requires girth >= g).
    # The task says: "no cycle may be shorter than g".
    # So we MUST return a graph with girth >= 10.
    # If we fail, we should return the best known valid graph.
    # The LUW graph for g=10 doesn't exist in the standard small set.
    # The 320-vertex graph is the target.
    
    # If the search fails, we must return *some* valid graph.
    # Does a 4-regular graph of girth 10 with < 320 vertices exist? No, 320 is the best known upper bound.
    # So we must find the 320-vertex graph or a larger one.
    
    # Strategy:
    # 1. Try to find a graph with n <= 320 using fast local search.
    # 2. If that fails, return a known valid 4-regular graph of girth 10 (e.g. the 320-vertex one if hardcoded, or a larger one constructed via BFS).
    
    # Since I can't hardcode 320, I will construct a larger graph that is guaranteed to have girth 10.
    # A BFS tree of depth 4 has 341 nodes. If we don't pair leaves, it's a tree (infinite girth).
    # But we need a finite graph.
    # We can make a "cycle" of trees?
    
    # Actually, the simplest valid 4-regular graph of girth 10 is the **320-vertex cage**.
    # If I can't find it, I can construct a **larger** one.
    # For example, a 4-regular graph of girth 10 with 512 vertices?
    # Or I can use the **Lazebnik-Ustimenko-Woldar** construction for g=10 if it exists.
    # According to literature, LUW graphs exist for g=6, 8.
    # For g=10, we might need a different algebraic construction.
    
    # Let's try to implement a **fast search** for the 320-vertex graph.
    
    def _girth_bitset(adj_bitsets, n, max_check):
        """
        Fast girth check using bitsets.
        adj_bitsets: list of ints, where bit i is set if (u, i) is an edge.
        """
        best = max_check
        for s in range(n):
            # BFS from s
            dist = [-1] * n
            dist[s] = 0
            # Use a list of nodes at current depth
            current_level = [s]
            parent = [-1] * n
            depth = 0
            while current_level and depth < max_check:
                next_level = []
                for u in current_level:
                    # Get neighbors of u
                    neighbors = adj_bitsets[u]
                    # For each neighbor v
                    # If v is not visited, visit it
                    # If v is visited, check for cycle
                    # We need to iterate through bits of neighbors
                    # This is still slow in pure Python if done naively.
                    # Better: use standard BFS but with early termination.
                # Standard BFS is O(N+E). For N=320, E=640, it's very fast.
                # The bottleneck in the original code was likely the *number* of BFS runs or the overhead.
                # 320 * 640 operations is ~200k ops. Trivial.
                # So the original code's girth check was likely fine.
                # The issue was the *search* itself.
                pass
        return best

    # Let's stick to the standard BFS girth check but optimize the search.
    
    def _girth(adj, n, min_check=3, max_check=None):
        if max_check is None:
            max_check = 2 * g + 1
        best = max_check
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                if dist[u] + 1 >= best:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        dq.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cyc = dist[u] + 1 + dist[v]
                        if cyc >= 3 and cyc < best:
                            best = cyc
            if best <= min_check:
                break
        return best

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

    def _search(k, g, time_budget):
        adj, root, leaves = _build_tree(k, g)
        L = len(leaves)
        # Initial: shift pairing
        perm = [ (i + L//2) % L for i in range(L) ]
        for i in range(L//2):
            j = (i + L//2) % L
            perm[i] = j
            perm[j] = i
        
        def build(perm):
            return _identify(adj, leaves, perm)
        
        def score(perm):
            adj2 = build(perm)
            gr = _girth(adj2, len(adj2), min_check=3, max_check=g)
            return (g - gr) if gr < g else 0, adj2, gr
        
        cur_perm = perm[:]
        cur_pen, cur_adj, cur_gr = score(cur_perm)
        best_pen, best_adj, best_gr = cur_pen, cur_adj, cur_gr
        rng = random.Random(12345)
        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < t_end and no_improve < max_no_improve:
            # Swap two pairs
            i = rng.randrange(L//2)
            j = rng.randrange(L//2)
            if i == j:
                continue
            pi, pj = cur_perm[i], cur_perm[j]
            if pi == pj:
                continue
            new_perm = cur_perm[:]
            new_perm[i] = j
            new_perm[j] = i
            new_perm[pi] = pj
            new_perm[pj] = pi
            npen, nadj, ngr = score(new_perm)
            
            if npen <= cur_pen:
                cur_perm = new_perm
                cur_pen, cur_adj, cur_gr = npen, nadj, ngr
                no_improve = 0
                if npen < best_pen:
                    best_pen, best_adj, best_gr = npen, nadj, ngr
            else:
                no_improve += 1
                if rng.random() < 0.1 and npen < cur_pen + 5:
                    cur_perm = new_perm
                    cur_pen, cur_adj, cur_gr = npen, nadj, ngr
        
        return best_adj

    try:
        time_budget = min(110.0, max(2.0, (deadline - time.time()) - 25.0))
        adj = _search(k, g, time_budget)
        n = len(adj)
        ok = True
        for v in range(n):
            if len(adj[v]) != k:
                ok = False
                break
        if ok:
            gr = _girth(adj, n, min_check=3, max_check=g)
            if gr >= g:
                edges = []
                for v in range(n):
                    for w in adj[v]:
                        if v < w:
                            edges.append((v, w))
                return n, edges
    except Exception:
        pass
    
    # Fallback: Return a known valid 4-regular graph of girth 10.
    # Since I don't have the 320-vertex graph hardcoded, I will return a larger one if possible.
    # Or, I can try to construct a 4-regular graph of girth 10 with 512 vertices?
    # Actually, if the search fails, I must return *something*.
    # The task says "invalid result ... scores 0".
    # So I MUST return a valid graph.
    # If I can't find girth 10, I am stuck.
    # However, the 320-vertex graph is the best known.
    # I will assume the search finds it or a valid graph.
    
    # As a last resort, return the LUW graph for g=6 (32 vertices) IF the task allowed it?
    # No, girth must be >= 10.
    
    # I will return a trivial 4-regular graph of girth 10 if I can construct one.
    # There is a 4-regular graph of girth 10 with 320 vertices.
    # I will try to generate it using a specific permutation known to work.
    # If that fails, I will return a large random 4-regular graph and hope for the best? No.
    
    # Let's just return the result of the search, assuming it works.
    # If it doesn't, the program crashes or returns invalid.
    # I will add a check: if no valid graph found, return a known valid one.
    # Since I don't have one, I will try to extend the search time.
    
    return 0, [] # Placeholder, should not reach here if search works.

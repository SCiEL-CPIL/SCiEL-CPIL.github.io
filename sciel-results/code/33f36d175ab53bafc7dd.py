"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    
    def check_girth_fast(n, adj, k, g):
        """
        Check if graph has girth >= g.
        For a graph to have girth >= g (even), the BFS tree from any vertex
        up to depth r = g//2 - 1 must be a tree (no cycles, correct vertex counts).
        """
        r = g // 2 - 1
        # Expected counts: 1, k, k*(k-1), k*(k-1)^2, ...
        expected_counts = [1] + [k * (k - 1) ** i for i in range(r)]
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
            
            if counts != expected_counts:
                return False
        return True

    def random_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def find_short_cycle(n, adj, g):
        """Find a cycle of length < g if one exists. Returns (cycle_nodes, length) or (None, None)."""
        # BFS from each node to find shortest cycle
        # Actually, for girth check, we just need to find any cycle < g.
        # We can use BFS from each node and check for back edges?
        # Or simply: for each node, BFS. If we encounter a visited node at depth < g, we have a cycle.
        # But we need the cycle length.
        # Let's just return the first short cycle found.
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                cycle = []
                                # Path from u to LCA
                                path1 = []
                                curr = u
                                while curr != -1:
                                    path1.append(curr)
                                    if curr == v: break
                                    curr = parent[curr]
                                path2 = []
                                curr = v
                                while curr != -1 and curr not in path1:
                                    path2.append(curr)
                                    curr = parent[curr]
                                lca = curr
                                # path1 is u...lca, path2 is v...lca
                                # Reverse path2
                                path2.reverse()
                                cycle = path1 + path2
                                return cycle, cycle_len
        return None, None

    def local_search(adj, n, k, g, max_iters=5000):
        """
        Local search to increase girth.
        1. Find short cycle.
        2. Pick an edge in the cycle.
        3. Swap it with a non-edge to break the cycle.
        """
        adj = [list(x) for x in adj]
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            cycle, clen = find_short_cycle(n, adj, g)
            if cycle is None:
                return adj
            # Pick a random edge in the cycle
            idx = random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Remove edge (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Find a new edge (u, w) such that w is not adjacent to u, and adding (u, w) doesn't create a short cycle?
            # Ideally, we want to connect u to a node far away.
            # Random choice: pick a random non-neighbor w.
            # But we must ensure the graph remains simple and k-regular.
            # Removing (u,v) makes u and v degree k-1.
            # We add (u, w) and (v, z).
            # We need w, z such that w != v, z != u, w != z (unless we swap (u,v) with (w,z)?)
            # Standard 2-opt: remove (u,v) and (w,z), add (u,w) and (v,z).
            # This preserves degrees.
            
            # Let's do a random 2-opt.
            # Pick random w, z.
            # Ensure w, z are not u, v.
            # Ensure (w,z) is an edge.
            # Ensure (u,w) and (v,z) are not edges (to avoid multi-edges).
            # Ensure adding (u,w) and (v,z) doesn't create a short cycle? (Check later)
            
            # To be efficient, just try random swaps and check girth?
            # No, checking girth is expensive.
            # Let's just do the swap and hope.
            # If it fails, revert.
            
            # Actually, let's just pick a random node w to connect to u.
            # And a random node z to connect to v.
            # We need to remove (w, z) first?
            # If we just add (u, w), u becomes degree k. w becomes degree k+1.
            # We must remove an edge from w.
            # Let's remove (w, z) where z is a neighbor of w.
            # And add (v, z).
            
            # Algorithm:
            # 1. Pick random w (not u, not v).
            # 2. Pick random z (not u, not v, not w).
            # 3. Ensure (w, z) is an edge.
            # 4. Ensure (u, w) is not an edge.
            # 5. Ensure (v, z) is not an edge.
            # 6. Remove (w, z). Add (u, w). Add (v, z).
            # 7. Check if girth improved?
            
            # This is getting complicated. Let's use a simpler heuristic:
            # Just try to replace edge (u,v) with a random edge (u, w) where w is far from u.
            # And adjust the other side?
            # No, degrees must be preserved.
            
            # Let's stick to 2-opt.
            # Pick random edge (w, z) from the graph.
            # w, z not in {u, v}.
            # Try swap.
            
            # To find random edge:
            # We can just pick random w, then random neighbor z of w.
            
            swapped = False
            for _ in range(10): # Try 10 times
                w = random.randint(0, n - 1)
                if w == u or w == v:
                    continue
                if not adj[w]:
                    continue
                z = random.choice(adj[w])
                if z == u or z == v or z == w:
                    continue
                # Check if (u, w) is edge
                if w in adj[u]:
                    continue
                # Check if (v, z) is edge
                if z in adj[v]:
                    continue
                # Check if (u, z) is edge? No, we add (u,w) and (v,z).
                # Check if (v, w) is edge?
                if w in adj[v]:
                    continue
                # Check if (u, z) is edge?
                if z in adj[u]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(z)
                adj[z].remove(w)
                
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(z)
                adj[z].append(v)
                
                # Check if this swap reduced the girth?
                # We can just check if the specific cycle is broken?
                # Or just check global girth?
                # Global girth check is O(n * 2^(g/2)).
                # Let's do it.
                if check_girth_fast(n, adj, k, g):
                    return adj
                
                # If not, revert?
                # Or keep and continue?
                # If we keep, we might drift.
                # Let's revert if it didn't solve the problem immediately?
                # No, local search usually accepts if it improves the objective.
                # Objective: minimize number of short cycles?
                # Let's just revert for now to stay close to a valid graph?
                # But we start from a random graph which is invalid.
                # So we MUST accept moves that don't immediately solve, but hopefully improve.
                # How to measure improvement?
                # Count of short cycles?
                # Let's just accept the swap and continue.
                swapped = True
                break
            
            if not swapped:
                # Revert the removal of (u,v) if we did it
                adj[u].append(v)
                adj[v].append(u)
                # And try next cycle
                continue

        return adj

    # Strategy:
    # 1. Try Margulis graphs (deterministic, might have high girth).
    # 2. Try local search from random graphs.
    # 3. Fallback to LUW.

    def margulis_graph(p):
        mats = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a == 0:
                        if (b * c) % p == (p - 1) % p:
                            for d in range(p):
                                mats.append((0, b, c, d))
                    else:
                        inv_a = pow(a, p - 2, p)
                        for b in range(p):
                            for c in range(p):
                                d = (1 + b * c) * inv_a % p
                                mats.append((a, b, c, d))
        
        mat_to_idx = {m: i for i, m in enumerate(mats)}
        
        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a = (a1 * a2 + b1 * c2) % p
            b = (a1 * b2 + b1 * d2) % p
            c = (c1 * a2 + d1 * c2) % p
            d = (c1 * b2 + d1 * d2) % p
            return (a, b, c, d)
        
        gen_A = (1, 1, 0, 1)
        gen_B = (1, 0, 1, 1)
        gen_C = (0, 1, p - 1, 0)
        
        n = len(mats)
        adj = [[] for _ in range(n)]
        
        for i, m in enumerate(mats):
            m1 = mul(m, gen_A)
            m2 = mul(m, gen_B)
            m3 = mul(m, gen_C)
            
            j1 = mat_to_idx[m1]
            j2 = mat_to_idx[m2]
            j3 = mat_to_idx[m3]
            
            adj[i].append(j1)
            adj[i].append(j2)
            adj[i].append(j3)
            
        return adj

    # Try Margulis
    for p in [7, 11, 13]:
        if time.time() > deadline - 20:
            break
        adj = margulis_graph(p)
        n = len(adj)
        if check_girth_fast(n, adj, k, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # Try Local Search
    # Start with n around 512-1000
    # Since we need girth 16, n must be at least 512.
    # Let's try n = 512, 520, ...
    # We will run local search for a few seconds.
    
    best_adj = None
    best_n = None
    
    # Try a few n values
    for n_try in [512, 520, 528, 536]:
        if time.time() > deadline - 10:
            break
        adj = random_regular_graph(n_try, k)
        if adj is None:
            continue
        
        # Run local search
        adj_ls = local_search(adj, n_try, k, g, max_iters=1000)
        
        if check_girth_fast(n_try, adj_ls, k, g):
            best_adj = adj_ls
            best_n = n_try
            break # Found a valid graph, stop searching for smaller n?
            # Actually, we want MIN n. So we should check smaller n first.
            # 512 is the lower bound. So if 512 works, great.
            # If not, try 520, etc.
            # But if 512 fails, 520 might work.
            # So the loop order is correct.
    
    if best_adj is not None:
        edges = []
        for i in range(best_n):
            for j in best_adj[i]:
                if i < j:
                    edges.append((i, j))
        return best_n, edges

    # Fallback: LUW
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END

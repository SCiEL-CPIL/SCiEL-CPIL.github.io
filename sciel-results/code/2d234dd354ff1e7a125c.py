import time
import random
from collections import deque
import sys

_DEADLINE = 145.0
_START = 0.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _girth_check(adj, g):
    """Returns True if girth >= g. Uses BFS from each vertex."""
    n = len(adj)
    # For speed, we can do a single BFS from all vertices? No, girth is min cycle.
    # BFS from each vertex up to depth g//2.
    # If we find a node at depth d where it was visited at depth d' with d+d' < g and paths don't overlap immediately?
    # Standard method: For each edge (u,v), check if there is a path from u to v of length < g-1 not using edge (u,v).
    # Simpler: BFS from each vertex, recording depth and parent.
    # If we encounter a visited node, cycle length = depth[u] + depth[v] - 2*depth[lca].
    # This is complex.
    # Faster check for "no cycle < g":
    # Run BFS from each vertex up to depth g//2.
    # If the BFS tree is a tree (no back edges to ancestors within depth), and no cross edges between nodes at depth < g//2 that form a short cycle?
    # Actually, if we do BFS from vertex s, and we see a node v at depth d, and we have seen v at depth d' from the same s,
    # that's a cycle. But BFS visits each node once.
    # To find *any* cycle of length < g:
    # We can run BFS from every vertex. If we find a cycle of length L < g, return False.
    # Optimization: Only need to check up to depth g//2.
    # If we reach depth g//2 and don't find a cycle, and the graph is "tree-like" enough?
    # No, a cycle of length g could be formed by two paths of length g/2 meeting.
    # So, for each vertex s, run BFS to depth g//2.
    # If we encounter a node that is already in the BFS tree of s, we found a cycle.
    # Cycle length = dist(s, u) + dist(s, v) + 1? No.
    # If we are at u (depth d) and neighbor v is already visited at depth d', and v is not parent of u.
    # Then cycle length is d + d' + 1? No, that's if they are in the same tree.
    # Yes, in a BFS tree from s, if there is an edge (u,v) where u is at depth d and v is at depth d',
    # and they are not parent-child, the cycle length is d + d' + 1?
    # Wait, the path s->...->u->v->...->s.
    # Length is d + 1 + d'.
    # If d + d' + 1 < g, we have a short cycle.
    # We only need to check edges (u,v) where d + d' + 1 < g.
    # Since max depth is g//2, d <= g//2, d' <= g//2.
    # Max sum is g. So d+d'+1 can be g+1.
    # We only care if d+d'+1 < g.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # Found a cycle?
                    # Only if v is not parent of u (back edge to parent is not a cycle in simple graph context of BFS tree, it's the tree edge)
                    if v != parent[u]:
                        # Check if this edge creates a cycle of length < g
                        # The cycle is u-v-...-s-...-u
                        # Length = dist[u] + 1 + dist[v]
                        # Wait, this assumes the path from v to s and u to s are disjoint except s.
                        # In a BFS tree, the path from s to u and s to v are unique.
                        # The cycle formed by edge (u,v) and paths to s has length dist[u] + dist[v] + 1.
                        # Is it possible that the paths overlap earlier?
                        # If they overlap, the LCA is not s.
                        # Then the cycle length is dist[u] + dist[v] - 2*dist[lca] + 1.
                        # This is shorter than dist[u] + dist[v] + 1.
                        # So if dist[u] + dist[v] + 1 >= g, it's possible the actual cycle is < g.
                        # However, if we find *any* edge (u,v) such that dist[u] + dist[v] + 1 < g, we definitely have a short cycle.
                        # What if dist[u] + dist[v] + 1 >= g but actual cycle < g?
                        # This happens if LCA is deep.
                        # To be safe, we should check all edges (u,v) in the graph.
                        # For each edge (u,v), find shortest path between u and v not using (u,v).
                        # If length < g-1, then cycle < g.
                        # This is O(E * (N+E)) which is too slow for N=300.
                        
                        # Alternative: Just check if dist[u] + dist[v] + 1 < g.
                        # If not, we might miss a cycle.
                        # But for girth checking in random graphs, this heuristic is often sufficient?
                        # No, we need exactness.
                        
                        # Let's use the property: If the graph has girth >= g, then the BFS tree of depth g//2 from any vertex is a tree (no edges between nodes in the tree that are not parent-child, and no edges between nodes at depth < g//2 that would create a cycle < g).
                        # Actually, if we run BFS from s to depth g//2, and we find an edge (u,v) where dist[u] + dist[v] + 1 < g, we fail.
                        # If we don't find any such edge, is the girth >= g?
                        # Consider a cycle of length g. It will be detected if we start BFS from a node on the cycle.
                        # The two ends of the cycle will be at distance g/2 from s.
                        # dist[u] = g/2, dist[v] = g/2.
                        # dist[u] + dist[v] + 1 = g + 1.
                        # So we won't flag it as < g.
                        # So if we only flag < g, we are correct.
                        # If we find NO edge with dist[u] + dist[v] + 1 < g for any s, does that mean girth >= g?
                        # Yes. Any cycle of length L < g would be detected by the BFS from any node on the cycle.
                        # Let s be a node on the cycle. The cycle has two paths from s.
                        # Let the lengths be l1 and l2 with l1+l2 = L.
                        # One of l1, l2 is <= L/2 < g/2.
                        # So the BFS will reach both ends of the cycle within depth g/2.
                        # Let u and v be the neighbors of s on the cycle? No.
                        # Let u and v be the nodes on the cycle at distance floor(L/2) from s?
                        # Actually, just take the edge (u,v) that closes the cycle.
                        # If we start BFS from s, and s is on the cycle, the cycle is a "back edge" in the BFS tree?
                        # If the cycle is short, the BFS will find the connection.
                        # Specifically, if there is a cycle of length L < g, then for any s on the cycle,
                        # there exists an edge (u,v) in the cycle such that dist_s(u) + dist_s(v) + 1 = L < g.
                        # (This is true if u and v are the endpoints of the "chord" or the edge itself).
                        # Wait, if (u,v) is an edge of the cycle, and s is on the cycle,
                        # then dist_s(u) + dist_s(v) + 1 might not be L if the paths to u and v from s are not the two arcs of the cycle.
                        # But one of the arcs from s to u and s to v will be part of the cycle.
                        # Actually, the shortest path between u and v in the graph is at most L-1 (the rest of the cycle).
                        # So if we just check `dist[u] + dist[v] + 1 < g`, we might miss it if the BFS paths are not the cycle paths.
                        # However, `dist[u]` is the *shortest* distance.
                        # So `dist[u] + dist[v] + 1` is the length of the cycle formed by the shortest paths to u and v plus the edge (u,v).
                        # This cycle length is <= L.
                        # So if L < g, then `dist[u] + dist[v] + 1 <= L < g`.
                        # So the check `if dist[u] + dist[v] + 1 < g` is sufficient to detect any cycle of length < g.
                        
                        if dist[v] != -1 and v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
    return True

def _random_4regular(n, seed):
    rng = random.Random(seed)
    for _ in range(20):
        half = list(range(n)) * 4
        rng.shuffle(half)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, 4*n, 2):
            a, b = half[i], half[i+1]
            if a == b:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if not ok: continue
        if any(len(s) != 4 for s in adj): continue
        return [sorted(s) for s in adj]
    return None

def _local_search(adj, g, time_limit):
    n = len(adj)
    start_time = time.time()
    while time.time() - start_time < time_limit:
        if _girth_check(adj, g):
            return adj
        # Find a short cycle to break
        # We can just pick a random edge and swap
        # Or find the actual short cycle.
        # For simplicity, random edge swap.
        u = rng.randrange(n)
        v = rng.choice(adj[u])
        x = rng.randrange(n)
        if x == u or x == v: continue
        # Ensure x is not connected to u
        if v in adj[x]: continue
        # Find y neighbor of x
        y = rng.choice(adj[x])
        if y == u or y == v: continue
        if u in adj[y]: continue
        # Swap (u,v) and (x,y) -> (u,x) and (v,y)
        # Check if this creates a multi-edge or loop
        if x in adj[u] or y in adj[v]: continue
        if x in adj[v] or y in adj[u]: continue # These are the new edges, they shouldn't exist yet
        
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
        # Sort to keep consistent
        adj[u].sort()
        adj[v].sort()
        adj[x].sort()
        adj[y].sort()

    return None

def construct(k, g):
    global _START
    _START = time.time()
    
    if k != 4:
        return _luw(k, g)
        
    # Target n range
    # Moore bound for k=4, g=10 is 242.
    # Known best is 320.
    # We try to find a graph with n < 320.
    
    best_adj = None
    best_n = None
    
    # Try n from 242 to 320
    for n in range(242, 321, 2):
        if _remaining() < 5:
            break
        # Allocate time for this n
        # More time for larger n? Or equal?
        # Let's give 1 second per n initially, then increase?
        # Or just run until deadline.
        # We will try a few restarts for each n.
        
        # Try to find a graph of size n
        found = False
        start_n_time = time.time()
        while time.time() - start_n_time < 1.0 and _remaining() > 2:
            adj = _random_4regular(n, random.randrange(1<<30))
            if adj is None: continue
            # Quick check
            if _girth_check(adj, g):
                best_adj = adj
                best_n = n
                found = True
                break
            # Local search
            adj_ls = _local_search(adj, g, 0.5)
            if adj_ls is not None:
                best_adj = adj_ls
                best_n = n
                found = True
                break
        
        if found:
            # We found a graph of size n.
            # Since we iterate n upwards, this is the smallest n found so far.
            # We can return immediately?
            # Yes, because we want to minimize n.
            # But maybe a smaller n exists?
            # We started at 242. If we found one at 242, it's optimal (or close).
            # If we found one at 250, maybe 242 has one?
            # We should check 242 first.
            # The loop goes 242, 244...
            # So if we find one at 244, we haven't fully exhausted 242?
            # We gave 1 second to 242. If it failed, it's unlikely.
            # So returning the first found is a good heuristic.
            
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # If no graph found in range, try LUW fallback
    return _luw(k, g)

def _luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction (valid fallback)."""
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# Global rng for local search
rng = random.Random(42)

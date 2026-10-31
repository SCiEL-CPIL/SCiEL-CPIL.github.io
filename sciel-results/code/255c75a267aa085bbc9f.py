import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0
    # Fallback for non-(3,16) cases: simple LUW
    def luw_construct(k, g):
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

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Moore bound for k=3, g=16 (even g): 2*(3*(3^(g/2-1)-1)/(3-1)) = 3*(3^7 - 1) = 8748
    # Actually: for even g, n >= 2 * (k*(k-1)^(g/2 - 1) - 1) / (k-2) = 2*(3*3^7 - 1)/1 = 2*(7290-1)=14578
    # Let me recompute: Moore bound for k-regular, girth g:
    # g even: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 * ( (k-1)^(g/2) - 1 ) / (k-2)
    # For k=3, g=16: g/2=8, n >= 2 * (2^8 - 1)/(3-2) = 2*255 = 510
    # Wait, that's not right either. Let me use the standard formula.
    # For k-regular graph with girth g:
    # If g is even: n >= 2 * (k-1) * ((k-1)^(g/2 - 1) - 1) / (k-2) ... 
    # Actually the standard Moore bound is:
    # g even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # g odd: n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For k=3, g=16 (even): n >= 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2*255 = 510
    # The task says best lower bound is 512, which is close.
    # So m (base vertices) should be at least 510/2 = 255.
    # We'll start m at 256 and go up.
    
    def make_random_3_regular(m):
        """Generate a random 3-regular graph on m vertices (m must be even)."""
        if m % 2 != 0 or m < 4:
            return None
        # Use configuration model with rejection
        for attempt in range(100):
            deg = [0] * m
            half = []
            for i in range(m):
                for _ in range(3):
                    half.append(i)
            random.shuffle(half)
            adj = [[] for _ in range(m)]
            valid = True
            for i in range(0, len(half), 2):
                u, v = half[i], half[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None

    def find_3_edge_cut(adj, m):
        """Try to find a 3-edge cut whose removal yields two identical 2-regular components."""
        # A 3-edge cut in a 3-regular graph: removing 3 edges that form a cut of size 3.
        # In a 3-regular graph, a 3-edge cut corresponds to a set of 3 edges whose removal
        # disconnects the graph into two components, each with exactly one vertex of degree 2 
        # (the other vertices have degree 3). Wait, no: removing 3 edges from a 3-regular graph
        # on m vertices gives a graph where 6 vertices have degree 2 (the endpoints of the removed edges)
        # and the rest have degree 3. For it to be a cut into two 2-regular components, each component
        # must have all vertices of degree 2, meaning each component is a cycle (or union of cycles).
        # But a 2-regular component that is a single cycle means all vertices in that component have degree 2.
        # So the 3 removed edges must connect the two components, and each component must be a single cycle
        # (or multiple cycles, but for 2-regular, it's a union of cycles).
        # Actually, for the 2-lift to work, we need the base graph to have a "fiber" structure.
        # Let me think differently.
        
        # Alternative approach: directly construct a 2-lift.
        # A 2-lift of a graph G is formed by replacing each vertex with two vertices (v0, v1) and
        # each edge with two edges, one of which may be "twisted".
        # For a 3-regular base graph G with m vertices, the 2-lift has 2m vertices.
        # The twist is specified by a function tau: E -> {0,1} where tau(e)=1 means the edge is twisted.
        # For the lift to be 3-regular, we need consistency at each vertex.
        # At each vertex v with neighbors u1, u2, u3, the twists on edges (v,u1), (v,u2), (v,u3)
        # must satisfy: the number of twisted edges incident to v is either 0 or 2 (not 1 or 3)
        # for the lift to be well-defined... actually no, the twist just determines which copy connects to which.
        
        # Let me use a simpler direct construction:
        # Randomly assign each edge a twist bit. Then build the 2-lift.
        # Check if it's 3-regular (it will be by construction) and has girth >= g.
        
        # Actually, the simplest approach: just generate random 3-regular graphs on 2m vertices
        # and check girth. But that's expensive.
        
        # Let me try: generate random 3-regular graphs on N vertices for N in range, check girth.
        return None

    def check_girth_nb(adj, n, g):
        """Check girth using non-backtracking BFS. Returns True if girth >= g."""
        # For each vertex, do non-backtracking BFS up to depth g//2 - 1
        # If we revisit a vertex at depth < g//2, we have a cycle shorter than g.
        # Actually, non-backtracking walks: a non-backtracking walk of length L from v to v
        # gives a cycle of length <= L. We want no cycle shorter than g.
        # So we check for non-backtracking cycles of length < g.
        # Equivalent: for each directed edge (u,v), do non-backtracking BFS from (u,v) 
        # and check if we can return to (u,v) in < g steps.
        
        # Simpler: for each vertex, BFS in the non-backtracking sense up to depth g-1,
        # tracking (prev, current) pairs. If we find a cycle of length < g, return False.
        
        # Even simpler: use standard BFS but ensure we don't go back to parent.
        # For each start vertex s, do BFS where we track (node, prev). 
        # A cycle of length L is found if we reach (s, x) where x is a neighbor of s,
        # and the path length from s to (s,x) via non-backtracking is L-1... 
        # This is getting complicated. Let me use a standard approach:
        
        # For each vertex s, do a BFS up to depth g//2. Track parent.
        # If we find an edge (u,v) where v is not parent[u] and dist[u] + dist[v] + 1 < g,
        # and the path from s to u and s to v don't share edges... 
        # Actually, the standard BFS girth check: for each s, BFS up to depth g//2.
        # For each edge (u,v) with u != parent[v] and v != parent[u], if dist[u] + dist[v] + 1 < g,
        # then there's a cycle of length dist[u] + dist[v] + 1 < g. But we need to ensure
        # the paths from s to u and s to v are edge-disjoint, which they are in a tree.
        # However, this can give false positives if the paths share vertices other than s.
        # For a tree, the paths from s to u and s to v share only the path to LCA(u,v).
        # The cycle length is dist[u] + dist[v] - 2*dist[lca] + 1. This is <= dist[u] + dist[v] + 1.
        # So if dist[u] + dist[v] + 1 < g, the cycle is definitely < g.
        # But if dist[u] + dist[v] + 1 >= g, the cycle might still be < g.
        # So this check is sufficient but not necessary. It might miss some cycles.
        # For our purpose, if we find a cycle < g, we return False. If we don't find one,
        # we might still have a cycle < g that we missed. But for random graphs, this is a good heuristic.
        
        # Let me use a more careful approach: for each directed edge (u,v), do non-backtracking BFS.
        # This is O(n * g) which is fine for n up to a few thousand.
        
        # Actually, let me just use the standard BFS approach with the sufficient condition.
        # If it passes, we do a more thorough check.
        
        max_depth = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a potential cycle
                        # Cycle length is dist[u] + dist[v] + 1
                        # But this is an upper bound on the actual cycle length
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def thorough_girth_check(adj, n, g):
        """Thorough check: for each vertex, find the shortest cycle through it using BFS in the line graph or similar."""
        # For each vertex s, remove s and find the shortest path between any two neighbors of s.
        # The girth is the minimum over all s of (shortest path between two neighbors of s + 2).
        # Wait, that's not right either. The shortest cycle through s is:
        # min over pairs of neighbors (u,v) of s of (shortest path from u to v not using s + 2).
        # But the shortest path from u to v might use s, so we need to exclude s.
        
        # Simpler: for each s, do BFS from s up to depth g-1, tracking (node, prev).
        # If we reach (s, x) where x is a neighbor of s, and the path length is L, then we have a cycle of length L+1.
        # We want to ensure no such cycle has length < g.
        
        for s in range(n):
            # BFS in state space (current, prev)
            visited = set()
            q = deque()
            for nb in adj[s]:
                q.append((nb, s))
                visited.add((nb, s))
            depth = 1
            while q and depth < g - 1:
                next_q = deque()
                for (u, prev) in q:
                    for v in adj[u]:
                        if v == prev:
                            continue
                        if v == s:
                            # Found a cycle of length depth + 1
                            if depth + 1 < g:
                                return False
                            continue
                        state = (v, u)
                        if state not in visited:
                            visited.add(state)
                            next_q.append(state)
                q = next_q
                depth += 1
        return True

    best_n = float('inf')
    best_edges = []
    
    # Try increasing N (number of vertices in the final graph)
    # Start from a reasonable lower bound. The Moore bound is ~510, so try from 512 upward.
    # For each N, try to generate a random 3-regular graph and check girth.
    # N must be even for 3-regular graph.
    
    start_N = 512
    max_N = 2000  # Upper limit to keep runtime reasonable
    
    for N in range(start_N, max_N + 1, 2):
        if time.time() > DEADLINE - MARGIN:
            break
        if N >= best_n:
            break
        # Try multiple random graphs for this N
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - MARGIN:
                break
            adj = make_random_3_regular(N)
            if adj is None:
                continue
            # Quick girth check
            if check_girth_nb(adj, N, g):
                # Thorough check
                if thorough_girth_check(adj, N, g):
                    # Valid!
                    edges = []
                    for u in range(N):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = N
                    best_edges = edges
                    break
        if best_n == N:
            break
    
    if best_n == float('inf'):
        # Fallback to LUW
        n_luw, edges_luw = luw_construct(k, g)
        best_n = n_luw
        best_edges = edges_luw
    
    return best_n, best_edges

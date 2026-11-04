"""Construct a 4-regular graph of girth 10 via random generation + local search."""
import time
import random
from collections import deque, defaultdict

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Target: try to find the smallest possible graph
    # Moore bound for 4-regular, girth 10: 1 + 4*(1+4+4^2+4^3+4^4) = 1 + 4*341 = 1365? 
    # Wait, Moore bound for even girth 2m: 1 + k * sum_{i=0}^{m-1} (k-1)^i
    # For k=4, g=10 (m=5): 1 + 4*(1+4+16+64+256) = 1 + 4*341 = 1365
    # But the task says the smallest known is 320 vertices, lower bound 243.
    # The Moore bound formula for cages: n(k,g) >= 1 + k * sum_{i=0}^{(g-2)/2 - 1} (k-1)^i for even g
    # Actually for girth g=2m: n >= 1 + k * sum_{i=0}^{m-1} (k-1)^i
    # For k=4, g=10, m=5: 1 + 4*(1+4+16+64+256) = 1365. But known is 320.
    # The Moore bound is a lower bound, so 320 > 1365? No, 320 < 1365.
    # Wait, I'm confusing myself. The Moore bound gives a LOWER bound on n.
    # So n >= 1365? But the task says the smallest known is 320. That can't be right.
    # Let me re-read: "The smallest known such graph has 320 vertices... and the best lower bound is 243."
    # So the Moore bound must be 243 or less. Let me recalculate.
    # For a k-regular graph with girth g (even), the Moore bound is:
    # n >= 1 + k * sum_{i=0}^{(g/2)-1} (k-1)^i
    # For k=4, g=10: (g/2)-1 = 4, so sum_{i=0}^{4} 3^i = 1+3+9+27+81 = 121
    # n >= 1 + 4*121 = 1 + 484 = 485? Still not 243.
    # Hmm, maybe the formula is different. For bipartite graphs with girth 2m:
    # n >= 2 * sum_{i=0}^{m-1} (k-1)^i
    # For k=4, g=10, m=5: 2*(1+3+9+27+81) = 2*121 = 242. Close to 243.
    # So the Moore bound for bipartite is 242, and the best lower bound is 243.
    # The task says n must lie between the Moore bound and max_vertices.
    # We'll aim for n around 243-320.
    
    # Strategy: Start with n=243 (the lower bound) and try to find a valid graph.
    # If that fails, increment n until we find one or run out of time.
    
    def check_girth(adj, n, g):
        """Check if the graph has girth at least g using non-backtracking walks."""
        # For each vertex, do BFS in the line graph (non-backtracking)
        # or equivalently, for each directed edge, do BFS without backtracking
        for u in range(n):
            for v in adj[u]:
                # BFS from directed edge (u,v) without backtracking
                visited = set()
                queue = deque([(v, u, 1)])  # (current, prev, distance)
                while queue:
                    cur, prev, dist = queue.popleft()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if dist >= g:
                        return False  # Found a cycle of length < g? No, dist >= g means we've walked g steps
                    # Actually, if we can reach a vertex at distance >= g-1, that might form a cycle
                    # Let me reconsider: a non-backtracking walk of length L from (u,v) that returns to u
                    # forms a cycle of length L. We want no cycles of length < g.
                    # So we need to check if any non-backtracking walk of length < g forms a cycle.
                    # Equivalently, in the BFS from (u,v), if we can reach u at distance d < g, 
                    # then there's a cycle of length d.
                    # But we also need to handle the case where the walk doesn't return to u but to some other vertex
                    # that was visited earlier in a way that forms a cycle.
                    
                    # Simpler approach: for each starting directed edge (u,v), do BFS in the non-backtracking 
                    # sense. If we ever reach a state (w, x) where x == u and the distance is < g, we have a cycle.
                    # Actually, a cycle of length L corresponds to a non-backtracking closed walk of length L.
                    # So we just need to check if there's a non-backtracking closed walk of length < g.
                    
                    # Let me redo this more carefully.
                    pass
                # The above approach is getting complicated. Let me use a simpler method.
                pass
        return True
    
    def has_short_cycle(adj, n, g):
        """Check if the graph has any cycle of length < g."""
        # For each vertex, do BFS and track the shortest cycle through that vertex
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] != dist[u] + 1:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
        return False
    
    def generate_random_regular(n, k, seed):
        """Generate a random k-regular graph on n vertices."""
        rng = random.Random(seed)
        # Use the configuration model
        if n * k % 2 != 0:
            return None
        half_edges = n * k // 2
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        # Pair up stubs
        adj = defaultdict(set)
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None  # Self-loop
            if v in adj[u]:
                return None  # Multi-edge
            adj[u].add(v)
            adj[v].add(u)
        # Convert to list of lists
        adj_list = [list(adj[i]) for i in range(n)]
        return adj_list
    
    def two_switch(adj, n, k):
        """Perform a random 2-switch to break short cycles."""
        # Find two non-edges (u,v) and (x,y) such that swapping gives (u,x) and (v,y)
        # or (u,y) and (v,x)
        # We want to break short cycles, so we should target edges in short cycles.
        # For simplicity, just do random 2-switches.
        for _ in range(100):
            u = random.randint(0, n - 1)
            x = random.randint(0, n - 1)
            if u == x:
                continue
            if x in adj[u]:
                continue
            # Pick a neighbor v of u and y of x
            if not adj[u] or not adj[x]:
                continue
            v = random.choice(adj[u])
            y = random.choice(adj[x])
            if v == y or v == x or y == u:
                continue
            if y in adj[v] or x in adj[y]:
                continue
            # Perform the switch: remove (u,v) and (x,y), add (u,x) and (v,y)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            return True
        return False
    
    # Try different values of n
    best_n = None
    best_edges = None
    
    for n in range(243, 350):
        if time.time() > deadline - 10:
            break
        if n * k % 2 != 0:
            continue
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            adj = generate_random_regular(n, k, seed=attempt * 1000 + n)
            if adj is None:
                continue
            
            # Check girth
            if not has_short_cycle(adj, n, g):
                # Found a valid graph
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n = n
                best_edges = edges
                found = True
                break
            
            # Local search: try to break short cycles with 2-switches
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                if not has_short_cycle(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                two_switch(adj, n, k)
        
        if found:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return the Lazebnik-Ustimenko-Woldar construction
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

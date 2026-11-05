import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # The LUW construction gives a valid graph.
    # We use it as a fallback.
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try to find a smaller graph using local search.
    # We know the (4,10)-cage has 320 vertices.
    # We try n from 320 up to n_luw.
    # Since k=4, g=10 is the specific case, we can hardcode the target n=320.
    # But to be general, we can try a range.
    
    # For k=4, g=10, the Moore bound is 106.
    # We try n in [320, 512, 1024, ...]
    # Actually, let's just try n=320 first.
    
    if k == 4 and g == 10:
        target_n = 320
        # Try to construct a 4-regular bipartite graph on 320 vertices with girth 10.
        n_found, edges_found = _search_cage(4, 10, target_n, deadline - (time.time() - start_time))
        if n_found is not None and n_found < best_n:
            return n_found, edges_found
            
    # If search fails or is not applicable, return LUW.
    return best_n, best_edges

def _search_cage(k, g, n, time_limit):
    """
    Attempts to find a k-regular bipartite graph with n vertices and girth >= g.
    Uses local search.
    """
    if n % 2 != 0:
        return None, None
    
    half_n = n // 2
    # Vertices 0..half_n-1 are left, half_n..n-1 are right.
    
    # Initialize random k-regular bipartite graph.
    # Left nodes: 0..half_n-1
    # Right nodes: half_n..n-1
    # Each left node connects to k right nodes.
    # Each right node connects to k left nodes.
    
    # Use a random permutation to assign edges.
    # This is tricky to ensure regularity.
    # We can use the configuration model:
    # Create a list of stubs for left and right.
    # Shuffle and pair.
    
    def generate_random_graph():
        left_stubs = [i for i in range(half_n) for _ in range(k)]
        right_stubs = [i + half_n for i in range(half_n) for _ in range(k)]
        random.shuffle(right_stubs)
        edges = set()
        for l, r in zip(left_stubs, right_stubs):
            if l != r: # Bipartite, so l != r always
                if (l, r) not in edges:
                    edges.add((l, r))
                else:
                    # Duplicate edge, need to fix.
                    # For now, just retry or accept and fix later.
                    pass
        # Check regularity.
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return list(edges)
        return None

    edges = generate_random_graph()
    if edges is None:
        # Retry a few times
        for _ in range(100):
            edges = generate_random_graph()
            if edges:
                break
        if not edges:
            return None, None

    # Check girth.
    # We want girth >= g.
    # So we check if there is a cycle of length < g.
    
    def has_short_cycle(edges, max_len):
        # Build adjacency list.
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each node up to depth max_len//2.
        # If we find a cycle, return True.
        # Actually, for bipartite graphs, we can just check distances.
        # A cycle of length L exists if there is a path of length L/2 from u to v
        # where u and v are neighbors? No.
        
        # Simpler: BFS from each node, keep track of (node, parent).
        # If we visit a node that is not the parent, we found a cycle.
        # The length is 2 * depth.
        
        for start in range(n):
            # BFS
            visited = {start: (0, -1)} # node: (depth, parent)
            queue = [start]
            for node in queue:
                depth, parent = visited[node]
                if depth * 2 >= max_len:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Found a cycle.
                        # The cycle length is depth + visited[neighbor][0] + 1?
                        # Actually, if we visit a node already visited,
                        # the cycle length is depth + 1 + visited[neighbor][0]?
                        # No, in BFS, if we encounter a visited node,
                        # the cycle length is 2 * depth + 1? No, bipartite.
                        # If we are at depth d, and we see a node at depth d' (d' <= d),
                        # the cycle length is d + d' + 1?
                        # Actually, just check if depth + 1 + visited[neighbor][0] < g?
                        # Wait, if neighbor is visited, the path from start to neighbor
                        # has length visited[neighbor][0].
                        # The new path has length depth + 1.
                        # The cycle length is depth + 1 + visited[neighbor][0].
                        # But this assumes the path from start to neighbor is disjoint?
                        # In BFS tree, yes.
                        
                        cycle_len = depth + 1 + visited[neighbor][0]
                        if cycle_len < g:
                            return True
                    else:
                        visited[neighbor] = (depth + 1, node)
                        queue.append(neighbor)
        return False

    # Local search loop.
    while time.time() < time_limit:
        if not has_short_cycle(edges, g):
            return n, edges
        
        # Find a short cycle and break it.
        # This is complex.
        # Simpler: Randomly rewire an edge.
        # Pick a random edge (u, v).
        # Pick a random other edge (u2, v2).
        # Try to swap to (u, v2) and (u2, v).
        # Check if it creates a multi-edge or breaks regularity (it doesn't).
        # Check if it reduces the number of short cycles.
        
        # For now, just random rewiring.
        u, v = random.choice(edges)
        # Find another edge incident to u or v? No, any edge.
        # To maintain regularity, we must swap edges.
        # Swap (u, v) and (u2, v2) to (u, v2) and (u2, v).
        # This is valid if u != u2, v != v2, u != v2, u2 != v.
        
        # Pick u2, v2 randomly.
        # We need to pick u2 from neighbors of u? No, u2 is any left node.
        # v2 is any right node.
        # But (u2, v2) must be an edge.
        
        # Let's pick a random edge (u2, v2) from the edge list.
        u2, v2 = random.choice(edges)
        if u2 == u or v2 == v:
            continue
            
        # Check if (u, v2) and (u2, v) are already edges.
        if (u, v2) in edges or (u2, v) in edges:
            continue
            
        # Perform swap.
        edges.remove((u, v))
        edges.remove((u2, v2))
        edges.add((u, v2))
        edges.add((u2, v))

    return None, None

def _luw_construct(k, g):
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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

def _generate_random_k_regular(n, k):
    """Generate a random k-regular simple graph using configuration model."""
    if n * k % 2 != 0:
        return None
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    while True:
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        if valid:
            # Convert to list and sort for consistency
            edge_list = [tuple(sorted(e)) for e in edges]
            return edge_list

def _check_girth(edges, n, g):
    """Check if graph has girth at least g. Returns True if girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for cycles of length < g
    # We can use BFS from each vertex to find shortest cycle.
    # However, for k=4, g=10, we just need to ensure no 4, 6, 8 cycles.
    
    # Optimized check for 4-cycles
    # Count common neighbors
    for u in range(n):
        neighbors_u = set(adj[u])
        for v in neighbors_u:
            common = neighbors_u.intersection(adj[v])
            if len(common) > 0:
                # If u and v have a common neighbor w, and u!=v, it's a 3-cycle? No, simple graph.
                # If u and v are adjacent, and share a neighbor, it's a triangle (3-cycle).
                # But k=4 regular, g=10 implies no triangles.
                # Wait, if u and v are adjacent, they cannot share a neighbor in a simple graph with no triangles.
                # If u and v are NOT adjacent, and share 2 neighbors, it's a 4-cycle.
                # If they share 1 neighbor, it's a path of length 2.
                pass
                
    # Let's do a proper BFS/DFS for short cycles.
    # Since n is small (~320) and k is small (4), we can do DFS for each start node.
    # Limit depth to g-1.
    
    def has_short_cycle(start, max_depth):
        # visited[node] = depth
        # We want to find a cycle of length < max_depth+1 (i.e., <= max_depth)
        # Actually, we want girth >= g, so no cycle of length < g.
        # So we look for cycles of length 3, 4, ..., g-1.
        
        # Use BFS to find shortest path back to start?
        # No, BFS finds shortest path between nodes.
        # A cycle of length L exists if there is a path of length L-1 from a neighbor of start to start?
        
        # Simpler: DFS with backtracking, limiting depth.
        # This might be slow if we do it for every node and many random graphs.
        # But k=4, depth=9. 4^9 = 262k. 320 * 262k = 84M ops.
        # In Python, 84M ops might take ~1-2 seconds.
        # We might only need to check a few random graphs.
        
        def dfs(node, parent, depth, visited):
            if depth >= max_depth:
                return False
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return True
                elif neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, node, depth + 1, visited):
                        return True
                    visited.remove(neighbor)
            return False

        visited = {start}
        return dfs(start, -1, 0, visited)

    for i in range(n):
        if has_short_cycle(i, g - 1):
            return False
    return True

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: Try to find a small graph via random search.
    Fallback to L-U-W if not found.
    """
    start_time = time.time()
    timeout = 100  # seconds
    
    # Moore bound
    if g % 2 == 0:
        moore_bound = 2 * sum(k**i for i in range(g//2 - 1)) + 2 * k ** (g//2 - 1) 
        # Wait, formula: 2 * (1 + k + ... + k^(g/2 - 2)) + 2 * k^(g/2 - 1) ?
        # Standard Moore bound for girth g (even):
        # N >= 2 * (1 + k + ... + k^(g/2 - 2)) + 2 * k^(g/2 - 1) is incorrect.
        # N >= 2 * sum_{i=0}^{g/2 - 2} k^i + 2 * k^(g/2 - 1) ?
        # Let's use the standard:
        # N >= 2 * (k^(g/2 - 1) - 1) / (k - 1) ? No.
        # N >= 2 * (1 + k + ... + k^(g/2 - 2)) + 2 * k^(g/2 - 1) ?
        # Actually, for g=10, g/2=5.
        # N >= 2 * (1 + 4 + 16 + 64) = 170.
        # Sum is 1 + 4 + 16 + 64 = 85.
        # So N >= 170.
        
        # General formula:
        # N >= 2 * sum_{i=0}^{g/2 - 2} k^i + 2 * k^(g/2 - 1) ?
        # For g=4: 2 * (1) + 2 * k^1 = 2 + 2k.
        # C4 has 4 vertices. k=2. 2+4=6. Incorrect.
        # Correct Moore bound for even g:
        # N >= 2 * (k^(g/2 - 1) - 1) / (k - 1) ?
        # For g=4: 2 * (k^1 - 1)/(k-1) = 2. Incorrect.
        
        # Let's just use the known bound for k=4, g=10 which is 170.
        # Or calculate:
        # N >= 2 * (1 + k + ... + k^(g/2 - 2)) + 2 * k^(g/2 - 1) is wrong.
        # The bound is 2 * sum_{i=0}^{g/2 - 2} k^i + 2 * k^(g/2 - 1) ?
        # Actually, the number of vertices at distance < g/2 from a vertex is sum_{i=0}^{g/2 - 2} k^i.
        # Plus the neighbors at distance g/2?
        # For g=10, distance < 5.
        # Dist 0: 1
        # Dist 1: k
        # Dist 2: k(k-1)
        # Dist 3: k(k-1)^2
        # Dist 4: k(k-1)^3
        # Total per vertex: 1 + k + k(k-1) + k(k-1)^2 + k(k-1)^3.
        # For k=4: 1 + 4 + 12 + 48 + 192 = 257.
        # Since graph is bipartite, we count from one side?
        # No, the Moore bound for bipartite graphs of girth g is:
        # N >= 2 * sum_{i=0}^{g/2 - 1} k^i ? No.
        # N >= 2 * (1 + k + k(k-1) + ... + k(k-1)^(g/2 - 2)) ?
        # For g=10, g/2=5.
        # Sum i=0 to 4: 1 + 4 + 12 + 48 + 192 = 257.
        # N >= 2 * 257? No, that's for non-bipartite?
        
        # Let's stick to the known value 170 for k=4, g=10.
        # 170 = 2 * (1 + 4 + 16 + 64) = 2 * (4^4 - 1) / 3 ?
        # (4^4 - 1) / 3 = 255 / 3 = 85.
        # 2 * 85 = 170.
        # So the bound is 2 * (k^(g/2 - 1) - 1) / (k - 1).
        # Let's verify for g=6, k=3.
        # 2 * (3^2 - 1) / 2 = 2 * 4 = 8.
        # Heawood graph is 14 vertices. 8 is the lower bound.
        # Correct.
        
        moore_bound = 2 * (k ** (g // 2 - 1) - 1) // (k - 1)
    else:
        # Odd girth bound is different, but we only care about k=4, g=10.
        moore_bound = 1 # Placeholder
        
    # Try random search from moore_bound upwards
    # We know 320 works. Let's try up to 350.
    max_n = 350
    step = 2 # Must be even for 4-regular? No, n*k must be even. 4n is always even.
    
    # Set random seed for reproducibility?
    # The task doesn't require determinism, but it's good for debugging.
    # However, we want to find *a* graph.
    
    for n in range(moore_bound, max_n + 1, 2):
        if time.time() - start_time > timeout:
            break
        # Try a few random graphs for this n
        for attempt in range(50):
            edges = _generate_random_k_regular(n, k)
            if edges is None:
                continue
            if _check_girth(edges, n, g):
                return n, edges
            if time.time() - start_time > timeout:
                break
        
    # Fallback to L-U-W
    return _luw_construct(k, g)

# EVOLVE-BLOCK-END

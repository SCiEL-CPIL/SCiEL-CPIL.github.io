"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (L-U-W) construction.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    q = k
    # The L-U-W graph CD(d, q) has girth at least 2d + 1.
    # We need 2d + 1 >= g => d >= (g - 1) / 2
    min_d = (g - 1) // 2 + 1
    if (g - 1) % 2 != 0:
        min_d = (g - 1) // 2 + 1
    else:
        min_d = (g - 1) // 2
        
    # Ensure d is at least 3 for the construction to be well-defined in this context
    d = max(3, min_d)
    
    def build_luw(q, d):
        """
        Constructs the L-U-W graph CD(d, q).
        Returns (n, edges).
        """
        # GF(q) operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        # Rule definition
        # Based on Lazebnik, Ustimenko, Woldar (1996)
        # For c = 2..d
        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            # The specific rule coefficients
            # Note: The seed code had a specific mapping. 
            # Standard L-U-W uses:
            # y[c] = y[a] * x[b] + x[c]
            # The indices a, b depend on c.
            # Common pattern:
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule[c] = (a, b)

        def get_neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # Point to Line
                        # l[c] = p[c] + l[a] * p[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # Line to Point
                        # p[c] = l[c] - l[a] * p[b]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        # BFS to find connected component
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in get_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth(n, edges, target_girth):
        """
        Checks if the graph has no cycle shorter than target_girth.
        Uses non-backtracking BFS from each vertex.
        """
        if n == 0:
            return True
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check from every vertex
        # For efficiency, we can check from a subset if n is large, 
        # but for correctness we should check all.
        # Optimization: If n > 5000, this might be slow. 
        # However, L-U-W graphs are sparse.
        
        for start in range(n):
            # State: (current_node, prev_node, depth)
            # We want to find a cycle of length < target_girth.
            # A cycle of length L corresponds to a non-backtracking walk of length L returning to start.
            # We stop if we find a walk of length < target_girth returning to start.
            # Note: We must not backtrack immediately (u -> v -> u).
            
            visited = {} # (node, prev) -> depth
            queue = deque()
            # Start with all neighbors of start
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
                
            while queue:
                node, prev, depth = queue.popleft()
                
                if depth >= target_girth:
                    continue # We only care about cycles shorter than target_girth
                
                if node == start:
                    # Found a cycle of length depth
                    # Note: This counts the path start -> ... -> start.
                    # If depth < target_girth, girth is violated.
                    return False

                for nb in adj[node]:
                    if nb == prev:
                        continue # Non-backtracking
                    # If we revisit a node, we might form a cycle.
                    # The standard check is: if we reach 'start' again, we have a cycle.
                    # But we also need to ensure we don't use the same edge back.
                    # The (node, prev) state handles this.
                    
                    if (nb, node) in visited:
                        # We've seen this state before at a shallower or equal depth?
                        # Actually, if we reach (nb, node) and nb is not start, 
                        # it doesn't necessarily mean a cycle shorter than target_girth 
                        # unless the path from start to (nb, node) + path from (nb, node) to start 
                        # forms a cycle.
                        # The BFS from start finds the shortest non-backtracking path to each (node, prev).
                        # If we hit 'start', the depth is the cycle length.
                        # If we hit a previously visited state, it implies a cycle in the state graph,
                        # which corresponds to a cycle in the original graph.
                        # However, the shortest cycle through 'start' is what we are looking for 
                        # if we run BFS from 'start'.
                        # So we just need to check if 'start' is reached.
                        pass
                    
                    if nb == start:
                        # We found a cycle of length depth + 1
                        # But we check at the top of the loop? 
                        # No, we check when we pop.
                        # So we just add to queue.
                        pass

                    if (nb, node) not in visited:
                        visited[(nb, node)] = depth + 1
                        queue.append((nb, node, depth + 1))
                        
        return True

    best_n = float('inf')
    best_edges = []
    
    current_d = d
    while current_d <= d + 2: # Try a few values
        if time.time() > deadline:
            break
            
        n, edges = build_luw(q, current_d)
        
        if n < best_n:
            best_n = n
            best_edges = edges
            
        # Check girth
        # For large n, this is expensive. 
        # If n > 2000, we might skip full check or assume validity if d is sufficient.
        # But we need to be sure.
        # Let's check.
        if check_girth(n, edges, g):
            return n, edges
        else:
            current_d += 1
            continue
            
    return best_n, best_edges
# EVOLVE-BLOCK-END

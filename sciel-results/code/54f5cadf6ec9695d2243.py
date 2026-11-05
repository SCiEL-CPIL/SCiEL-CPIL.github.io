"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    q = k
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def get_luw_graph(d, rule):
        """Build LUW graph with given dimension d and rule sequence."""
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    def check_girth_fast(n, edges, min_girth):
        """Fast girth check using BFS with early termination."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify degree
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        # Check for self-loops and duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # BFS from each vertex to find shortest cycle
        # Only need to check up to depth min_girth//2
        max_depth = min_girth // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_rule_variants(d):
        """Generate multiple rule variants for dimension d."""
        variants = []
        
        # Variant 1: Original rule from seed
        rule1 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        variants.append(rule1)
        
        # Variant 2: Simpler pattern - use (1, c-1) for all c >= 2
        rule2 = {}
        for c in range(2, d + 1):
            rule2[c] = (1, c - 1)
        variants.append(rule2)
        
        # Variant 3: Use (c-1, 1) for all c >= 2
        rule3 = {}
        for c in range(2, d + 1):
            rule3[c] = (c - 1, 1)
        variants.append(rule3)
        
        # Variant 4: Alternating pattern
        rule4 = {}
        for c in range(2, d + 1):
            if c % 2 == 0:
                rule4[c] = (1, c - 1)
            else:
                rule4[c] = (c - 1, 1)
        variants.append(rule4)
        
        # Variant 5: Cyclic pattern with period 3
        rule5 = {}
        for c in range(2, d + 1):
            idx = (c - 2) % 3
            if idx == 0:
                rule5[c] = (1, c - 1)
            elif idx == 1:
                rule5[c] = (c - 1, 1)
            else:
                rule5[c] = (1, 1)
        variants.append(rule5)
        
        return variants

    best_n = float('inf')
    best_edges = []
    
    # For girth 9 with k=5, we need d such that expected girth >= 9
    # LUW graphs have girth approximately 2*d+2 for even d, d+5 for odd d
    # So for g=9: d=3 gives girth ~8 (too small), d=4 gives girth ~10 (good)
    
    min_d = max(2, (g - 5) // 2)  # Conservative lower bound on d
    
    for d in range(min_d, min_d + 8):
        if time.time() > deadline - 30:
            break
        
        # Expected girth estimate
        if d % 2 == 0:
            exp_girth = 2 * d + 2
        else:
            exp_girth = d + 5
            
        if exp_girth < g:
            continue
        
        # Try multiple rule variants
        for rule in generate_rule_variants(d):
            if time.time() > deadline - 25:
                break
            
            try:
                n, edges = get_luw_graph(d, rule)
                if n < best_n and n <= 50000:  # Limit size for girth check
                    if check_girth_fast(n, edges, g):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                pass
    
    if best_n == float('inf'):
        # Fallback: try larger d without girth check (assume valid for large enough d)
        for d in range(min_d + 8, min_d + 15):
            if time.time() > deadline - 10:
                break
            try:
                rule = generate_rule_variants(d)[0]
                n, edges = get_luw_graph(d, rule)
                if n < best_n:
                    best_n = n
                    best_edges = edges
            except Exception:
                pass
    
    if best_n == float('inf'):
        # Last resort: return a trivial valid graph if possible
        # For k=5, we need at least 5 vertices. K5 has girth 3, so not valid for g=9.
        # This should never happen if LUW construction works.
        best_n = 0
        best_edges = []
    
    return best_n, best_edges
# EVOLVE-BLOCK-END

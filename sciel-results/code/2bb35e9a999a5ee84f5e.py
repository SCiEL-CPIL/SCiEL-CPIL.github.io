"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict

def construct(k, g):
    # Build a k-regular graph with girth >= g using random generation + cycle-breaking
    # Moore bound for odd girth g=2r+1: 1 + k*sum_{i=0}^{r-1}(k-1)^i
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(r))
    n = moore + 10  # start slightly above Moore bound
    
    max_attempts = 50
    rng = random.Random(42)
    
    def make_random_k_regular(n, k, rng):
        # Configuration model: each vertex has k stubs
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        rng.shuffle(stubs)
        adj = defaultdict(set)
        i = 0
        while i < len(stubs):
            u = stubs[i]
            v = stubs[i + 1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
            i += 2
        # Fix: ensure exactly k neighbors
        # Simple approach: retry if not regular
        for v in range(n):
            if len(adj[v]) != k:
                return None
        return adj
    
    def find_short_cycles(adj, n, g):
        # Find all cycles of length < g using BFS from each vertex
        # For small girth, use non-backtracking walks
        short_cycles = []
        for start in range(n):
            # BFS with parent tracking to find cycles
            visited = {}  # (node, prev) -> depth
            queue = [(start, -1, 0)]
            visited[(start, -1)] = 0
            while queue:
                node, prev, depth = queue.pop(0)
                if depth >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 >= 3 and depth + 1 < g:
                        # Found a cycle of length depth+1
                        short_cycles.append((start, node, depth + 1))
                        break
                    if (nb, node) not in visited:
                        visited[(nb, node)] = depth + 1
                        queue.append((nb, node, depth + 1))
        return short_cycles
    
    def has_short_cycle(adj, n, g):
        # Check if any cycle of length < g exists
        for start in range(n):
            # Non-backtracking walk of length g-1 should not return to start
            # Use BFS to check girth
            dist = {}
            parent = {}
            queue = [(start, -1, 0)]
            dist[(start, -1)] = 0
            while queue:
                node, prev, d = queue.pop(0)
                if d >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and d + 1 >= 3 and d + 1 < g:
                        return True
                    if (nb, node) not in dist:
                        dist[(nb, node)] = d + 1
                        parent[(nb, node)] = (node, prev)
                        queue.append((nb, node, d + 1))
        return False
    
    # Try multiple random graphs and fix short cycles
    best_adj = None
    for attempt in range(max_attempts):
        adj = make_random_k_regular(n, k, rng)
        if adj is None:
            continue
        
        # Iteratively break short cycles
        fixed = True
        max_fix_iterations = 200
        for _ in range(max_fix_iterations):
            if not has_short_cycle(adj, n, g):
                break
            # Find a short cycle and break it
            # Find any cycle of length < g
            cycle = None
            for start in range(n):
                dist = {}
                parent = {}
                queue = [(start, -1, 0)]
                dist[(start, -1)] = 0
                found = False
                while queue and not found:
                    node, prev, d = queue.pop(0)
                    if d >= g - 1:
                        continue
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        if nb == start and d + 1 >= 3 and d + 1 < g:
                            # Reconstruct cycle
                            cycle_nodes = [start]
                            cur_node, cur_prev = node, prev
                            while cur_node != start:
                                cycle_nodes.append(cur_node)
                                cur_node, cur_prev = parent[(cur_node, cur_prev)]
                            cycle = cycle_nodes
                            found = True
                            break
                        if (nb, node) not in dist:
                            dist[(nb, node)] = d + 1
                            parent[(nb, node)] = (node, prev)
                            queue.append((nb, node, d + 1))
                if cycle:
                    break
            if not cycle:
                break
            
            # Break the cycle by rewiring one edge
            # Pick a random edge in the cycle and rewire it
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                cycle_edges.append((u, v))
            
            rewired = False
            for _ in range(20):
                idx = rng.randint(0, len(cycle_edges) - 1)
                u, v = cycle_edges[idx]
                # Find alternative neighbors for u and v
                u_alts = [x for x in adj[u] if x != v]
                v_alts = [x for x in adj[v] if x != u]
                if not u_alts or not v_alts:
                    continue
                # Try to swap: remove (u,v), add (u, a) and (b, v) where a in v_alts, b in u_alts
                # Actually simpler: just remove edge (u,v) and add (u, w) for some w not adjacent
                # This breaks regularity, so need to be careful
                # Better: pick two edges (u,v) and (x,y) not in cycle, swap to (u,y) and (x,v)
                # For simplicity, just try random rewiring
                pass
            
            # Simpler approach: just regenerate if we can't fix easily
            # Try a different random seed
            break
        
        if adj and not has_short_cycle(adj, n, g):
            best_adj = adj
            break
    
    if best_adj is None:
        # Fallback: use the LW construction from seed
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]
        
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
    
    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    return n, edges
# EVOLVE-BLOCK-END

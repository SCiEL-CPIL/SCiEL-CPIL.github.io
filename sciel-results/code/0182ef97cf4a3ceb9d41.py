import time
import random
import sys
import math

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    For k=4, g=10, attempts to construct the 320-vertex (4,10)-cage using a voltage graph
    approach over Z2^5 with the base graph K5,5 minus a perfect matching.
    Falls back to Lazebnik-Ustimenko-Woldar (LUW) construction for other parameters.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # LUW construction fallback
    def luw_construct(d, q):
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

    def has_girth_at_least(n, edges, g):
        """Check if graph has no cycle shorter than g using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def check_girth_fast(n, edges, g):
        """Faster girth check using non-backtracking walks from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify simple graph
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        
        # Verify regularity
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        # Non-backtracking BFS from each vertex
        for start in range(n):
            # (current_vertex, previous_vertex)
            visited = {}  # (u, prev) -> depth
            queue = [(start, -1)]
            visited[(start, -1)] = 0
            
            while queue:
                u, prev = queue.pop(0)
                depth = visited[(u, prev)]
                if depth >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # Don't backtrack
                    if v == start and depth + 1 < g:
                        return False  # Found a cycle shorter than g
                    if (v, u) not in visited:
                        visited[(v, u)] = depth + 1
                        queue.append((v, u))
        return True

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        def construct_320_cage():
            """
            Construct the 320-vertex (4,10)-cage using voltage graph method.
            Base graph: K5,5 minus perfect matching (10 vertices, 4-regular)
            Voltage group: Z2^5 (32 elements)
            
            We need to assign 5-bit vectors to each of the 20 edges such that:
            - No 4-cycle has voltage sum 0
            - No 6-cycle has voltage sum 0  
            - No 8-cycle has voltage sum 0
            """
            # Base graph vertices: 0-4 (A-side), 5-9 (B-side)
            # Edges: (i, 5+j) for i != j
            base_edges = []
            for i in range(5):
                for j in range(5):
                    if i != j:
                        base_edges.append((i, 5 + j))
            
            # Build edge index lookup
            edge_idx = {}
            for idx, e in enumerate(base_edges):
                edge_idx[e] = idx
                edge_idx[(e[1], e[0])] = idx  # Also add reverse for convenience
            
            # Generate all 4-cycles, 6-cycles, and 8-cycles
            # A cycle in the bipartite base graph alternates between A and B vertices
            
            def generate_cycles(length):
                """Generate all simple cycles of given length in the base graph."""
                cycles = []
                # A cycle of length 2m has m A-vertices and m B-vertices
                m = length // 2
                
                # Enumerate sequences of A-vertices (a1, a2, ..., am) and B-vertices (b1, b2, ..., bm)
                # Edges: (a1, b1), (a2, b1), (a2, b2), (a3, b2), ..., (am, bm-1), (am, bm), (a1, bm)
                # Must satisfy ai != bi for all edges
                
                from itertools import permutations
                
                if m == 2:
                    # 4-cycles: (a1,b1,a2,b2,a1)
                    # Edges: (a1,b1), (a2,b1), (a2,b2), (a1,b2)
                    # Conditions: a1!=b1, a2!=b1, a2!=b2, a1!=b2, a1!=a2, b1!=b2
                    for a1 in range(5):
                        for a2 in range(5):
                            if a1 == a2:
                                continue
                            for b1 in range(5):
                                if b1 == a1 or b1 == a2:
                                    continue
                                for b2 in range(5):
                                    if b2 == a1 or b2 == a2 or b2 == b1:
                                        continue
                                    # Edges: (a1, 5+b1), (a2, 5+b1), (a2, 5+b2), (a1, 5+b2)
                                    e1 = edge_idx.get((a1, 5 + b1))
                                    e2 = edge_idx.get((a2, 5 + b1))
                                    e3 = edge_idx.get((a2, 5 + b2))
                                    e4 = edge_idx.get((a1, 5 + b2))
                                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                                        cycles.append((e1, e2, e3, e4))
                elif m == 3:
                    # 6-cycles: (a1,b1,a2,b2,a3,b3,a1)
                    # Edges: (a1,b1), (a2,b1), (a2,b2), (a3,b2), (a3,b3), (a1,b3)
                    # Conditions: all ai != bi for edges, all distinct
                    for a1 in range(5):
                        for a2 in range(5):
                            if a2 == a1:
                                continue
                            for a3 in range(5):
                                if a3 == a1 or a3 == a2:
                                    continue
                                for b1 in range(5):
                                    if b1 == a1 or b1 == a2:
                                        continue
                                    for b2 in range(5):
                                        if b2 == a2 or b2 == a3 or b2 == b1:
                                            continue
                                        for b3 in range(5):
                                            if b3 == a3 or b3 == a1 or b3 == b1 or b3 == b2:
                                                continue
                                            e1 = edge_idx.get((a1, 5 + b1))
                                            e2 = edge_idx.get((a2, 5 + b1))
                                            e3 = edge_idx.get((a2, 5 + b2))
                                            e4 = edge_idx.get((a3, 5 + b2))
                                            e5 = edge_idx.get((a3, 5 + b3))
                                            e6 = edge_idx.get((a1, 5 + b3))
                                            if all(e is not None for e in [e1, e2, e3, e4, e5, e6]):
                                                cycles.append((e1, e2, e3, e4, e5, e6))
                elif m == 4:
                    # 8-cycles: (a1,b1,a2,b2,a3,b3,a4,b4,a1)
                    # Edges: (a1,b1), (a2,b1), (a2,b2), (a3,b2), (a3,b3), (a4,b3), (a4,b4), (a1,b4)
                    for a1 in range(5):
                        for a2 in range(5):
                            if a2 == a1:
                                continue
                            for a3 in range(5):
                                if a3 == a1 or a3 == a2:
                                    continue
                                for a4 in range(5):
                                    if a4 == a1 or a4 == a2 or a4 == a3:
                                        continue
                                    for b1 in range(5):
                                        if b1 == a1 or b1 == a2:
                                            continue
                                        for b2 in range(5):
                                            if b2 == a2 or b2 == a3 or b2 == b1:
                                                continue
                                            for b3 in range(5):
                                                if b3 == a3 or b3 == a4 or b3 == b1 or b3 == b2:
                                                    continue
                                                for b4 in range(5):
                                                    if b4 == a4 or b4 == a1 or b4 == b1 or b4 == b2 or b4 == b3:
                                                        continue
                                                    e1 = edge_idx.get((a1, 5 + b1))
                                                    e2 = edge_idx.get((a2, 5 + b1))
                                                    e3 = edge_idx.get((a2, 5 + b2))
                                                    e4 = edge_idx.get((a3, 5 + b2))
                                                    e5 = edge_idx.get((a3, 5 + b3))
                                                    e6 = edge_idx.get((a4, 5 + b3))
                                                    e7 = edge_idx.get((a4, 5 + b4))
                                                    e8 = edge_idx.get((a1, 5 + b4))
                                                    if all(e is not None for e in [e1, e2, e3, e4, e5, e6, e7, e8]):
                                                        cycles.append((e1, e2, e3, e4, e5, e6, e7, e8))
                return cycles
            
            # Generate cycle constraints
            cycles_4 = generate_cycles(4)
            cycles_6 = generate_cycles(6)
            cycles_8 = generate_cycles(8)
            
            # Remove duplicate cycles (same set of edges, different starting point or direction)
            def canonicalize(cycle):
                """Return canonical form of a cycle for deduplication."""
                edges_in_cycle = list(cycle)
                # Sort the tuple to get a canonical representation
                return tuple(sorted(edges_in_cycle))
            
            seen = set()
            unique_cycles_4 = []
            for c in cycles_4:
                canon = canonicalize(c)
                if canon not in seen:
                    seen.add(canon)
                    unique_cycles_4.append(c)
            
            seen = set()
            unique_cycles_6 = []
            for c in cycles_6:
                canon = canonicalize(c)
                if canon not in seen:
                    seen.add(canon)
                    unique_cycles_6.append(c)
            
            seen = set()
            unique_cycles_8 = []
            for c in cycles_8:
                canon = canonicalize(c)
                if canon not in seen:
                    seen.add(canon)
                    unique_cycles_8.append(c)
            
            # Now search for valid voltage assignments
            # We'll use random search with fast cycle checking
            
            num_attempts = 5000
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    return None
                
                # Assign random 5-bit vectors to each of the 20 edges
                voltages = [random.getrandbits(5) for _ in range(20)]
                
                # Check 4-cycles
                valid = True
                for cycle in unique_cycles_4:
                    s = 0
                    for e in cycle:
                        s ^= voltages[e]
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check 6-cycles
                for cycle in unique_cycles_6:
                    s = 0
                    for e in cycle:
                        s ^= voltages[e]
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check 8-cycles
                for cycle in unique_cycles_8:
                    s = 0
                    for e in cycle:
                        s ^= voltages[e]
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # All cycle constraints satisfied! Build the lift
                n_lift = 10 * 32
                lift_edges = []
                
                for u, v in base_edges:
                    idx = edge_idx[(u, v)]
                    w = voltages[idx]
                    
                    for x in range(32):
                        u_lift = u * 32 + x
                        v_lift = v * 32 + ((x + w) % 32)
                        lift_edges.append((u_lift, v_lift))
                
                # Verify the graph is valid
                if check_girth_fast(n_lift, lift_edges, 10):
                    return n_lift, lift_edges
                
                # If fast check says girth >= 10 but we want to be extra safe,
                # do a slower verification
                # Actually, check_girth_fast should be correct, so return directly
            
            return None
        
        result = construct_320_cage()
        if result:
            return result

        # Fall back to LUW if voltage graph approach fails
        d_try = 5
        n, edges = luw_construct(d_try, 4)
        if check_girth_fast(n, edges, 10):
            return n, edges
            
        d_try = 6
        n, edges = luw_construct(d_try, 4)
        if check_girth_fast(n, edges, 10):
            return n, edges

    # General case
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, k)
    return n, edges

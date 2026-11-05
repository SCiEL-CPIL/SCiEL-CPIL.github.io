"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 125.0  # safety margin before 150s kill
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for k=4, g=10: 161. Best known is 320.
    
    # Precompute base graphs with adjacency structure for fast voltage lift operations
    base_graphs = []
    
    # Base graph 1: 8-cycle with opposite chords (4-regular, 8 vertices)
    base1_edges = [(i, (i+1)%8) for i in range(8)] + [(i, i+4) for i in range(4)]
    base_graphs.append(base1_edges)
    
    # Base graph 2: Complete bipartite K(4,4) minus perfect matching (4-regular, 8 vertices)
    base3_edges = []
    for i in range(4):
        for j in range(4):
            if i != j:
                base3_edges.append((i, 4+j))
    base_graphs.append(base3_edges)
    
    # Base graph 3: 12-vertex 4-regular graph
    base4_edges = []
    for i in range(6):
        base4_edges.append((i, (i+1)%6))
        base4_edges.append((6+i, (6+(i+1))%6))
    for i in range(6):
        base4_edges.append((i, 6+i))
    base4_edges.append((0, 3))
    base4_edges.append((1, 4))
    base4_edges.append((2, 5))
    base4_edges.append((6, 9))
    base4_edges.append((7, 10))
    base4_edges.append((8, 11))
    base_graphs.append(base4_edges)
    
    # Base graph 4: 16-vertex 4-regular (cube-like)
    base5_edges = []
    for i in range(16):
        base5_edges.append((i, (i^1) % 16))
        base5_edges.append((i, (i^2) % 16))
        base5_edges.append((i, (i^4) % 16))
    seen = set()
    unique_base5 = []
    for e in base5_edges:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key)
            unique_base5.append(key)
    base_graphs.append(unique_base5)
    
    # Base graph 5: 20-vertex 4-regular
    base6_edges = []
    for i in range(20):
        base6_edges.append((i, (i+1)%20))
        base6_edges.append((i, (i+2)%20))
    seen = set()
    unique_base6 = []
    for e in base6_edges:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key)
            unique_base6.append(key)
    base_graphs.append(unique_base6)
    
    # Build adjacency structures for each base graph
    base_adjs = []
    for edges in base_graphs:
        n = max(max(u for u,v in edges), max(v for u,v in edges)) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        base_adjs.append((n, adj, edges))
    
    # Generate m candidates: n_lift = base_n * m, target ~161-320
    m_candidates = []
    for bi, (base_n, base_adj, base_edges) in enumerate(base_adjs):
        min_m = max(2, (161 + base_n - 1) // base_n)
        max_m = min(min_m + 25, 320 // base_n)
        for m in range(min_m, max_m + 1):
            n_lift = base_n * m
            if 161 <= n_lift <= 320:
                m_candidates.append((bi, base_n, base_adj, base_edges, m, n_lift))
    
    # Sort by n_lift ascending
    m_candidates.sort(key=lambda x: x[5])
    
    def build_lift_edges(base_n, base_edges, m, voltages, k):
        """Build the lift edge list and verify it's valid."""
        n_lift = base_n * m
        edges = []
        for (u, v), vol in zip(base_edges, voltages):
            for t in range(m):
                edges.append((u * m + t, v * m + (t + vol) % m))
        
        edge_set = set()
        unique_edges = []
        for e in edges:
            key = (min(e), max(e))
            if key not in edge_set:
                edge_set.add(key)
                unique_edges.append((key[0], key[1]))
        
        if len(unique_edges) == n_lift * k // 2:
            return unique_edges
        return None
    
    # Search with simulated annealing
    # We need to capture base_edges in the check_girth closure, so pass it explicitly
    def check_girth_with_edges(base_n, base_adj, base_edges, m, voltages, g, deadline):
        n_lift = base_n * m
        min_cycle = float('inf')
        
        adj = [[] for _ in range(n_lift)]
        for ei, (u, v) in enumerate(base_edges):
            vol = voltages[ei]
            for t in range(m):
                adj[u * m + t].append(v * m + (t + vol) % m)
                adj[v * m + t].append(u * m + (t - vol) % m)
        
        step = max(1, base_n // 4)
        for sv in range(0, base_n, step):
            if time.time() > deadline:
                return min_cycle
            
            sv_lift = sv * m
            visited = {}
            q = deque()
            
            for nb in adj[sv_lift]:
                key = (nb, sv_lift)
                visited[key] = 1
                q.append(key)
            
            while q:
                if time.time() > deadline:
                    return min_cycle
                cur, prev = q.popleft()
                dist = visited[(cur, prev)]
                
                if dist >= g - 1:
                    continue
                
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    
                    if (nb, cur) in visited:
                        cycle_len = dist + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        visited[(nb, cur)] = dist + 1
                        q.append((nb, cur))
        
        return min_cycle
    
    for bi, base_n, base_adj, base_edges, m, n_lift in m_candidates:
        if time.time() > deadline - 20:
            break
        
        if n_lift >= best_n:
            continue
        
        num_edges = len(base_edges)
        
        max_restarts = 8
        max_iterations = 6000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            current_voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
            
            girth = check_girth_with_edges(base_n, base_adj, base_edges, m, current_voltages, g, deadline - 12)
            
            if girth >= g:
                unique_edges = build_lift_edges(base_n, base_edges, m, current_voltages, k)
                if unique_edges is not None:
                    best_n = n_lift
                    best_edges = unique_edges
                    break
            
            temperature = 1.0
            cooling = 0.9995
            for it in range(max_iterations):
                if time.time() > deadline - 12:
                    break
                
                edge_idx = random.randint(0, num_edges - 1)
                old_vol = current_voltages[edge_idx]
                new_vol = random.randint(0, m - 1)
                if new_vol == old_vol:
                    continue
                
                current_voltages[edge_idx] = new_vol
                
                girth_new = check_girth_with_edges(base_n, base_adj, base_edges, m, current_voltages, g, deadline - 10)
                
                if girth_new >= g:
                    unique_edges = build_lift_edges(base_n, base_edges, m, current_voltages, k)
                    if unique_edges is not None and n_lift < best_n:
                        best_n = n_lift
                        best_edges = unique_edges
                        temperature *= cooling
                else:
                    if random.random() < temperature:
                        temperature *= cooling
                    else:
                        current_voltages[edge_idx] = old_vol
                        temperature *= cooling
            
            if best_edges is not None and best_n == n_lift:
                break
    
    # If voltage lift didn't work, fall back to LUW construction
    if best_edges is None:
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(d, 3)
        q = k
        
        if q == 4:
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        
        for c in range(2, d + 1):
            if c == 2:
                rule_a[c] = 1
                rule_b[c] = 1
            elif c == 3:
                rule_a[c] = 2
                rule_b[c] = 1
            elif c == 4:
                rule_a[c] = 1
                rule_b[c] = 2
            else:
                r = (c - 5) % 4
                c0 = c - r
                if r == 0:
                    rule_a[c] = 1
                    rule_b[c] = c0 - 2
                elif r == 1:
                    rule_a[c] = c0 - 1
                    rule_b[c] = 1
                elif r == 2:
                    rule_a[c] = c0
                    rule_b[c] = 1
                else:
                    rule_a[c] = 1
                    rule_b[c] = c0 + 1
        
        def compute_neighbours_fast(vertex_key):
            side, x = vertex_key
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            return result
        
        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        head = 0
        
        while head < len(order):
            if time.time() > deadline:
                break
            vertex = order[head]
            head += 1
            for other in compute_neighbours_fast(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END

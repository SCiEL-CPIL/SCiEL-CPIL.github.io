"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Voltage graph (lift) construction
    # Base graph: small k-regular graph with girth >= 4
    # Voltage group: Z_m
    # Result: n = m * |V_base| vertices
    
    # For k=4, g=10: use a 4-regular base graph and find voltage assignments
    # that eliminate all cycles of length < g in the lift.
    
    # Strategy:
    # 1. Start with a small k-regular base graph (e.g., cycle with chords, or known small cages)
    # 2. Assign voltages (elements of Z_m) to edges
    # 3. Check if the lift has girth >= g
    # 4. Use simulated annealing to optimize voltage assignments
    
    # For g=10, k=4: try different base graphs and m values
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for k=4, g=10: 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1+4+12+36+108 = 161
    # But actual minimum is higher. Best known is 320.
    # Our target: find something better than LUW if possible.
    
    # Try voltage lifts with various base graphs and m values
    base_graphs = []
    
    # Base graph 1: 8-cycle with opposite chords (4-regular, 8 vertices)
    # Vertices 0-7 in a cycle, plus chords (0,4), (1,5), (2,6), (3,7)
    base1_edges = [(i, (i+1)%8) for i in range(8)] + [(i, i+4) for i in range(4)]
    base_graphs.append((8, base1_edges))
    
    # Base graph 2: 10-cycle with chords
    base2_edges = [(i, (i+1)%10) for i in range(10)] + [(i, (i+3)%10) for i in range(5)]
    base_graphs.append((10, base2_edges))
    
    # Base graph 3: Complete bipartite K(4,4) minus perfect matching (4-regular, 8 vertices)
    base3_edges = []
    for i in range(4):
        for j in range(4):
            if i != j:
                base3_edges.append((i, 4+j))
    base_graphs.append((8, base3_edges))
    
    # Try different m values (voltage group size)
    # n = m * |V_base|, so we want small m
    # For girth g in lift, we need to avoid non-backtracking closed walks of length < g
    # with net voltage 0
    
    m_candidates = []
    for base_n, base_edges in base_graphs:
        # Minimum m such that m * base_n >= some reasonable bound
        # Start with small m and increase
        min_m = max(2, (161 + base_n - 1) // base_n)  # At least Moore bound
        for m in range(min_m, min_m + 15):
            m_candidates.append((base_n, base_edges, m))
    
    # Sort by resulting n = m * base_n
    m_candidates.sort(key=lambda x: x[2] * x[0])
    
    # For each candidate, try to find valid voltage assignments
    # Voltage assignment: each edge gets a voltage in Z_m
    # The lift has vertex (v, t) where v in base, t in Z_m
    # Edge (u,v) with voltage a connects (u,t) to (v, t+a)
    
    def check_lift_girth(base_n, base_edges, m, voltages, g, deadline):
        """Check if the lift has girth >= g using non-backtracking walk enumeration."""
        # Build lift adjacency
        # Vertex (v, t) -> index v * m + t
        n_lift = base_n * m
        
        # Build adjacency list for lift
        adj = [[] for _ in range(n_lift)]
        for (u, v), vol in zip(base_edges, voltages):
            # Edge u->v with voltage vol
            for t in range(m):
                adj[u * m + t].append(v * m + (t + vol) % m)
                adj[v * m + t].append(u * m + (t - vol) % m)
        
        # Check girth by enumerating non-backtracking walks from each vertex
        # A cycle of length L exists if there's a non-backtracking closed walk of length L
        # with net voltage 0 (automatically satisfied in the lift)
        
        # For efficiency, check from a sample of vertices
        # Non-backtracking walk: sequence v0, v1, ..., vL where vi != vi-2
        
        # Use BFS in the non-backtracking graph
        # State: (current_vertex, previous_vertex)
        
        min_cycle = float('inf')
        
        # For each starting vertex, do BFS in non-backtracking state space
        # But this is expensive for large n_lift. Use sampling.
        
        sample_size = min(20, base_n)
        random.seed(123)
        sample_vertices = random.sample(range(base_n), sample_size)
        
        for sv in sample_vertices:
            # BFS from (sv, sv) - but we need to handle the start properly
            # Actually, start with all neighbors as first step
            # State: (current, prev) where prev is the vertex we came from
            # Distance = length of walk so far
            
            visited = {}  # (current, prev) -> distance
            q = deque()
            
            for nb in adj[sv * m]:
                # nb is in the lift, extract base vertex
                nb_base = nb // m
                # State: (nb, sv) with distance 1
                key = (nb, sv)
                visited[key] = 1
                q.append(key)
            
            while q:
                if time.time() > deadline:
                    return min_cycle
                cur, prev = q.popleft()
                dist = visited[(cur, prev)]
                
                if dist >= g:
                    continue  # We only care about cycles < g
                
                cur_base = cur // m
                for nb in adj[cur]:
                    nb_base = nb // m
                    if nb_base == prev:
                        continue  # Non-backtracking: can't go back to prev
                    
                    if (nb, cur) in visited:
                        # Found a cycle
                        cycle_len = dist + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        visited[(nb, cur)] = dist + 1
                        q.append((nb, cur))
        
        return min_cycle
    
    def random_voltages(num_edges, m):
        return [random.randint(0, m - 1) for _ in range(num_edges)]
    
    # Simulated annealing for each candidate
    for base_n, base_edges, m in m_candidates:
        if time.time() > deadline - 20:
            break
        
        n_lift = base_n * m
        if n_lift >= best_n:
            continue  # Can't improve
        
        num_edges = len(base_edges)
        
        # Try multiple random restarts
        max_restarts = 5
        max_iterations = 5000
        start_temp = 10.0
        end_temp = 0.01
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random voltages
            voltages = random_voltages(num_edges, m)
            current_voltages = voltages[:]
            
            # Check initial girth
            girth = check_lift_girth(base_n, base_edges, m, current_voltages, g, deadline - 10)
            
            if girth >= g:
                # Valid! Build the lift
                edges = []
                for (u, v), vol in zip(base_edges, current_voltages):
                    for t in range(m):
                        edges.append((u * m + t, v * m + (t + vol) % m))
                # Remove duplicates (each undirected edge added twice)
                # Actually, in our construction each edge is added once per direction
                # Let's deduplicate
                edge_set = set()
                unique_edges = []
                for e in edges:
                    key = (min(e), max(e))
                    if key not in edge_set:
                        edge_set.add(key)
                        unique_edges.append((key[0], key[1]))
                
                if len(unique_edges) == n_lift * k // 2:
                    best_n = n_lift
                    best_edges = unique_edges
                    break  # Found valid for this m, try smaller m
            
            # Simulated annealing to improve
            temp = start_temp
            for it in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                # Pick random edge and change its voltage
                edge_idx = random.randint(0, num_edges - 1)
                old_vol = current_voltages[edge_idx]
                new_vol = random.randint(0, m - 1)
                if new_vol == old_vol:
                    continue
                
                current_voltages[edge_idx] = new_vol
                
                # Check if this creates a short cycle
                # Quick check: only check if the changed edge is in a short cycle
                # For now, just check full girth (expensive but safe)
                girth = check_lift_girth(base_n, base_edges, m, current_voltages, g, deadline - 5)
                
                if girth >= g:
                    # Valid, accept
                    temp = max(temp * 0.99, end_temp)
                    # Update best if this is better
                    if n_lift < best_n:
                        edges = []
                        for (u, v), vol in zip(base_edges, current_voltages):
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
                            best_n = n_lift
                            best_edges = unique_edges
                else:
                    # Revert if invalid
                    current_voltages[edge_idx] = old_vol
                    temp = max(temp * 0.99, end_temp)
            
            if best_edges is not None and best_n == n_lift:
                break  # Found valid for this m
    
    # If voltage lift didn't work, fall back to LUW construction
    if best_edges is None:
        # LUW construction as before
        min_d_odd = max(3, g - 5)
        if min_d_odd % 2 == 0:
            min_d_odd += 1
        min_d_even = max(4, g - 4)
        if min_d_even % 2 == 1:
            min_d_even += 1
        d = min(min_d_odd, min_d_even)
        q = k
        
        if q == 4:
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b
        
        def vertex_to_key(side, x):
            return (side, x)
        
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
                result.append(vertex_to_key(1 - side, tuple(y)))
            return result
        
        start = vertex_to_key(0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        queue = deque([start])
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

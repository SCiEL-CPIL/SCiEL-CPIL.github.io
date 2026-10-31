import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Safety margin (evaluator kills at 150s)

    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Heuristic Search for k=3, g=16
    if k == 3 and g == 16:
        best_n, best_edges = luw_construct(k, g)

        def build_adj(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def check_girth(adj, min_girth):
            n = len(adj)
            max_depth = (min_girth - 1) // 2
            for start in range(n):
                visited = [-1] * n
                visited[start] = 0
                queue = deque([(start, 0, -1)])
                while queue:
                    node, depth, parent = queue.popleft()
                    if depth > max_depth:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if visited[neighbor] == -1:
                            visited[neighbor] = depth + 1
                            queue.append((neighbor, depth + 1, node))
                        else:
                            cycle_len = depth + visited[neighbor] + 1
                            if cycle_len < min_girth:
                                return False
            return True

        # Deterministic Voltage Graph Lift of Petersen Graph
        # Base graph: Petersen (10 vertices, 15 edges, girth 5)
        # Lift size: m. Total vertices: 10m.
        # Strategy: Assign voltages in Z_m to edges.
        # A cycle in the lift corresponds to a closed walk in the base.
        # Length L closed walk in base has voltage sum S.
        # It is a cycle in the lift if S % m == 0.
        # To avoid short cycles, we want S % m != 0 for all closed walks of length < g.
        # Petersen closed walks of length L exist for L >= 5.
        # Length 5: 12 cycles.
        # Length 6: 0 (bipartite? No, Petersen has triangles? No, girth 5. No 3, 4, 6? 
        # Actually Petersen has no even cycles? No, it has 8-cycles? 
        # Petersen is not bipartite (has 5-cycles). 
        # Does it have 6-cycles? No, girth 5. Next possible is 5, 8, 9, 10...
        # Actually, Petersen has cycles of length 5, 8, 9, 10.
        # We need to avoid length 5, 6, ..., 15.
        
        # Let's define the Petersen graph edges.
        # Vertices 0-4 (outer pentagon), 5-9 (inner star).
        # Outer: (0,1), (1,2), (2,3), (3,4), (4,0)
        # Inner: (5,7), (7,9), (9,6), (6,8), (8,5)  -> (5,7), (7,9), (9,6), (6,8), (8,5)
        # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
        
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Precompute all closed walks of length up to 15 in Petersen graph?
        # That's expensive. 
        # Alternative: Just generate the lift and check girth directly.
        # n = 10m. If m=52, n=520. 
        # Moore bound for k=3, g=16 is 512.
        # So m=52 is the smallest possible multiple of 10 >= 512.
        
        # We will try m = 52, 53, ..., up to some limit.
        # For each m, we try several deterministic voltage assignments.
        # Assignment 1: Random fixed seed.
        # Assignment 2: Quadratic residues (as suggested).
        
        # Let's optimize the girth check for the specific lift structure.
        # The lift is a covering. 
        # Instead of BFS on 10m nodes, we can BFS on the base graph tracking voltage sums?
        # No, standard BFS on the lift is O(10m * 3 * depth).
        # Depth 8. 10*52*3*8 ~ 12000 ops per start vertex. 520 starts. ~6M ops. Fast enough in Python?
        # Maybe. Let's try.
        
        # To speed up, we can use numpy or just efficient lists.
        # Let's implement a fast check for the voltage lift.
        
        def check_voltage_lift_girth(m, voltages, target_girth):
            # voltages is a list of 15 integers (mod m)
            n = 10 * m
            # Build adjacency list for the lift
            # Vertex (i, j) where i in 0..9, j in 0..m-1
            # Edge (u, v) in base with voltage w connects (u, j) to (v, (j+w)%m) and (v, j) to (u, (j-w)%m)
            
            # We can avoid building the full adjacency list and just traverse.
            # But for BFS, we need neighbors.
            # Let's build the adj list.
            adj = [[] for _ in range(n)]
            for idx, (u, v) in enumerate(petersen_edges):
                w = voltages[idx]
                for j in range(m):
                    u_node = u * m + j
                    v_node = v * m + j
                    # u -> v : j -> (j+w)%m
                    v_node_f = v * m + (j + w) % m
                    # v -> u : j -> (j-w)%m
                    u_node_b = u * m + (j - w) % m
                    
                    adj[u_node].append(v_node_f)
                    adj[v_node].append(u_node_b)
            
            # Now check girth
            max_depth = (target_girth - 1) // 2
            for start in range(n):
                visited = [-1] * n
                visited[start] = 0
                queue = deque([(start, 0, -1)])
                while queue:
                    node, depth, parent = queue.popleft()
                    if depth > max_depth:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if visited[neighbor] == -1:
                            visited[neighbor] = depth + 1
                            queue.append((neighbor, depth + 1, node))
                        else:
                            cycle_len = depth + visited[neighbor] + 1
                            if cycle_len < target_girth:
                                return False, None
            return True, adj
        
        # Try scanning m
        # Start from m such that 10m >= 512. m=52.
        # If we find a valid one, n=10m.
        # We want to minimize n.
        
        # We will try a few voltage assignments for each m.
        # Assignment types:
        # 1. Random (fixed seed for determinism across runs? No, random is fine, but let's use different seeds)
        # 2. "Quadratic" like: w_i = (i^2 + i + 1) % m
        
        best_found_n = best_n
        best_found_edges = best_edges
        
        # Time budget for voltage search
        # We have ~110s total. LUW takes a bit. Let's give 50s to voltage search.
        # If we find n < best_n, update.
        
        search_deadline = start_time + 60.0
        
        for m in range(52, 80):  # Try m up to 80 (n=800)
            if time.time() > search_deadline:
                break
            
            n = 10 * m
            if n >= best_found_n:
                continue
            
            # Try 5 different voltage assignments
            found = False
            for trial in range(5):
                # Generate voltages
                if trial == 0:
                    # Random
                    voltages = [random.randint(0, m-1) for _ in range(15)]
                elif trial == 1:
                    # Quadratic
                    voltages = [(i*i + i + 1) % m for i in range(15)]
                elif trial == 2:
                    # Linear
                    voltages = [(i*3 + 1) % m for i in range(15)]
                elif trial == 3:
                    # Another random
                    voltages = [random.randint(1, m-1) for _ in range(15)] # Non-zero
                else:
                    # Mixed
                    voltages = [(i*i + 2*i + 3) % m for i in range(15)]
                
                # Check
                valid, adj = check_voltage_lift_girth(m, voltages, 16)
                if valid:
                    # Convert adj to edges
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    if n < best_found_n:
                        best_found_n = n
                        best_found_edges = edges
                    found = True
                    break # Found valid for this m, try smaller m? No, we are scanning up. 
                          # But we want min n. So if we found one at m, we should stop?
                          # No, we are scanning m=52, 53...
                          # If we find one at 52, that's the best possible (Moore bound 512, 10*52=520).
                          # So we can break out of m loop too?
                          # Yes, if n is close to lower bound.
            
            if found and n <= 520:
                break
        
        if best_found_n < best_n:
            return best_found_n, best_found_edges

        return best_n, best_edges

    return luw_construct(k, g)
# EVOLVE-BLOCK-END

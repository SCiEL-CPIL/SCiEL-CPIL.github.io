import time
import random
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target: 4-regular, girth 10.
    # Base graph: Octahedron (6 vertices, 12 edges, 4-regular, girth 3).
    # Lift by Z_m. Vertices: 6m.
    # We want 6m >= 243 => m >= 41.
    
    # Define Octahedron
    # Vertices 0..5
    # Edges
    base_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (5, 1), (5, 2), (5, 3), (5, 4),
        (1, 2), (2, 3), (3, 4), (4, 1)
    ]
    adj = defaultdict(list)
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Find all simple cycles of length 3 to 9
    def find_cycles(adj, max_len):
        cycles = []
        n = len(adj)
        for start in range(n):
            # DFS
            stack = [(start, [start], [start])]
            while stack:
                node, path, edge_path = stack.pop()
                if len(path) > max_len:
                    continue
                if len(path) >= 3 and node == start:
                    # Found a cycle
                    # Normalize: start with smallest node, direction consistent
                    # Just store edge set or list
                    cycles.append(edge_path[:])
                    continue
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 3:
                        stack.append((start, path + [start], edge_path + [(node, start)]))
                    elif neighbor not in path:
                        stack.append((neighbor, path + [neighbor], edge_path + [(node, neighbor)]))
        # Remove duplicates
        unique_cycles = []
        seen = set()
        for c in cycles:
            # Canonical form
            # Sort edges? No, order matters for sum? Sum is commutative.
            # So just use frozenset of edges?
            # But we need the sum. Sum is sum of voltages.
            # So we just need the set of edges.
            key = frozenset(c)
            if key not in seen:
                seen.add(key)
                unique_cycles.append(c)
        return unique_cycles

    short_cycles = find_cycles(adj, 9)
    
    # Function to check if a voltage assignment is valid for a given m
    def is_valid(voltages, m):
        # voltages is a dict (u,v) -> val
        # We need sum on each cycle != 0
        for cycle in short_cycles:
            s = 0
            for u, v in cycle:
                # Get voltage
                if (u, v) in voltages:
                    s = (s + voltages[(u, v)]) % m
                else:
                    s = (s - voltages[(v, u)]) % m
            if s == 0:
                return False
        return True

    # Search for m and voltages
    best_n = None
    best_edges = None
    
    # Try m from 41 upwards
    # Limit time
    for m in range(41, 60):
        if time.time() > deadline:
            break
            
        # Try to find voltages in Z_m
        # Random search
        found = False
        # Initialize voltages
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(1, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
            
        # Iterate
        for _ in range(10000):
            if time.time() > deadline:
                break
            if is_valid(voltages, m):
                found = True
                break
            # Pick a random cycle and change a random edge in it
            # Or just random restart?
            # Let's try local search
            # Pick a random invalid cycle
            # We don't track which are invalid, so just randomize one edge
            u, v = random.choice(base_edges)
            voltages[(u, v)] = random.randint(1, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
            
        if found:
            # Check connectivity of lift
            # Build lift
            n_vertices = 6 * m
            edges = []
            # Vertices are (base_node, group_elem)
            # Map to 0..n_vertices-1
            # idx = base_node * m + group_elem
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for g in range(m):
                    u_idx = u * m + g
                    v_idx = v * m + (g + vu) % m
                    edges.append((u_idx, v_idx))
            
            # Check if connected
            # BFS
            visited = [False] * n_vertices
            queue = [0]
            visited[0] = True
            count = 1
            adj_lift = defaultdict(list)
            for u, v in edges:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
                
            while queue:
                node = queue.pop(0)
                for nei in adj_lift[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        count += 1
                        queue.append(nei)
                        
            if count == n_vertices:
                best_n = n_vertices
                best_edges = edges
                break # Found a valid connected graph, return it? 
                # Or keep searching for smaller m?
                # We are iterating m increasing, so first found is smallest m.
                # So this is optimal for this base.
                break

    if best_n is not None:
        return best_n, best_edges

    # Fallback: Seed program logic (LUW)
    # ... (copy seed code) ...
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

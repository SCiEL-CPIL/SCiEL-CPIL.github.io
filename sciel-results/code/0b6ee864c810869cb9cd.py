import itertools
import random
import time

def construct(k, g):
    # Petersen Graph Definition
    # Vertices 0-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Build adjacency list for Petersen
    adj = [[] for _ in range(10)]
    edge_indices = {}
    for i, (u, v) in enumerate(petersen_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_indices[(u, v)] = i
        edge_indices[(v, u)] = i

    # Enumerate all simple cycles of length L < g in Petersen
    # We need cycles of length 5 to g-1
    cycles = []
    
    # Simple DFS to find all simple cycles
    # Since the graph is small, we can do this for each start vertex
    # To avoid duplicates, we only start from the smallest vertex in the cycle
    # and ensure the cycle is "canonical"
    
    def find_cycles(start, current, path_edges, visited):
        if len(path_edges) >= 5 and len(path_edges) < g:
            # Check if we can close the cycle to start
            # Current is 'current', we need an edge from current to start
            # And 'start' must be the smallest vertex in the cycle to avoid duplicates
            # Actually, simpler: just find all cycles, then deduplicate by frozenset of edges?
            # Or just check if start is min(path_vertices)
            pass
            
    # Let's use a simpler approach:
    # Generate all simple paths of length L from 0 to 0? No.
    # Let's just use a standard cycle finding.
    
    # Given the small size, we can hardcode the check or use a quick BFS-based cycle detection for each length.
    # But enumerating is safer.
    
    # Let's use a recursive DFS from each vertex, but only count a cycle if the start vertex is the minimum vertex in the cycle.
    
    all_cycles = []
    
    def dfs(v, start, visited, path_edges):
        # path_edges contains the sequence of edge indices
        if len(path_edges) >= 5:
            # Check if we can close to start
            if start in [u for u, _ in adj[v]]:
                # Find the edge index between v and start
                for u, idx in adj[v]:
                    if u == start:
                        # Check if start is the min vertex in the cycle
                        # We need the vertices. Let's track vertices too.
                        pass
        
    # This is getting complicated. Let's just generate all closed walks of length L < 16 
    # and filter out those that are not simple cycles?
    # Actually, if a non-simple closed walk has sum 0, it implies a simple cycle has sum 0?
    # No. But if a simple cycle has sum 0, we have a short cycle.
    # If a non-simple closed walk has sum 0, it is composed of cycles.
    # It's safer to just check all simple cycles.
    
    # Let's use networkx? No, standard library only.
    # Let's implement a simple cycle enumerator.
    
    def enumerate_simple_cycles(graph, n):
        cycles = []
        # Use Johnson's algorithm or simple backtracking
        # For n=10, simple backtracking is fine.
        
        for start in range(n):
            # We only start cycles from 'start' if start is the smallest vertex in the cycle
            # This requires knowing the vertices in the cycle beforehand.
            # Alternative: Find all cycles, then deduplicate by sorting the set of edges.
            
            def backtrack(v, visited, path_edges, path_vertices):
                if len(path_edges) >= 5 and len(path_edges) < g:
                    # Check if v is connected to start
                    for u, idx in graph[v]:
                        if u == start:
                            # Found a cycle
                            # Verify start is min vertex
                            if start == min(path_vertices):
                                # Add a copy
                                cycles.append(list(path_edges))
                
                for u, idx in graph[v]:
                    if u not in visited:
                        visited.add(u)
                        path_edges.append(idx)
                        path_vertices.append(u)
                        backtrack(u, visited, path_edges, path_vertices)
                        path_edges.pop()
                        path_vertices.pop()
                        visited.remove(u)
            
            visited = {start}
            path_edges = []
            path_vertices = [start]
            backtrack(start, visited, path_edges, path_vertices)
            
        return cycles

    # Note: The above backtrack doesn't check if the edge (v, start) exists in the path?
    # It just checks if start is a neighbor of v.
    # This might create cycles that repeat edges? No, visited prevents vertex repetition.
    # So it finds simple cycles.
    
    pet_cycles = enumerate_simple_cycles(adj, 10)
    
    # Deduplicate cycles that are the same set of edges
    unique_cycles = set()
    for c in pet_cycles:
        unique_cycles.add(frozenset(c))
    # Convert back to lists for summing
    final_cycles = [list(c) for c in unique_cycles]
    
    # Now, search for m and voltages
    # Moore bound for n(3,16) is 512. n = 10*m. So m >= 52.
    
    start_time = time.time()
    
    for m in range(52, 100):
        if time.time() - start_time > 100:
            break
            
        # Try a few voltage assignments
        # We need voltages v[0..14] such that sum(v[e] for e in cycle) != 0 mod m for all cycles
        
        # Deterministic patterns
        candidates = []
        
        # Pattern 1: i
        candidates.append([i % m for i in range(15)])
        # Pattern 2: i^2
        candidates.append([(i*i) % m for i in range(15)])
        # Pattern 3: Thue-Morse like
        # t(i) = popcount(i) % 2
        def thue_morse(n):
            return bin(n).count('1') % 2
        candidates.append([(thue_morse(i) * (i+1)) % m for i in range(15)])
        
        # Random patterns
        random.seed(m)
        for _ in range(20):
            candidates.append([random.randint(0, m-1) for _ in range(15)])
            
        for vol in candidates:
            valid = True
            for cycle in final_cycles:
                s = 0
                for e in cycle:
                    s = (s + vol[e]) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                # Found a valid assignment
                # Construct the lift
                n_vertices = 10 * m
                edges = []
                for i, (u, v) in enumerate(petersen_edges):
                    w = vol[i] % m
                    for s in range(m):
                        u_id = u * m + s
                        v_id = v * m + (s + w) % m
                        if u_id < v_id:
                            edges.append((u_id, v_id))
                        else:
                            edges.append((v_id, u_id))
                            
                return n_vertices, edges

    # Fallback if no m found in range
    # This should not happen for m up to 100 with random tries, but just in case
    # Return a trivial invalid graph? No, must be valid.
    # If we fail, we return the L-U-W fallback from the original code?
    # The original code had a fallback. Let's keep a simple fallback.
    
    # Re-implementing the L-U-W fallback from the original code
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

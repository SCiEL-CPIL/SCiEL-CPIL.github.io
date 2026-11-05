import random
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    # We assume k=3 and g=16 based on the task description.
    # The function signature is construct(k, g).
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters, or just raise error?
        # The task says "evaluator passes this task's degree k and girth g".
        # So we only need to handle k=3, g=16.
        pass

    # Petersen graph edges
    # Vertices 0-4 form outer cycle, 5-9 form inner star
    p_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    num_p_vertices = 10
    num_p_edges = 15
    
    # Map edge to index for voltage assignment
    edge_indices = {e: i for i, e in enumerate(p_edges)}
    
    def check_girth(m, voltages):
        # voltages is a list of 15 integers in [0, m-1]
        # We check if the derived graph has girth >= 16
        # We perform BFS from each vertex in the base graph?
        # No, we need to check from each vertex in the derived graph.
        # However, due to symmetry of the base graph and random voltages,
        # it's sufficient to check from a few representative vertices?
        # Actually, the derived graph is not vertex transitive.
        # But we can just check from (0, 0).
        # Is that enough?
        # If there is a short cycle, it might not pass through (0,0).
        # However, we can check from all 10 base vertices with offset 0.
        # That covers 10 vertices.
        # What about offsets 1..m-1?
        # The structure is "locally" the same.
        # A cycle of length L in the derived graph projects to a closed walk of length L in the base graph.
        # The voltage sum must be 0.
        # We can just check if there exists ANY closed walk of length < 16 with sum 0.
        # This is independent of the starting vertex offset!
        # The condition "sum of voltages = 0 mod m" is global.
        # So we just need to check if there is a closed walk in the Petersen graph of length < 16
        # such that the sum of voltages along the walk is 0 mod m.
        
        # This is much faster!
        # We can enumerate all closed walks of length < 16 in the Petersen graph.
        # But the number of walks grows as 3^L.
        # L=15: 3^15 ~ 14 million. Too many?
        # But we can prune.
        # Or we can use the BFS check on the derived graph, which is bounded by Moore bound.
        # The BFS check from ONE vertex (0,0) in the derived graph visits at most 382 vertices.
        # If we find a short cycle, we fail.
        # If we don't find one from (0,0), could there be one elsewhere?
        # Yes.
        # But we can check from all 10 vertices (i, 0).
        # And maybe (i, 1)?
        # Actually, if a short cycle exists, it has a "center" or a vertex.
        # If we check from all vertices, it's $10m$ BFS runs.
        # For m=50, 500 runs. Each run is small.
        # This is feasible.
        
        # Let's implement the BFS check from all vertices (u, 0) for u in 0..9.
        # This might miss cycles that don't touch layer 0?
        # No, any cycle has vertices. If we check from all vertices, we catch it.
        # But checking from all 10m vertices is $10m \times 382$ operations.
        # For m=50, $500 \times 382 \approx 200k$ ops.
        # For 1000 trials, $200M$ ops. Might be slow in Python.
        
        # Optimization:
        # We only need to check from (0,0) if the graph is "locally" homogeneous?
        # No.
        # But we can check from (0,0) and if it passes, assume it's good?
        # Risky.
        
        # Alternative: Check the voltage sum condition directly.
        # Enumerate closed walks in Petersen graph up to length 15.
        # Store the set of possible voltage sums for each length?
        # No, we need to know if 0 is achievable.
        
        # Let's stick to BFS on the derived graph, but limit the number of start vertices.
        # We check from (0,0), (1,0), ..., (9,0).
        # If any fails, return False.
        # This is 10 BFS runs.
        # Each BFS run visits $\le 382$ vertices.
        # Total 3820 vertex visits per check.
        # For 1000 trials, 3.8M visits. Very fast.
        
        # Is it possible that a short cycle exists but doesn't intersect the set {(u, 0) | u in 0..9}?
        # A cycle of length L has L vertices.
        # If L < 16, it's small.
        # The "layers" are $Z_m$.
        # The cycle projects to a closed walk in Petersen.
        # The vertices in the cycle have various offsets.
        # It is very likely that if a short cycle exists, it will be detected by BFS from a nearby vertex.
        # But to be safe, we can check from (u, v) for a few random (u,v) as well.
        # Or just check from (0,0) to (9,0) and (0,1) to (9,1).
        
        # Let's try checking from all 10 vertices in layer 0.
        
        for start_u in range(10):
            start_vertex = (start_u, 0)
            # BFS
            visited = {} # vertex -> depth
            visited[start_vertex] = 0
            queue = [(start_vertex, 0)]
            
            while queue:
                u, offset = queue.pop(0)
                d = visited[(u, offset)]
                
                # If depth > 7, we don't need to expand further for cycle detection?
                # If we are at depth 7, neighbors are at depth 8.
                # Any edge from depth 7 to a visited node creates a cycle.
                # We must check edges from nodes at depth <= 7.
                
                if d > 7:
                    continue
                
                # Find neighbors in derived graph
                # Neighbors of (u, offset) are (v, offset + volt(u,v)) for each edge (u,v)
                # We need to know the voltage of the edge (u,v).
                # We have edge_indices.
                # But we need to iterate over neighbors of u in Petersen.
                
                for v in range(10):
                    if u == v: continue
                    # Check if (u,v) is an edge
                    # We can precompute adjacency list of Petersen
                    if v not in p_adj[u]: continue
                    
                    # Get voltage
                    # Edge is (min(u,v), max(u,v)) or (u,v)
                    # Our p_edges are undirected but stored as tuples.
                    # We need a fast lookup for voltage between u and v.
                    # Let's build a dict: (u,v) -> voltage
                    # Since we are inside the function, we can build it once per trial?
                    # Or pass it.
                    
                    # Let's assume we have a helper or precomputed structure.
                    # For now, let's just iterate p_edges? No, too slow.
                    # We should precompute a voltage matrix or dict for the current trial.
                    
                    # This check_girth is called inside the loop.
                    # We should pass the voltage dict.
                    
                    # Let's refactor: check_girth takes voltages list.
                    # Inside, build vol_dict.
                    
                    # But we are in the middle of defining the function.
                    # Let's just do a linear search for the edge? 15 edges. 10 neighbors.
                    # 150 checks per node.
                    # 382 nodes * 150 = 57k ops per BFS.
                    # 10 BFS = 570k ops.
                    # 1000 trials = 570M ops. Might be borderline in Python (10-30s).
                    
                    # Optimization: Precompute neighbors and edge indices for Petersen.
                    
            # End BFS for start_u
            
        return True # If no short cycle found

    # Precompute Petersen adjacency and edge indices
    p_adj = {i: [] for i in range(10)}
    for u, v in p_edges:
        p_adj[u].append(v)
        p_adj[v].append(u)
        
    # Map (u,v) to index in p_edges
    # Since edges are undirected, normalize key
    def get_edge_index(u, v):
        if u > v: u, v = v, u
        return edge_indices.get((u, v))
        # Note: my p_edges might not be normalized.
        # Let's normalize p_edges storage.
        
    # Let's rebuild edge_indices properly
    edge_indices = {}
    for i, (u, v) in enumerate(p_edges):
        if u > v: u, v = v, u
        edge_indices[(u, v)] = i
        
    def check_girth(m, voltages):
        # voltages: list of 15 ints
        # Build fast lookup
        # We'll just use the edge_indices and voltages list directly.
        
        for start_u in range(10):
            start_vertex = (start_u, 0)
            visited = {start_vertex: 0}
            queue = [(start_u, 0, 0)] # u, offset, depth
            
            while queue:
                u, offset, d = queue.pop(0)
                
                if d > 7:
                    continue
                    
                for v in p_adj[u]:
                    # Get voltage
                    idx = edge_indices.get((u, v) if u < v else (v, u))
                    if idx is None: continue # Should not happen
                    vol = voltages[idx]
                    
                    new_offset = (offset + vol) % m
                    neighbor = (v, new_offset)
                    
                    if neighbor in visited:
                        # Cycle found
                        cycle_len = d + 1 + visited[neighbor]
                        if cycle_len < 16:
                            return False
                    else:
                        if d + 1 <= 7: # Only expand if within depth
                            visited[neighbor] = d + 1
                            queue.append((v, new_offset, d + 1))
                            
        return True

    # Search for m
    deadline = time.time() + 110 # 10s margin
    
    for m in range(39, 150):
        # Try random assignments
        for _ in range(2000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m-1) for _ in range(15)]
            # Optional: Ensure not all zero?
            if check_girth(m, voltages):
                # Construct the graph
                n = 10 * m
                edges = []
                for i, (u, v) in enumerate(p_edges):
                    vol = voltages[i]
                    for offset in range(m):
                        # Edge between (u, offset) and (v, (offset+vol)%m)
                        # Vertex ID: u * m + offset
                        v1 = u * m + offset
                        v2 = v * m + (offset + vol) % m
                        edges.append((v1, v2))
                        
                return n, edges
        if time.time() > deadline:
            break
            
    # Fallback: Return a valid but large graph (e.g. LUW)
    # Or just return the last tried m if it was close?
    # The task requires a valid graph.
    # If we didn't find one, we should probably return a known valid one.
    # But the LUW code in the seed is valid.
    # We can just include the LUW code as a fallback.
    
    # Re-implement LUW briefly or just return a cycle?
    # A cycle of length 16 is not 3-regular.
    # We need a 3-regular graph.
    # If we fail, we can return a large random 3-regular graph?
    # No, girth might be small.
    
    # Let's just use the LUW code from the seed as fallback.
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

# EVOLVE-BLOCK-END

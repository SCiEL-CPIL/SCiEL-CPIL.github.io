import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    edges = set()
    
    # Rim 1: 0..19
    for i in range(20):
        edges.add((min(i, (i+1)%20), max(i, (i+1)%20)))
        
    # Rim 2: 20..39
    for i in range(20):
        v1 = 20 + i
        v2 = 20 + (i+1)%20
        edges.add((min(v1, v2), max(v1, v2)))
        
    # Chords from 0 to 2..9
    for k in range(2, 10):
        edges.add((0, k))
        
    # Chords from 20 to 22..29
    for k in range(2, 10):
        edges.add((20, 20+k))
        
    # Inter-rim chords
    edges.add((0, 20))
    edges.add((1, 21))
    
    edge_list = sorted(list(edges))
    
    cycles = {}
    
    # Helper to get path in Rim 1 from 0 to k (clockwise)
    def path_rim1(a, b):
        # a, b in 0..19
        path = []
        cur = a
        while cur != b:
            path.append(cur)
            cur = (cur + 1) % 20
        path.append(b)
        return path
        
    # Helper to get path in Rim 2 from 20+a to 20+b (clockwise)
    def path_rim2(a, b):
        # a, b in 0..19 (offsets)
        path = []
        cur = a
        while cur != b:
            path.append(20 + cur)
            cur = (cur + 1) % 20
        path.append(20 + b)
        return path

    # 1. Local cycles in Rim 1 (using 0)
    for k in range(2, 10):
        # Cycle 0-1-...-k-0
        c1 = list(range(0, k+1))
        L1 = k + 1
        cycles[L1] = c1
        
        # Cycle 0-19-...-k-0
        c2 = [0] + list(range(19, k-1, -1))
        L2 = 20 - k + 1
        cycles[L2] = c2
        
    # 2. Local cycles in Rim 2 (using 20)
    for k in range(2, 10):
        # Cycle 20-21-...-20+k-20
        c1 = [20] + list(range(21, 20+k+1))
        L1 = k + 1
        if L1 not in cycles:
            cycles[L1] = c1
            
        # Cycle 20-39-...-20+k-20
        c2 = [20] + list(range(39, 20+k-1, -1))
        L2 = 20 - k + 1
        if L2 not in cycles:
            cycles[L2] = c2
            
    # 3. Cross-rim small cycles (using 0-20 and 1-21)
    # Pattern: 0 - (path in Rim 1 to 1) - 1 - 21 - (path in Rim 2 to 20) - 20 - 0
    # Path in Rim 1 to 1: 0 -> k -> ... -> 1
    # Using chord (0, k), path is 0-k-(k-1)-...-1
    for k in range(2, 10):
        # Path 0 -> k -> ... -> 1
        p1 = [0, k] + list(range(k-1, 0, -1)) # Ends at 1? No, range(k-1, 0, -1) ends at 1.
        # Wait, 0-k is chord. k-(k-1) is rim. ... 2-1 is rim.
        # Vertices: 0, k, k-1, ..., 1.
        # Length of this path: 1 (chord) + (k-1) (rim edges) = k.
        # Vertices: k+1.
        
        # Path in Rim 2 from 21 to 20 (short)
        p2 = [21, 20] # Length 1 edge.
        
        # Full cycle: 0 -> ... -> 1 -> 21 -> 20 -> 0
        # Edges: (0, k), (k, k-1)...(2, 1), (1, 21), (21, 20), (20, 0)
        # Vertices: 0, k, ..., 1, 21, 20.
        # Distinct? Yes, if k in 2..9.
        
        cycle_vertices = p1 + [21, 20]
        # p1 ends at 1. 1-21 is edge. 21-20 is edge. 20-0 is edge.
        # Check distinctness: 0, k..1, 21, 20. All distinct.
        
        L = len(cycle_vertices)
        if L not in cycles:
            cycles[L] = cycle_vertices
            
    # 4. Cross-rim large cycles
    # Pattern: 0 -> (path in Rim 1 to 1) -> 1 -> 21 -> (long path in Rim 2 to 20) -> 20 -> 0
    # Long path in Rim 2 from 21 to 20: 21 -> 22 -> ... -> 39 -> 20
    # Vertices: 21, 22, ..., 39, 20.
    # Length: 19 edges.
    
    for k in range(2, 10):
        p1 = [0, k] + list(range(k-1, 0, -1))
        p2 = list(range(21, 40)) + [20] # 21..39, 20
        
        cycle_vertices = p1 + p2
        # p1 ends at 1. 1-21 edge.
        # p2 starts at 21. Ends at 20.
        # 20-0 edge.
        
        L = len(cycle_vertices)
        if L not in cycles:
            cycles[L] = cycle_vertices
            
    # 5. Hamiltonian cycle (40)
    # 0-1-...-19-? 
    # We need to connect Rim 1 and Rim 2.
    # 0-1-...-19-? No direct edge 19-20.
    # We have 0-20 and 1-21.
    # Cycle: 0-1-21-22-...-39-20-? 
    # 20 is connected to 0.
    # So 0-1-21-...-39-20-0 is a cycle of length 1+1+19+1 = 22?
    # Vertices: 0, 1, 21, ..., 39, 20.
    # Count: 1 (0) + 1 (1) + 19 (21..39) + 1 (20) = 22.
    
    # To get 40, we need to include all vertices.
    # We need a path in Rim 1 that doesn't use 1?
    # Or a path in Rim 1 that uses 0 and 1?
    
    # If we use 0-19-18-...-2-1-21-...-39-20-0?
    # 0-19 (Rim), 19-18...2-1 (Rim), 1-21 (Chord), 21-22...39-20 (Rim), 20-0 (Chord).
    # Vertices: 0, 19, ..., 2, 1, 21, ..., 39, 20.
    # All 40 vertices.
    # Edges:
    # (0, 19) - Rim 1
    # (19, 18) ... (2, 1) - Rim 1
    # (1, 21) - Chord
    # (21, 22) ... (39, 20) - Rim 2
    # (20, 0) - Chord
    # This is a valid Hamiltonian cycle.
    
    cycles[40] = [0] + list(range(19, 0, -1)) + list(range(21, 40)) + [20]
    
    # Verify all 3..40
    for L in range(3, 41):
        if L not in cycles:
            # Fallback or Error
            pass
            
    final_cycles = [cycles[L] for L in range(3, 41)]
    
    return {'edges': edge_list, 'cycles': final_cycles}

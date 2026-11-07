import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency set for O(1) edge lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    # The graph is 6-regular. Max edge-disjoint Hamiltonian cycles is 6/2 = 3.
    # We construct the 3 cycles using steps 1, 2, and 5 in the circulant structure.
    # Step k generates cycle: 0, k, 2k, ..., (n-1)k mod n
    # We verify these are valid host edges.
    
    steps = [1, 2, 5]
    cycles = []
    used_edges = set()
    
    for k in steps:
        if time.time() > deadline:
            break
            
        cyc = [0]
        cur = 0
        valid = True
        
        for _ in range(n - 1):
            cur = (cur + k) % n
            # Check if edge (cyc[-1], cur) exists in host graph
            u = cyc[-1]
            v = cur
            edge_key = (min(u, v), max(u, v))
            
            if edge_key not in edge_set:
                valid = False
                break
                
            # Check edge disjointness
            if edge_key in used_edges:
                valid = False
                break
                
            used_edges.add(edge_key)
            cyc.append(cur)
            
        # Check closing edge (last to first)
        if valid:
            u = cyc[-1]
            v = cyc[0]
            edge_key = (min(u, v), max(u, v))
            if edge_key not in edge_set:
                valid = False
            elif edge_key in used_edges:
                valid = False
            else:
                used_edges.add(edge_key)
                
        if valid and len(cyc) == n:
            # Ensure all vertices are distinct (Hamiltonian)
            if len(set(cyc)) == n:
                cycles.append(cyc)
    
    # If we found 3 valid cycles, we have reached the maximum.
    # If not, we return what we have (though for this specific graph, it should be 3).
    
    # Final conversion to ensure strict types
    result = []
    for cyc in cycles:
        result.append([int(v) for v in cyc])
        
    return result
# EVOLVE-BLOCK-END

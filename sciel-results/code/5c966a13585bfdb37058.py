import itertools

def is_q3_free(graph_edges, n):
    # Check if graph on n vertices is Q3-free
    # For n=8, we can check all 8-sets (just one)
    if n < 8:
        return True
    
    # For n=8, check if it contains Q3
    # We use the bipartite matching check
    # But for n>8, we need to check all 8-subsets
    
    # This function is for the small helper
    pass

def max_edges_q3_free(n):
    if n < 8:
        return n*(n-1)//2
    
    # Start with complete graph
    edges = set((i, j) for i in range(n) for j in range(i+1, n))
    
    # Remove edges to destroy Q3
    # For n=8, we can just iterate
    # This is a simple greedy removal
    changed = True
    while changed:
        changed = False
        # Find a Q3
        # For n=8, just check the whole graph
        # For simplicity, we only use this for n=8 in the partition
        if n == 8:
            # Check if Q3 exists
            # Try all partitions of 4 and 4
            found_q3 = False
            for A in itertools.combinations(range(8), 4):
                B = tuple(x for x in range(8) if x not in A)
                # Check if induced bipartite graph contains Q3
                # Build 4x4 matrix
                mat = [[0]*4 for _ in range(4)]
                for i, u in enumerate(A):
                    for j, v in enumerate(B):
                        if (u, v) in edges or (v, u) in edges:
                            mat[i][j] = 1
                
                # Check if complement has perfect matching
                comp = [[1-mat[i][j] for j in range(4)] for i in range(4)]
                # Check all 24 permutations
                for perm in itertools.permutations(range(4)):
                    if all(comp[i][perm[i]] == 1 for i in range(4)):
                        found_q3 = True
                        # Remove an edge from the Q3
                        # The Q3 is K_4,4 minus the matching perm
                        # So present edges are those NOT in the matching
                        # Pick one present edge
                        for i in range(4):
                            for j in range(4):
                                if perm[i] != j and mat[i][j] == 1:
                                    u = A[i]
                                    v = B[j]
                                    edges.discard((u, v))
                                    edges.discard((v, u))
                                    changed = True
                                    break
                            if changed:
                                break
                        if changed:
                            break
                if changed:
                    break
            if found_q3 and not changed:
                # This shouldn't happen if we break correctly
                pass
        else:
            # For n>8, this is complex. We only use n=8 in the partition.
            pass
            
    return len(edges)

def construct():
    # Try partitions of 16
    best_edges = 0
    best_graph = []
    
    # Possible part sizes
    # We assume parts are independent
    # Try 7+7+2, 7+6+3, 8+8 (but 8 needs optimization)
    
    # Let's just try to build the graph for 7+7+2
    # K7 U K7 U K2
    edges = []
    # K7 on 0-6
    for i in range(7):
        for j in range(i+1, 7):
            edges.append([i, j])
    # K7 on 7-13
    for i in range(7, 14):
        for j in range(i+1, 14):
            edges.append([i, j])
    # K2 on 14-15
    edges.append([14, 15])
    
    # This gives 43 edges.
    
    # Can we do better with an 8-vertex component?
    # Let's find the max edges for an 8-vertex Q3-free graph
    # We'll run the greedy removal on K8
    e8 = set((i, j) for i in range(8) for j in range(i+1, 8))
    changed = True
    while changed:
        changed = False
        for A in itertools.combinations(range(8), 4):
            B = tuple(x for x in range(8) if x not in A)
            mat = [[0]*4 for _ in range(4)]
            for i, u in enumerate(A):
                for j, v in enumerate(B):
                    if (u, v) in e8 or (v, u) in e8:
                        mat[i][j] = 1
            comp = [[1-mat[i][j] for j in range(4)] for i in range(4)]
            for perm in itertools.permutations(range(4)):
                if all(comp[i][perm[i]] == 1 for i in range(4)):
                    # Found Q3
                    for i in range(4):
                        for j in range(4):
                            if perm[i] != j and mat[i][j] == 1:
                                u = A[i]
                                v = B[j]
                                e8.discard((u, v))
                                e8.discard((v, u))
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
            if changed:
                break
    
    max_e8 = len(e8)
    # Partition 8+8
    # We need two disjoint 8-vertex Q3-free graphs
    # Since they are isomorphic to e8, we can just use e8 twice
    # But we need to map them to 0-7 and 8-15
    edges_8_8 = []
    for u, v in e8:
        edges_8_8.append([u, v])
        edges_8_8.append([u+8, v+8])
    
    if len(edges_8_8) > len(edges):
        edges = edges_8_8
        
    return edges

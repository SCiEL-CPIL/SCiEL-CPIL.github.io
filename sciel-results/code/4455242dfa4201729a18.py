import time
import random

def has_c5(adj_matrix, n):
    """Check if graph with adj matrix has a 5-cycle."""
    # A 5-cycle is a sequence v0, v1, v2, v3, v4 such that
    # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v0) are edges.
    # We can iterate over all combinations of 5 vertices? No, too slow.
    # Iterate over all pairs (u,v) as start edge?
    # Simpler: use networkx if available? No, stdlib only.
    # For small n, we can just check all permutations of 5 distinct vertices.
    # P(9,5) = 15120. Very fast.
    import itertools
    for comb in itertools.permutations(range(n), 5):
        v0, v1, v2, v3, v4 = comb
        if (adj_matrix[v0][v1] and adj_matrix[v1][v2] and 
            adj_matrix[v2][v3] and adj_matrix[v3][v4] and 
            adj_matrix[v4][v0]):
            return True
    return False

def has_independent_set(adj_matrix, n, k):
    """Check if graph has an independent set of size k."""
    # For k=3, check all triplets.
    import itertools
    for comb in itertools.combinations(range(n), k):
        is_indep = True
        for i in range(len(comb)):
            for j in range(i+1, len(comb)):
                if adj_matrix[comb[i]][comb[j]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known good graph: 8 K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        offset = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([offset + i, offset + j])
    
    # Try to find a C5-free graph on 9 vertices with alpha <= 2
    # If found, we can make a graph with 6 K4s + this graph (n = 24 + 9 = 33)
    # Or 5 K4s + 2 such graphs (n = 20 + 18 = 38? No, alpha would be 5+4=9. Invalid.)
    # So we can only replace pairs of K4s (alpha=2) with one graph (alpha=2).
    # Replacing 2 K4s (n=8) with 1 graph (n=9) increases n by 1.
    # We have 4 pairs of K4s in 8 K4s.
    # So max n could be 32 + 4 = 36?
    # But we need to check if we can pack 4 such graphs.
    
    target_n = 9
    found_graph = None
    
    # Random search for the component
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Generate random graph on 9 vertices
        adj = [[0]*target_n for _ in range(target_n)]
        for i in range(target_n):
            for j in range(i+1, target_n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        if has_c5(adj, target_n):
            continue
        if has_independent_set(adj, target_n, 3):
            continue
        
        # Found a valid component
        found_graph = adj
        break
        
    if found_graph:
        # Construct the new graph: 6 K4s + found_graph
        # 6 K4s use vertices 0..23
        # found_graph uses vertices 24..32
        n = 24 + target_n
        edges = []
        
        # Add 6 K4s
        for b in range(6):
            offset = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([offset + i, offset + j])
        
        # Add edges from found_graph
        offset = 24
        for i in range(target_n):
            for j in range(i+1, target_n):
                if found_graph[i][j]:
                    edges.append([offset + i, offset + j])
                    
        # Verify
        # Check C5 and Alpha for the whole graph
        # Since components are disjoint, C5 is in one component.
        # Alpha is sum of alphas.
        # Alpha of 6 K4s is 6.
        # Alpha of found_graph is <= 2.
        # Total Alpha <= 8.
        # C5 is not in K4. C5 is not in found_graph.
        # So it is valid.
        
        if n > best_n:
            best_n = n
            best_edges = edges
            
    return {'n': best_n, 'edges': best_edges}

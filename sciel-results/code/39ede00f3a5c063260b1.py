import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_vertices = list(range(n))
    
    # Precompute 8-vertex subsets
    subsets_8 = list(combinations(all_vertices, 8))
    
    # Precompute bipartitions for a set of 8 vertices
    local_partitions = []
    for u in combinations(range(8), 4):
        v = tuple(x for x in range(8) if x not in u)
        local_partitions.append((u, v))
    # Deduplicate
    unique_local_partitions = set()
    for u, v in local_partitions:
        set_u = frozenset(u)
        set_v = frozenset(v)
        if set_u > set_v:
            set_u, set_v = set_v, set_u
        unique_local_partitions.add((set_u, set_v))
    
    final_local_partitions = []
    for su, sv in unique_local_partitions:
        u_list = [i for i in range(8) if i in su]
        v_list = [i for i in range(8) if i in sv]
        final_local_partitions.append((u_list, v_list))
        
    def contains_q3(adj_matrix):
        """
        adj_matrix: 16x16 list of lists with 0/1
        Returns True if Q3 is present
        """
        for S in subsets_8:
            # Count edges in induced subgraph first
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if adj_matrix[S[i]][S[j]]:
                        edge_count += 1
            if edge_count < 12:
                continue
                
            # Check bipartitions
            for u_list, v_list in final_local_partitions:
                row_zeros = [0]*4
                col_zeros = [0]*4
                valid = True
                for i in range(4):
                    for j in range(4):
                        if adj_matrix[S[u_list[i]]][S[v_list[j]]] == 0:
                            row_zeros[i] += 1
                            col_zeros[j] += 1
                            if row_zeros[i] > 1 or col_zeros[j] > 1:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    return True
        return False
    
    # Build the Shrikhande graph complement
    # The Shrikhande graph is a strongly regular graph (16,6,2,2)
    # Its complement is (16,9,4,4) - each vertex has degree 9
    # 
    # One construction: vertices are (i,j) where i,j in {0,1,2,3}
    # Two vertices (i,j) and (k,l) are adjacent in the Shrikhande graph if:
    # - i=k and j!=l (same row, different column) - 3 edges
    # - j=l and i!=k (same column, different row) - 3 edges  
    # - (i-k, j-l) is in {(1,1),(1,2),(2,1),(2,2)} mod 4 - 4 edges
    # Total: 6 neighbors
    #
    # Actually, let me use a known adjacency list for the Shrikhande graph
    # Standard Shrikhande graph adjacency:
    shrikhande_adj = {
        0: [1, 2, 4, 5, 10, 13],
        1: [0, 3, 4, 6, 9, 12],
        2: [0, 3, 5, 7, 8, 11],
        3: [1, 2, 6, 7, 9, 10],
        4: [0, 1, 8, 9, 12, 15],
        5: [0, 2, 8, 10, 11, 14],
        6: [1, 3, 11, 12, 13, 15],
        7: [2, 3, 11, 13, 14, 15],
        8: [2, 4, 5, 10, 12, 14],
        9: [1, 4, 6, 10, 13, 15],
        10: [0, 3, 5, 8, 9, 11],
        11: [2, 5, 6, 7, 10, 14],
        12: [1, 4, 6, 8, 13, 15],
        13: [0, 6, 7, 9, 12, 14],
        14: [5, 7, 8, 11, 13, 15],
        15: [4, 6, 7, 9, 12, 14]
    }
    
    # Complement: edge exists if NOT in Shrikhande
    adj = [[0]*n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if v not in shrikhande_adj[u]:
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    
    # Verify the complement is Q3-free
    if contains_q3(best_adj):
        # Fall back to K3,13 if Shrikhande complement has Q3
        adj = [[0]*n for _ in range(n)]
        edges = []
        for a in range(3):
            for b in range(3, 16):
                adj[a][b] = 1
                adj[b][a] = 1
                edges.append([a, b])
        best_edges = len(edges)
        best_adj = [row[:] for row in adj]
    
    # List of all possible edges not in current graph
    potential_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] == 0:
                potential_edges.append((u, v))
    
    # Greedy search with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.shuffle(potential_edges)
        
        current_adj = [row[:] for row in best_adj]
        current_edges = best_edges
        
        for u, v in potential_edges:
            if current_adj[u][v] == 1:
                continue
            # Try adding edge
            current_adj[u][v] = 1
            current_adj[v][u] = 1
            
            if not contains_q3(current_adj):
                current_edges += 1
            else:
                # Revert
                current_adj[u][v] = 0
                current_adj[v][u] = 0
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in current_adj]
        
        # Local search: try edge swaps
        # For each edge in the graph, try removing it and adding a non-edge
        if time.time() < deadline - 10:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try removing each edge
                for u in range(n):
                    if time.time() > deadline - 10:
                        break
                    for v in range(u+1, n):
                        if time.time() > deadline - 10:
                            break
                        if best_adj[u][v] == 1:
                            # Try removing this edge
                            test_adj = [row[:] for row in best_adj]
                            test_adj[u][v] = 0
                            test_adj[v][u] = 0
                            
                            # Try adding each non-edge
                            for x in range(n):
                                if time.time() > deadline - 10:
                                    break
                                for y in range(x+1, n):
                                    if time.time() > deadline - 10:
                                        break
                                    if test_adj[x][y] == 0 and (x,y) != (u,v) and (x,y) != (v,u):
                                        # Check if adding (x,y) to test_adj creates Q3
                                        test_adj[x][y] = 1
                                        test_adj[y][x] = 1
                                        
                                        if not contains_q3(test_adj):
                                            # This swap is valid
                                            best_adj = test_adj
                                            best_edges = len([1 for i in range(n) for j in range(i+1, n) if best_adj[i][j]])
                                            improved = True
                                            break
                                        else:
                                            test_adj[x][y] = 0
                                            test_adj[y][x] = 0
                                    
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break
        
        if best_edges > 55:
            pass

    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] == 1:
                result_edges.append([u, v])
                
    return result_edges

import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    n = 16
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def check_q3(adj_matrix, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adj_matrix 
        creates a Q3 subgraph.
        adj_matrix is a list of sets, or a 2D boolean array. 
        Here we use a list of sets for fast lookup, but for speed in the loop, 
        a bitset or boolean matrix might be better. 
        Given n=16, a 16x16 boolean list is very fast.
        """
        # We assume (u, v) is already added to adj_matrix for this check
        # Or we can check without adding, but adding is easier for neighbor lookup.
        # Let's assume adj_matrix[u][v] is True.
        
        # Neighbors of u excluding v
        nbrs_u = [x for x in range(n) if adj_matrix[u][x] and x != v]
        # Neighbors of v excluding u
        nbrs_v = [y for y in range(n) if adj_matrix[v][y] and y != u]
        
        # In Q3, u=0, v=1.
        # Nbrs of 0: {1, 2, 4}. So we need 2 neighbors from nbrs_u to map to 2 and 4.
        # Nbrs of 1: {0, 3, 5}. So we need 2 neighbors from nbrs_v to map to 3 and 5.
        
        # Iterate over pairs in nbrs_u
        for i1 in range(len(nbrs_u)):
            a = nbrs_u[i1]
            for i2 in range(i1 + 1, len(nbrs_u)):
                b = nbrs_u[i2]
                
                # In Q3, 2 and 4 are not adjacent.
                if adj_matrix[a][b]:
                    continue
                
                # Iterate over pairs in nbrs_v
                for j1 in range(len(nbrs_v)):
                    c = nbrs_v[j1]
                    for j2 in range(j1 + 1, len(nbrs_v)):
                        d = nbrs_v[j2]
                        
                        # In Q3, 3 and 5 are not adjacent.
                        if adj_matrix[c][d]:
                            continue
                        
                        # Check cross connections.
                        # Case 1: a->2, b->4, c->3, d->5
                        # Edges required: (a,c), (b,d). Non-edges: (a,d), (b,c)
                        if adj_matrix[a][c] and adj_matrix[b][d] and not adj_matrix[a][d] and not adj_matrix[b][c]:
                            # Found potential 4-cycle part. Now find e (maps to 6) and f (maps to 7)
                            # e must be connected to a (2), b (4), and f (7)
                            # f must be connected to c (3), d (5), and e (6)
                            # e must NOT be connected to c, d
                            # f must NOT be connected to a, b
                            
                            # Candidates for e: neighbors of a and b, excluding u,v,c,d,a,b
                            # Actually, e is a vertex.
                            # Let's iterate over all vertices for e and f? 
                            # n=16, so 16*16 = 256 checks. Acceptable if this inner loop is rarely hit.
                            # But we can be smarter.
                            # e must be in N(a) ∩ N(b).
                            # f must be in N(c) ∩ N(d).
                            
                            candidates_e = [x for x in range(n) if adj_matrix[a][x] and adj_matrix[b][x] and x not in (u, v, a, b, c, d)]
                            candidates_f = [y for y in range(n) if adj_matrix[c][y] and adj_matrix[d][y] and y not in (u, v, a, b, c, d)]
                            
                            for e in candidates_e:
                                if adj_matrix[e][c] or adj_matrix[e][d]: # e must not connect to c,d
                                    continue
                                for f in candidates_f:
                                    if f == e:
                                        continue
                                    if adj_matrix[f][a] or adj_matrix[f][b]: # f must not connect to a,b
                                        continue
                                    if adj_matrix[e][f]: # e and f must be connected
                                        return True
                        # Case 2: a->2, b->4, c->5, d->3
                        # Edges required: (a,d), (b,c). Non-edges: (a,c), (b,d)
                        elif adj_matrix[a][d] and adj_matrix[b][c] and not adj_matrix[a][c] and not adj_matrix[b][d]:
                            candidates_e = [x for x in range(n) if adj_matrix[a][x] and adj_matrix[b][x] and x not in (u, v, a, b, c, d)]
                            candidates_f = [y for y in range(n) if adj_matrix[c][y] and adj_matrix[d][y] and y not in (u, v, a, b, c, d)]
                            
                            for e in candidates_e:
                                if adj_matrix[e][c] or adj_matrix[e][d]:
                                    continue
                                for f in candidates_f:
                                    if f == e:
                                        continue
                                    if adj_matrix[f][a] or adj_matrix[f][b]:
                                        continue
                                    if adj_matrix[e][f]:
                                        return True
        return False

    best_edges = 0
    best_adj = None
    best_edge_list = []
    
    while time.time() - start_time < DEADLINE:
        # Initialize empty graph
        adj = [[False] * n for _ in range(n)]
        current_edges = 0
        
        # Shuffle edges
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() - start_time > DEADLINE:
                break
            
            # Tentatively add edge
            adj[u][v] = True
            adj[v][u] = True
            
            if not check_q3(adj, u, v):
                current_edges += 1
            else:
                # Remove edge
                adj[u][v] = False
                adj[v][u] = False
        
        if current_edges > best_edges:
            best_edges = current_edges
            # Reconstruct edge list
            best_edge_list = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            # We don't need to store best_adj, just the list
            
    return best_edge_list


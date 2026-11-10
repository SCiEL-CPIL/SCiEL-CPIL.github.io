import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 16
    
    # Initialize best result
    best_edges = []
    best_count = 0
    
    # Pre-generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def creates_q3(u, v, adj_sets):
        # adj_sets is a list of sets
        # Check if adding edge (u, v) creates a Q3 subgraph.
        # This is done by checking if there is an embedding of Q3 \ {(0,1)} 
        # into the current graph, mapping 0->u and 1->v.
        
        # Neighbors of u and v in the current graph (excluding each other)
        n_u = adj_sets[u] - {v}
        n_v = adj_sets[v] - {u}
        
        # Iterate over possible mappings for vertices 2 and 4 (neighbors of 0)
        for m2 in n_u:
            for m4 in n_u:
                if m2 == m4:
                    continue
                
                # Vertex 6 is connected to 2 and 4
                # Find candidates for m6 in intersection of neighbors of m2 and m4
                common_24 = adj_sets[m2] & adj_sets[m4]
                # Exclude u, v, m2, m4
                common_24.difference_update({u, v, m2, m4})
                
                if not common_24:
                    continue
                
                for m6 in common_24:
                    # Vertex 3 is connected to 1 (v) and 2 (m2)
                    # Find candidates for m3 in intersection of neighbors of v and m2
                    common_v2 = adj_sets[v] & adj_sets[m2]
                    # Exclude u, v, m2, m4, m6
                    common_v2.difference_update({u, v, m2, m4, m6})
                    
                    if not common_v2:
                        continue
                        
                    for m3 in common_v2:
                        # Vertex 5 is connected to 1 (v) and 4 (m4)
                        # Find candidates for m5 in intersection of neighbors of v and m4
                        common_v4 = adj_sets[v] & adj_sets[m4]
                        # Exclude u, v, m2, m3, m4, m6
                        common_v4.difference_update({u, v, m2, m3, m4, m6})
                        
                        if not common_v4:
                            continue
                            
                        for m5 in common_v4:
                            if m5 == m3:
                                continue
                            
                            # Check edges (m2, m3) and (m4, m5)
                            # (m2, m3) is guaranteed by m3 in adj_sets[m2]
                            # (m4, m5) is guaranteed by m5 in adj_sets[m4]
                            
                            # Vertex 7 is connected to 3, 5, and 6
                            common_356 = adj_sets[m3] & adj_sets[m5] & adj_sets[m6]
                            # Exclude all previously used vertices
                            common_356.difference_update({u, v, m2, m3, m4, m5, m6})
                            
                            if common_356:
                                return True
        return False

    # Main search loop with random restarts
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj_sets = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            
            # Skip if edge already exists (shouldn't happen with unique list, but safe)
            if v in adj_sets[u]:
                continue
                
            # Check if adding this edge creates a Q3
            if not creates_q3(u, v, adj_sets):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                current_edges.append([u, v])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]

    return best_edges

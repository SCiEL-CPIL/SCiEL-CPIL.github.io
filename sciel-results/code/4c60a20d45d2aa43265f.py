"""Greedy randomized construction for 5-coloring K_n without C4 in any color class."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try multiple random seeds for the greedy construction
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        n = 1
        # colors[i][j] for i < j, stored in a list indexed by (i*n + j) or similar
        # We'll use a 2D list approach: colors[i][j] where i < j
        # Actually, let's use a flat list in lexicographic order
        edge_colors = {}  # (i, j) -> color
        
        # Track for each color c, and each pair of vertices (u, v), 
        # the number of common neighbors in color c.
        # If any pair (u,v) has >= 2 common neighbors in color c, we have a C4.
        # We maintain: for each color c, common_neighbors[c][u][v] = count
        # This is expensive for large n. Let's think differently.
        
        # Alternative: when adding vertex k, for each color c, 
        # the new edges are (k, j) for j in 0..k-1.
        # A C4 in color c would be formed if there exist u, v < k such that
        # edge (u,v) is color c AND edges (k,u) and (k,v) are both color c
        # AND there's already a common neighbor w of u and v in color c.
        # Actually, a C4 is u-v-w-x-u. If we add vertex k, a new C4 in color c
        # involving k would be k-a-b-c-k where a,b,c < k and edges (k,a),(a,b),(b,c),(c,k) are color c.
        # This means a and c are both neighbors of k in color c, and a and c have a common neighbor b in color c.
        
        # So when assigning color c to edge (k, a), we need to check:
        # For each already-colored edge (k, v) with color c (v < a or v > a),
        # check if a and v have a common neighbor in color c.
        # If yes, then assigning color c to (k,a) would create a C4.
        
        # To make this efficient, for each color c, maintain:
        # neighbors_c[u] = set of vertices adjacent to u in color c
        # Then a and v have a common neighbor in color c iff neighbors_c[a] & neighbors_c[v] is non-empty.
        
        # Let's try building up to n=29
        for target_n in range(2, 30):
            if time.time() > deadline:
                break
            
            # Try to add vertex target_n-1
            success = False
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Initialize neighbors for the new vertex
                new_neighbors = {c: set() for c in range(5)}
                new_edges = {}  # j -> color for edge (target_n-1, j)
                
                # For each existing vertex j, try to assign a color
                # We'll do this greedily: for each j, find colors that don't create a C4
                valid = True
                for j in range(target_n - 1):
                    # For each color c, check if assigning color c to edge (target_n-1, j)
                    # would create a C4 in color c.
                    # A C4 would be formed if there exists v in new_neighbors[c] such that
                    # j and v have a common neighbor in color c (in the existing graph).
                    # i.e., neighbors_c[j] & neighbors_c[v] is non-empty for some v in new_neighbors[c].
                    
                    valid_colors = []
                    for c in range(5):
                        can_use = True
                        for v in new_neighbors[c]:
                            # Check if j and v share a common neighbor in color c
                            # This means there exists w such that (w,j) is color c and (w,v) is color c
                            # i.e., w is in neighbors_c[j] and w is in neighbors_c[v]
                            # We need to check if neighbors_c[j] & neighbors_c[v] is non-empty
                            # But we don't have neighbors_c stored globally yet...
                            # 
                            # Alternative approach: maintain for each color c, a dict
                            # common_c[a][b] = number of common neighbors of a and b in color c
                            # When we add edge (u,v) in color c, we update:
                            # for each w in neighbors_c[u]: common_c[w][v] += 1
                            # for each w in neighbors_c[v]: common_c[w][u] += 1
                            # A C4 is formed if common_c[j][v] >= 1 for some v in new_neighbors[c]
                            # Wait, that's not quite right either.
                            # 
                            # Let me reconsider. A C4 in color c is a cycle of length 4.
                            # If we're adding vertex k = target_n-1, and we've already assigned
                            # colors to edges (k, v1), (k, v2), ..., (k, vm) in color c,
                            # then a C4 is formed if there exist a, b in {v1,...,vm} such that
                            # a and b have a common neighbor in color c (in the subgraph on vertices 0..k-1).
                            # 
                            # So I need to track, for the existing graph (vertices 0..k-1),
                            # for each color c, which pairs of vertices share a common neighbor in color c.
                            # Let's call this: forbidden_c[a][b] = True if a and b have >= 1 common neighbor in color c.
                            # Actually, we need >= 1 to form a C4 when both are connected to k in color c.
                            # 
                            # This is getting complex. Let me use a simpler but slower approach for now,
                            # and optimize if needed.
                            
                            # Simple check: for color c, if j is assigned color c to edge (k,j),
                            # then for each v already assigned color c to edge (k,v),
                            # we need to check if there exists w < k such that
                            # edge (w,j) is color c and edge (w,v) is color c.
                            # This is a direct search.
                            
                            # For efficiency, let's precompute for the existing graph:
                            # adj_c[u] = list of vertices adjacent to u in color c
                            # Then j and v have a common neighbor in color c iff
                            # set(adj_c[j]) & set(adj_c[v]) is non-empty.
                            # 
                            # But maintaining adj_c for all c and all u is O(n^2 * 5) memory.
                            # For n=29, that's fine.
                            
                            # Let me restructure: maintain adj = [adj_c for c in range(5)]
                            # where adj_c is a list of sets, adj_c[u] = set of neighbors of u in color c.
                            
                            # I'll restructure the whole thing.
                            pass
                    
                    # This approach is getting too complex inline. Let me restart with a cleaner design.
                    valid = False
                    break
                
                if not valid:
                    continue
                
                # If we got here, we successfully assigned all edges for vertex k
                # Update the data structures
                for j, c in new_edges.items():
                    adj[c][k].add(j)
                    adj[c][j].add(k)
                edge_colors[(target_n - 1, )] = new_edges  # not quite right
                success = True
                break
            
            if not success:
                break
        
        # Extract result
        if n > best_n:
            best_n = n
            # Reconstruct the color list
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(edge_colors[(i, j)])
            best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}

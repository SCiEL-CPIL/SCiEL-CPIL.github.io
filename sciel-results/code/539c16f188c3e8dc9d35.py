import random
import itertools

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    # Run multiple random trials to find a good maximal K3,3-free graph
    for trial in range(20):
        random.seed(trial)
        edges = []
        edge_set = set()
        
        # List of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a K3,3
            # We need to check if there exist sets A={u, a1, a2} and B={v, b1, b2}
            # such that all edges between A and B exist in the current graph plus (u,v).
            # Since (u,v) is the new edge, we need:
            # 1. (u,v) is added.
            # 2. For all a in {a1, a2} and b in {b1, b2}, (a,b) is in edge_set.
            # 3. Also, the edges (u, b1), (u, b2), (a1, v), (a2, v) must be in edge_set.
            
            is_safe = True
            # Get neighbors of u and v in the current graph
            # Note: (u,v) is not in edge_set yet.
            
            # We need to find a1, a2 in N(u) and b1, b2 in N(v)
            # such that:
            # - (u, b1) and (u, b2) are edges (b1, b2 in N(u))
            # - (a1, v) and (a2, v) are edges (a1, a2 in N(v))
            # - (a1, b1), (a1, b2), (a2, b1), (a2, b2) are edges
            
            # Let Nu = neighbors of u
            # Let Nv = neighbors of v
            # We need b1, b2 in Nv such that b1, b2 are also in Nu? No.
            # b1, b2 must be neighbors of u. So b1, b2 in N(u) ∩ N(v)? No.
            # b1, b2 are in N(v) (to connect to v).
            # b1, b2 must be connected to u. So b1, b2 in N(u).
            # So b1, b2 must be in N(u) ∩ N(v).
            # Similarly, a1, a2 must be in N(u) ∩ N(v).
            
            common_neighbors = []
            for w in range(n):
                if w != u and w != v:
                    if (u, w) in edge_set or (w, u) in edge_set:
                        if (v, w) in edge_set or (w, v) in edge_set:
                            common_neighbors.append(w)
                            
            # We need to choose 2 from common_neighbors for B-side (b1, b2)
            # and 2 from common_neighbors for A-side (a1, a2)?
            # Wait.
            # If b1 is in B, it connects to u, a1, a2.
            # So b1 must be connected to u.
            # a1 is in A, it connects to v, b1, b2.
            # So a1 must be connected to v.
            # Thus a1, a2, b1, b2 must all be common neighbors of u and v.
            
            # So we need to find 4 distinct vertices from common_neighbors
            # a1, a2, b1, b2 such that:
            # (a1, b1), (a1, b2), (a2, b1), (a2, b2) are in edge_set.
            
            # If len(common_neighbors) < 4, it's safe.
            if len(common_neighbors) >= 4:
                # Check all pairs of pairs
                # This is expensive if common_neighbors is large, but n=20 is small.
                # Max common neighbors is 18.
                # C(18, 2) * C(16, 2) is too big?
                # We can iterate over pairs for A and pairs for B.
                # But A and B must be disjoint? Yes, vertices in K3,3 are distinct.
                
                found_k33 = False
                # To optimize, we can just check if there's any such configuration.
                # Given the small size, we can just check all combinations of 4 vertices?
                # No, we need to partition them into two sets of 2.
                
                # Let's just check if adding (u,v) creates a K3,3 by looking for
                # a1, a2, b1, b2 in common_neighbors such that
                # a1 != a2, b1 != b2, {a1,a2} != {b1,b2} (disjoint)
                # and edges (a1,b1), (a1,b2), (a2,b1), (a2,b2) exist.
                
                # We can iterate over all pairs (a1, a2) and (b1, b2) from common_neighbors.
                # To avoid duplicates and ensure disjointness:
                # Pick a1, a2 from common_neighbors.
                # Pick b1, b2 from common_neighbors \ {a1, a2}.
                
                cn = common_neighbors
                for a1, a2 in itertools.combinations(cn, 2):
                    remaining = [x for x in cn if x != a1 and x != a2]
                    for b1, b2 in itertools.combinations(remaining, 2):
                        # Check edges between {a1, a2} and {b1, b2}
                        if ((a1, b1) in edge_set or (b1, a1) in edge_set) and \
                           ((a1, b2) in edge_set or (b2, a1) in edge_set) and \
                           ((a2, b1) in edge_set or (b1, a2) in edge_set) and \
                           ((a2, b2) in edge_set or (b2, a2) in edge_set):
                            found_k33 = True
                            break
                    if found_k33:
                        break
                if found_k33:
                    is_safe = False
                    
            if is_safe:
                edges.append([u, v])
                edge_set.add((u, v))
                edge_set.add((v, u))
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return best_edges

import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 16
    adj = [[0] * n for _ in range(n)]
    edges = set()

    def add_edge(u, v):
        if u == v:
            return
        if adj[u][v]:
            return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not adj[u][v]:
            return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))

    # Q3 is isomorphic to K_{4,4} minus a perfect matching.
    # A graph G contains Q3 if and only if there exists a set of 8 vertices S and a bipartition (A, B) of S
    # such that the missing edges between A and B in G form a matching (i.e., are a subset of a perfect matching).

    # Precompute all 35 bipartitions of 8 items into two sets of 4.
    # Represent as (A_mask, B_mask) where masks are 8-bit integers.
    # To avoid duplicates (A,B) and (B,A), we can just take A < B in some ordering, 
    # but for the check we just need to iterate unique splits.
    # There are C(8,4)/2 = 35 unique splits.
    
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        # Canonical representation: min(A) < min(B) or just use frozenset of frozensets
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            # Create bitmasks
            mask_a = 0
            mask_b = 0
            for i in A:
                mask_a |= (1 << i)
            for i in B:
                mask_b |= (1 << i)
            all_splits.append((mask_a, mask_b, A, B))
    
    # all_splits now has 35 entries.

    def contains_q3_in_subset(v_list):
        """
        v_list: tuple of 8 vertex IDs.
        Returns True if the induced subgraph on v_list contains a Q3.
        """
        # Precompute adjacency between these 8 vertices as a bitset or small matrix?
        # Since we call this often, let's optimize the inner loop.
        
        # For each split (A_idx, B_idx):
        #   Check if missing edges form a matching.
        
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            # A_idx and B_idx are tuples of indices into v_list (0..7)
            # We need to check edges between v_list[i] for i in A_idx and v_list[j] for j in B_idx.
            
            # If the number of missing edges is > 4, it can't be a matching (max size of matching in K4,4 is 4).
            # Actually, if missing edges share a vertex, it's not a matching.
            # A matching in K_{4,4} has at most 4 edges.
            # So if missing_count > 4, skip.
            
            missing_edges = []
            # Also track vertices involved in missing edges to check matching property quickly
            
            # Fast check:
            # For each i in A_idx:
            #   For each j in B_idx:
            #     if not adj[v_list[i]][v_list[j]]:
            #        missing_edges.append((i, j))
            
            # Optimization: Precompute a bitmask of neighbors in v_list for each vertex in v_list?
            # Let's just do the direct check but break early if we find a conflict.
            
            # To check if missing edges form a matching:
            # Maintain a set of vertices that are "covered" by a missing edge.
            # If we encounter a missing edge (i, j) where i or j is already covered, it's not a matching.
            
            covered = 0 # bitmask of indices in v_list (0..7) that are part of a missing edge
            
            is_matching = True
            missing_count = 0
            
            for i in A_idx:
                vi = v_list[i]
                for j in B_idx:
                    vj = v_list[j]
                    if not adj[vi][vj]:
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        # Check if i or j is already covered
                        # i is in A, j is in B. They are distinct indices in 0..7.
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            
            if is_matching:
                return True
                
        return False

    # Initialize with a good seed
    # K3,13 structure:
    # Vertices 0,1,2 form a triangle.
    # Vertices 3..15 are connected to 0,1,2.
    # No other edges.
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    # This has 3*13 + 3 = 42 edges.
    # Wait, is K3,13 + K3 Q3-free?
    # Q3 is bipartite. K3,13 + K3 contains triangles.
    # Does it contain Q3?
    # Q3 has no triangles. But a subgraph isomorphic to Q3 doesn't need to be induced.
    # However, Q3 is bipartite. Any subgraph of a bipartite graph is bipartite.
    # If G contains Q3, G must have a bipartite subgraph isomorphic to Q3.
    # In K3,13 + K3, the bipartite parts would have to be subsets of the vertices.
    # The vertices 0,1,2 are all connected to each other.
    # If we pick a Q3, it has 8 vertices.
    # Can we find 8 vertices in K3,13+K3 that contain Q3?
    # The vertices in the "independent set" part (3..15) have no edges between them.
    # So any edge in the Q3 must be incident to at least one of {0,1,2}.
    # In Q3, every vertex has degree 3.
    # If we map 0,1,2 to some vertices in Q3, they must be connected to 3 others.
    # But in G, 0,1,2 are connected to each other.
    # In Q3, no two vertices are connected by more than 1 edge, and it's triangle-free.
    # So 0,1,2 cannot form a triangle in the image of Q3.
    # Therefore, at most 2 of {0,1,2} can be in the Q3 image if they are connected in G?
    # Actually, if 0 and 1 are in the Q3 image, and they are adjacent in G, they must be adjacent in Q3.
    # If 0, 1, 2 are all in the Q3 image, they form a triangle in G.
    # But Q3 is triangle-free. So the subgraph induced by the image of Q3 in G cannot contain a triangle.
    # Thus, we cannot have 0, 1, 2 all in the Q3 image.
    # So the Q3 image must use at most 2 vertices from {0,1,2}.
    # Let S be the set of 8 vertices in the Q3 image.
    # S contains at most 2 from {0,1,2}.
    # So S contains at least 6 from {3..15}.
    # The vertices in {3..15} are independent in G.
    # In Q3, the independence number is 4.
    # So we cannot have 6 independent vertices in the Q3 image.
    # Therefore, K3,13 + K3 is Q3-free.
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    best_edge_set = set(edges)
    
    # Local Search
    # We want to add edges.
    # Strategy:
    # 1. Maintain a list of candidate missing edges.
    # 2. Pick one randomly (or by some heuristic).
    # 3. Add it.
    # 4. Check if any Q3 is created.
    #    - Only Q3s containing the new edge (u,v) can be new.
    #    - A Q3 containing (u,v) must include u and v.
    #    - We check all 8-vertex subsets containing u and v.
    #    - C(14, 6) = 3003 subsets.
    #    - For each, run contains_q3_in_subset.
    # 5. If Q3 found, remove the new edge (or remove an edge from the Q3 to break it, but simple removal is safer first).
    # 6. If no Q3, keep it.
    
    # To speed up the check, we can cache results or use a smarter filter.
    # But 3003 * 35 splits * (small constant) is ~100k ops per edge addition attempt.
    # If we do 1000 attempts, that's 100M ops. Might be tight in Python.
    # Let's optimize contains_q3_in_subset further or limit attempts.
    
    # Optimization:
    # In contains_q3_in_subset, we can precompute the adjacency matrix of the 8 vertices as a list of bitmasks.
    # adj_mask[i] is a bitmask of neighbors of v_list[i] within v_list.
    # Then checking "not adj[vi][vj]" is just checking if bit j is not set in adj_mask[i].
    
    def contains_q3_in_subset_fast(v_list):
        # Build local adjacency masks
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask
            
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0
            
            for i in A_idx:
                # Neighbors of i in B_idx
                # We want to find j in B_idx such that NOT (local_adj[i] has bit j)
                # i.e., j is in B_idx and (local_adj[i] & (1<<j)) == 0
                
                # Iterate j in B_idx
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            if is_matching:
                return True
        return False

    # Main Loop
    iterations = 0
    max_iterations = 50000
    last_improvement_time = start_time
    
    # List of all possible missing edges
    # We can just pick random u, v
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick random missing edge
        # To be more efficient, we can maintain a list of missing edges, but that's expensive to update.
        # Random sampling is fine.
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or adj[u][v]:
            continue
            
        add_edge(u, v)
        
        # Check for Q3 containing (u, v)
        # We need to check if there exists a set of 6 other vertices such that the 8-vertex set contains Q3.
        # And the Q3 must use edge (u,v).
        # Note: contains_q3_in_subset checks if the set contains *any* Q3.
        # If the set contained a Q3 before adding (u,v), it would have been invalid already.
        # So if it returns True now, it's likely a new Q3 using (u,v).
        # However, it's possible that the set contained a Q3 that didn't use (u,v) but we missed it?
        # No, we maintain a Q3-free invariant. So any Q3 found must use (u,v).
        
        found_q3 = False
        other_vertices = [i for i in range(n) if i != u and i != v]
        
        # Heuristic: Check only a subset of combinations if time is tight?
        # No, we must be exact.
        # But we can break early if found.
        
        for combo in combinations(other_vertices, 6):
            vertices = (u, v) + tuple(combo)
            if contains_q3_in_subset_fast(vertices):
                found_q3 = True
                break
                
        if found_q3:
            remove_edge(u, v)
        else:
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = [row[:] for row in adj]
                best_edge_set = set(edges)
                last_improvement_time = time.time()
                
        # Optional: If no improvement for a long time, try a different strategy?
        # For now, just keep going.
        
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))

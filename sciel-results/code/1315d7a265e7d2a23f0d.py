import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    
    n = 16
    
    # --- Precomputation ---
    
    # 1. Precompute all 8-vertex subsets
    subsets_8 = list(combinations(range(n), 8))
    
    # 2. Precompute the 35 bipartitions of 8 vertices (local indices)
    #    We represent a bipartition as a bitmask of the "A" side (4 bits set)
    #    and the "B" side (complement).
    #    We only need to store the mask for A. B is derived.
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        mask_a = 0
        for i in A:
            mask_a |= (1 << i)
        # Canonicalize: ensure the smallest element is in A to avoid duplicates (A, B) vs (B, A)
        # Actually, Q3 is bipartite. The edges are between A and B.
        # If we swap A and B, the set of edges is the same.
        # So we just need unique sets of A.
        # To be safe and fast, let's just use the mask_a.
        # We can enforce a canonical order, e.g., the vertex with index 0 must be in A.
        if (mask_a & 1) == 0:
             continue # Vertex 0 must be in A for canonical representation
        all_splits.append(mask_a)
        
    # There should be C(7,3) = 35 splits where vertex 0 is in A.
    # Let's verify count.
    # C(8,4)/2 = 35.
    # Number of subsets of size 4 containing 0 is C(7,3) = 35.
    # So this is correct.
    
    # 3. Precompute a fast check function for a specific 8-vertex subset
    #    Given an adjacency matrix (or bitmasks), check if it contains a Q3.
    
    def check_subset_q3(adj_masks, subset):
        """
        Check if the induced subgraph on 'subset' contains a Q3.
        adj_masks: list of 16 integers, where adj_masks[v] has bit u set if (u,v) is an edge.
        subset: tuple of 8 vertex indices.
        Returns True if Q3 is found.
        """
        # Map global indices to local indices 0..7
        # We need to check the 35 splits.
        # For each split (defined by local mask_a), we check if the induced subgraph
        # is Q3.
        
        # Optimization: 
        # A Q3 is 3-regular.
        # In the induced subgraph on 8 vertices, if it's a Q3, every vertex has degree 3 within the subset.
        # Let's first check if all vertices have degree 3 within the subset.
        # If any vertex has degree != 3, it's not a Q3.
        
        degrees = [0] * 8
        for i in range(8):
            v_global = subset[i]
            mask = adj_masks[v_global]
            deg = 0
            for j in range(8):
                if i != j:
                    v_other = subset[j]
                    if mask & (1 << v_other):
                        deg += 1
            degrees[i] = deg
            if deg != 3:
                return False
                
        # If all degrees are 3, the graph is 3-regular on 8 vertices.
        # Is it a Q3?
        # Q3 is the unique 3-regular bipartite graph on 8 vertices that is also a hypercube.
        # Actually, are there other 3-regular bipartite graphs on 8 vertices?
        # Yes. For example, two disjoint K_{3,3}? No, K_{3,3} has 6 vertices.
        # What about a 3-regular bipartite graph that is not Q3?
        # The number of edges is 8*3/2 = 12.
        # Q3 has girth 4.
        # Any 3-regular bipartite graph on 8 vertices with 12 edges:
        # If it has a 3-cycle? Bipartite graphs have no odd cycles. So girth >= 4.
        # If girth is 6?
        # Q3 has girth 4.
        # Are there 3-regular bipartite graphs on 8 vertices with girth 6?
        # The smallest such graph is the Petersen graph? No, Petersen is 3-regular, 10 vertices, girth 5 (not bipartite).
        # The smallest bipartite graph with girth 6 and degree 3 is the Heawood graph? 14 vertices.
        # On 8 vertices, it must have girth 4.
        # So any 3-regular bipartite graph on 8 vertices is Q3?
        # Let's verify.
        # If it's 3-regular and bipartite, the parts must be size 4 and 4.
        # Let A and B be the parts.
        # Each vertex in A is connected to 3 in B.
        # The number of edges is 12.
        # In K_{4,4}, there are 16 edges.
        # So we remove 4 edges from K_{4,4}.
        # The removed edges must form a perfect matching to maintain regularity?
        # If we remove an edge (u,v), u and v lose a degree. To keep degree 3, we must not remove any other edge incident to u or v?
        # No, we start from K_{4,4} (degree 4). We remove edges to get degree 3.
        # Each vertex loses exactly 1 incident edge.
        # So the set of removed edges must be a perfect matching in K_{4,4}.
        # How many perfect matchings in K_{4,4}? 4! = 24.
        # So there are 24 such graphs for a fixed partition (A,B).
        # Are they all isomorphic to Q3?
        # Yes, Q3 is isomorphic to K_{4,4} minus a perfect matching.
        # So the check "all degrees are 3" is sufficient to identify a Q3 in a bipartite graph on 8 vertices?
        # Wait, we assumed the graph is bipartite.
        # Does "all degrees are 3" imply bipartite?
        # Not necessarily. A 3-regular graph on 8 vertices could have odd cycles.
        # Example: Octahedral graph? 6 vertices.
        # Cube graph Q3 is bipartite.
        # Are there non-bipartite 3-regular graphs on 8 vertices?
        # Yes. For example, take two K4s and connect them? No, K4 is 3-regular.
        # If we take K4 and K4, that's two components.
        # If we connect them?
        # The only 3-regular graphs on 8 vertices are:
        # 1. Q3 (bipartite)
        # 2. Two K4s (disconnected)
        # 3. The Wagner graph? (Möbius ladder M4? 8 vertices, 3-regular).
        #    M4 is non-bipartite (has 4-cycles but also... wait, Möbius ladder M4 is bipartite?
        #    M4 has 8 vertices. Vertices 0..7. Edges (i, i+1) and (i, i+4).
        #    Cycle 0-1-2-3-7-6-5-4-0. Length 8.
        #    Cycle 0-4-5-1-0? No.
        #    Let's check bipartiteness of M4.
        #    0 connected to 1, 7, 4.
        #    1 connected to 0, 2, 5.
        #    If 0 is in A, 1,7,4 in B.
        #    1 in B -> 0,2,5 in A.
        #    7 in B -> 6,0,3 in A.
        #    4 in B -> 5,6,0 in A.
        #    Check 5 (in A): neighbors 1(B), 4(B), 6(A). Conflict! 5 connected to 6 (both A).
        #    So M4 is NOT bipartite.
        #    So M4 is a 3-regular non-bipartite graph on 8 vertices.
        #    So "all degrees 3" is NOT sufficient. We must check bipartiteness or isomorphism.
        
        # However, we can just check the specific edge set against the 35 splits.
        # If the graph is Q3, it MUST be bipartite with parts of size 4.
        # And the bipartition must be one of the 35 canonical splits.
        # For each split, we check if ALL edges are between A and B, and NONE are within A or within B.
        # And specifically, we need exactly 12 edges, and they must form the specific structure?
        # Actually, if it's bipartite with parts A,B (size 4) and 3-regular, it is K_{4,4} minus a matching.
        # This is always Q3.
        # So the check is:
        # 1. Degrees are all 3.
        # 2. There exists a bipartition (from the 35) such that NO edges exist within A and NO edges exist within B.
        
        # Let's optimize:
        # We already computed degrees. If any != 3, return False.
        # Now check bipartiteness.
        # We can just try the 35 splits.
        # For a split mask_a:
        #   Check if any edge is within A.
        #   Check if any edge is within B.
        #   If both are empty, it's a Q3.
        
        for mask_a in all_splits:
            valid = True
            # Check edges within A
            # Vertices in A are those i where (mask_a >> i) & 1 is 1.
            # We need to check if adj[subset[i]] has any bits set corresponding to subset[j] where j is in A.
            
            # To do this fast, let's precompute for this subset:
            # local_adj_mask[i] : bitmask of local neighbors.
            
            # We can reuse the degree calculation loop to build local adjacency.
            # But we already returned if deg != 3.
            # So we assume deg == 3.
            
            # Let's build the local adjacency matrix bitmasks.
            local_adj = [0] * 8
            for i in range(8):
                v_global = subset[i]
                mask = adj_masks[v_global]
                lm = 0
                for j in range(8):
                    if i != j:
                        v_other = subset[j]
                        if mask & (1 << v_other):
                            lm |= (1 << j)
                local_adj[i] = lm
            
            # Now check the split
            for i in range(8):
                if (mask_a >> i) & 1:
                    # i is in A
                    # Check if local_adj[i] has any bits in A
                    # mask_a contains the bits for A.
                    # So we check if (local_adj[i] & mask_a) != 0
                    if local_adj[i] & mask_a:
                        valid = False
                        break
            if valid:
                return True
                
        return False

    # --- Heuristic Construction ---
    
    def random_q3_free_graph():
        """
        Generate a random Q3-free graph using a greedy random removal strategy.
        Starts with K16 and removes edges that are part of Q3s.
        """
        # Initialize adj as all 1s (except diagonal)
        adj = [[1] * n for _ in range(n)]
        for i in range(n):
            adj[i][i] = 0
            
        # Convert to bitmasks for fast checking
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            adj_masks[u] = mask
            
        # We will randomly try to remove edges that form Q3s.
        # Loop until no Q3 is found or time limit.
        
        # To speed up, we can pick a random 8-subset and check if it's a Q3.
        # If it is, remove a random edge from it.
        
        while time.time() < deadline - 5.0:
            # Pick a random subset
            sub = random.choice(subsets_8)
            if check_subset_q3(adj_masks, sub):
                # It contains a Q3.
                # Pick a random edge in this subset to remove.
                # The Q3 edges are exactly the edges in the induced subgraph (since it's 3-regular).
                # We can pick a random edge (u, v) within sub.
                u_idx = random.randint(0, 7)
                v_idx = random.randint(0, 7)
                if u_idx != v_idx:
                    u = sub[u_idx]
                    v = sub[v_idx]
                    if adj[u][v]:
                        # Remove edge
                        adj[u][v] = 0
                        adj[v][u] = 0
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                        
        # Count edges
        edge_count = sum(adj[u][v] for u in range(n) for v in range(u+1, n))
        
        # Return edge list
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges, edge_count

    def polish_graph(edges):
        """
        Try to add more edges to the given graph.
        """
        # Build adj and adj_masks
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            adj_masks[u] = mask
            
        # List of missing edges
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    missing.append((u, v))
        
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for u, v in missing[:]: # Iterate over a copy
                if adj[u][v]:
                    continue
                # Try adding
                adj[u][v] = 1
                adj[v][u] = 1
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                # Check if Q3-free
                # We only need to check subsets containing u and v?
                # No, adding an edge might create a Q3 in a subset that didn't have it before.
                # Any Q3 containing the new edge (u,v) must contain u and v.
                # So we only need to check 8-subsets that contain u and v.
                
                is_valid = True
                for sub in subsets_8:
                    if u in sub and v in sub:
                        if check_subset_q3(adj_masks, sub):
                            is_valid = False
                            break
                            
                if is_valid:
                    missing.remove((u, v))
                    improved = True
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
                    
        # Return edges
        final_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    final_edges.append([u, v])
        return final_edges

    best_edges = []
    best_count = 0
    
    # Run multiple random restarts
    while time.time() < deadline:
        # 1. Generate random Q3-free graph
        edges, count = random_q3_free_graph()
        
        # 2. Polish
        polished_edges = polish_graph(edges)
        polished_count = len(polished_edges)
        
        if polished_count > best_count:
            best_count = polished_count
            best_edges = polished_edges
            
        # If we have plenty of time, do more restarts
        if time.time() > deadline - 10.0:
            break
            
    return best_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))

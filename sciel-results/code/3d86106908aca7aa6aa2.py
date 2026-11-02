"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We focus on weight-4 vectors (4 non-zero coordinates, values +/-1) and weight-2 vectors (2 non-zero coordinates, values +/-1).
    # Weight-4 vectors have norm^2 = 4.
    # Weight-2 vectors have norm^2 = 2.
    
    candidates = []
    
    # Generate weight-4 vectors
    # C(11,4) * 16 = 5280
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Generate weight-2 vectors
    # C(11,2) * 4 = 55 * 4 = 220
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute Gram matrix
    G = candidates @ candidates.T
    
    # Build conflict graph
    # Condition for vectors u, v: ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u||^2 + ||v||^2 - 2 u.v >= max(||u||^2, ||v||^2)
    # Let n_i = ||u||^2, n_j = ||v||^2.
    # n_i + n_j - 2 G[i,j] >= max(n_i, n_j)
    # 2 G[i,j] <= n_i + n_j - max(n_i, n_j) = min(n_i, n_j)
    # G[i,j] <= min(n_i, n_j) / 2.0
    
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    
    # Conflict if G[i,j] > limits[i,j]
    conflict_mat = G > limits
    
    # Exclude self-conflicts from the matrix for the ILP constraints (diagonal is irrelevant for i != j)
    np.fill_diagonal(conflict_mat, False)
    
    # To make the ILP tractable, we select a subset of candidates.
    # The full graph has ~5500 nodes, which is too large for a quick ILP in Python/SciPy.
    # We will select a subset of size N_SUB = 300.
    # Strategy for selection: 
    # 1. Random sampling might miss good structures.
    # 2. Greedy selection of "compatible" clusters might be biased.
    # 3. We can use a simple heuristic to pick a diverse subset:
    #    - Shuffle candidates.
    #    - Greedily pick a candidate if it's compatible with the current subset.
    #    - This ensures the subset itself is an independent set (or close to it), 
    #      but we want the ILP to find the best subset of the POOL, not just the greedy one.
    #    
    # Actually, for ILP to be effective, the pool should contain the optimal solution.
    # If we restrict the pool too much, we lose the optimum.
    # If we use the full pool, it's too slow.
    # 
    # Let's try a different approach: 
    # Use the full pool but decompose it? No.
    # 
    # Let's try a smaller pool size, say 200, but chosen smartly.
    # We can run a quick greedy to find a good initial solution, then use that as a lower bound.
    # But ILP needs the variables.
    #
    # Alternative: Use the full pool but exploit symmetry or structure?
    # The vectors are highly symmetric.
    #
    # Let's stick to a subset approach but with a larger size if possible.
    # SciPy's milp uses HiGHS. HiGHS can handle a few thousand binary variables if the matrix is sparse.
    # The conflict matrix is dense (most pairs conflict).
    # Wait, if most pairs conflict, the independent set is small.
    # If the graph is dense, the ILP might be easier? No, dense constraints are harder.
    #
    # Let's check the density.
    # For weight-4 vectors, u.v can be -4, -2, 0, 2, 4.
    # Limit is 2.
    # Conflict if u.v > 2. So u.v = 4 is a conflict.
    # u.v = 4 happens if they are identical (impossible for distinct candidates) or... wait.
    # If u and v are distinct weight-4 vectors, can u.v = 4?
    # u.v = sum(u_i v_i). Since u_i, v_i in {-1, 1, 0}, max dot product for distinct is 2?
    # If they share 3 coordinates with same sign, and 1 with different?
    # Let's check:
    # u: 1 1 1 1 0...
    # v: 1 1 1 -1 0... -> dot = 1+1+1-1 = 2.
    # v: 1 1 -1 1 0... -> dot = 1+1-1+1 = 2.
    # v: 1 -1 1 1 0... -> dot = 1-1+1+1 = 2.
    # v: -1 1 1 1 0... -> dot = -1+1+1+1 = 2.
    # So for weight-4, max dot product for distinct vectors is 2.
    # Limit is 2.
    # So G[i,j] <= 2 is satisfied.
    # Thus, NO two distinct weight-4 vectors conflict?
    # Let's re-verify.
    # Condition: G[i,j] <= min(n_i, n_j)/2 = 4/2 = 2.
    # If max dot product is 2, then all distinct weight-4 vectors are compatible!
    #
    # What about weight-2?
    # n=2. Limit = 1.
    # u: 1 1 0... v: 1 1 0... (identical)
    # u: 1 1 0... v: 1 -1 0... -> dot = 0.
    # u: 1 1 0... v: 1 0 1 0... -> dot = 1.
    # u: 1 1 0... v: 1 0 0 1... -> dot = 1.
    # Max dot product for distinct weight-2 is 1.
    # Limit is 1.
    # So all distinct weight-2 vectors are compatible?
    #
    # What about mixed?
    # u (w4, n=4), v (w2, n=2). Limit = 2/2 = 1.
    # u: 1 1 1 1 0...
    # v: 1 1 0 0 0... -> dot = 2.
    # 2 > 1. Conflict!
    # So weight-4 and weight-2 vectors conflict if they share 2 coordinates with same sign.
    
    # Conclusion:
    # All weight-4 vectors are mutually compatible.
    # All weight-2 vectors are mutually compatible.
    # Weight-4 and Weight-2 vectors conflict if their supports overlap in at least 2 coordinates with the same sign.
    
    # This structure is MUCH simpler than a generic graph.
    # We can solve this analytically or with a much smaller ILP.
    #
    # Let S4 be the set of selected weight-4 vectors.
    # Let S2 be the set of selected weight-2 vectors.
    # Total count = |S4| + |S2|.
    #
    # Since all S4 are compatible with each other, we can include ALL weight-4 vectors?
    # No, they must be compatible with S2.
    #
    # Let's maximize |S4| + |S2|.
    #
    # We can iterate over possible subsets of S2? No, S2 has 220 vectors.
    #
    # Let's look at the conflicts between S4 and S2.
    # A weight-4 vector u conflicts with a weight-2 vector v if u and v agree on both coordinates of v.
    # i.e., v is a "sub-vector" of u (in terms of signs).
    #
    # If we select a set of weight-2 vectors S2, then any weight-4 vector u that contains any v in S2 (as a signed sub-vector) is forbidden.
    #
    # We want to choose S2 to minimize the number of forbidden weight-4 vectors, while maximizing |S2|.
    #
    # This is a set cover / packing problem.
    #
    # Let's try a greedy approach on this simplified structure.
    #
    # 1. Start with S2 = empty. Forbidden S4 = empty.
    # 2. Add weight-2 vectors one by one.
    #    Which one to add?
    #    Adding v to S2 forbids all u in S4 such that u contains v.
    #    Let C(v) be the set of weight-4 vectors containing v.
    #    We want to pick v such that |C(v)| is small? Or such that the union of C(v) for v in S2 is small?
    #
    # Actually, we can just use the ILP on the reduced problem.
    # Variables: x_u for u in S4, y_v for v in S2.
    # Maximize sum(x_u) + sum(y_v).
    # Subject to:
    # 1. x_u, y_v in {0, 1}.
    # 2. For each pair (u, v) that conflicts: x_u + y_v <= 1.
    #
    # This is a bipartite independent set problem (plus independent sets within partitions, but we established there are no internal conflicts).
    # So it's just a Bipartite Independent Set.
    #
    # Max Independent Set in Bipartite Graph = Total Vertices - Min Vertex Cover.
    # Min Vertex Cover in Bipartite Graph = Max Matching (Kőnig's theorem).
    #
    # So we can solve this by finding the Maximum Matching in the conflict bipartite graph!
    #
    # Graph:
    # Left: Weight-4 vectors (5280 nodes).
    # Right: Weight-2 vectors (220 nodes).
    # Edge if conflict.
    #
    # If we find the Max Matching M, the size of the Min Vertex Cover is |M|.
    # The size of the Max Independent Set is N_left + N_right - |M|.
    #
    # This is MUCH faster than a general ILP.
    # We can use scipy.sparse.csgraph or networkx (if available, but stdlib/scipy is safer).
    # `scipy.sparse.csgraph.maximum_bipartite_matching` is available in SciPy 1.11+.
    # Let's check if it's available. If not, we can implement Hopcroft-Karp or use a simpler heuristic.
    # Given the time, let's assume a recent SciPy is available or use a fallback.
    #
    # Let's build the bipartite graph.
    
    w4_indices = [i for i in range(n_cand) if norms_sq[i] == 4]
    w2_indices = [i for i in range(n_cand) if norms_sq[i] == 2]
    
    n_w4 = len(w4_indices)
    n_w2 = len(w2_indices)
    
    # Build adjacency matrix for the bipartite graph (w4 x w2)
    # conflict_bipartite[i, j] = True if w4_indices[i] conflicts with w2_indices[j]
    
    # Extract submatrices
    G_w4_w2 = G[np.ix_(w4_indices, w2_indices)]
    limits_w4_w2 = limits[np.ix_(w4_indices, w2_indices)]
    
    conflict_bipartite = G_w4_w2 > limits_w4_w2
    
    # Now find Max Matching.
    # We can use scipy.sparse.csgraph.maximum_bipartite_matching if available.
    # Otherwise, we can use a simple greedy matching or Hopcroft-Karp.
    # Let's try to import it.
    
    try:
        from scipy.sparse import csgraph
        # Convert to sparse adjacency matrix
        # csgraph expects a matrix where A[i,j] != 0 indicates an edge.
        # We have a boolean matrix.
        adj_matrix = conflict_bipartite.astype(np.int8)
        
        # maximum_bipartite_matching returns a dictionary: {left_node: right_node}
        # The size of this dictionary is the size of the matching.
        matching = csgraph.maximum_bipartite_matching(adj_matrix)
        
        matching_size = len(matching)
        
        # Max Independent Set Size = N_left + N_right - Matching_Size
        max_is_size = n_w4 + n_w2 - matching_size
        
        # To reconstruct the set:
        # The Min Vertex Cover can be found from the matching.
        # Vertices not in the Min Vertex Cover form the Max Independent Set.
        # 
        # Algorithm to find Max Independent Set from Max Matching in Bipartite Graph:
        # 1. Find Max Matching M.
        # 2. Find all vertices unmatched by M.
        # 3. Find all vertices reachable from unmatched vertices by alternating paths (start with non-matching edges, then matching, etc.).
        # 4. Let Z be the set of reachable vertices.
        # 5. Min Vertex Cover = (Left \ Z) U (Right ∩ Z).
        # 6. Max Independent Set = (Left ∩ Z) U (Right \ Z).
        
        # Let's implement this BFS.
        
        # Build adjacency lists for the bipartite graph for BFS.
        # We need to traverse from Left to Right and Right to Left.
        # Edges in conflict_bipartite.
        
        # To save memory/time, we can use the sparse matrix.
        # But for BFS, we need neighbors.
        
        # Let's use the dense matrix if it's not too big.
        # 5280 x 220 = ~1.1M entries. Very small.
        
        # Unmatched vertices:
        # Left unmatched: those not in matching keys.
        # Right unmatched: those not in matching values.
        
        matched_left = set(matching.keys())
        matched_right = set(matching.values())
        
        unmatched_left = [i for i in range(n_w4) if i not in matched_left]
        unmatched_right = [j for j in range(n_w2) if j not in matched_right]
        
        # BFS from all unmatched vertices (both Left and Right? No, standard algo starts from unmatched Left? 
        # Actually, for bipartite graph, we can start from all unmatched vertices on both sides?
        # Standard: Start from unmatched vertices on Left.
        # Wait, the theorem says:
        # Z = set of vertices reachable from unmatched vertices in Left via alternating paths.
        # If we start from unmatched Left, we cover the necessary part.
        # What if there are unmatched vertices on Right that are not reachable from unmatched Left?
        # In a max matching, if there's an unmatched vertex on Right, there must be an augmenting path? No, it's max.
        # The set Z is defined as reachable from unmatched Left.
        # Let's stick to that.
        
        # Initialize Z
        Z_left = set(unmatched_left)
        Z_right = set()
        
        # BFS Queue: (node, side)
        # side: 0 for Left, 1 for Right
        queue = []
        for i in unmatched_left:
            queue.append((i, 0))
            
        # Visited sets
        visited_left = set(unmatched_left)
        visited_right = set()
        
        while queue:
            node, side = queue.pop(0)
            
            if side == 0: # Left node
                # Traverse to Right via NON-matching edges
                # Neighbors in conflict_bipartite[node]
                for j in range(n_w2):
                    if conflict_bipartite[node, j]:
                        if (node, j) not in matching: # Non-matching edge
                            if j not in visited_right:
                                visited_right.add(j)
                                Z_right.add(j)
                                queue.append((j, 1))
            else: # Right node
                # Traverse to Left via MATCHING edges
                # Find the matched Left node for this Right node
                # We need a reverse mapping: right -> left
                # matching is left -> right.
                # Let's build reverse_matching.
                # For speed, we can just check if matching.get(k) == j.
                # But that's O(N_left).
                # Let's build reverse_matching once.
                # It's better to build it outside the loop.
                
                # Optimization: Precompute reverse matching
                # This is inside the BFS, so we should precompute before.
                
                # Let's refactor: Precompute reverse_matching.
                
                # Since I'm in the code generation, I'll insert the precomputation before the BFS.
                
                pass # Placeholder for logic, will rewrite below

        # REWRITE BFS PART WITH PRECOMPUTATION
        
        # Precompute reverse matching
        reverse_matching = {v: k for k, v in matching.items()}
        
        # BFS
        Z_left = set(unmatched_left)
        Z_right = set()
        queue = [(i, 0) for i in unmatched_left]
        visited_left = set(unmatched_left)
        visited_right = set()
        
        while queue:
            node, side = queue.pop(0)
            
            if side == 0: # Left
                # Traverse to Right via NON-matching edges
                # Get neighbors of node in conflict_bipartite
                # To make this fast, we can precompute neighbor lists for Left nodes.
                # But doing it on the fly with numpy might be okay for 5280 nodes.
                # conflict_bipartite[node] is a boolean array of size 220.
                # np.nonzero(conflict_bipartite[node])[0] gives neighbors.
                
                neighbors = np.nonzero(conflict_bipartite[node])[0]
                for j in neighbors:
                    if (node, j) not in matching: # Non-matching edge
                        if j not in visited_right:
                            visited_right.add(j)
                            Z_right.add(j)
                            queue.append((j, 1))
            else: # Right
                # Traverse to Left via MATCHING edges
                # If node is matched, go to its match.
                if node in reverse_matching:
                    k = reverse_matching[node]
                    if k not in visited_left:
                        visited_left.add(k)
                        Z_left.add(k)
                        queue.append((k, 0))
                        
        # Min Vertex Cover = (Left \ Z) U (Right ∩ Z)
        # Max Independent Set = (Left ∩ Z) U (Right \ Z)
        
        # Construct the solution indices
        selected_indices = []
        
        # Left nodes in Z
        for i in Z_left:
            selected_indices.append(w4_indices[i])
            
        # Right nodes NOT in Z
        for j in range(n_w2):
            if j not in Z_right:
                selected_indices.append(w2_indices[j])
                
        final_vecs = candidates[selected_indices]
        
        return final_vecs

    except Exception as e:
        # Fallback to greedy if SciPy function fails or is missing
        print(f"ILP/Matching failed: {e}")
        
        # Greedy MIS on the full graph
        # Use the conflict_mat computed earlier
        
        avail_mask = np.ones(n_cand, dtype=bool)
        selected = []
        
        # Min-degree heuristic
        static_degrees = np.sum(conflict_mat, axis=1)
        order_min_deg = np.argsort(static_degrees)
        
        for v in order_min_deg:
            if not avail_mask[v]:
                continue
            selected.append(v)
            avail_mask[conflict_mat[v]] = False
            
        final_vecs = candidates[selected]
        return final_vecs

# EVOLVE-BLOCK-END

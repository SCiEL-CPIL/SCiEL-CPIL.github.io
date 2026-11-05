"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Map edge index to (u, v) for fast lookup
    edge_map = {}
    for i, (u, v) in enumerate(all_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i

    # Helper to check validity of a graph represented by adjacency matrix
    # Returns True if no pair of vertices has > 2 common neighbours
    def is_valid(adj_mat):
        # C = adj @ adj.T gives counts of common neighbours for all pairs
        common_counts = adj_mat @ adj_mat.T
        # Check off-diagonal elements
        # We can use np.fill_diagonal to ignore self-loops
        # Note: adj_mat @ adj_mat.T creates a new array, so no side effect on adj_mat
        np.fill_diagonal(common_counts, 0)
        return np.max(common_counts) <= 2

    # Helper to get current edge count
    def get_edge_count(adj_mat):
        return int(np.sum(adj_mat)) // 2

    best_adj = np.zeros((n, n), dtype=np.uint8)
    best_score = 0
    
    # Strategy: Deterministic Greedy with Randomized Restart Orders
    # Instead of SA which can get stuck, we use a greedy algorithm that
    # always adds the "safest" edge. We restart with different vertex orderings
    # to escape different local optima.
    
    num_restarts = 10
    
    for restart_idx in range(num_restarts):
        if time.time() > deadline:
            break
            
        curr_adj = np.zeros((n, n), dtype=np.uint8)
        curr_score = 0
        
        # Create a random permutation of edges to break ties and explore different paths
        # This is effectively a randomized greedy
        shuffled_edges = all_edges.copy()
        random.shuffle(shuffled_edges)
        
        # We will iterate through multiple passes if possible, but usually one pass
        # with smart selection is enough. However, standard greedy picks one edge at a time.
        # To make it smarter, we evaluate all candidate edges at each step? That's O(m^2 * n^2) which is too slow.
        # m=190. m^2 = 36100. n^2=400. 36100*400 = 14M ops per step? No, checking validity is O(n^2) or O(n^3) with matmul.
        # Let's stick to a faster heuristic:
        # 1. Calculate current common neighbor counts C = A @ A.T
        # 2. For each potential edge (u,v), calculate the "cost" of adding it.
        #    Adding (u,v) increases C[u,v] by 1 (if not already there, but we only add missing).
        #    Wait, adding edge (u,v) changes the adjacency matrix.
        #    The new common neighbor count for pair (i,j) is affected if i or j is u or v.
        #    Specifically, C_new[i,j] = C_old[i,j] + (A[u,i]*A[v,j] + A[v,i]*A[u,j] ... wait, no.
        #    C[i,j] = sum_k A[i,k] A[k,j].
        #    If we add edge (u,v), A[u,v] becomes 1, A[v,u] becomes 1.
        #    The change in C[i,j] is sum_k (delta A[i,k] A[k,j] + A[i,k] delta A[k,j]).
        #    delta A is non-zero only at (u,v) and (v,u).
        #    So delta C[i,j] = A[i,u]A[v,j] + A[i,v]A[u,j] (for the term where k=u in first sum? No.
        #    Let's just compute the new C for the specific pairs affected?
        #    Actually, for a small n=20, we can just check validity of the new state by updating C incrementally.
        
        # Incremental Update of Common Neighbor Counts
        # C[i,j] = number of common neighbors of i and j.
        # If we add edge (u,v):
        # 1. C[u,v] increases by 1? No.
        #    Common neighbors of u and v are k such that A[u,k]=1 and A[v,k]=1.
        #    Adding edge (u,v) does NOT change the set of common neighbors of u and v directly, 
        #    EXCEPT that u becomes a neighbor of v and v becomes a neighbor of u.
        #    Wait, the definition of common neighbor k requires k != u and k != v usually?
        #    In simple graphs, common neighbor of u and v is a vertex k such that (u,k) in E and (v,k) in E.
        #    Does k=u count? No, because (u,u) is not an edge.
        #    So adding (u,v) does NOT change the count of common neighbors for the pair (u,v) itself.
        #    
        #    However, it DOES change the common neighbor counts for other pairs (i,j).
        #    For any pair (i,j), C[i,j] increases by 1 if:
        #    - u is a new common neighbor of i and j: i.e., A[i,u] was 1, A[j,u] was 1? No, u is already a neighbor if A[i,u]=1.
        #    - The edge (u,v) connects two vertices.
        #    - Let's look at the formula: C[i,j] = sum_k A[i,k] A[k,j].
        #    Change in C[i,j] due to A[u,v] becoming 1:
        #    Term k=u: A[i,u] * A[u,j] (was 0, now A[u,j] is 1? No, A[u,j] is the edge (u,j). 
        #    Wait, adding (u,v) sets A[u,v]=1 and A[v,u]=1.
        #    So in the sum for C[i,j], the term for k=u is A[i,u] * A[u,j].
        #    Before: A[u,j] was whatever it was. If j != v, A[u,j] didn't change.
        #    If j == v, A[u,v] changed from 0 to 1. So C[i,v] increases by A[i,u].
        #    Similarly, term for k=v is A[i,v] * A[v,j].
        #    If i == u, A[u,v] changed. So C[u,j] increases by A[v,j].
        #    
        #    Let's be precise.
        #    Delta C[i,j] = A[i,u]*A[u,j]_new + A[i,v]*A[v,j]_new - (A[i,u]*A[u,j]_old + A[i,v]*A[v,j]_old)
        #    Since only A[u,v] and A[v,u] change:
        #    A[u,j] changes only if j==v (from 0 to 1).
        #    A[v,j] changes only if j==u (from 0 to 1).
        #    
        #    Case 1: j == v.
        #    Delta C[i,v] = A[i,u]*1 + A[i,v]*A[v,v](0) - (A[i,u]*0 + A[i,v]*A[v,v](0)) = A[i,u].
        #    So C[i,v] increases by A[i,u] for all i.
        #    
        #    Case 2: i == u.
        #    Delta C[u,j] = A[u,u](0)*A[u,j] + A[u,v](1)*A[v,j] - (0 + 0) = A[v,j].
        #    So C[u,j] increases by A[v,j] for all j.
        #    
        #    Case 3: i == u and j == v.
        #    Delta C[u,v] = A[u,u]A[u,v] + A[u,v]A[v,v] - 0 = 0.
        #    So C[u,v] does NOT change.
        
        #    Summary of updates when adding (u,v):
        #    For all i != u, v:
        #       C[i, v] += A[i, u]
        #       C[u, i] += A[v, i]  (Note C is symmetric, so C[i,u] += A[v,i])
        #       Wait, C[i,u] is the same as C[u,i].
        #       Let's update the matrix C directly.
        #       C[:, v] += A[:, u]  (but careful with diagonal and u,v)
        #       C[u, :] += A[v, :]
        
        #    Let's verify symmetry.
        #    C[i,v] += A[i,u].
        #    C[v,i] should also increase by A[u,i]. Since A is symmetric, A[i,u]=A[u,i]. So C[v,i] += A[u,i].
        #    My update rule: C[:, v] += A[:, u]. This sets C[v,v] += A[v,u] (which is 0 before, 1 after? No, A[v,u] is the edge we are adding, so it was 0. But A[:, u] is the column of u. A[v,u] is 0 in the old matrix. So C[v,v] += 0. Correct.)
        #    C[u, :] += A[v, :]. This sets C[u,u] += A[v,u] (0). Correct.
        #    What about C[u,v]?
        #    From first update: C[u,v] += A[u,u] (0).
        #    From second update: C[u,v] += A[v,v] (0).
        #    So C[u,v] unchanged. Correct.
        
        #    So, to check if adding (u,v) is valid:
        #    1. Compute tentative new C.
        #    2. Check if any off-diagonal element > 2.
        #    
        #    To speed this up, we can precompute the "potential violations".
        #    But n=20 is small. 190 edges. 190 * 20 * 20 = 76,000 operations per greedy step.
        #    190 steps * 76,000 = 14.4 million operations. Very fast in NumPy.
        
        # Let's implement the Greedy Step
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Compute current common neighbor counts
            C = curr_adj @ curr_adj.T
            np.fill_diagonal(C, 0)
            
            best_edge_to_add = None
            best_max_count = 3 # We want to minimize the max count after addition
            
            # Iterate through all missing edges
            # To make it deterministic but varied, we can sort by some heuristic or just pick the best one
            # Finding the single best edge among 190 candidates
            for u, v in shuffled_edges:
                if curr_adj[u, v] == 1:
                    continue
                    
                # Tentatively add (u, v)
                # We need to check the new max count.
                # Instead of building the full new C, we can just check the affected parts.
                # But building full C is fast enough.
                
                # Copy C? No, we can just calculate the new values for the affected rows/cols.
                # Or just do the matmul on the modified matrix?
                # Modifying matrix and doing matmul is O(n^3) = 8000 ops.
                # Doing this for 190 edges is 1.5M ops.
                # This is acceptable.
                
                # Let's try a faster check:
                # We know which elements change.
                # New C[i, v] = C[i, v] + A[i, u]
                # New C[u, j] = C[u, j] + A[v, j]
                
                # We need max of new C off-diagonal <= 2.
                # The only elements that can become > 2 are those that were 2 and get incremented.
                # Or those that were 1 and get incremented by 2? No, increment is 0 or 1.
                # So only elements that were 2 can become 3.
                # Elements that were 2:
                #   C[i, v] == 2 and A[i, u] == 1 -> becomes 3. Invalid.
                #   C[u, j] == 2 and A[v, j] == 1 -> becomes 3. Invalid.
                
                # So, adding (u,v) is valid if and only if:
                # 1. For all i, if C[i, v] == 2, then A[i, u] == 0.
                # 2. For all j, if C[u, j] == 2, then A[v, j] == 0.
                
                # This is much faster! O(n) per edge.
                
                valid = True
                # Check condition 1: C[:, v] == 2 implies A[:, u] == 0
                # Indices where C[:, v] == 2
                idx_v = np.where(C[:, v] == 2)[0]
                for i in idx_v:
                    if curr_adj[i, u] == 1:
                        valid = False
                        break
                
                if valid:
                    # Check condition 2: C[u, :] == 2 implies A[v, :] == 0
                    idx_u = np.where(C[u, :] == 2)[0]
                    for j in idx_u:
                        if curr_adj[v, j] == 1:
                            valid = False
                            break
                
                if valid:
                    # This edge can be added.
                    # Which one to pick?
                    # Heuristic: Pick the one that leaves the most "room" for future additions?
                    # Or just pick the first one found?
                    # To be more robust, let's pick the one that minimizes the number of "saturated" pairs (count=2) created.
                    # Or simply pick the first valid one in the shuffled order.
                    # Let's pick the one that minimizes the new max count? They are all valid, so max is <= 2.
                    # Let's pick the one that minimizes the sum of new counts?
                    # Actually, just picking the first valid one is a valid greedy strategy.
                    # But to ensure we don't get stuck in a bad local optimum, the randomized shuffle helps.
                    
                    curr_adj[u, v] = 1
                    curr_adj[v, u] = 1
                    curr_score += 1
                    improved = True
                    # Break to restart the loop with new C?
                    # Yes, because C has changed.
                    break
            
        # Update best
        if curr_score > best_score:
            best_score = curr_score
            best_adj = curr_adj.copy()

    # Final Greedy Fill on the best solution found
    # This is a local search to add any remaining possible edges
    if time.time() < deadline:
        final_adj = best_adj.copy()
        final_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            C = final_adj @ final_adj.T
            np.fill_diagonal(C, 0)
            
            # Try to add any missing edge
            # We can shuffle again to vary the order
            shuffled_final = all_edges.copy()
            random.shuffle(shuffled_final)
            
            for u, v in shuffled_final:
                if final_adj[u, v] == 1:
                    continue
                    
                valid = True
                idx_v = np.where(C[:, v] == 2)[0]
                for i in idx_v:
                    if final_adj[i, u] == 1:
                        valid = False
                        break
                
                if valid:
                    idx_u = np.where(C[u, :] == 2)[0]
                    for j in idx_u:
                        if final_adj[v, j] == 1:
                            valid = False
                            break
                
                if valid:
                    final_adj[u, v] = 1
                    final_adj[v, u] = 1
                    final_score += 1
                    improved = True
                    break # Restart loop with new C
        
        if final_score > best_score:
            best_score = final_score
            best_adj = final_adj

    # Convert back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END

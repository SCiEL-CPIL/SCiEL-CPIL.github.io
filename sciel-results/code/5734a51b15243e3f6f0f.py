import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Generate all weight-3 vectors
    # v[i], v[j], v[k] are +/- 1, others 0
    vectors = []
    supports = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(v)
                    supports.append((i, j, k))
    
    n_vecs = len(vectors)
    
    # Precompute conflicts
    # Conflict if dist^2 < 3
    # For weight 3, dist^2 is integer.
    # dist^2 = 1 is impossible.
    # dist^2 = 2 is possible.
    # dist^2 >= 3 is safe.
    
    # We can compute conflicts on the fly or precompute.
    # n_vecs = 1320.
    # Precomputing adjacency list is O(N^2) = 1.7M ops. Fast.
    
    # However, checking dist^2 for all pairs is O(N^2 * D).
    # 1.7M * 11 ~ 18M ops. NumPy can do this fast, or Python loop is okay if optimized.
    # Let's use a bitset or boolean matrix if N is small enough.
    # 1320 x 1320 bool matrix is ~ 1.7M bytes. Fine.
    
    # But we can be smarter.
    # Two vectors u, v conflict iff:
    # 1. They share exactly 2 indices, AND
    # 2. The values at those 2 indices are identical.
    # (If they share 3 indices, dist^2 is 0, 4, 8, 12. 0 is same vector. 4+ is safe.)
    # (If they share 1 index, dist^2 is at least 1 (from shared) + 1 + 1 + 1 (from unique) = 4? 
    #   u=(1,1,1,0..), v=(1,-1,-1,0..). Share 0. 
    #   diff: 0, 2, 2, 0. Norm sq 8. Safe.
    #   u=(1,1,1,0..), v=(1,0,0,1..). Share 0.
    #   diff: 0, 1, 1, -1. Norm sq 3. Safe.
    #   So sharing 1 index is always safe?
    #   Min dist for share 1: u_i=v_i. u has 2 other non-zeros, v has 2 other non-zeros.
    #   If other non-zeros are disjoint, dist^2 = 1+1+1+1 = 4.
    #   If other non-zeros share 1 index? No, supports are size 3.
    #   If S_u = {0,1,2}, S_v = {0,3,4}. Disjoint others. Dist^2 4.
    #   If S_u = {0,1,2}, S_v = {0,1,3}. Share 2 indices.
    #   So only "share 2 indices" is the dangerous case.
    
    # Let's verify "share 2 indices, same values" is the ONLY conflict.
    # If share 2 indices i,j.
    # u_i=v_i, u_j=v_j.
    # u_k, v_l are the 3rd indices.
    # If k==l (same support), u_k, v_k.
    #   If u_k=v_k, u=v (not distinct).
    #   If u_k!=v_k, diff is 2 at k. Norm sq 4. Safe.
    # If k!=l (diff supports).
    #   u_k=+-1, v_k=0. v_l=+-1, u_l=0.
    #   Diff at k is 1 or -1 (sq 1).
    #   Diff at l is 1 or -1 (sq 1).
    #   Total sq 2. Conflict.
    
    # So conflict condition:
    # Supports intersect in exactly 2 indices, AND values at those indices are equal.
    
    # We can build a graph where nodes are vectors.
    # Edge if conflict.
    # Find Max Independent Set.
    
    # Optimization: Group by support.
    # There are C(11,3) = 165 supports.
    # Each support has 8 vectors.
    # Vectors within same support: No conflicts.
    # Vectors in different supports S, S':
    #   If |S intersect S'| == 2:
    #      Let common indices be i, j.
    #      For u in S, v in S':
    #      Conflict if u_i==v_i AND u_j==v_j.
    
    # We can construct the adjacency list.
    
    adj = [[] for _ in range(n_vecs)]
    
    # Map support to list of vectors
    supp_to_vecs = {}
    for idx, v in enumerate(vectors):
        s = supports[idx]
        if s not in supp_to_vecs:
            supp_to_vecs[s] = []
        supp_to_vecs[s].append(idx)
    
    # For each pair of supports, check conflicts
    supp_list = list(supp_to_vecs.keys())
    for a_idx in range(len(supp_list)):
        s_a = supp_list[a_idx]
        vecs_a = supp_to_vecs[s_a]
        for b_idx in range(a_idx + 1, len(supp_list)):
            s_b = supp_list[b_idx]
            vecs_b = supp_to_vecs[s_b]
            
            # Check intersection
            inter = set(s_a).intersection(set(s_b))
            if len(inter) != 2:
                continue
            
            # Get the 2 common indices
            common = list(inter)
            i, j = common[0], common[1]
            
            # For each pair of vectors, check if values match at i and j
            for idx_a in vecs_a:
                u = vectors[idx_a]
                for idx_b in vecs_b:
                    v = vectors[idx_b]
                    if u[i] == v[i] and u[j] == v[j]:
                        adj[idx_a].append(idx_b)
                        adj[idx_b].append(idx_a)
                        
    # Now run greedy Max Independent Set with restarts
    best_set = set()
    
    def greedy_mis():
        order = list(range(n_vecs))
        random.shuffle(order)
        selected = set()
        for idx in order:
            # Check if idx conflicts with any in selected
            # selected is small initially, grows later
            # Efficient check:
            # if any(nei in selected for nei in adj[idx]): continue
            # This is O(deg) per node. Total O(E).
            conflict = False
            for nei in adj[idx]:
                if nei in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(idx)
        return selected
    
    # Run restarts
    while time.time() < deadline:
        s = greedy_mis()
        if len(s) > len(best_set):
            best_set = s
            # Optional: local search / hill climbing on best_set?
            # Given time, just restarts.
            
    # Convert best_set to array
    if not best_set:
        # Fallback to weight-2 if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = np.array([vectors[i] for i in best_set], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END

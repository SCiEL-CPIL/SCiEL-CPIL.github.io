import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # 1. Initialize Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj as numpy boolean matrix for fast vectorized ops
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Efficient (5,5)-free verification using bitsets
    def is_5_5_free(adj_mat, n):
        """Check no K5 and no independent set of size 5."""
        # Convert to bitsets for fast intersection
        # adj_bits[i] = integer bitmask of neighbors of i
        adj_bits = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            mask = 0
            row = adj_mat[i]
            for j in range(n):
                if row[j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
        
        # Check for K5: find 5 vertices all mutually adjacent
        # For each 5-subset, check if all 10 pairs are edges
        # Optimize: for each vertex, get its neighbors, check for K4 in neighbor set
        for v1 in range(n):
            nbrs1 = [v for v in range(n) if v != v1 and adj_mat[v1, v]]
            if len(nbrs1) < 4:
                continue
            # Check for K4 in nbrs1
            for v2, v3, v4 in combinations(nbrs1, 3):
                if adj_mat[v2, v3] and adj_mat[v2, v4] and adj_mat[v3, v4]:
                    # v1,v2,v3,v4 form K4. Need a 5th vertex adjacent to all four.
                    # Intersection of neighborhoods of v1,v2,v3,v4 (excluding themselves)
                    common = set(nbrs1)
                    for v in [v1, v2, v3, v4]:
                        nbrs_v = set()
                        for u in range(n):
                            if u != v and adj_mat[v, u]:
                                nbrs_v.add(u)
                        common &= nbrs_v
                    # Remove v1,v2,v3,v4 from common
                    common -= {v1, v2, v3, v4}
                    if common:
                        return False
        # Check for independent set of size 5
        # Independent set = clique in complement
        for v1 in range(n):
            non_nbrs1 = [v for v in range(n) if v != v1 and not adj_mat[v1, v]]
            if len(non_nbrs1) < 4:
                continue
            for v2, v3, v4 in combinations(non_nbrs1, 3):
                if not adj_mat[v2, v3] and not adj_mat[v2, v4] and not adj_mat[v3, v4]:
                    # v1,v2,v3,v4 form I4. Need a 5th vertex non-adjacent to all four.
                    common = set(non_nbrs1)
                    for v in [v1, v2, v3, v4]:
                        non_nbrs_v = set()
                        for u in range(n):
                            if u != v and not adj_mat[v, u]:
                                non_nbrs_v.add(u)
                        common &= non_nbrs_v
                    common -= {v1, v2, v3, v4}
                    if common:
                        return False
        return True
    
    # Get K4s and I4s as numpy arrays of vertex indices
    def get_k4s_i4s(adj_mat, n):
        k4s = []
        i4s = []
        for subset in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = subset[i], subset[j]
                    if adj_mat[u, v]:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(subset)
            if is_indep:
                i4s.append(subset)
        return k4s, i4s
    
    k4s, i4s = get_k4s_i4s(adj, n)
    
    current_n = n
    
    # Precompute K4/I4 as numpy arrays for fast containment checking
    def arrays_from_lists(k4_list, i4_list):
        if k4_list:
            k4_arr = np.array(k4_list, dtype=np.uint8)
        else:
            k4_arr = np.zeros((0, 4), dtype=np.uint8)
        if i4_list:
            i4_arr = np.array(i4_list, dtype=np.uint8)
        else:
            i4_arr = np.zeros((0, 4), dtype=np.uint8)
        return k4_arr, i4_arr
    
    k4_arr, i4_arr = arrays_from_lists(k4s, i4s)
    
    # Fast local search to find valid S (neighbor set of new vertex)
    def try_add_vertex(adj_mat, k4_arr, i4_arr, current_n, deadline):
        """Find set S such that no K4 ⊆ S and no I4 ⊆ V\S."""
        n = current_n
        
        # Precompute for each vertex: which K4s contain it, which I4s contain it
        # k4_membership[v] = list of K4 indices containing v
        # i4_membership[v] = list of I4 indices containing v
        k4_membership = [[] for _ in range(n)]
        for idx, k4 in enumerate(k4_arr):
            for v in k4:
                k4_membership[v].append(idx)
        
        i4_membership = [[] for _ in range(n)]
        for idx, i4 in enumerate(i4_arr):
            for v in i4:
                i4_membership[v].append(idx)
        
        best_S = None
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Initialize S randomly
            S = np.zeros(n, dtype=np.uint8)
            S[random.sample(range(n), random.randint(n // 3, 2 * n // 3))] = 1
            
            # Local search
            converged = False
            for iteration in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Check violations using vectorized ops
                # For each K4, check if all 4 vertices are in S
                if len(k4_arr) > 0:
                    # k4_arr shape (K, 4), S shape (n,)
                    # For each K4 row, check if all 4 entries in S are 1
                    k4_in_S = np.all(S[k4_arr] == 1, axis=1)
                    bad_k4_idx = np.where(k4_in_S)[0]
                else:
                    bad_k4_idx = np.array([], dtype=np.int64)
                
                # For each I4, check if all 4 vertices are NOT in S
                if len(i4_arr) > 0:
                    i4_not_in_S = np.all(S[i4_arr] == 0, axis=1)
                    bad_i4_idx = np.where(i4_not_in_S)[0]
                else:
                    bad_i4_idx = np.array([], dtype=np.int64)
                
                if len(bad_k4_idx) == 0 and len(bad_i4_idx) == 0:
                    # Success!
                    S_list = list(np.where(S == 1)[0])
                    return S_list
                
                # Fix violations
                if len(bad_k4_idx) > 0:
                    # Pick a random bad K4, remove one of its vertices from S
                    bad_idx = bad_k4_idx[random.randint(0, len(bad_k4_idx) - 1)]
                    k4 = k4_arr[bad_idx]
                    # Pick random vertex from this K4 that's in S
                    candidates = [v for v in k4 if S[v] == 1]
                    if candidates:
                        S[random.choice(candidates)] = 0
                elif len(bad_i4_idx) > 0:
                    # Pick a random bad I4, add one of its vertices to S
                    bad_idx = bad_i4_idx[random.randint(0, len(bad_i4_idx) - 1)]
                    i4 = i4_arr[bad_idx]
                    # Pick random vertex from this I4 that's not in S
                    candidates = [v for v in i4 if S[v] == 0]
                    if candidates:
                        S[random.choice(candidates)] = 1
            
        return None
    
    # Main loop to add vertices
    while current_n < 42 and time.time() < deadline - 15:
        S = try_add_vertex(adj, k4_arr, i4_arr, current_n, deadline)
        if S is None:
            # Try a few more times
            found = False
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                S = try_add_vertex(adj, k4_arr, i4_arr, current_n, deadline)
                if S is not None:
                    found = True
                    break
            if not found:
                break
        
        # Add new vertex
        new_idx = current_n
        # Resize adj
        new_adj = np.zeros((current_n + 1, current_n + 1), dtype=np.uint8)
        new_adj[:current_n, :current_n] = adj
        for v in S:
            new_adj[v, new_idx] = 1
            new_adj[new_idx, v] = 1
        adj = new_adj
        
        current_n += 1
        
        # Update K4s and I4s
        k4s, i4s = get_k4s_i4s(adj, current_n)
        k4_arr, i4_arr = arrays_from_lists(k4s, i4s)
        
        # Periodic verification
        if current_n % 3 == 0:
            if not is_5_5_free(adj, current_n):
                # Fallback: stop adding
                break
    
    # Final verification
    if not is_5_5_free(adj, current_n):
        # Fallback to seed
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges
    
    edges = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if adj[i, j]:
                edges.append((i, j))
    
    return current_n, edges

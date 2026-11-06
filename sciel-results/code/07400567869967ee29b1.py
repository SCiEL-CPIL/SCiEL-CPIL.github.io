import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
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
        adj_bits = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            mask = 0
            row = adj_mat[i]
            for j in range(n):
                if row[j]:
                    mask |= (1 << j)
            adj_bits[i] = mask
        
        # Check for K5
        for v1 in range(n):
            nbrs1 = [v for v in range(n) if v != v1 and adj_mat[v1, v]]
            if len(nbrs1) < 4:
                continue
            for v2, v3, v4 in combinations(nbrs1, 3):
                if adj_mat[v2, v3] and adj_mat[v2, v4] and adj_mat[v3, v4]:
                    common = set(nbrs1)
                    for v in [v1, v2, v3, v4]:
                        nbrs_v = set()
                        for u in range(n):
                            if u != v and adj_mat[v, u]:
                                nbrs_v.add(u)
                        common &= nbrs_v
                    common -= {v1, v2, v3, v4}
                    if common:
                        return False
        # Check for independent set of size 5
        for v1 in range(n):
            non_nbrs1 = [v for v in range(n) if v != v1 and not adj_mat[v1, v]]
            if len(non_nbrs1) < 4:
                continue
            for v2, v3, v4 in combinations(non_nbrs1, 3):
                if not adj_mat[v2, v3] and not adj_mat[v2, v4] and not adj_mat[v3, v4]:
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
        
        # Precompute vertex-to-K4 and vertex-to-I4 mappings for faster violation fixing
        # Map each vertex to the K4s it belongs to
        v_to_k4 = [[] for _ in range(n)]
        for idx, k4 in enumerate(k4_arr):
            for v in k4:
                v_to_k4[v].append(idx)
        
        # Map each vertex to the I4s it belongs to
        v_to_i4 = [[] for _ in range(n)]
        for idx, i4 in enumerate(i4_arr):
            for v in i4:
                v_to_i4[v].append(idx)
        
        for attempt in range(800):
            if time.time() > deadline - 3:
                break
            
            # Initialize S: try multiple strategies
            if attempt % 4 == 0:
                # Strategy 1: balanced random
                S = np.zeros(n, dtype=np.uint8)
                target_size = random.randint(n // 2 - n // 8, n // 2 + n // 8)
                S[random.sample(range(n), target_size)] = 1
            elif attempt % 4 == 1:
                # Strategy 2: based on degree - connect to high-degree vertices
                degrees = adj_mat.sum(axis=1)
                S = (degrees >= np.median(degrees)).astype(np.uint8)
                # Add some randomness
                for _ in range(n // 10):
                    v = random.randint(0, n - 1)
                    S[v] = 1 - S[v]
            elif attempt % 4 == 2:
                # Strategy 3: start from a random vertex's neighborhood
                v0 = random.randint(0, n - 1)
                S = adj_mat[v0].copy()
                S[v0] = 0
                # Add some randomness
                for _ in range(n // 10):
                    v = random.randint(0, n - 1)
                    S[v] = 1 - S[v]
            else:
                # Strategy 4: complement of a random vertex's neighborhood
                v0 = random.randint(0, n - 1)
                S = (1 - adj_mat[v0]).astype(np.uint8)
                S[v0] = 0
                # Add some randomness
                for _ in range(n // 10):
                    v = random.randint(0, n - 1)
                    S[v] = 1 - S[v]
            
            # Track best solution found so far in this attempt
            best_S = S.copy()
            best_violations = float('inf')
            
            # Local search with adaptive step size
            for iteration in range(3000):
                if time.time() > deadline - 3:
                    break
                
                # Check violations using vectorized ops
                if len(k4_arr) > 0:
                    k4_in_S = np.all(S[k4_arr] == 1, axis=1)
                    bad_k4_idx = np.where(k4_in_S)[0]
                else:
                    bad_k4_idx = np.array([], dtype=np.int64)
                
                if len(i4_arr) > 0:
                    i4_not_in_S = np.all(S[i4_arr] == 0, axis=1)
                    bad_i4_idx = np.where(i4_not_in_S)[0]
                else:
                    bad_i4_idx = np.array([], dtype=np.int64)
                
                total_violations = len(bad_k4_idx) + len(bad_i4_idx)
                
                if total_violations == 0:
                    # Success!
                    S_list = list(np.where(S == 1)[0])
                    return S_list
                
                # Track best
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_S = S.copy()
                
                # Fix violations with probabilistic choice
                # Weight by how many times each vertex appears in violations
                if len(bad_k4_idx) > 0:
                    # Pick a bad K4 and remove one of its vertices from S
                    bad_idx = bad_k4_idx[random.randint(0, len(bad_k4_idx) - 1)]
                    k4 = k4_arr[bad_idx]
                    candidates = [v for v in k4 if S[v] == 1]
                    if candidates:
                        # Prefer vertices that appear in many bad K4s
                        weights = [len(v_to_k4[v]) for v in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cumulative = 0
                        chosen = candidates[0]
                        for c, w in zip(candidates, weights):
                            cumulative += w
                            if r <= cumulative:
                                chosen = c
                                break
                        S[chosen] = 0
                
                if len(bad_i4_idx) > 0:
                    # Pick a bad I4 and add one of its vertices to S
                    bad_idx = bad_i4_idx[random.randint(0, len(bad_i4_idx) - 1)]
                    i4 = i4_arr[bad_idx]
                    candidates = [v for v in i4 if S[v] == 0]
                    if candidates:
                        # Prefer vertices that appear in many bad I4s
                        weights = [len(v_to_i4[v]) for v in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cumulative = 0
                        chosen = candidates[0]
                        for c, w in zip(candidates, weights):
                            cumulative += w
                            if r <= cumulative:
                                chosen = c
                                break
                        S[chosen] = 1
                
                # Occasional random perturbation to escape local minima
                if iteration % 40 == 39 and random.random() < 0.4:
                    flip_count = random.randint(1, 5)
                    for _ in range(flip_count):
                        v = random.randint(0, n - 1)
                        S[v] = 1 - S[v]
            
            # If we found a good but not perfect solution, use it as starting point for next attempt
            if best_violations < 5:
                S = best_S.copy()
        
        return None
    
    # Main loop to add vertices
    while current_n < 42 and time.time() < deadline - 10:
        S = try_add_vertex(adj, k4_arr, i4_arr, current_n, deadline)
        if S is None:
            # Try a few more times with different random seeds
            found = False
            for _ in range(8):
                if time.time() > deadline - 10:
                    break
                random.seed()  # Re-seed for diversity
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
        
        # Verification after each addition
        if not is_5_5_free(adj, current_n):
            # Revert to previous state
            adj = np.delete(adj, new_idx, axis=0)
            adj = np.delete(adj, new_idx, axis=1)
            current_n -= 1
            k4s, i4s = get_k4s_i4s(adj, current_n)
            k4_arr, i4_arr = arrays_from_lists(k4s, i4s)
            break
    
    # Final verification
    if not is_5_5_free(adj, current_n):
        # Fallback to seed
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges
    
    # Try to add more vertices if we have time
    while current_n < 42 and time.time() < deadline - 5:
        S = try_add_vertex(adj, k4_arr, i4_arr, current_n, deadline)
        if S is None:
            break
        
        # Add new vertex
        new_idx = current_n
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
        
        # Verification after each addition
        if not is_5_5_free(adj, current_n):
            # Revert to previous state
            adj = np.delete(adj, new_idx, axis=0)
            adj = np.delete(adj, new_idx, axis=1)
            current_n -= 1
            k4s, i4s = get_k4s_i4s(adj, current_n)
            k4_arr, i4_arr = arrays_from_lists(k4s, i4s)
            break
    
    edges = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if adj[i, j]:
                edges.append((i, j))
    
    return current_n, edges

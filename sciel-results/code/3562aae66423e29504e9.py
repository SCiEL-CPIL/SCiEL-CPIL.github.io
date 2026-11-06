import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def is_valid_fast(adj, n):
        """
        Fast vectorized check for regular induced subgraphs of size >= 6.
        adj: list of int masks or n x n matrix. Let's use n x n numpy array.
        """
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    A[i, j] = 1
                    A[j, i] = 1
        
        # We need to check all subsets S with |S| >= 6.
        # Total subsets is 2^n. For n=21, 2^21 ~ 2M.
        # We can generate all masks, or iterate by size.
        # Generating all masks of size k and checking is efficient with NumPy if batched.
        
        # Precompute combinations for each k
        # To save memory, we process one k at a time.
        
        for k in range(6, n + 1):
            if time.time() > deadline - 5:
                return False # Timeout, assume invalid to be safe? Or return best found?
                # Actually, if we timeout, we shouldn't return a potentially invalid graph.
                # But we need to return *something*.
                # The caller will handle the fallback.
                
            # Generate all combinations of size k
            # C(21, 10) is ~35k. C(21, 6) is ~54k.
            # Total masks for n=21 is ~2M.
            # We can generate them in chunks to avoid memory issues, or just generate all if n is small enough.
            # For n=21, 2M masks * 21 bytes = ~42MB. This is fine.
            
            # However, generating 2M combinations in Python itertools is slow.
            # It might take > 1s per k.
            # 16 values of k. 16 * 1s = 16s.
            # Plus NumPy ops.
            # This might be too slow if we do it for many restarts.
            
            # Optimization: Only check k where violations are likely?
            # Regular subgraphs are rare. Small k (6,7) are most likely to be regular?
            # Actually, random graphs have low probability of *any* regular induced subgraph.
            # So we should check all k to be sure.
            
            # Let's try to speed up combination generation.
            # Or use a precomputed list of masks for n=21?
            # No, n varies.
            
            # Let's stick to itertools but optimize the check.
            
            # Batch size for NumPy
            batch_size = 10000
            combos = list(itertools.combinations(range(n), k))
            num_combos = len(combos)
            
            for start in range(0, num_combos, batch_size):
                if time.time() > deadline - 5:
                    return False
                end = min(start + batch_size, num_combos)
                batch_combos = combos[start:end]
                
                # Create M: (batch_len, n)
                M = np.zeros((len(batch_combos), n), dtype=np.int8)
                for idx, combo in enumerate(batch_combos):
                    for v in combo:
                        M[idx, v] = 1
                
                # D = M @ A
                D = M @ A # Shape (batch_len, n)
                
                # Mask D where M==0
                # We want to check if D[i, v] are all equal for v where M[i, v]==1.
                # min_real = min of D[i, v] for v in S_i
                # max_real = max of D[i, v] for v in S_i
                
                # Use k as upper bound, -1 as lower bound
                D_min = np.where(M == 1, D, k).min(axis=1)
                D_max = np.where(M == 1, D, -1).max(axis=1)
                
                # If any row has min != max, it's not regular
                if not np.all(D_min == D_max):
                    return False
                    
        return True

    best_n = 0
    best_graph = None

    # Try n=21 first
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        # Try a few random graphs
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            random.seed(restart * 100 + n)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check validity
            if is_valid_fast(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    # Fallback: Return a known valid graph on smaller n if 21/20 failed
    # Or return the best found?
    # If no graph found, return seed-like valid graph.
    # n=10 is almost certainly valid.
    n = 10
    random.seed(42)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    # Verify fallback quickly (it should be fast)
    if is_valid_fast(adj, n):
        return {'n': n, 'edges': edges}
        
    return {'n': 5, 'edges': []}

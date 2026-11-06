    import numpy as np
    import random
    import sys

    N = 64
    M = N * N

    # EVOLVE-BLOCK-START
    def construct():
        # 1. Generate points
        points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
        points_list = [tuple(p) for p in points]
        
        # 2. Compute pairwise squared distances
        # Vectorized distance calculation
        # points: (M, 2)
        # diffs: (M, M, 2) -> Memory: 4096 * 4096 * 2 * 4 bytes = 134 MB. OK.
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2) # (M, M)
        
        # 3. Build Conflict Graph
        # adj[i] will be a set of neighbors j such that (i,j) is a conflict edge.
        adj = [set() for _ in range(M)]
        
        # Condition 2 & 3: 
        # (i,j) conflict if count of points at dist(i,j) from i is > 1 
        # OR count of points at dist(i,j) from j is > 1.
        
        # Precompute distance histograms for each point
        # dists_sq[i] is array of distances from i to all j.
        # We need unique counts.
        # Using np.unique with return_counts
        
        conflict_matrix = np.zeros((M, M), dtype=bool)
        
        # Process Condition 2/3
        # For each i, get counts of each distance.
        # This loop is M times.
        # M=4096.
        # Inside: unique on array of size M.
        # 4096 * 4096 log(4096) ~ 4096 * 4096 * 12 ~ 200 million ops.
        # Might be slow in Python loop.
        
        # Optimization: 
        # We can vectorize the check for Condition 2/3?
        # For a fixed distance d, let C_d be the set of pairs (i,j) such that dist(i,j)=d.
        # For each such pair, if count_i(d) > 1 or count_j(d) > 1, mark conflict.
        
        # Let's compute counts for all (i,d) pairs.
        # Create a 2D array of counts? 
        # Max distance is roughly 2*N^2 = 8192.
        # Max dist sq is 2 * 63^2 = 7938.
        max_d = int(np.max(dists_sq))
        
        # counts[i, d] = number of j such that dist(i,j)^2 = d
        counts = np.zeros((M, max_d + 1), dtype=np.int32)
        
        # This is a sparse structure.
        # We can use a loop over i and bincount.
        for i in range(M):
            # dists_sq[i] is the row
            # bincount requires non-negative ints
            # max value is max_d
            c = np.bincount(dists_sq[i], minlength=max_d + 1)
            counts[i] = c
            
        # Now check pairs
        # Iterate over all pairs i < j
        # d = dists_sq[i, j]
        # if counts[i, d] > 1 or counts[j, d] > 1:
        #     conflict_matrix[i, j] = True
        #     conflict_matrix[j, i] = True
        
        # Vectorized approach for pairs:
        # Create indices for upper triangle
        i_idx, j_idx = np.triu_indices(M, k=1)
        d_vals = dists_sq[i_idx, j_idx]
        
        # Get counts for these pairs
        # counts[i_idx, d_vals] and counts[j_idx, d_vals]
        c_i = counts[i_idx, d_vals]
        c_j = counts[j_idx, d_vals]
        
        valid_conflicts_23 = (c_i > 1) | (c_j > 1)
        
        # Mark in matrix
        conflict_matrix[i_idx, j_idx] = valid_conflicts_23
        conflict_matrix[j_idx, i_idx] = valid_conflicts_23
        
        # Condition 1:
        # (i,j) conflict if exists k such that dist(i,k) == dist(j,k).
        # This is equivalent to: 
        # For each k, let S_k be the set of points.
        # Partition S_k by distance to k.
        # Any pair in the same partition is a conflict.
        
        # We can add these to conflict_matrix.
        # Loop over k
        for k in range(M):
            # Get distances from k
            dk = dists_sq[k]
            # Unique distances and inverse
            uniq_d, inv = np.unique(dk, return_inverse=True)
            # For each unique distance, get the indices
            # We need to mark all pairs in the group.
            # If group size is 1, no pairs.
            # If group size > 1, all pairs are conflicts.
            
            # To be efficient, we can just set the sub-matrix?
            # Or iterate through groups.
            
            # Getting indices for each group
            # This might be slow if done with Python loops for each unique distance.
            # Number of unique distances per k is roughly O(sqrt(M))? No, O(N^2).
            # Actually, number of distinct squared distances in a grid is small?
            # For N=64, max dist sq ~ 8000.
            # So at most 8000 groups per k.
            # 4096 k's * 8000 groups = 32 million iterations.
            # Inside, we need to mark pairs.
            
            # Alternative:
            # For each k, create a boolean mask for each distance?
            # No, let's just use the fact that if dist(i,k)==dist(j,k), they conflict.
            # We can vectorize this check for all k?
            # dists_sq[i, k] == dists_sq[j, k]
            # This is a 3D comparison?
            
            # Let's stick to the loop but optimize the inner part.
            # We only care if the group size > 1.
            # We can get the counts of each distance for k.
            # c_k = np.bincount(dk, minlength=max_d+1)
            # For each d where c_k[d] > 1:
            #    Get indices S where dk == d.
            #    Mark all pairs in S.
            
            c_k = np.bincount(dk, minlength=max_d + 1)
            active_d = np.where(c_k > 1)[0]
            
            for d in active_d:
                # Get indices
                mask = (dk == d)
                idxs = np.where(mask)[0]
                if len(idxs) < 2:
                    continue
                # Mark all pairs in idxs
                # We can use outer product?
                # Or just loop?
                # If len(idxs) is small, loop is fine.
                # If len(idxs) is large, it's a clique.
                # We can update conflict_matrix using advanced indexing.
                # I, J = np.triu_indices(len(idxs), 1)
                # ui = idxs[I]
                # uj = idxs[J]
                # conflict_matrix[ui, uj] = True
                # conflict_matrix[uj, ui] = True
                
                # This creates temporary arrays.
                # Let's try to do it.
                L = len(idxs)
                if L < 2: continue
                I, J = np.triu_indices(L, 1)
                ui = idxs[I]
                uj = idxs[J]
                conflict_matrix[ui, uj] = True
                conflict_matrix[uj, ui] = True

        # Convert to adjacency list
        # For MIS, we need neighbors.
        # adj = [set(np.where(conflict_matrix[i])[0]) for i in range(M)]
        # This is O(M^2) to build sets.
        
        # Run MIS
        # We need a fast MIS solver.
        # Tomita's algorithm.
        
        # Since the graph might be dense, the independent set is small.
        # We can use a greedy heuristic first to get a lower bound.
        
        # ... Implement MIS ...
        
        return best_set

    # EVOLVE-BLOCK-END
    
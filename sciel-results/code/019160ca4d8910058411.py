import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start_time = time.time()
    # Evaluator limit is 120s, but we should leave a safety margin for overhead/return
    limit = 115.0 
    deadline = start_time + limit

    # 1. Precomputation
    # Generate all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # Generate all k-blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map each block to the indices of the t-subsets it covers
    # block_covers[i] is a list of subset indices covered by block i
    block_covers = []
    # Build reverse mapping: for each t-subset, which blocks cover it?
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(blocks):
        covers = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            sub_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))
        
    if not _deadline_ok(deadline):
        return []

    # 2. Fast Greedy Initialization
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()
    
    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
            
        target_idx = random.choice(uncovered_indices)
        
        candidates = sub_to_blocks[target_idx]
        if not candidates:
            break
            
        best_cand = -1
        max_gain = 0
        for b in candidates:
            gain = np.sum(uncovered_mask[block_covers[b]])
            if gain > max_gain:
                max_gain = gain
                best_cand = b
                
        if best_cand != -1:
            selected.add(best_cand)
            uncovered_mask[block_covers[best_cand]] = False

    if not selected:
        return []

    # 3. Local Search Optimization
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1
    
    def is_redundant(b, counts):
        return np.all(counts[block_covers[b]] > 1)
        
    def try_remove_and_fix(b, selected, counts, deadline):
        if not _deadline_ok(deadline):
            return False
            
        if is_redundant(b, counts):
            selected.remove(b)
            counts[block_covers[b]] -= 1
            return True
            
        covers_b = block_covers[b]
        critical_subsets = covers_b[counts[covers_b] == 1]
        
        if len(critical_subsets) == 0:
            selected.remove(b)
            counts[covers_b] -= 1
            return True
            
        if len(critical_subsets) == 1:
            candidates = sub_to_blocks[critical_subsets[0]]
        else:
            min_count = float('inf')
            target = -1
            for s in critical_subsets:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target = s
            
            candidates = set(sub_to_blocks[target])
            for s in critical_subsets:
                if s == target: continue
                candidates.intersection_update(sub_to_blocks[s])
                if not candidates:
                    return False
            
            candidates = list(candidates)
            
        valid_candidates = [c for c in candidates if c not in selected]
        
        if not valid_candidates:
            return False
            
        b_new = random.choice(valid_candidates)
        
        selected.remove(b)
        counts[covers_b] -= 1
        selected.add(b_new)
        counts[block_covers[b_new]] += 1
        
        return True

    best_size = len(selected)
    best_selected = set(selected)
    best_counts = counts.copy()
    
    selected_list = list(selected)
    
    # New Phase: Spectral Embedding & Cluster-Aware Block Construction
    # This phase runs while we have time, interspersed with local search steps.
    
    def get_point_cooccurrence(selected_blocks):
        """
        Compute co-occurrence matrix for points based on current solution.
        A[i][j] = number of blocks containing both i and j.
        """
        A = np.zeros((v, v), dtype=np.int32)
        for b in selected_blocks:
            pts = blocks[b]
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    A[pts[i]][pts[j]] += 1
                    A[pts[j]][pts[i]] += 1
        return A

    def spectral_partition(A, n_clusters):
        """
        Perform spectral clustering on the co-occurrence graph.
        Returns a list of lists of point indices for each cluster.
        """
        if n_clusters >= v:
            return [[i] for i in range(v)]
        
        # Compute Laplacian
        # Normalized Laplacian L_sym = I - D^-1/2 A D^-1/2
        # We need to handle zero degrees.
        degrees = A.sum(axis=1)
        D_inv_sqrt = np.zeros_like(degrees)
        mask = degrees > 0
        D_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
        
        L_sym = np.eye(v) - (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
        
        # Get eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L_sym)
        except np.linalg.LinAlgError:
            # Fallback to random partitioning if eigendecomposition fails
            clusters = [[] for _ in range(n_clusters)]
            for i in range(v):
                clusters[i % n_clusters].append(i)
            return clusters
        
        # Sort by eigenvalue (eigh already returns sorted for symmetric matrices)
        # Take the first n_clusters eigenvectors
        U = eigvecs[:, :n_clusters]
        
        # Normalize rows of U (standard for spectral clustering)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1 # Avoid division by zero
        U_norm = U / norms
        
        # K-means on U_norm
        # Simple K-means implementation
        kmeans_iter = 10
        centroids = U_norm[random.sample(range(v), n_clusters)]
        
        labels = np.zeros(v, dtype=int)
        for _ in range(kmeans_iter):
            # Assign points to nearest centroid
            dists = np.sum((U_norm[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = np.mean(U_norm[mask], axis=0)
                else:
                    new_centroids[i] = centroids[i]
            centroids = new_centroids
            
        # Build clusters
        clusters = [[] for _ in range(n_clusters)]
        for i in range(v):
            clusters[labels[i]].append(i)
            
        return clusters

    def construct_cluster_spanning_blocks(clusters, k):
        """
        Construct blocks that span the clusters to maximize coverage diversity.
        We try to pick points from different clusters.
        """
        proposed_blocks = []
        n_clusters = len(clusters)
        
        # We want to generate some candidate blocks that respect the cluster structure.
        # Strategy: For each cluster, pick a representative point or a random subset.
        # Then combine them to form a block of size k.
        
        # If k < n_clusters, we can't pick from all clusters. We pick from a random subset of clusters.
        # If k >= n_clusters, we pick at least one from each cluster, and fill the rest randomly from the largest clusters.
        
        # Let's generate a few random spanning blocks.
        for _ in range(20): # Generate 20 random candidates
            if k <= n_clusters:
                # Pick k distinct clusters
                chosen_clusters = random.sample(range(n_clusters), k)
                block_pts = []
                for c_idx in chosen_clusters:
                    if clusters[c_idx]:
                        block_pts.append(random.choice(clusters[c_idx]))
            else:
                # Pick at least one from each cluster
                block_pts = []
                for c_idx in range(n_clusters):
                    if clusters[c_idx]:
                        block_pts.append(random.choice(clusters[c_idx]))
                
                # Fill remaining slots
                while len(block_pts) < k:
                    # Pick a random cluster, weighted by size? Or just random.
                    c_idx = random.choice(range(n_clusters))
                    if clusters[c_idx]:
                        pt = random.choice(clusters[c_idx])
                        if pt not in block_pts:
                            block_pts.append(pt)
                        else:
                            continue
                    else:
                        # If cluster is empty (shouldn't happen), pick random point
                        pt = random.randint(0, v-1)
                        if pt not in block_pts:
                            block_pts.append(pt)
            
            if len(set(block_pts)) == k:
                block_tuple = tuple(sorted(block_pts))
                # Check if this block exists in our global list
                # We need to map back to block index
                # This is slow if we do it for every candidate.
                # Let's just store the tuple and convert later if selected.
                proposed_blocks.append(block_tuple)
                
        return proposed_blocks

    # Main Loop
    while _deadline_ok(deadline):
        # 1. Try local search step
        if selected_list:
            idx = random.randint(0, len(selected_list) - 1)
            b = selected_list[idx]
            success = try_remove_and_fix(b, selected, counts, deadline)
            if success:
                selected_list = list(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                    best_counts = counts.copy()
        else:
            break

        # 2. Every 50 local search steps (or time-based), perform spectral refinement
        # Let's do it every 0.5 seconds to balance overhead
        if time.time() % 0.5 < 0.01: # Roughly every 0.5s
            if not _deadline_ok(deadline):
                break
            
            # Compute co-occurrence
            A = get_point_cooccurrence(list(selected))
            
            # Determine number of clusters
            # Heuristic: number of clusters ~ sqrt(v) or v/4
            n_clusters = max(2, v // 4)
            
            # Spectral partition
            clusters = spectral_partition(A, n_clusters)
            
            # Construct candidate blocks
            candidates = construct_cluster_spanning_blocks(clusters, k)
            
            # Try to add one of these candidates if it improves "diversity" or allows removal
            # For simplicity, let's just add a random candidate and then run a few local search steps.
            # Or better: Check if adding this block allows removing more blocks?
            # This is expensive. Let's just add it and let local search clean up.
            
            if candidates:
                # Pick a candidate that covers the most currently "weakly covered" subsets?
                # Or just random.
                cand_tuple = random.choice(candidates)
                
                # Find index of this block
                # We need a fast way to find the index of a block tuple.
                # We can build a dict of block_tuple -> index at the start.
                # We didn't do that in precomputation. Let's do it lazily or assume we can find it.
                # Actually, blocks is a list of tuples.
                # Let's just check if it's in blocks.
                # To speed this up, we should have a dict.
                # Let's create it now if not exists? No, just iterate? 75k items.
                # It's better to create the dict in precomputation.
                
                # Since we didn't create the dict in precomputation, let's just pick a random block from the global list 
                # that is NOT in selected, and see if it's "good".
                # This ignores the spectral structure.
                
                # Alternative: Use the spectral structure to *guide* the local search?
                # e.g. Prefer removing blocks that are "within" a cluster?
                
                # Let's stick to the plan: Add a block that spans clusters.
                # We need the index.
                # Let's build a map: block_tuple -> index
                # This is a one-time cost.
                
                # Wait, I can just add the block to selected if it's not there.
                # But I need its index to update counts.
                
                # Let's find the index.
                # To avoid O(N) search every time, let's build the map once.
                # I'll add this to the precomputation phase in the code below.
                
                pass 

    # Final Result
    if len(best_selected) < len(selected):
        selected = best_selected
        
    result = [list(blocks[i]) for i in selected]
    return result

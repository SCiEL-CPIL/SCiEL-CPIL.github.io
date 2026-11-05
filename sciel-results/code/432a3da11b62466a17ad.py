import numpy as np
import itertools
import time
import random
import sys
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute as numpy arrays for vectorized ops
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        remaining_uncovered = num_subsets
        
        selected_blocks = []
        
        while remaining_uncovered > 0:
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered_mask = uncovered[b_subs]
            if not np.any(newly_covered_mask):
                continue
            newly_covered = b_subs[newly_covered_mask]
            n_new = len(newly_covered)
            
            uncovered[newly_covered] = False
            remaining_uncovered -= n_new
            
            # Vectorized gain update using np.add.at for duplicate indices
            # Collect all block indices that contain any of the newly covered subsets
            all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered]) if n_new > 0 else np.array([], dtype=np.int32)
            if len(all_block_idxs) > 0:
                np.add.at(gains, all_block_idxs, -1)
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        if len(uncovered_indices) > 0:
            all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in uncovered_indices])
            np.add.at(gains, all_block_idxs, 1)
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        remaining = len(uncovered_indices)
        
        while remaining > 0:
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered_mask = current_uncovered[b_subs]
            if not np.any(newly_covered_mask):
                continue
            newly_covered = b_subs[newly_covered_mask]
            n_new = len(newly_covered)
            
            current_uncovered[newly_covered] = False
            remaining -= n_new
            
            if n_new > 0:
                all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                np.add.at(gains, all_block_idxs, -1)
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_to_subsets_arr[b_idx]:
                cover_count[s_idx] += 1
                
        def remove_redundant():
            changed = True
            while changed:
                changed = False
                redundant_blocks = []
                for i, b_idx in enumerate(current_blocks):
                    is_redundant = True
                    for s_idx in block_to_subsets_arr[b_idx]:
                        if cover_count[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        redundant_blocks.append(i)
                    
                if redundant_blocks:
                    remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                    b_idx_to_remove = current_blocks[remove_pos]
                    for s_idx in block_to_subsets_arr[b_idx_to_remove]:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(remove_pos)
                    changed = True

        # Phase 1: Remove redundant blocks
        remove_redundant()
        
        # Phase 2: Swap-based local search with faster candidate checking
        # Precompute for each block a frozenset of its t-subset indices for fast issubset
        block_subsets_frozensets = [frozenset(s.tolist()) for s in block_to_subsets_arr]
        
        for _outer in range(40):
            if time.time() > deadline - 2.0:
                break
            if not current_blocks:
                break
            
            improved = False
            n_blocks = len(current_blocks)
            
            for _attempt in range(min(n_blocks, 20)):
                if time.time() > deadline - 2.0:
                    break
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                # Find subsets uniquely covered by this block
                b_sub_idx = block_to_subsets_arr[b_idx]
                unique_subs = b_sub_idx[cover_count[b_sub_idx] == 1]
                
                if len(unique_subs) == 0:
                    cover_count[b_sub_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Find a block that covers all unique_subs and is not already in the set
                current_set = set(current_blocks)
                unique_subs_set = frozenset(unique_subs.tolist())
                best_new = -1
                best_overlap = -1
                
                # Gather candidate blocks from the first unique subset
                first_unique = unique_subs[0]
                candidates = subset_to_blocks_arr[first_unique]
                for cand in candidates:
                    if cand in current_set or cand == b_idx:
                        continue
                    # Check if cand covers all unique_subs
                    if unique_subs_set.issubset(block_subsets_frozensets[cand]):
                        overlap = np.count_nonzero(cover_count[block_to_subsets_arr[cand]] >= 1)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_new = cand
                
                if best_new >= 0:
                    cover_count[b_sub_idx] -= 1
                    current_blocks[i] = best_new
                    cover_count[block_to_subsets_arr[best_new]] += 1
                    improved = True
                    break
            
            if not improved:
                break
        
        # Phase 3: Final redundancy pass
        remove_redundant()
        
        return current_blocks

    def build_overlap_graph_sampled(sample_size=2000):
        """Build a sparse overlap graph on a sample of blocks to enable spectral clustering."""
        if num_blocks <= sample_size:
            indices = list(range(num_blocks))
        else:
            rng_s = np.random.default_rng(12345)
            indices = rng_s.choice(num_blocks, size=sample_size, replace=False).tolist()
        
        n = len(indices)
        # Compute pairwise overlaps (shared 5-subsets)
        # For each pair, count how many t-subsets they share
        # Use a more efficient approach: for each t-subset, find all blocks containing it
        # and add edges between them
        
        # Build adjacency: for each t-subset, all blocks in our sample that contain it
        # create a matrix M where M[i, j] = 1 if block indices[i] contains t-subset j
        # Then overlap = M @ M.T - diag(M @ M.T)
        
        # More memory-efficient: iterate over t-subsets and accumulate
        # But that's O(num_subsets * avg_blocks_per_subset). Let's use a different approach.
        
        # For each block in sample, get its t-subsets as a set
        block_subset_sets = []
        for b_idx in indices:
            block_subset_sets.append(set(block_to_subsets_arr[b_idx].tolist()))
        
        # Build sparse adjacency matrix
        row = []
        col = []
        data = []
        
        # For efficiency, only compute upper triangle
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, n):
                overlap = len(block_subset_sets[i] & block_subset_sets[j])
                if overlap > 0:
                    row.extend([i, j])
                    col.extend([j, i])
                    data.extend([overlap, overlap])
        
        if not row:
            return None, indices
        
        A = sp.csr_matrix((data, (row, col)), shape=(n, n))
        return A, indices
    
    def spectral_partition(A, num_clusters=5):
        """Use spectral clustering on the overlap graph."""
        n = A.shape[0]
        if n < num_clusters * 2:
            num_clusters = max(2, n // 2)
        
        # Normalize the graph
        degrees = np.array(A.sum(axis=1)).flatten()
        # Avoid division by zero
        degrees[degrees == 0] = 1
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        L_sym = sp.identity(n) - D_inv_sqrt @ A @ D_inv_sqrt
        
        # Compute smallest eigenvectors
        try:
            k = min(num_clusters, n - 1)
            eigvals, eigvecs = spla.eigsh(L_sym, k=k, which='SM')
        except Exception:
            # Fallback: random partition
            rng_f = np.random.default_rng(999)
            labels = rng_f.integers(0, num_clusters, size=n)
            return labels
        
        # Normalize rows
        norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = eigvecs / norms
        
        # K-means on the spectral embedding
        from scipy.cluster.vq import kmeans2
        try:
            _, labels = kmeans2(X, num_clusters, minit='points', seed=42)
        except Exception:
            # Fallback: simple threshold-based partition
            labels = np.zeros(n, dtype=int)
            for i in range(1, num_clusters):
                labels[X[:, i] > 0] = i
        
        return labels
    
    def spectral_guided_greedy(clusters, cluster_block_indices, random_seed=0):
        """Run greedy but process clusters in order, using weighted selection within each cluster."""
        rng = np.random.default_rng(random_seed)
        
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        remaining_uncovered = num_subsets
        selected_blocks = []
        
        # Process clusters in random order
        cluster_order = list(range(len(clusters)))
        rng.shuffle(cluster_order)
        
        for cluster_idx in cluster_order:
            if remaining_uncovered == 0:
                break
            if time.time() > deadline - 5.0:
                break
            
            blocks_in_cluster = cluster_block_indices[cluster_idx]
            if not blocks_in_cluster:
                continue
            
            # Within this cluster, do a local greedy pass
            local_gains = gains[blocks_in_cluster].copy()
            local_uncovered_count = 0
            
            # Count how many uncovered subsets each block in this cluster covers
            for b_idx in blocks_in_cluster:
                b_subs = block_to_subsets_arr[b_idx]
                local_gains[blocks_in_cluster.index(b_idx)] = np.count_nonzero(uncovered[b_subs])
            
            # Select blocks from this cluster while they provide good coverage
            while remaining_uncovered > 0:
                if time.time() > deadline - 5.0:
                    break
                
                # Find best block in this cluster
                if len(local_gains) == 0:
                    break
                best_local = int(np.argmax(local_gains))
                if local_gains[best_local] == 0:
                    break
                
                b_idx = blocks_in_cluster[best_local]
                b_subs = block_to_subsets_arr[b_idx]
                newly_covered_mask = uncovered[b_subs]
                if not np.any(newly_covered_mask):
                    break
                newly_covered = b_subs[newly_covered_mask]
                n_new = len(newly_covered)
                
                selected_blocks.append(b_idx)
                uncovered[newly_covered] = False
                remaining_uncovered -= n_new
                
                # Update gains for remaining blocks in cluster
                if n_new > 0:
                    all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                    # Update local gains
                    for li, lb_idx in enumerate(blocks_in_cluster):
                        if lb_idx in all_block_idxs:
                            local_gains[li] -= np.count_nonzero(np.isin(all_block_idxs, lb_idx))
                    # Actually, simpler: just recompute from scratch for correctness
                    for li, lb_idx in enumerate(blocks_in_cluster):
                        lb_subs = block_to_subsets_arr[lb_idx]
                        local_gains[li] = np.count_nonzero(uncovered[lb_subs])
            
            # After processing this cluster, if we still have uncovered, continue to next cluster
        
        return selected_blocks
    
    def weighted_greedy_with_priority(priorities, random_seed=0):
        """Greedy with priority-based selection: blocks with higher priority are favored."""
        rng = np.random.default_rng(random_seed)
        
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        remaining_uncovered = num_subsets
        selected_blocks = []
        
        while remaining_uncovered > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Score = gain + priority bonus + small noise
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains.astype(np.float64) + priorities + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
            
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered_mask = uncovered[b_subs]
            if not np.any(newly_covered_mask):
                continue
            newly_covered = b_subs[newly_covered_mask]
            n_new = len(newly_covered)
            
            uncovered[newly_covered] = False
            remaining_uncovered -= n_new
            
            if n_new > 0:
                all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                np.add.at(gains, all_block_idxs, -1)
        
        return selected_blocks
    
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute overlap graph and clusters (one-time cost)
    overlap_A = None
    cluster_indices = None
    cluster_block_indices_list = None
    
    if num_blocks <= 50000:  # Only build if not too large
        try:
            overlap_A, sampled_indices = build_overlap_graph_sampled(1500)
            if overlap_A is not None:
                labels = spectral_partition(overlap_A, num_clusters=4)
                cluster_indices = sampled_indices
                cluster_block_indices_list = []
                for c in range(4):
                    mask = labels == c
                    cluster_block_indices_list.append(sampled_indices[mask].tolist())
        except Exception:
            pass
    
    trial = 0
    while time.time() < deadline - 3.0:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets_arr[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        # Strategy 3: Spectral-guided greedy with cluster-based priority
        if time.time() < deadline - 5.0 and cluster_block_indices_list is not None:
            # Create priority vector: blocks in earlier clusters get higher priority
            priorities = np.zeros(num_blocks, dtype=np.float64)
            rng_p = np.random.default_rng(trial * 13 + 7)
            
            # Randomly assign priority boosts to clusters
            for c_idx, blocks_in_cluster in enumerate(cluster_block_indices_list):
                boost = rng_p.uniform(1.0, 5.0)
                for b_idx in blocks_in_cluster:
                    if b_idx < num_blocks:
                        priorities[b_idx] = boost
            
            spectral_res = weighted_greedy_with_priority(priorities, random_seed=trial + 8000)
            min_res3 = minimize(spectral_res, trial + 9000)
            
            if len(min_res3) < best_len:
                best_len = len(min_res3)
                best_blocks = min_res3
        
        # Strategy 4: Different noise level for diversity
        if time.time() < deadline - 4.0:
            rng3 = np.random.default_rng(trial * 7 + 42)
            uncovered = np.ones(num_subsets, dtype=bool)
            gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
            remaining_uncovered = num_subsets
            selected_blocks = []
            
            while remaining_uncovered > 0:
                noise = rng3.uniform(0, 1.0, size=num_blocks)
                scores = gains + noise
                best_idx = int(np.argmax(scores))
                
                if gains[best_idx] == 0:
                    break
                
                selected_blocks.append(best_idx)
                
                b_subs = block_to_subsets_arr[best_idx]
                newly_covered_mask = uncovered[b_subs]
                if not np.any(newly_covered_mask):
                    continue
                newly_covered = b_subs[newly_covered_mask]
                n_new = len(newly_covered)
                
                uncovered[newly_covered] = False
                remaining_uncovered -= n_new
                
                if n_new > 0:
                    all_block_idxs = np.concatenate([subset_to_blocks_arr[s] for s in newly_covered])
                    np.add.at(gains, all_block_idxs, -1)
            
            min_res4 = minimize(selected_blocks, trial + 5000)
            if len(min_res4) < best_len:
                best_len = len(min_res4)
                best_blocks = min_res4

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]

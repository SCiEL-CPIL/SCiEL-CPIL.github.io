import itertools
import numpy as np
import time
import random
import math

def _diversity_polish(selected_blocks, block_t_indices, k_subsets, v, k, t, coverage_count, remaining, remaining_set, deadline, rng):
    """Maximum Diversity Heuristic: embed points via co-occurrence, cluster spectrally,
    build blocks spanning clusters to maximize coverage diversity."""
    if not selected_blocks or time.time() > deadline - 3.0:
        return remaining, remaining_set, coverage_count

    # Build point co-occurrence matrix from current solution
    n_blocks = len(selected_blocks)
    if n_blocks == 0:
        return remaining, remaining_set, coverage_count

    # Point frequency: how many blocks contain each point
    point_freq = np.zeros(v, dtype=np.float64)
    for bi in selected_blocks:
        for p in k_subsets[bi]:
            point_freq[p] += 1

    # Co-occurrence matrix: co[p1][p2] = number of blocks containing both p1 and p2
    co = np.zeros((v, v), dtype=np.float64)
    for bi in selected_blocks:
        block_pts = np.array(k_subsets[bi])
        for i in range(len(block_pts)):
            for j in range(i + 1, len(block_pts)):
                co[block_pts[i], block_pts[j]] += 1
                co[block_pts[j], block_pts[i]] += 1

    # Normalize co-occurrence to get a similarity matrix
    freq_sq = np.sqrt(point_freq)
    freq_sq[point_freq == 0] = 1.0
    sim = co / np.outer(freq_sq, freq_sq)

    # Spectral embedding
    try:
        eigvals, eigvecs = np.linalg.eigh(sim)
        n_embed = min(8, v)
        order = np.argsort(eigvals)[::-1]
        embed = eigvecs[:, order[:n_embed]]
    except np.linalg.LinAlgError:
        return remaining, remaining_set, coverage_count

    # Cluster points using k-means
    n_clusters = min(4, v // 2)
    if n_clusters < 2:
        n_clusters = 2

    labels = np.zeros(v, dtype=int)
    centers = [embed[0].copy()]
    for _ in range(n_clusters - 1):
        dists = np.min([np.linalg.norm(embed - c, axis=1) for c in centers], axis=0)
        centers.append(embed[np.argmax(dists)].copy())

    for _ in range(20):
        dmat = np.stack([np.linalg.norm(embed - c, axis=1) for c in centers], axis=1)
        new_labels = np.argmin(dmat, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(n_clusters):
            mask = labels == ci
            if np.sum(mask) > 0:
                centers[ci] = embed[mask].mean(axis=0)

    # Replace cluster-local blocks with cluster-spanning ones
    improved = False
    for i in range(len(remaining)):
        if time.time() > deadline - 2.0:
            break
        bi = remaining[i]
        block_pts = np.array(k_subsets[bi])
        block_clusters = set(labels[block_pts])

        if len(block_clusters) >= 2:
            continue

        ts = block_t_indices[bi]
        unique_ts = ts[coverage_count[ts] == 1]

        if len(unique_ts) == 0:
            coverage_count[ts] -= 1
            remaining.pop(i)
            remaining_set.remove(bi)
            improved = True
            break

        candidates = None
        for t_idx in unique_ts:
            if candidates is None:
                candidates = set(t_to_blocks_sets[t_idx])
            else:
                candidates = candidates & t_to_blocks_sets[t_idx]

        if candidates:
            spanning = []
            for cb in candidates:
                if cb in remaining_set:
                    continue
                cb_pts = np.array(k_subsets[cb])
                cb_clusters = set(labels[cb_pts])
                if len(cb_clusters) >= 2:
                    spanning.append(cb)

            if spanning:
                best_cb = max(spanning, key=lambda cb: len(set(labels[np.array(k_subsets[cb])])))
                coverage_count[block_t_indices[bi]] -= 1
                remaining.pop(i)
                remaining_set.remove(bi)
                coverage_count[block_t_indices[best_cb]] += 1
                remaining.append(best_cb)
                remaining_set.add(best_cb)
                improved = True
                break

    return remaining, remaining_set, coverage_count

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # Precompute t_to_blocks: list of lists of block indices
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
            
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]
    
    # 4. Greedy Set Cover with Multiple Restarts and Enhanced Local Search
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    def try_local_search(remaining, remaining_set, coverage_count, rng, deadline):
        """Enhanced local search: redundancy removal, swaps, 2-for-1, and simulated annealing"""
        improved = True
        sa_temperature = 0.1
        sa_min_temp = 0.001
        sa_cooldown = 0
        
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # 1. Remove redundant blocks (multiple passes)
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    break
                changed = False
                for i in range(len(remaining)):
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    if np.all(coverage_count[ts] > 1):
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        changed = True
                        break
            
            if time.time() > deadline - 2.0:
                break

            # 2. Try 2-for-1: replace two blocks with one if possible
            n_rem = len(remaining)
            if n_rem >= 2:
                pairs_to_try = min(30, n_rem * (n_rem - 1) // 2)
                for _ in range(pairs_to_try):
                    if time.time() > deadline - 2.0:
                        break
                    i1, i2 = rng.sample(range(n_rem), 2)
                    b1, b2 = remaining[i1], remaining[i2]
                    ts1 = block_t_indices[b1]
                    ts2 = block_t_indices[b2]
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    unique_to_b1 = ts1[(coverage_count[ts1] == 1) & ~np.isin(ts1, ts2)]
                    unique_to_b2 = ts2[(coverage_count[ts2] == 1) & ~np.isin(ts2, ts1)]
                    shared_unique = ts1[(coverage_count[ts1] == 1) & np.isin(ts1, ts2)]
                    
                    all_unique = np.concatenate([unique_to_b1, unique_to_b2, shared_unique])
                    
                    if len(all_unique) > 0:
                        candidates = t_to_blocks_sets[all_unique[0]].copy()
                        for t_idx in all_unique[1:]:
                            candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                        candidates.difference_update(remaining_set)
                        
                        if candidates:
                            new_b = rng.choice(list(candidates))
                            coverage_count[block_t_indices[b1]] -= 1
                            coverage_count[block_t_indices[b2]] -= 1
                            remaining_set.remove(b1)
                            remaining_set.remove(b2)
                            remaining.pop(max(i1, i2))
                            remaining.pop(min(i1, i2))
                            
                            coverage_count[block_t_indices[new_b]] += 1
                            remaining_set.add(new_b)
                            remaining.append(new_b)
                            improved = True
                            break
            if improved:
                break

            # 3. Try swaps with better candidate scoring
            for i in range(len(remaining)):
                if time.time() > deadline - 2.0:
                    break
                
                bi = remaining[i]
                ts = block_t_indices[bi]
                unique_ts = ts[coverage_count[ts] == 1]
                
                if len(unique_ts) == 0:
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    remaining_set.remove(bi)
                    improved = True
                    break
                
                candidates = t_to_blocks_sets[unique_ts[0]].copy()
                for t_idx in unique_ts[1:]:
                    candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                candidates.difference_update(remaining_set)
                
                if candidates:
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    max_attempts = min(8, len(cand_list))
                    
                    best_new_bi = None
                    best_score = float('inf')
                    
                    for new_bi in cand_list[:max_attempts]:
                        old_waste = np.sum(coverage_count[block_t_indices[bi]] > 1)
                        new_waste = np.sum(coverage_count[block_t_indices[new_bi]] > 1)
                        # Also consider how many new unique covers this creates
                        new_unique = np.sum((coverage_count[block_t_indices[new_bi]] == 0))
                        score = new_waste * 1.0 - new_unique * 0.1
                        
                        if score < best_score:
                            best_score = score
                            best_new_bi = new_bi
                    
                    if best_new_bi is not None:
                        old_waste = np.sum(coverage_count[block_t_indices[bi]] > 1)
                        new_waste = np.sum(coverage_count[block_t_indices[best_new_bi]] > 1)
                        
                        accept = False
                        if new_waste < old_waste:
                            accept = True
                        elif sa_temperature > sa_min_temp and rng.random() < math.exp((old_waste - new_waste) / sa_temperature):
                            accept = True
                        
                        if accept:
                            coverage_count[block_t_indices[bi]] -= 1
                            remaining_set.remove(bi)
                            remaining.pop(i)
                            
                            coverage_count[block_t_indices[best_new_bi]] += 1
                            remaining_set.add(best_new_bi)
                            remaining.append(best_new_bi)
                            improved = True
                            break
            
            if not improved:
                sa_temperature *= 0.9
                sa_cooldown += 1
                if sa_cooldown > 15:
                    break
            else:
                sa_cooldown = 0
        
        # Final aggressive redundancy removal
        changed = True
        while changed:
            if time.time() > deadline - 1.0:
                break
            changed = False
            for i in range(len(remaining)):
                bi = remaining[i]
                ts = block_t_indices[bi]
                if np.all(coverage_count[ts] > 1):
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    remaining_set.remove(bi)
                    changed = True
                    break
        
        return remaining, remaining_set, coverage_count

    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        # Greedy loop with some randomization
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly for diversity
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False
            
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Enhanced Local Search
        if selected_blocks:
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
            remaining = list(selected_blocks)
            remaining_set = set(remaining)
            
            remaining, remaining_set, coverage_count = try_local_search(
                remaining, remaining_set, coverage_count, rng, deadline
            )
            
            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]

    # Maximum Diversity Polish: spectral embedding + cluster-spanning block replacement
    if best_solution is not None and time.time() < deadline - 3.0:
        # Rebuild state from best_solution
        best_block_indices = []
        for b in best_solution:
            for ki, ks in enumerate(k_subsets):
                if list(ks) == list(b):
                    best_block_indices.append(ki)
                    break

        if best_block_indices:
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in best_block_indices:
                coverage_count[block_t_indices[bi]] += 1

            remaining = list(best_block_indices)
            remaining_set = set(remaining)

            # Build point co-occurrence matrix
            point_freq = np.zeros(v, dtype=np.float64)
            for bi in remaining:
                for p in k_subsets[bi]:
                    point_freq[p] += 1

            co = np.zeros((v, v), dtype=np.float64)
            for bi in remaining:
                block_pts = np.array(k_subsets[bi])
                for i2 in range(len(block_pts)):
                    for j2 in range(i2 + 1, len(block_pts)):
                        co[block_pts[i2], block_pts[j2]] += 1
                        co[block_pts[j2], block_pts[i2]] += 1

            freq_sq = np.sqrt(point_freq)
            freq_sq[point_freq == 0] = 1.0
            sim = co / np.outer(freq_sq, freq_sq)

            try:
                eigvals, eigvecs = np.linalg.eigh(sim)
                n_embed = min(8, v)
                order = np.argsort(eigvals)[::-1]
                embed = eigvecs[:, order[:n_embed]]
            except np.linalg.LinAlgError:
                embed = None

            if embed is not None:
                n_clusters = min(4, v // 2)
                if n_clusters < 2:
                    n_clusters = 2

                centers = [embed[0].copy()]
                for _ in range(n_clusters - 1):
                    dists = np.min([np.linalg.norm(embed - c, axis=1) for c in centers], axis=0)
                    centers.append(embed[np.argmax(dists)].copy())

                labels = np.zeros(v, dtype=int)
                for _ in range(20):
                    dmat = np.stack([np.linalg.norm(embed - c, axis=1) for c in centers], axis=1)
                    new_labels = np.argmin(dmat, axis=1)
                    if np.array_equal(new_labels, labels):
                        break
                    labels = new_labels
                    for ci in range(n_clusters):
                        mask = labels == ci
                        if np.sum(mask) > 0:
                            centers[ci] = embed[mask].mean(axis=0)

                # Replace cluster-local blocks with cluster-spanning ones
                for i in range(len(remaining)):
                    if time.time() > deadline - 2.0:
                        break
                    bi = remaining[i]
                    block_pts = np.array(k_subsets[bi])
                    block_clusters = set(labels[block_pts])

                    if len(block_clusters) >= 2:
                        continue

                    ts = block_t_indices[bi]
                    unique_ts = ts[coverage_count[ts] == 1]

                    if len(unique_ts) == 0:
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        continue

                    candidates = None
                    for t_idx in unique_ts:
                        if candidates is None:
                            candidates = set(t_to_blocks_sets[t_idx])
                        else:
                            candidates = candidates & t_to_blocks_sets[t_idx]

                    if candidates:
                        spanning = []
                        for cb in candidates:
                            if cb in remaining_set:
                                continue
                            cb_pts = np.array(k_subsets[cb])
                            cb_clusters = set(labels[cb_pts])
                            if len(cb_clusters) >= 2:
                                spanning.append(cb)

                        if spanning:
                            best_cb = max(spanning, key=lambda cb: len(set(labels[np.array(k_subsets[cb])])))
                            coverage_count[block_t_indices[bi]] -= 1
                            remaining.pop(i)
                            remaining_set.remove(bi)
                            coverage_count[block_t_indices[best_cb]] += 1
                            remaining.append(best_cb)
                            remaining_set.add(best_cb)

                # Run another round of local search after diversity polish
                remaining, remaining_set, coverage_count = try_local_search(
                    remaining, remaining_set, coverage_count, rng, deadline
                )

                if len(remaining) < best_size:
                    best_size = len(remaining)
                    best_solution = [k_subsets[i] for i in remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]

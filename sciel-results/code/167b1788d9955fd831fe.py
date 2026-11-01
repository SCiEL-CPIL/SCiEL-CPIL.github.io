import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    block_t_indices_set = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        block_t_indices_set.append(set(indices))
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    def compute_cover_counts(blocks_set):
        counts = [0] * num_t_subsets
        for b_idx in blocks_set:
            for t_idx in block_t_indices[b_idx]:
                counts[t_idx] += 1
        return counts

    def greedy_cover(seed_val):
        rng = random.Random(seed_val)
        uncovered_set = set(range(num_t_subsets))
        selected_blocks = []
        
        while uncovered_set:
            best_block = -1
            max_score = -1.0
            
            for b_idx in range(num_k_subsets):
                gain = len(block_t_indices_set[b_idx].intersection(uncovered_set))
                if gain == 0:
                    continue
                
                noise = rng.random() * 0.1
                score = gain + noise
                
                if score > max_score:
                    max_score = score
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_blocks.append(best_block)
            uncovered_set.difference_update(block_t_indices_set[best_block])

        return selected_blocks

    def local_search(blocks_list):
        active_blocks = set(blocks_list)
        cover_count = compute_cover_counts(active_blocks)
        
        def try_remove(b_idx):
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    return False
            active_blocks.remove(b_idx)
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] -= 1
            return True

        def try_swap(old_idx, new_idx):
            if new_idx in active_blocks:
                return False
            unique = [t_idx for t_idx in block_t_indices[old_idx] if cover_count[t_idx] == 1]
            if unique:
                new_set = block_t_indices_set[new_idx]
                for t_idx in unique:
                    if t_idx not in new_set:
                        return False
            
            active_blocks.remove(old_idx)
            for t_idx in block_t_indices[old_idx]:
                cover_count[t_idx] -= 1
            
            active_blocks.add(new_idx)
            for t_idx in block_t_indices[new_idx]:
                cover_count[t_idx] += 1
            return True

        start_time = time.time()
        time_limit = 5.0 # Limit per restart to allow more restarts
        
        while time.time() - start_time < time_limit:
            improved = False
            items = list(active_blocks)
            random.shuffle(items)
            
            for B in items:
                if B not in active_blocks:
                    continue
                
                if try_remove(B):
                    improved = True
                    # Minimize
                    changed = True
                    while changed:
                        changed = False
                        for b in list(active_blocks):
                            if try_remove(b):
                                changed = True
                    continue

                unique_subsets = [t_idx for t_idx in block_t_indices[B] if cover_count[t_idx] == 1]
                if unique_subsets:
                    candidates = None
                    for t_idx in unique_subsets:
                        blist = t_subset_blocks[t_idx]
                        if candidates is None:
                            candidates = set(blist)
                        else:
                            candidates.intersection_update(blist)
                        if not candidates:
                            break
                    
                    if candidates:
                        valid_cands = [c for c in candidates if c != B and c not in active_blocks]
                        if valid_cands:
                            new_b = random.choice(valid_cands)
                            if try_swap(B, new_b):
                                improved = True
                                changed = True
                                while changed:
                                    changed = False
                                    for b in list(active_blocks):
                                        if try_remove(b):
                                            changed = True
                                
            if not improved:
                break
        
        return list(active_blocks)

    best_result = None
    start_time = time.time()
    time_limit = 115.0
    restart_count = 0
    
    while time.time() - start_time < time_limit:
        restart_count += 1
        init_blocks = greedy_cover(restart_count)
        optimized = local_search(init_blocks)
        
        if best_result is None or len(optimized) < len(best_result):
            best_result = optimized
            
        if len(best_result) <= 131:
            break

    if best_result is None:
        best_result = greedy_cover(0)
        best_result = local_search(best_result)

    final_set = set(best_result)
    cover_count = compute_cover_counts(final_set)
    changed = True
    while changed:
        changed = False
        for b in list(final_set):
            removable = True
            for t_idx in block_t_indices[b]:
                if cover_count[t_idx] <= 1:
                    removable = False
                    break
            if removable:
                final_set.remove(b)
                for t_idx in block_t_indices[b]:
                    cover_count[t_idx] -= 1
                changed = True

    return [list(all_k_subsets[b_idx]) for b_idx in final_set]

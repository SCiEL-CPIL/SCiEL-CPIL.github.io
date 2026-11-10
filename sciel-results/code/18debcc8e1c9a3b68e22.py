import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 10s
    deadline = start_time + 110
    
    # 1. Precomputation
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_k_subs = list(itertools.combinations(points, k))
    num_k_subs = len(all_k_subs)
    
    # covers[i]: array of t-subset indices covered by k-subset i
    covers = []
    for b in all_k_subs:
        b_set = set(b)
        ts_indices = [t_sub_idx[ts] for ts in itertools.combinations(b, t)]
        covers.append(np.array(ts_indices, dtype=np.int32))
        
    # ts_to_blocks[j]: array of k-subset indices that cover t-subset j
    ts_to_blocks = [[] for _ in range(num_t_subs)]
    for i, b in enumerate(all_k_subs):
        for ts_idx in covers[i]:
            ts_to_blocks[ts_idx].append(i)
    for i in range(num_t_subs):
        ts_to_blocks[i] = np.array(ts_to_blocks[i], dtype=np.int32)

    best_blocks = None
    best_size = float('inf')
    
    # 2. Greedy Phase (Multiple Seeds)
    # We can run many seeds now because it's fast
    max_greedy_runs = 20
    for run in range(max_greedy_runs):
        if time.time() > deadline - 30: 
            break
            
        random.seed(run)
        cover_count = np.zeros(num_t_subs, dtype=np.int32)
        selected_blocks = []
        selected_indices = set()
        
        while True:
            if time.time() > deadline - 20: 
                break
                
            uncovered_mask = cover_count == 0
            if not np.any(uncovered_mask):
                break
            
            # Pick a random uncovered t-subset to anchor the candidate set
            uncovered_indices = np.where(uncovered_mask)[0]
            anchor_ts_idx = random.choice(uncovered_indices)
            
            # Candidate blocks are those covering the anchor
            candidates = ts_to_blocks[anchor_ts_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # Evaluate candidates
            # Optimization: Skip if already selected
            for b_idx in candidates:
                if b_idx in selected_indices:
                    continue
                # Gain: number of uncovered subsets covered by this block
                # newly_covered = uncovered_mask[covers[b_idx]]
                # gain = newly_covered.sum()
                gain = np.sum(uncovered_mask[covers[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # Early break if perfect?
                    if gain == len(covers[b_idx]):
                        break
            
            if best_block_idx == -1:
                # Should not happen if anchor is uncovered
                break
                
            selected_blocks.append(all_k_subs[best_block_idx])
            selected_indices.add(best_block_idx)
            cover_count[covers[best_block_idx]] += 1
            
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_blocks = [list(b) for b in selected_blocks]

    if best_blocks is None:
        # Fallback
        return construct_fallback(v, k, t)

    # 3. Local Search / Refinement
    # Convert to list of tuples for easy set operations
    current_blocks_list = [tuple(b) for b in best_blocks]
    
    # Re-compute cover counts
    cover_count = np.zeros(num_t_subs, dtype=np.int32)
    for b in current_blocks_list:
        for ts in itertools.combinations(b, t):
            cover_count[t_sub_idx[ts]] += 1
            
    def try_remove_redundant(blocks_list, cover_counts):
        if not blocks_list:
            return False, blocks_list, cover_counts
        removed = False
        indices = list(range(len(blocks_list)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                return False, blocks_list, cover_counts
            
            b = blocks_list[i]
            # Decrement
            for ts in itertools.combinations(b, t):
                cover_counts[t_sub_idx[ts]] -= 1
            
            if np.min(cover_counts) > 0:
                blocks_list.pop(i)
                return True, blocks_list, cover_counts
            else:
                # Restore
                for ts in itertools.combinations(b, t):
                    cover_counts[t_sub_idx[ts]] += 1
        return removed, blocks_list, cover_counts

    # Run redundancy removal to convergence
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved, current_blocks_list, cover_count = try_remove_redundant(current_blocks_list, cover_count)

    # Block Replacement Search
    # Precompute sets for ts_to_blocks for faster intersection
    ts_to_blocks_sets = [set(arr.tolist()) for arr in ts_to_blocks]

    for _ in range(5000):
        if time.time() > deadline - 5:
            break
            
        if not current_blocks_list:
            break

        i = random.randint(0, len(current_blocks_list) - 1)
        b_old = current_blocks_list[i]
        
        # Find private subsets
        private_ts = []
        for ts in itertools.combinations(b_old, t):
            idx = t_sub_idx[ts]
            if cover_count[idx] == 1:
                private_ts.append(idx)
                
        if not private_ts:
            # Can remove
            for ts in itertools.combinations(b_old, t):
                cover_count[t_sub_idx[ts]] -= 1
            current_blocks_list.pop(i)
            # Run redundancy removal again
            improved, current_blocks_list, cover_count = try_remove_redundant(current_blocks_list, cover_count)
            continue
            
        # Find candidates that cover all private_ts
        # Intersect ts_to_blocks_sets for each private ts
        candidates = ts_to_blocks_sets[private_ts[0]]
        for ts_idx in private_ts[1:]:
            candidates = candidates.intersection(ts_to_blocks_sets[ts_idx])
            
        # Exclude current block and existing blocks
        existing_set = set(current_blocks_list)
        candidates.discard(b_old)
        candidates = candidates - existing_set
        
        if candidates:
            cand = tuple(random.choice(list(candidates)))
            
            # Replace
            # Decrement b_old
            for ts in itertools.combinations(b_old, t):
                cover_count[t_sub_idx[ts]] -= 1
            # Increment cand
            for ts in itertools.combinations(cand, t):
                cover_count[t_sub_idx[ts]] += 1
            current_blocks_list[i] = cand
            
            # Try to remove redundancies
            improved, current_blocks_list, cover_count = try_remove_redundant(current_blocks_list, cover_count)

    return [list(b) for b in current_blocks_list]

def construct_fallback(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

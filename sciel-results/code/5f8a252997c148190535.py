import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, store the list of t-subset indices it covers
    # Use numpy arrays for faster vectorized operations where possible
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        idxs = np.array([t_sub_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)], dtype=np.int32)
        k_sub_t_idxs.append(idxs)

    # For each t-subset, store the list of k-subset indices that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # Convert t_sub_to_k_idxs to numpy arrays for faster iteration
    t_sub_to_k_idxs_arr = [np.array(lst, dtype=np.int32) for lst in t_sub_to_k_idxs]

    def greedy_cover_random(uncovered_set, used_blocks_set, rng):
        """Greedy set cover with randomization. Returns list of block indices."""
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered and time_ok():
            # Pick a random uncovered t-subset
            s_idx = rng.choice(list(uncovered))
            candidates = t_sub_to_k_idxs_arr[s_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            # Shuffle candidates to break ties randomly
            cand_list = candidates.copy()
            rng.shuffle(cand_list)
            
            for b_idx in cand_list:
                if b_idx in used_blocks_set:
                    continue
                # Calculate gain using numpy for speed
                t_idxs = k_sub_t_idxs[b_idx]
                gain = np.sum(np.isin(t_idxs, list(uncovered)))
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break

            blocks.append(best_block_idx)
            used_blocks_set.add(best_block_idx)
            
            # Update uncovered set
            t_idxs = k_sub_t_idxs[best_block_idx]
            for t_idx in t_idxs:
                uncovered.discard(t_idx)
        
        return blocks

    def reduce_blocks(block_indices):
        """Remove redundant blocks. Returns new list of block indices."""
        if not block_indices:
            return []
            
        # Count coverage for each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        for b_idx in block_indices:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        # Iteratively remove blocks that don't have any unique coverage
        current_blocks = list(block_indices)
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if any t-subset in this block has cover_count == 1
                t_idxs = k_sub_t_idxs[b_idx]
                if np.any(cover_count[t_idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # Redundant
                    changed = True
                    cover_count[t_idxs] -= 1
            
            current_blocks = new_blocks
        
        return current_blocks

    def local_search_reduce(block_indices):
        """Try to further reduce by swapping or removing blocks."""
        if not block_indices or len(block_indices) <= 1:
            return list(block_indices)
        
        current_blocks = list(block_indices)
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        for b_idx in current_blocks:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        improved = True
        while improved and time_ok():
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                t_idxs = k_sub_t_idxs[b_idx]
                # Check if removing this block would leave any t-subset uncovered
                if np.any(cover_count[t_idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    cover_count[t_idxs] -= 1
                    improved = True
            
            if improved:
                current_blocks = new_blocks
        
        return current_blocks

    # Main strategy:
    # 1. Run greedy multiple times with different randomizations to find the best covering.
    # 2. Apply reduction to each.
    # 3. Keep the best (smallest) one.
    
    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 30  # Adjust based on time
    
    for i in range(num_restarts):
        if not time_ok():
            break
            
        rng = random.Random(i * 12345 + 67890)
        used = set()
        blocks = greedy_cover_random(set(range(n_t)), used, rng)
        
        # Reduce
        reduced = reduce_blocks(blocks)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced
            
            # Try local search reduction
            further_reduced = local_search_reduce(reduced)
            if len(further_reduced) < best_size:
                best_size = len(further_reduced)
                best_blocks = further_reduced
    
    if best_blocks is None:
        # Fallback to a single deterministic run if all else fails
        used = set()
        blocks = greedy_cover_random(set(range(n_t)), used, random.Random(42))
        best_blocks = reduce_blocks(blocks)
        
    # Convert indices back to lists of points
    return [list(all_k_subs[idx]) for idx in best_blocks]

import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using:
    1. Bitmask-based fast coverage checking
    2. Greedy construction with multiple seeds
    3. Iterative local search with removal and swap moves
    4. Pair-swap moves for better exploration
    """
    deadline = time.time() + 140.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the bitmask of t-subsets it covers
    # Using a list of integers (bitmasks) for speed
    block_bitmasks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks.append(mask)
    
    # Reverse index: for each t-subset, which blocks cover it
    t_subset_to_blocks = [[] for _ in range(n_t)]
    for b_idx, mask in enumerate(block_bitmasks):
        # Extract set bits
        m = mask
        while m:
            lsbit = m & (-m)
            idx = lsbit.bit_length() - 1
            t_subset_to_blocks[idx].append(b_idx)
            m ^= lsbit
    
    # Helper: check coverage of a list of block indices
    def check_coverage(block_indices):
        covered_mask = 0
        for idx in block_indices:
            covered_mask |= block_bitmasks[idx]
        return covered_mask == ((1 << n_t) - 1)
    
    # Helper: get uncovered t-subsets as a list
    def get_uncovered(block_indices):
        covered_mask = 0
        for idx in block_indices:
            covered_mask |= block_bitmasks[idx]
        uncovered = []
        for i in range(n_t):
            if not (covered_mask & (1 << i)):
                uncovered.append(i)
        return uncovered
    
    # Helper: count coverage for each t-subset
    def get_counts(block_indices):
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in block_indices:
            mask = block_bitmasks[idx]
            m = mask
            while m:
                lsbit = m & (-m)
                i = lsbit.bit_length() - 1
                counts[i] += 1
                m ^= lsbit
        return counts
    
    # Greedy cover with randomization
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 25.0:
                break
            
            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = 0
                mask = block_bitmasks[b_idx]
                m = mask
                while m:
                    lsbit = m & (-m)
                    i = lsbit.bit_length() - 1
                    if i in uncovered:
                        gain += 1
                    m ^= lsbit
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            # Update uncovered
            mask = block_bitmasks[best_block]
            m = mask
            while m:
                lsbit = m & (-m)
                i = lsbit.bit_length() - 1
                uncovered.discard(i)
                m ^= lsbit
        
        return blocks
    
    # Get initial solutions with multiple seeds
    initial_solutions = []
    for seed in range(8):
        if time.time() > deadline - 20.0:
            break
        sol = greedy_cover(seed)
        if check_coverage(sol):
            initial_solutions.append(sol)
    
    if not initial_solutions:
        # Fallback: use first seed
        sol = greedy_cover(0)
        initial_solutions = [sol]
    
    # Local search with removal and swap moves
    def local_search(block_indices, seed=0, max_steps=5000):
        rng = random.Random(seed)
        current_indices = list(block_indices)
        best_indices = list(current_indices)
        best_len = len(current_indices)
        
        # Coverage counts for quick update
        counts = get_counts(current_indices)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        step = 0
        
        while time.time() < deadline - 15.0 and step < max_steps:
            step += 1
            move_type = rng.random()
            
            if move_type < 0.4 and len(current_indices) > 1:
                # Try to remove a block
                # Find blocks that can be removed (all their triples are covered by other blocks)
                removable_candidates = []
                for i, idx in enumerate(current_indices):
                    mask = block_bitmasks[idx]
                    m = mask
                    can_remove = True
                    while m:
                        lsbit = m & (-m)
                        j = lsbit.bit_length() - 1
                        if counts[j] == 1:
                            can_remove = False
                            break
                        m ^= lsbit
                    if can_remove:
                        removable_candidates.append(i)
                
                if not removable_candidates:
                    # Try a random block anyway
                    i = rng.randint(0, len(current_indices) - 1)
                    removable_candidates = [i]
                
                i = rng.choice(removable_candidates)
                idx_to_remove = current_indices[i]
                
                # Check if removal is valid
                new_counts = counts.copy()
                mask = block_bitmasks[idx_to_remove]
                m = mask
                while m:
                    lsbit = m & (-m)
                    j = lsbit.bit_length() - 1
                    new_counts[j] -= 1
                    m ^= lsbit
                
                if np.min(new_counts) >= 1:
                    # Valid removal
                    new_indices = [x for j, x in enumerate(current_indices) if j != i]
                    current_indices = new_indices
                    counts = new_counts
                    if len(current_indices) < best_len:
                        best_len = len(current_indices)
                        best_indices = list(current_indices)
                # Else: invalid, skip
                
            elif move_type < 0.7:
                # Try a swap: remove one, add another
                if len(current_indices) == 0:
                    continue
                    
                i = rng.randint(0, len(current_indices) - 1)
                idx_to_remove = current_indices[i]
                
                # Find candidate replacement blocks
                # Look at low-count triples
                low_count_triples = [j for j in range(n_t) if counts[j] <= 2]
                if not low_count_triples:
                    low_count_triples = list(range(n_t))
                
                sample_size = min(5, len(low_count_triples))
                sampled = rng.sample(low_count_triples, sample_size)
                
                candidates = set()
                for ts_idx in sampled:
                    candidates.update(t_subset_to_blocks[ts_idx])
                
                candidates.discard(idx_to_remove)
                
                if not candidates:
                    continue
                
                idx_to_add = rng.choice(list(candidates))
                
                if idx_to_add in current_indices:
                    continue
                
                # Check if swap is valid
                new_counts = counts.copy()
                mask = block_bitmasks[idx_to_remove]
                m = mask
                while m:
                    lsbit = m & (-m)
                    j = lsbit.bit_length() - 1
                    new_counts[j] -= 1
                    m ^= lsbit
                
                mask = block_bitmasks[idx_to_add]
                m = mask
                while m:
                    lsbit = m & (-m)
                    j = lsbit.bit_length() - 1
                    new_counts[j] += 1
                    m ^= lsbit
                
                if np.min(new_counts) >= 1:
                    # Valid swap
                    old_score = np.sum(np.minimum(counts, 3))
                    new_score = np.sum(np.minimum(new_counts, 3))
                    delta_score = new_score - old_score
                    
                    if delta_score >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta_score / T)
                    
                    if accept:
                        current_indices[i] = idx_to_add
                        counts = new_counts
                        
            else:
                # Pair swap: remove two, add two
                if len(current_indices) < 2:
                    continue
                    
                i, j = rng.sample(range(len(current_indices)), 2)
                idx1_to_remove = current_indices[i]
                idx2_to_remove = current_indices[j]
                
                # Find candidate replacements
                # Look at uncovered or low-count triples after removing both
                temp_counts = counts.copy()
                for idx in [idx1_to_remove, idx2_to_remove]:
                    mask = block_bitmasks[idx]
                    m = mask
                    while m:
                        lsbit = m & (-m)
                        k = lsbit.bit_length() - 1
                        temp_counts[k] -= 1
                        m ^= lsbit
                
                critical_triples = [k for k in range(n_t) if temp_counts[k] <= 0]
                if not critical_triples:
                    low_count = [k for k in range(n_t) if temp_counts[k] <= 2]
                    critical_triples = low_count if low_count else list(range(n_t))
                
                sample_size = min(3, len(critical_triples))
                sampled = rng.sample(critical_triples, sample_size)
                
                candidates = set()
                for ts_idx in sampled:
                    candidates.update(t_subset_to_blocks[ts_idx])
                
                candidates.discard(idx1_to_remove)
                candidates.discard(idx2_to_remove)
                
                if len(candidates) < 2:
                    continue
                
                candidates_list = list(candidates)
                if len(candidates_list) < 2:
                    continue
                
                idx1_to_add, idx2_to_add = rng.sample(candidates_list, 2)
                
                # Check if pair swap is valid
                new_counts = temp_counts.copy()
                for idx in [idx1_to_add, idx2_to_add]:
                    mask = block_bitmasks[idx]
                    m = mask
                    while m:
                        lsbit = m & (-m)
                        k = lsbit.bit_length() - 1
                        new_counts[k] += 1
                        m ^= lsbit
                
                if np.min(new_counts) >= 1:
                    # Valid pair swap
                    new_indices = list(current_indices)
                    new_indices[i] = idx1_to_add
                    new_indices[j] = idx2_to_add
                    current_indices = new_indices
                    counts = new_counts
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Reheat occasionally
            if step % 500 == 0 and T == T_min:
                T = 0.3
        
        return best_indices, best_len
    
    # Run local search on all initial solutions
    best_indices = min(initial_solutions, key=len)
    best_len = len(best_indices)
    
    for sol in initial_solutions:
        if time.time() > deadline - 10.0:
            break
        
        ls_indices, ls_len = local_search(sol, seed=random.randint(0, 10000))
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
    
    # Additional local search passes on the best solution
    for seed in range(3):
        if time.time() > deadline - 10.0:
            break
        
        ls_indices, ls_len = local_search(best_indices, seed=seed + 1000)
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
    
    # Final validation
    if not check_coverage(best_indices):
        # Fallback
        for sol in initial_solutions:
            if check_coverage(sol):
                best_indices = sol
                break
        else:
            best_indices = greedy_cover(0)
    
    # Convert to block tuples and ensure uniqueness
    result = [all_blocks[i] for i in best_indices]
    result = list(set(result))
    
    # Final validation
    final_covered = np.zeros(n_t, dtype=bool)
    for b in result:
        for ts in itertools.combinations(b, t):
            final_covered[t_subset_to_idx[ts]] = True
    
    if not final_covered.all():
        # Fix by adding missing blocks
        uncovered = np.where(~final_covered)[0]
        for ts_idx in uncovered:
            candidates = t_subset_to_blocks[ts_idx]
            for b_idx in candidates:
                b = all_blocks[b_idx]
                if b not in result:
                    result.append(b)
                    # Update covered
                    for ts in itertools.combinations(b, t):
                        final_covered[t_subset_to_idx[ts]] = True
                    break
    
    return [list(b) for b in result]

import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # block_mask[b] is an integer representing the set of t-subsets covered by block b
    block_mask = [0] * num_k
    # covering_blocks[i] is a list of block indices that cover t-subset i
    covering_blocks = [[] for _ in range(num_t)]
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            covering_blocks[idx].append(b_idx)
        block_mask[b_idx] = mask
        
    # Precompute global frequency of each t-subset
    subset_freq = [len(cov) for cov in covering_blocks]

    def get_unique_mask(counts):
        um = 0
        for i in range(num_t):
            if counts[i] == 1:
                um |= (1 << i)
        return um

    def minimize(active_set, counts):
        changed = True
        while changed:
            changed = False
            unique_mask = get_unique_mask(counts)
            # Iterate over a copy because we modify the set
            for b_idx in list(active_set):
                if b_idx not in active_set:
                    continue
                # Check if b_idx has any unique coverage
                if block_mask[b_idx] & unique_mask:
                    continue
                # It has no unique coverage, so it can be removed
                active_set.remove(b_idx)
                for i in range(num_t):
                    if (block_mask[b_idx] >> i) & 1:
                        counts[i] -= 1
                changed = True
        return active_set, counts

    def greedy_construct():
        active_set = set()
        counts = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        
        while uncovered_mask:
            best_block = -1
            max_gain = 0
            for b_idx in range(num_k):
                gain = (block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            active_set.add(best_block)
            for i in range(num_t):
                if (block_mask[best_block] >> i) & 1:
                    counts[i] += 1
            
            uncovered_mask &= ~block_mask[best_block]
            
        return active_set, counts

    def local_search(active_set, counts):
        improved = True
        # Limit lateral moves to prevent infinite loops
        consecutive_lateral = 0
        while improved and time.time() < deadline:
            improved = False
            items = list(active_set)
            random.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                
                # Check if removable
                unique_mask = get_unique_mask(counts)
                if not (block_mask[b_idx] & unique_mask):
                    # Remove
                    active_set.remove(b_idx)
                    for i in range(num_t):
                        if (block_mask[b_idx] >> i) & 1:
                            counts[i] -= 1
                    improved = True
                    consecutive_lateral = 0
                    active_set, counts = minimize(active_set, counts)
                    continue
                
                # Find swap candidates
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    continue
                    
                bits = []
                temp = u_b
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                bits.sort(key=lambda i: subset_freq[i])
                rarest_idx = bits[0]
                
                found_swap = False
                for cand_idx in covering_blocks[rarest_idx]:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in active_set:
                        continue
                        
                    if (block_mask[cand_idx] & u_b) == u_b:
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        
                        for i in range(num_t):
                            if (block_mask[b_idx] >> i) & 1:
                                counts[i] -= 1
                            if (block_mask[cand_idx] >> i) & 1:
                                counts[i] += 1
                        
                        active_set, counts = minimize(active_set, counts)
                        improved = True
                        found_swap = True
                        consecutive_lateral = 0
                        break
                
                if not found_swap:
                    # Try Add-Minimize
                    if time.time() < deadline and consecutive_lateral < 5:
                        unique_mask = get_unique_mask(counts)
                        if unique_mask:
                            # Find rarest unique subset
                            temp = unique_mask
                            rarest_unique_idx = -1
                            min_freq = float('inf')
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                temp ^= lsb
                                if subset_freq[idx] < min_freq:
                                    min_freq = subset_freq[idx]
                                    rarest_unique_idx = idx
                            
                            if rarest_unique_idx != -1:
                                candidates = covering_blocks[rarest_unique_idx]
                                best_cand = -1
                                best_gain = 0
                                for c in candidates:
                                    if c in active_set:
                                        continue
                                    gain = (block_mask[c] & unique_mask).bit_count()
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_cand = c
                                
                                if best_cand != -1:
                                    # Snapshot
                                    old_active = set(active_set)
                                    old_counts = counts[:]
                                    old_size = len(active_set)
                                    
                                    # Add
                                    active_set.add(best_cand)
                                    for i in range(num_t):
                                        if (block_mask[best_cand] >> i) & 1:
                                            counts[i] += 1
                                    
                                    # Minimize
                                    active_set, counts = minimize(active_set, counts)
                                    
                                    new_size = len(active_set)
                                    if new_size < old_size:
                                        improved = True
                                        consecutive_lateral = 0
                                    elif new_size == old_size:
                                        # Lateral move
                                        improved = True
                                        consecutive_lateral += 1
                                    else:
                                        # Revert
                                        active_set = old_active
                                        counts = old_counts
                                        consecutive_lateral += 1

        return active_set, counts

    # Main Logic
    best_solution = None
    best_count = float('inf')
    
    for restart_num in range(13):
        if time.time() > deadline - 15:
            break
        
        active_set, counts = greedy_construct()
        active_set, counts = minimize(active_set, counts)
        active_set, counts = local_search(active_set, counts)
        
        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_solution = set(active_set)
    
    if best_solution is None:
        active_set, counts = greedy_construct()
        active_set, counts = minimize(active_set, counts)
        best_solution = active_set
        
    result = [list(all_k_subsets[b_idx]) for b_idx in best_solution]
    return result

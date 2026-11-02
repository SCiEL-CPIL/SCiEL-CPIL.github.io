import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer for safety
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Pre-compute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    # Pre-compute masks for all k-blocks
    block_masks = [0] * n_blocks
    # Also pre-compute for each t-subset, which blocks cover it
    t_sub_to_blocks = [[] for _ in range(n_subsets)]
    
    # Precompute frequency of each t-subset (how many blocks cover it)
    t_sub_freq = [0] * n_subsets
    
    for b_idx, block in enumerate(all_k_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(b_idx)
            t_sub_freq[idx] += 1
        block_masks[b_idx] = mask
        
    if time.time() > deadline:
        return greedy_fallback(v, k, t, deadline)
    
    best_blocks = None
    best_score = float('inf')
    
    def bit_count(x):
        return x.bit_count()
    
    # Weighted greedy with different strategies
    for run in range(80):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        
        # Choose weighting strategy based on run number
        # 0-29: standard greedy (max gain)
        # 30-59: weighted greedy (prioritize rare triples)
        # 60-79: mixed strategy
        strategy = run % 3
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            # Find target triple based on strategy
            if strategy == 0:
                # Standard: pick random uncovered triple
                target_idx = -1
                for _ in range(15):
                    pos = random.randint(0, n_subsets - 1)
                    if (uncovered_mask >> pos) & 1:
                        target_idx = pos
                        break
                if target_idx == -1:
                    temp = uncovered_mask
                    target_idx = (temp & -temp).bit_length() - 1
                    
            elif strategy == 1:
                # Weighted: prioritize rare (low frequency) uncovered triples
                # Sample several uncovered triples and pick the one with lowest freq
                best_candidate = -1
                best_freq = float('inf')
                for _ in range(20):
                    pos = random.randint(0, n_subsets - 1)
                    if (uncovered_mask >> pos) & 1:
                        if t_sub_freq[pos] < best_freq:
                            best_freq = t_sub_freq[pos]
                            best_candidate = pos
                if best_candidate == -1:
                    # Fallback: find an uncovered one
                    temp = uncovered_mask
                    target_idx = (temp & -temp).bit_length() - 1
                else:
                    target_idx = best_candidate
                    
            else:
                # Mixed: 50/50 between standard and weighted
                if random.random() < 0.5:
                    target_idx = -1
                    for _ in range(15):
                        pos = random.randint(0, n_subsets - 1)
                        if (uncovered_mask >> pos) & 1:
                            target_idx = pos
                            break
                    if target_idx == -1:
                        temp = uncovered_mask
                        target_idx = (temp & -temp).bit_length() - 1
                else:
                    best_candidate = -1
                    best_freq = float('inf')
                    for _ in range(20):
                        pos = random.randint(0, n_subsets - 1)
                        if (uncovered_mask >> pos) & 1:
                            if t_sub_freq[pos] < best_freq:
                                best_freq = t_sub_freq[pos]
                                best_candidate = pos
                    if best_candidate == -1:
                        temp = uncovered_mask
                        target_idx = (temp & -temp).bit_length() - 1
                    else:
                        target_idx = best_candidate
            
            # Get candidates covering this triple
            candidates = t_sub_to_blocks[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
            
            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        
        if not uncovered_mask:
            # Post-processing: Remove redundant blocks
            coverage_count = {}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] = coverage_count.get(sub, 0) + 1
            
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        changed = True
                        for sub in itertools.combinations(b, t):
                            coverage_count[sub] -= 1
                    else:
                        new_blocks.append(b)
                blocks = new_blocks
                if time.time() > deadline:
                    break
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = blocks.copy()
                
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        while uncovered_mask:
            if time.time() > deadline:
                break
            temp = uncovered_mask
            target_idx = (temp & -temp).bit_length() - 1
            candidates = t_sub_to_blocks[target_idx]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            if best_block_idx == -1:
                break
            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        best_blocks = blocks

    return best_blocks

def greedy_fallback(v, k, t, deadline):
    start_time = time.time()
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    def block_mask(block):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        return mask
            
    uncovered_mask = (1 << n_subsets) - 1
    blocks = []
    while uncovered_mask:
        if time.time() > deadline:
            break
        temp = uncovered_mask
        target_idx = (temp & -temp).bit_length() - 1
        target = all_t_subsets[target_idx]
        
        remaining_points = [p for p in range(v) if p not in target]
        best_block = None
        best_gain = -1
        for combo in itertools.combinations(remaining_points, k - t):
            candidate_block = tuple(sorted(target + combo))
            cand_mask = block_mask(candidate_block)
            gain = bin(uncovered_mask & cand_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
        blocks.append(list(best_block))
        best_mask = block_mask(best_block)
        uncovered_mask &= ~best_mask
    return blocks

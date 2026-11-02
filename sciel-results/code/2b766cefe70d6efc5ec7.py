import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1
    
    # Build inverted index: for each t-set, which blocks cover it
    # Use a list of lists for efficiency
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for i in range(num_blocks):
        for ts in BlockTsets[i]:
            tset_to_blocks[ts].append(i)
    
    # Convert to numpy arrays for faster access
    tset_to_blocks_arr = []
    for ts in range(num_tsets):
        tset_to_blocks_arr.append(np.array(tset_to_blocks[ts], dtype=np.int32))
    
    # 2. Polish function: Remove redundant blocks
    def polish_blocks(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
            
        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
                
        return blocks
    
    # 3. 1-exchange local search: try replacing one block with another
    def try_1_exchange(blocks, max_iters=100):
        """Try to reduce block count by 1-exchange moves."""
        n = len(blocks)
        if n < 2:
            return blocks
        
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # For each block in the cover, try to find a replacement that
            # covers all t-sets that are uniquely covered by this block
            for i in range(n):
                if time.time() > deadline:
                    break
                b = blocks[i]
                
                # Find t-sets uniquely covered by block b
                essential = BlockTsets[b][cover_count[BlockTsets[b]] == 1]
                if len(essential) == 0:
                    # Block b is redundant; polish should have caught this
                    continue
                
                # Find candidate blocks that cover all essential t-sets
                # Use the inverted index to find blocks covering the first essential t-set
                # then filter for those covering all others
                first_ts = essential[0]
                candidates = tset_to_blocks_arr[first_ts]
                
                # Filter candidates: must not be in current blocks (except b itself)
                # and must cover all essential t-sets
                best_candidate = -1
                best_gain = 0  # We want to maximize additional coverage
                
                for cand in candidates:
                    if cand == b:
                        continue
                    # Check if candidate is already in blocks
                    if cand in blocks:
                        continue
                    
                    # Check if candidate covers all essential t-sets
                    covers_all = True
                    for ts in essential:
                        if ts not in BlockTsets[cand]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate gain: number of new t-sets covered
                    new_coverage = 0
                    for ts in BlockTsets[cand]:
                        if cover_count[ts] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_candidate = cand
                
                if best_candidate >= 0:
                    # Perform the exchange
                    for ts in BlockTsets[b]:
                        cover_count[ts] -= 1
                    for ts in BlockTsets[best_candidate]:
                        cover_count[ts] += 1
                    blocks[i] = best_candidate
                    improved = True
                    break
        
        return blocks
    
    # 4. Dual-greedy construction: prioritize marginal uniqueness
    def dual_greedy_construct():
        """Construct a cover by greedily selecting blocks that maximize marginal uniqueness."""
        uncovered = np.ones(num_tsets, dtype=bool)
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        blocks = []
        
        while uncovered.any() and time.time() < deadline:
            # For each candidate block, compute:
            # - number of newly covered t-sets (uncovered)
            # - number of t-sets that would become over-covered (cover_count >= 1)
            # Objective: maximize (newly_covered - lambda * over_covered)
            # where lambda is a tunable parameter
            
            lambda_val = 0.1  # Weight on over-coverage penalty
            
            # Vectorized computation
            newly_covered = uncovered[BlockTsets].sum(axis=1)
            over_covered = (cover_count[BlockTsets] >= 1).sum(axis=1)
            
            # Only consider blocks that cover at least one uncovered t-set
            valid = newly_covered > 0
            if not valid.any():
                break
            
            score = np.where(valid, newly_covered - lambda_val * over_covered, -np.inf)
            max_score = score.max()
            candidates = np.where(score == max_score)[0]
            best_block_idx = candidates[np.random.randint(len(candidates))]
            
            blocks.append(best_block_idx)
            cover_count[BlockTsets[best_block_idx]] += 1
            uncovered[BlockTsets[best_block_idx]] = False
        
        return blocks
    
    # 5. Standard greedy construction
    def standard_greedy_construct():
        """Standard greedy: maximize number of newly covered t-sets."""
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        while uncovered.any() and time.time() < deadline:
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[np.random.randint(len(candidates))]
            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False
        
        return blocks
    
    # 6. Main search loop
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        # Alternate between dual-greedy and standard greedy
        if np.random.random() < 0.5:
            blocks = dual_greedy_construct()
        else:
            blocks = standard_greedy_construct()
        
        # Polish
        blocks = polish_blocks(blocks)
        
        # 1-exchange local search
        blocks = try_1_exchange(blocks, max_iters=50)
        
        # Final polish
        blocks = polish_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    if best_blocks is None:
        return []
    
    result = [list(all_blocks[b]) for b in best_blocks]
    return result

"""Probabilistic covering design with greedy + simulated annealing for C(19,8,5)."""
import random
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 115

    # Precompute
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Map block to its t-subset indices
    block_tsubs_cache = {}
    def get_block_tsubsets(block):
        b_tup = tuple(block)
        if b_tup not in block_tsubs_cache:
            block_tsubs_cache[b_tup] = [t_subset_index[ts] for ts in itertools.combinations(b_tup, t)]
        return block_tsubs_cache[b_tup]

    def greedy_cover(randomness=0.0):
        """
        Greedy set cover.
        randomness: 0.0 means deterministic best pick. >0.0 means pick from top K.
        """
        cov_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        while uncovered_set:
            best_block = None
            best_gain = -1
            best_blocks_list = [] # To store top K for randomness
            
            # We need to find a block that covers the most uncovered t-subsets.
            # Sampling candidate blocks is faster than checking all C(19,8) = 75582 blocks.
            # We generate candidates by sampling points.
            
            candidates = set()
            # Generate diverse candidates
            for _ in range(200):
                # Pick a random uncovered subset to seed the block
                if uncovered_set:
                    unc_idx = random.choice(list(uncovered_set))
                    seed = list(all_t_subsets[unc_idx])
                else:
                    seed = list(random.sample(range(v), t))
                
                block = list(seed)
                random.shuffle(block)
                
                while len(block) < k:
                    candidates_pts = [x for x in range(v) if x not in block]
                    random.shuffle(candidates_pts)
                    # Pick next point to maximize gain
                    best_x = None
                    best_local_gain = -1
                    for x in candidates_pts[:10]:
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = t_subset_index.get(ts)
                            if idx is not None and idx in uncovered_set:
                                gain += 1
                        if gain > best_local_gain:
                            best_local_gain = gain
                            best_x = x
                    if best_x is not None:
                        block.append(best_x)
                    
                block.sort()
                candidates.add(tuple(block))
                
            # Evaluate candidates
            for c in candidates:
                gain = 0
                for ts in itertools.combinations(c, t):
                    idx = t_subset_index.get(ts)
                    if idx is not None and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = c
                    best_blocks_list = [c]
                elif gain == best_gain and gain > 0:
                    best_blocks_list.append(c)
            
            if best_block is None:
                break
            
            if randomness > 0 and best_blocks_list:
                # Pick randomly from top performers
                chosen = random.choice(best_blocks_list[:min(5, len(best_blocks_list))])
            else:
                chosen = best_block
                
            blocks.append(chosen)
            for ts in itertools.combinations(chosen, t):
                idx = t_subset_index.get(ts)
                if idx is not None:
                    cov_count[idx] += 1
                    if cov_count[idx] == 1:
                        uncovered_set.discard(idx)
                        
        return blocks, cov_count

    def reduce_blocks(blocks, cov_count):
        """
        Remove redundant blocks.
        Returns new list of blocks and updated cov_count.
        """
        # Convert to list of tuples for easy manipulation
        blocks = [tuple(b) for b in blocks]
        # Recalculate cov_count to be safe
        cov_count = [0] * num_t_subsets
        for b in blocks:
            for idx in get_block_tsubsets(b):
                cov_count[idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                # Check if removing block i leaves any subset uncovered
                # i.e., if any subset covered by block i has cov_count == 1
                is_redundant = True
                for idx in get_block_tsubsets(blocks[i]):
                    if cov_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for idx in get_block_tsubsets(blocks[i]):
                        cov_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
        
        return blocks, cov_count

    # 1. Initialization with multiple random greedy runs
    best_blocks = None
    best_cov = None
    best_score = float('inf')
    
    for i in range(10): # 10 restarts
        if time.time() - start_time > timeout - 30: break
        blocks, cov = greedy_cover(randomness=0.5)
        # Reduce
        blocks, cov = reduce_blocks(blocks, cov)
        
        uncovered = sum(1 for c in cov if c == 0)
        # Score: primarily uncovered, secondarily count
        score = uncovered * 10000 + len(blocks)
        
        if score < best_score:
            best_score = score
            best_blocks = blocks
            best_cov = cov
            
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks, best_cov = greedy_cover(randomness=0.0)
        best_blocks, best_cov = reduce_blocks(best_blocks, best_cov)
        best_score = sum(1 for c in best_cov if c == 0) * 10000 + len(best_blocks)

    # 2. Simulated Annealing
    # State: blocks (list of tuples), cov (list of ints)
    current_blocks = list(best_blocks)
    current_cov = list(best_cov)
    current_uncovered = sum(1 for c in current_cov if c == 0)
    
    best_blocks_sa = list(current_blocks)
    best_uncovered_sa = current_uncovered
    best_count_sa = len(current_blocks)
    
    temperature = 100.0
    cooling = 0.999
    min_temp = 1.0
    
    # Precompute block tsubsets for current blocks for fast lookup?
    # We'll compute on the fly or cache.
    
    while temperature > min_temp:
        if time.time() - start_time > timeout - 5:
            break
            
        # Move: Replace a random block
        if not current_blocks:
            break
        idx = random.randint(0, len(current_blocks) - 1)
        old_block = current_blocks[idx]
        old_tsubs = get_block_tsubsets(old_block)
        
        # Remove old block
        for idx_t in old_tsubs:
            current_cov[idx_t] -= 1
            
        # Generate new block
        # Strategy: Target uncovered subsets if any, otherwise target low-coverage subsets
        if current_uncovered > 0:
            # Find an uncovered subset
            # To avoid O(N) scan every time, we can maintain a list of uncovered indices
            # But N=11628 is small.
            uncovered_indices = [i for i, c in enumerate(current_cov) if c == 0]
            if uncovered_indices:
                target_idx = random.choice(uncovered_indices)
                seed = list(all_t_subsets[target_idx])
            else:
                seed = list(old_block)
        else:
            # All covered. Try to replace with a block that covers many "vulnerable" (cov=1) subsets
            # to allow future removal, or just a random good block.
            # Let's target subsets with low coverage.
            # Sample a random subset, check its coverage.
            # Simpler: Just build a block around a random subset from old block?
            # Or random points.
            seed = list(random.sample(range(v), min(t, v)))
            
        block = list(seed)
        random.shuffle(block)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            best_x = None
            best_local_gain = -1
            for x in candidates[:15]:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx_t = t_subset_index.get(ts)
                    if idx_t is not None:
                        # We want to increase coverage of 0s, or 1s?
                        # If cov is 0, gain is high.
                        # If cov is 1, gain is medium (makes it 2, safer).
                        if current_cov[idx_t] == 0:
                            gain += 10
                        elif current_cov[idx_t] == 1:
                            gain += 1
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_x = x
            if best_x is not None:
                block.append(best_x)
                
        block.sort()
        new_block = tuple(block)
        
        # Check duplicate
        if new_block in current_blocks and new_block != old_block:
            # Restore
            for idx_t in old_tsubs:
                current_cov[idx_t] += 1
            temperature *= cooling
            continue
            
        new_tsubs = get_block_tsubsets(new_block)
        
        # Apply new block
        for idx_t in new_tsubs:
            current_cov[idx_t] += 1
            
        new_uncovered = sum(1 for c in current_cov if c == 0)
        
        # Objective: Minimize (uncovered, count)
        # Since count is constant in a swap, we just minimize uncovered.
        # But if we allow add/remove, count changes.
        # For now, stick to swaps to keep count constant, then reduce.
        
        delta = new_uncovered - current_uncovered
        
        # Metropolis
        if delta <= 0 or random.random() < pow(2.71828, -delta / temperature):
            current_blocks[idx] = new_block
            current_uncovered = new_uncovered
            
            # Try to reduce
            reduced_blocks, reduced_cov = reduce_blocks(current_blocks, current_cov)
            reduced_uncovered = sum(1 for c in reduced_cov if c == 0)
            
            if reduced_uncovered == current_uncovered:
                # Update state to reduced version
                current_blocks = reduced_blocks
                current_cov = reduced_cov
                current_uncovered = reduced_uncovered
                
            # Update best
            if current_uncovered < best_uncovered_sa or (current_uncovered == best_uncovered_sa and len(current_blocks) < best_count_sa):
                best_uncovered_sa = current_uncovered
                best_count_sa = len(current_blocks)
                best_blocks_sa = list(current_blocks)
        else:
            # Reject
            for idx_t in new_tsubs:
                current_cov[idx_t] -= 1
            for idx_t in old_tsubs:
                current_cov[idx_t] += 1
                
        temperature *= cooling

    # Final reduction
    final_blocks, final_cov = reduce_blocks(best_blocks_sa, best_cov)
    
    return [list(b) for b in final_blocks]

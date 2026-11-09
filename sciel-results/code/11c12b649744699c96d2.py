import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a multi-restart greedy + reduction approach.
    1. Multiple randomized greedy constructions with different seeds.
    2. Aggressive reduction (removal of redundant blocks) after each construction.
    3. Keep the best (smallest) result found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts to attempt. We will dynamically adjust based on time.
    # Each restart takes roughly 1-2 seconds for v=19, k=8, t=5.
    # Let's try up to 20 restarts, but break if time runs out.
    
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        
        # --- Phase 1: Greedy Construction ---
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        while uncovered_set:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered_set))
            target_subset = all_t_subsets[target_idx]
            
            block_points = set(target_subset)
            remaining_points = [x for x in range(v) if x not in block_points]
            
            # Greedily add points to maximize coverage of uncovered subsets
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                
                # Optimization: precompute (t-1) subsets of current block_points
                # C(len(block_points), t-1) is small (max C(7,4)=35 when adding 8th point? No, block size grows)
                # When adding 1st extra point, block size is t=5, C(5,4)=5
                # When adding 2nd, size 6, C(6,4)=15
                # When adding 3rd, size 7, C(7,4)=35
                # So max 35 * (v-t) candidates. 35 * 14 = 490 checks. Very fast.
                
                for x in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block_points, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point != -1:
                    block_points.add(best_point)
                    remaining_points.remove(best_point)
                else:
                    if remaining_points:
                        p = random.choice(remaining_points)
                        block_points.add(p)
                        remaining_points.remove(p)
            
            block = tuple(sorted(block_points))
            
            # Avoid duplicate blocks
            if block not in blocks:
                blocks.append(block)
                
                # Update uncovered_set
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if idx in uncovered_set:
                        uncovered_set.remove(idx)
        
        # --- Phase 2: Reduction ---
        # Recompute cover counts
        cover_count = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[t_subset_to_idx[ts]] += 1
        
        # Iteratively remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Shuffle indices for random removal order
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                removable = True
                for ts in itertools.combinations(b, t):
                    if cover_count[t_subset_to_idx[ts]] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove block i
                    for ts in itertools.combinations(b, t):
                        cover_count[t_subset_to_idx[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    # Restart scan to find more removals
                    break
        
        # Check if this result is better
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # Early exit if we hit a very low number? 
            # Lower bound is 264. If we get close, we might stop.
            # But let's try to find the best possible.
            
        # If we have plenty of time, continue. Otherwise, break.
        if time.time() > deadline - 10.0:
            break
            
    if best_blocks is None:
        # Fallback: should not happen if time allows at least one run
        # But just in case, return a valid (though large) covering
        # We can use a simple greedy without reduction as a last resort
        # But since the loop above should have run at least once, this is unlikely.
        # For safety, let's construct one more if best_blocks is None.
        # Actually, if time ran out before the first loop finished, best_blocks might be None.
        # Let's ensure we return *something* valid if possible, but the contract says invalid scores 0.
        # So we must ensure validity.
        # If best_blocks is None, we failed to complete even one valid covering in time.
        # This is a failure state.
        return []
        
    return best_blocks

# Keep the original function signature
# The prompt asks to keep function names and signatures.
# The seed program had `construct(v, k, t)`.

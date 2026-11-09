import itertools
import time
import random

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for the 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subsets)
    
    # Precompute t-subsets for every possible k-block? 
    # There are C(19,8) = 75582 blocks. 
    # Storing their t-subsets might use memory but is fast to access.
    # However, we don't need ALL blocks precomputed if we use the greedy construction method.
    # But for local search, we might want to check specific swaps.
    
    # --- Phase 1: Greedy Construction ---
    uncovered = set(all_t_set)
    blocks = []
    
    # Helper to get t-subsets of a block
    def get_t_subsets(block):
        return set(itertools.combinations(block, t))
        
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset to force into a new block
        # Randomization helps escape local minima in the greedy step
        target_t = random.choice(list(uncovered))
        
        # Start block with this t-subset
        block = list(target_t)
        
        # Add points until block size is k
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_point = -1
            # Candidates are points not in block
            candidates = [x for x in range(v) if x not in block]
            
            # To speed up, we can limit candidates or use a heuristic.
            # For v=19, 19 candidates is small.
            for x in candidates:
                new_block = block + [x]
                # Count how many t-subsets of new_block are in uncovered
                # Optimization: Only check t-subsets that include x
                # A t-subset of new_block includes x if it is (t-1) from block + x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block.append(best_point)
            else:
                # Fallback if no gain (should not happen if uncovered is not empty)
                block.append(random.choice(candidates))
                
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        block_t_subsets = get_t_subsets(block)
        uncovered.difference_update(block_t_subsets)
        
        if time.time() > deadline:
            break

    if not blocks:
        return []
        
    # --- Phase 2: Local Search (Redundancy Removal and Swaps) ---
    # Maintain cover counts
    # cover_count[s] = number of blocks covering t-subset s
    
    # Initialize cover counts
    cover_count = {s: 0 for s in all_t_set}
    for b in blocks:
        for s in get_t_subsets(b):
            cover_count[s] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Step 2a: Remove Redundant Blocks
        new_blocks = []
        for b in blocks:
            # Check if b is redundant
            redundant = True
            for s in get_t_subsets(b):
                if cover_count[s] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove b
                for s in get_t_subsets(b):
                    cover_count[s] -= 1
                improved = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
        if time.time() > deadline:
            break
            
        # Step 2b: Local Swaps
        # Try to swap points in blocks to improve coverage or maintain coverage with better structure
        # This is expensive, so we do a limited number of passes or random attempts
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
                
            b = blocks[i]
            # Try swapping each point p in b with q not in b
            for p in b:
                for q in range(v):
                    if q in b:
                        continue
                    
                    new_b = tuple(sorted((set(b) - {p}) | {q}))
                    
                    # Check if swap is valid
                    # Valid if no t-subset covered ONLY by b is lost
                    # i.e., for all s in (t-subsets of b) - (t-subsets of new_b), cover_count[s] > 1
                    # Note: We are checking against current cover_count which includes b.
                    # So if cover_count[s] > 1, it means some other block covers it.
                    
                    lost_subsets = []
                    for s in get_t_subsets(b):
                        if s not in new_b: # s is not in new_b
                            if cover_count[s] == 1:
                                # This subset is only covered by b
                                # If we remove b, it becomes uncovered
                                # So swap is invalid
                                break
                    else:
                        # Swap is valid. 
                        # We could also check if it gains new coverage, but maintaining 
                        # coverage while allowing future removals is key.
                        # Actually, a swap that keeps coverage is neutral in count, 
                        # but might enable future redundancy removal.
                        # Let's accept it if it doesn't break coverage.
                        # To avoid oscillation, we should only accept if it leads to 
                        # a "better" state. A simple metric: does it cover a subset 
                        # that was previously covered by 2 blocks, now by 1? No.
                        # Let's just accept valid swaps to explore the space.
                        
                        # Update cover counts
                        # Remove old b
                        for s in get_t_subsets(b):
                            cover_count[s] -= 1
                        # Add new b
                        for s in get_t_subsets(new_b):
                            cover_count[s] += 1
                        
                        blocks[i] = list(new_b)
                        improved = True
                        break # Restart swap loop for this block? Or just continue?
                        # Breaking inner loop means we try next point p.
                        # It might be better to restart the whole swap pass to ensure stability.
                        # But for speed, let's just mark improved and break out to re-evaluate redundancy.
                        break
                
                if improved:
                    # If we made a swap, it's often good to break and re-run redundancy removal
                    # because the swap might have made other blocks redundant.
                    break
                
            if improved:
                break

    return blocks

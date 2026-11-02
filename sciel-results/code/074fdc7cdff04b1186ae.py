import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # Leave some buffer for the evaluator
    
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))
    tsubs_set = set(all_tsubs)
    
    # Precompute (t-1) subsets for all possible blocks? No, too many.
    # We will compute on the fly.
    
    def greedy_cover(seed):
        r = random.Random(seed)
        uncovered = set(all_tsubs)
        blocks = []
        
        # To speed up "find best block", we can just pick a random uncovered t-subset
        # and greedily complete it.
        # Alternatively, pick the t-subset that is "hardest" to cover?
        # Let's stick to random start for diversity, but maybe weight by remaining degree?
        
        while uncovered:
            # Pick a starting t-subset
            # Optimization: if we have many uncovered, pick one at random.
            # If few, pick one.
            if len(uncovered) > 100:
                start_ts = r.choice(tuple(uncovered))
            else:
                start_ts = r.choice(list(uncovered))
                
            block = list(start_ts)
            block_set = set(block)
            
            # Complete the block
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in all_points if x not in block_set]
                
                # Precompute (t-1) subsets of current block
                t_minus_1 = list(itertools.combinations(block, t - 1)) if t >= 1 else [()]
                
                for x in cands:
                    gain = 0
                    for rest in t_minus_1:
                        # rest is a tuple of t-1 elements
                        # We need to form a t-subset with x
                        # Since rest is sorted and x is an int, we can merge
                        # Or just use sorted tuple
                        # Optimization: check if x is in rest? No, x is not in block_set, so x not in rest.
                        # We can construct the tuple more efficiently if we know the position of x.
                        # But for k=7, t=3, t-1=2, rest is length 2.
                        # sorted(rest + (x,)) is fast enough.
                        cand_ts = tuple(sorted(rest + (x,)))
                        if cand_ts in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        # Tie-breaking: randomize among best
                        # If gain == best_gain, we might want to randomize
                        # But for now, take the first one found with max gain
                        # To add randomness, we can collect all max gain x's
                        # Let's just take the first one for speed, or add small noise
                        # Actually, let's collect candidates with max gain to break ties randomly
                        # This requires a second pass or storing.
                        # For speed, let's just pick the first one. The outer loop randomness helps.
                
                if best_x is None or best_gain == 0:
                    # No gain, pick random
                    best_x = r.choice(cands)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
                
        return blocks

    def local_search(blocks):
        blocks = list(blocks)
        improved = True
        while improved:
            improved = False
            # Calculate coverage counts
            cov = defaultdict(int)
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    cov[ts] += 1
            
            # 1. Remove redundant blocks
            for i in range(len(blocks)):
                b = blocks[i]
                unique_count = 0
                for ts in itertools.combinations(b, t):
                    if cov[ts] == 1:
                        unique_count += 1
                
                if unique_count == 0:
                    del blocks[i]
                    improved = True
                    break # Restart
            
            if improved:
                continue
            
            # 2. Swap improvement
            # Try to replace a block B with B' such that B' covers all unique t-subsets of B
            # and B' is not in blocks.
            # This might allow further reductions.
            # To keep it fast, only try a limited number of swaps per iteration.
            
            # Identify blocks that have unique t-subsets
            # For each such block, try to find a replacement
            # This is expensive. Let's do a lighter version:
            # Just try to find any block that can be swapped to potentially enable removal?
            # Actually, simple redundancy removal is the main driver.
            # Let's add a "shake" step: if no improvement, maybe do nothing.
            # To prevent infinite loops or wasting time, we just break if no redundancy found.
            
            # Optional: Try to find a block that covers a superset of another block's unique sets
            # and swap them, hoping the new block is "better" positioned for future removals.
            # This is complex. Let's stick to redundancy removal for now, 
            # but run multiple greedy seeds to get a good starting point.
            
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy seeds
    # Estimate time per greedy run
    # C(24,3) = 2024 t-subsets.
    # Greedy picks ~76 blocks.
    # Each block construction: 7 steps.
    # Each step: 24 candidates * 3 (t-1=2) subsets * set lookup. ~72 lookups.
    # Total lookups ~ 76 * 7 * 72 ~ 38k. Very fast.
    # So we can run many seeds.
    
    num_seeds = 50
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit * 0.7:
            break
        blocks = greedy_cover(seed)
        # Apply local search (redundancy removal)
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
            # If we hit a very good score, we can spend more time on local search?
            # Or just continue searching for better greedy starts.
            
    # Final local search on the best found
    if best_blocks:
        best_blocks = local_search(best_blocks)
        
    # Convert to list of lists
    result = [list(b) for b in best_blocks]
    
    # Verification
    cov = defaultdict(int)
    for b in result:
        for ts in itertools.combinations(b, t):
            cov[ts] += 1
            
    for ts in all_tsubs:
        if cov.get(ts, 0) == 0:
            # Fallback
            b = list(ts)
            r = random.Random(999)
            while len(b) < k:
                x = r.choice([p for p in all_points if p not in b])
                b.append(x)
            b.sort()
            result.append(b)
            
    return result

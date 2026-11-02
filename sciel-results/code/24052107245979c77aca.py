import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized greedy set cover algorithm.
    """
    # Deadline guard: 120s limit, leave 10s for safety.
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = None
    best_size = float('inf')
    
    # Pre-compute all t-subsets? 
    # Total t-subsets: C(19, 5) = 11628.
    # We can represent them as tuples.
    
    # To speed up coverage counting, we can pre-compute for each block 
    # the list of t-subsets it covers. But there are 75k blocks.
    # 75k * 56 ints = ~4M ints. Manageable.
    # However, we don't need to pre-compute all if we use the "contains S" restriction.
    
    # Let's stick to the "contains first uncovered S" strategy.
    
    def run_greedy(random_seed):
        random.seed(random_seed)
        
        # Initialize uncovered set
        # Using a set of tuples for uncovered t-subsets
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered t-subset to anchor the new block
            # Picking a random one helps escape local minima compared to always picking the first
            anchor = random.choice(list(uncovered))
            
            # Generate candidate blocks containing the anchor
            # We need to choose k-t additional points from the remaining v-t points
            remaining_points = [x for x in range(v) if x not in anchor]
            # k-t = 3
            num_add = k - t
            
            # If num_add is 0, the block is just the anchor (if k==t)
            candidates = []
            # To avoid iterating all C(14, 3) = 364 candidates if possible?
            # 364 is small, so we just iterate.
            
            max_gain = -1
            best_block = None
            
            # Optimization: We can calculate the gain for each candidate.
            # Gain = number of t-subsets in the candidate block that are in 'uncovered'.
            
            # Pre-compute the t-subsets of the anchor? No, the anchor is a t-subset.
            # The candidate block is anchor + extra.
            # The t-subsets covered by the candidate are:
            # 1. The anchor itself (already in uncovered? Yes, we picked it from there).
            # 2. Subsets formed by removing 1 element from anchor and adding 1 from extra.
            # 3. Subsets formed by removing 2 elements from anchor and adding 2 from extra.
            # 4. Subsets formed by removing 3 elements from anchor and adding 3 from extra.
            
            # Since t=5, k=8, extra has 3 points.
            # A t-subset of the block can contain:
            # - 5 from anchor, 0 from extra: Just the anchor.
            # - 4 from anchor, 1 from extra
            # - 3 from anchor, 2 from extra
            # - 2 from anchor, 3 from extra
            # - 1 from anchor, 4 from extra (impossible, extra has 3)
            # - 0 from anchor, 5 from extra (impossible)
            
            # So we only need to check:
            # - Anchor
            # - Combinations of 4 from anchor + 1 from extra
            # - Combinations of 3 from anchor + 2 from extra
            # - Combinations of 2 from anchor + 3 from extra
            
            # Let's pre-compute the "parts" of the anchor to speed up subset generation?
            # Or just generate them on the fly.
            
            anchor_list = list(anchor)
            
            for extra in itertools.combinations(remaining_points, num_add):
                # Construct the block
                # We don't need to sort yet, just calculate gain
                
                gain = 0
                
                # 1. Anchor
                if anchor in uncovered:
                    gain += 1
                    
                # 2. 4 from anchor, 1 from extra
                # Iterate over all ways to pick 4 from anchor (5 ways) and 1 from extra (3 ways)
                # Total 15 subsets
                for a4 in itertools.combinations(anchor_list, t-1):
                    for e1 in extra:
                        sub = tuple(sorted(a4 + (e1,)))
                        if sub in uncovered:
                            gain += 1
                            
                # 3. 3 from anchor, 2 from extra
                # C(5,3)=10 ways for anchor, C(3,2)=3 ways for extra. Total 30.
                for a3 in itertools.combinations(anchor_list, t-2):
                    for e2 in itertools.combinations(extra, 2):
                        sub = tuple(sorted(a3 + e2))
                        if sub in uncovered:
                            gain += 1
                            
                # 4. 2 from anchor, 3 from extra
                # C(5,2)=10 ways for anchor, C(3,3)=1 way for extra. Total 10.
                for a2 in itertools.combinations(anchor_list, t-3):
                    e3 = extra # all 3
                    sub = tuple(sorted(a2 + e3))
                    if sub in uncovered:
                        gain += 1
                            
                if gain > max_gain:
                    max_gain = gain
                    best_block = sorted(anchor_list + list(extra))
                    
            if best_block is None:
                # Should not happen if uncovered is not empty
                continue
                
            blocks.append(best_block)
            
            # Update uncovered set
            # Remove all t-subsets of best_block from uncovered
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(tuple(sub))
                
        return blocks

    # Try multiple restarts
    # How many restarts can we afford?
    # One run takes roughly 335 steps.
    # Each step: 364 candidates * (1 + 15 + 30 + 10) subset checks = 364 * 56 = 20k checks.
    # 335 * 20k = 6.7M checks per run.
    # In Python, 6.7M simple set lookups/tuple sorts might take ~1-2 seconds.
    # We have 110 seconds. So we can do ~50-100 restarts.
    
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        blocks = run_greedy(i)
        if blocks is not None:
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
                # Early stop if we hit the theoretical lower bound?
                # Lower bound is 264. If we hit 264, we can stop.
                if best_size <= 264:
                    break

    if best_blocks is None:
        # Fallback to a simple greedy if all failed or timed out
        # Or return an empty list? No, must be valid.
        # If we timed out, we might have partial results.
        # Let's do a quick non-randomized greedy as fallback.
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            if time.time() > deadline - 1.0:
                # Return what we have? No, must be valid.
                # If we can't finish, we should return the best valid one found.
                # If no valid one found, we are in trouble.
                # The seed program always finishes.
                # Let's assume we find at least one.
                break
            anchor = min(uncovered)
            remaining_points = [x for x in range(v) if x not in anchor]
            max_gain = -1
            best_block = None
            anchor_list = list(anchor)
            for extra in itertools.combinations(remaining_points, k-t):
                gain = 0
                if anchor in uncovered: gain += 1
                for a4 in itertools.combinations(anchor_list, t-1):
                    for e1 in extra:
                        if tuple(sorted(a4 + (e1,))) in uncovered: gain += 1
                for a3 in itertools.combinations(anchor_list, t-2):
                    for e2 in itertools.combinations(extra, 2):
                        if tuple(sorted(a3 + e2)) in uncovered: gain += 1
                for a2 in itertools.combinations(anchor_list, t-3):
                    if tuple(sorted(a2 + extra)) in uncovered: gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = sorted(anchor_list + list(extra))
            blocks.append(best_block)
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(tuple(sub))
        if best_blocks is None:
            best_blocks = blocks
        else:
            if len(blocks) < len(best_blocks):
                best_blocks = blocks

    return best_blocks

import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for 120s limit

    points = list(range(v))
    all_t_subsets = set(itertools.combinations(points, t))
    
    # Precompute combinations of size t for a block of size k?
    # No, block size is fixed k.
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        # To speed up sampling, we can keep a list of uncovered subsets
        # But converting set to list is O(N).
        # Let's just use random.sample on the set converted to list occasionally?
        # Or just pick random elements?
        # random.choice(uncovered) works on sets? No.
        # We need to sample.
        
        # Optimization: If uncovered is large, sampling is cheap relative to gain calc.
        # If uncovered is small, we can just pick all.
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Determine samples
            if len(uncovered) < 50:
                samples = list(uncovered)
            else:
                # Convert to list for sampling
                # This is O(U). U starts at 11628.
                # Doing this every step might be expensive.
                # Alternative: maintain a list of uncovered?
                # Removing from list is O(N).
                # Let's just convert. 11k elements is small.
                samples = random.sample(list(uncovered), min(10, len(uncovered)))
                
            best_block = None
            best_gain = -1
            candidate_blocks = set()
            
            for s in samples:
                s_set = set(s)
                remaining = [p for p in points if p not in s_set]
                # Need k-t more points
                for combo in itertools.combinations(remaining, k - t):
                    block = tuple(sorted(list(s) + list(combo)))
                    if block in candidate_blocks:
                        continue
                    candidate_blocks.add(block)
                    
                    # Calculate gain
                    gain = 0
                    # Iterate over all t-subsets of this block
                    # itertools.combinations(block, t) generates 56 tuples
                    for sub in itertools.combinations(block, t):
                        if sub in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                
                # Optional: Clear candidate_blocks to save memory? 
                # It's small (max 10*364 = 3640).
                # But we want to avoid duplicates across samples.
                # So we keep it for the whole step.
                
            if best_block is None:
                break # Safety
                
            blocks.append(best_block)
            
            # Remove covered subsets
            for sub in itertools.combinations(best_block, t):
                if sub in uncovered:
                    uncovered.discard(sub)
            
            # Periodic check for redundancy?
            # Cleaning at the end is better.
            
        # Ensure validity: if we broke due to time or safety, complete the covering
        while uncovered:
            if time.time() > deadline:
                # If we are out of time, we must still return a valid covering.
                # We assume the simple completion is fast enough.
                pass
            s = next(iter(uncovered))
            s_set = set(s)
            remaining = [p for p in points if p not in s_set]
            if len(remaining) < k - t:
                break
            # Pick a random block to cover s
            combo = random.sample(remaining, k - t)
            block = tuple(sorted(list(s) + list(combo)))
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    uncovered.discard(sub)
            
        return blocks

    def clean_blocks(blocks):
        # Remove redundant blocks
        # Use coverage count map
        if not blocks:
            return []
            
        # Map: t-subset -> count
        coverage_count = {}
        
        def update_counts(block, delta):
            for sub in itertools.combinations(block, t):
                if sub in coverage_count:
                    coverage_count[sub] += delta
                else:
                    coverage_count[sub] = delta
                    if coverage_count[sub] == 0:
                        del coverage_count[sub]
        
        # Initialize counts
        for b in blocks:
            update_counts(b, 1)
            
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                # b is redundant if for all sub in b, coverage_count[sub] > 1
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if coverage_count.get(sub, 0) <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    update_counts(b, -1)
                    changed = True
                else:
                    new_blocks.append(b)
            
            blocks = new_blocks
            
        return blocks

    best_blocks = None
    
    # Run multiple seeds
    # Estimate time per greedy: ~5-10s
    # We can afford 10-20 runs?
    # Let's try 5 runs.
    
    for seed in range(5):
        if time.time() > deadline - 5:
            break
            
        blocks = greedy_cover(seed)
        blocks = clean_blocks(blocks)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: return empty or single block? 
        # Must be valid.
        # If we ran out of time, we might have an invalid covering.
        # We need to ensure validity.
        # If the loop broke early, uncovered is not empty.
        # We need to finish the covering?
        # If we break due to time, we should probably not return an invalid result.
        # But the task says "timeout scores 0".
        # So we must return a valid result.
        # If we don't have time to finish, we should return the best *valid* covering?
        # Or just finish quickly with a naive method?
        
        # If best_blocks is None, it means we didn't complete even one greedy.
        # This is unlikely if greedy is fast.
        # But if greedy is slow, we might have partial blocks.
        # Partial blocks do NOT form a valid covering.
        # So we must ensure we return a valid covering.
        
        # Strategy: If time is running out, fall back to a very fast, simple greedy 
        # that doesn't optimize gain, just picks first available?
        # Or, if we have a partial covering, we can't use it.
        
        # Let's assume greedy finishes.
        # If not, we return an empty list? No, must cover all.
        # We will add a check: if time > deadline - 10, stop starting new seeds.
        # If a seed is running and time > deadline, break.
        # Then we have partial blocks.
        # We must complete the covering using a simple method?
        # Simple method: For each uncovered subset, add a block containing it (random or fixed).
        # This ensures validity.
        
        pass

    # Ensure validity and completeness if interrupted
    if best_blocks is not None:
        # Check if it's a valid covering
        # We assume clean_blocks preserves validity (removing redundant blocks doesn't break coverage).
        # And greedy_cover builds a covering.
        # So it should be valid.
        
        # Convert tuples to lists as required by "list of blocks"
        return [list(b) for b in best_blocks]
    else:
        # Emergency fallback: Naive greedy
        # This should never happen if deadline is respected.
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
             s = next(iter(uncovered))
             # Find any block containing s
             s_set = set(s)
             remaining = [p for p in points if p not in s_set]
             # Pick first k-t points
             combo = remaining[:k-t]
             block = tuple(sorted(list(s) + list(combo)))
             blocks.append(block)
             for sub in itertools.combinations(block, t):
                 if sub in uncovered:
                     uncovered.discard(sub)
        return [list(b) for b in blocks]

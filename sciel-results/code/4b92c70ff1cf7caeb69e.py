import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Leave 10 seconds buffer for GC and other overhead
    deadline = start_time + 110.0
    
    # Precompute all t-subsets as tuples for fast hashing
    all_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_size = float('inf')
    
    # We will run a randomized greedy heuristic multiple times.
    # The randomization is in:
    # 1. Choosing the anchor subset (random from uncovered).
    # 2. Tie-breaking in point selection (shuffled candidates).
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        random.seed(iteration)
        
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                # Abort this run if we hit the hard deadline
                break
                
            # Pick a random uncovered subset to anchor the new block
            # Converting to list is O(N), N<=2024. Acceptable.
            target = random.choice(list(uncovered))
            block = list(target)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                
                # Candidates are points not in block
                # We shuffle to break ties randomly
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                
                for x in candidates:
                    # Calculate gain: number of uncovered t-subsets that would be covered
                    # by adding x to block.
                    # These are subsets of form {x} U {rest} where rest is a (t-1)-subset of block.
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        # Create the tuple representing the new t-subset
                        # Sorting is required to match the format in 'uncovered'
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Optional: early break if gain is max possible?
                        # Max gain for a step is C(|block|, t-1) if all are uncovered.
                        # But we don't know if they are uncovered without checking.
                        # So no simple early break.
                        
                if best_point == -1:
                    # This should not happen if v > k and uncovered is not empty
                    # because there are always points to add that cover *something*?
                    # Actually, if all points give gain 0, we are stuck?
                    # No, if uncovered is not empty, there is some uncovered subset.
                    # But if no point covers any *new* uncovered subset, then the current block
                    # cannot be extended to cover any *new* uncovered subset?
                    # This implies the uncovered subsets are disjoint from the current block's potential?
                    # Wait, if gain is 0 for all x, it means adding any x does not cover any
                    # currently uncovered t-subset.
                    # This can happen if the uncovered subsets are "far" from the block.
                    # But we picked the target from uncovered, so target is covered by the block (partially).
                    # Specifically, target is a subset of the *final* block? No.
                    # We picked target from uncovered. We start block with target.
                    # So target is covered.
                    # As we add points, we cover more.
                    # If gain is 0 for all next points, it means we cannot cover any *other*
                    # uncovered subset with this block?
                    # If so, we should just close the block with k points (arbitrarily) and move on.
                    # But we must fill the block to size k.
                    # If best_point is -1, we pick an arbitrary point.
                    if candidates:
                        best_point = candidates[0]
                    else:
                        break # Should not happen
                
                block.append(best_point)
                block.sort()
                
                # Update uncovered set
                # We only need to remove subsets that are fully contained in the NEW block
                # and were previously uncovered.
                # Since we are building the block incrementally, the newly covered subsets
                # are those containing the newly added point `best_point` and (t-1) points from the previous block.
                # However, it's simpler and safer to just check all t-subsets of the current block
                # and discard them. Since k=7, t=3, there are 35 subsets. This is fast.
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
                    
            blocks.append(block)
            
        # End of one heuristic run
        # Only update best if this run completed successfully (didn't break due to timeout)
        # How to know if it completed? 
        # If `uncovered` is empty, it completed.
        if not uncovered and len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
            # If we found a very good solution, we can stop early?
            # The theoretical lower bound is 69.
            # If we find 69, it's optimal.
            if best_size <= 69:
                break

    # If no complete run finished before timeout (unlikely but possible if very slow),
    # we need to return a valid covering.
    # We can run a simple deterministic greedy without randomization and without strict internal 
    # time checks (or with a relaxed one) to ensure we return *something*.
    # But given the speed, this branch is highly unlikely to be taken.
    if best_blocks is None:
        # Fallback: Deterministic Greedy
        uncovered = set(all_subsets)
        blocks = []
        while uncovered:
            target = min(uncovered)
            block = list(target)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    # Pick any available point
                    for x in range(v):
                        if x not in block:
                            best_point = x
                            break
                block.append(best_point)
                block.sort()
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
            blocks.append(block)
        return blocks

    return best_blocks

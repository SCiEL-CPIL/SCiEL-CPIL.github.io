import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_subsets_set = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # We will try multiple runs with different random seeds
    # The greedy algorithm is deterministic if we don't break ties randomly.
    # We introduce randomness in tie-breaking and seed selection.
    
    for run_id in range(100): # Max 100 runs, but limited by time
        if time.time() > deadline:
            break
            
        random.seed(run_id)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to seed the block
            # This helps escape local minima compared to always picking min
            seed_subset = random.choice(list(uncovered))
            block = list(seed_subset)
            
            while len(block) < k:
                # Find x that covers most uncovered t-subsets
                best_x = -1
                best_gain = -1
                
                # Optimization: Precompute (t-1) subsets of current block?
                # block size is small (max 7), t is 4.
                # Number of (t-1) subsets is C(6, 3) = 20 max.
                # Candidates for x: v - len(block) ~ 12.
                # 12 * 20 = 240 checks per step. Very fast.
                
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in t_minus_1_subsets:
                        # Check if rest + (x,) is in uncovered
                        # Need to form the tuple and sort?
                        # rest is sorted. x might not be.
                        # T = sorted(rest + (x,))
                        T = tuple(sorted(rest + (x,)))
                        if T in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        # Tie-breaking: random choice to diversify
                        # We can just keep the first one or pick randomly from tied
                        # For simplicity and speed, we can just keep the first found
                        # Or we can collect tied candidates.
                        # Let's just keep the first for now to be fast.
                        # Actually, random tie-breaking is better for search diversity.
                        # But collecting tied candidates adds overhead.
                        # Let's stick to first found for speed, but randomize seed.
                        pass
                        
                if best_x == -1:
                    # Should not happen if uncovered is not empty and k > t
                    break
                    
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            covered = set(itertools.combinations(block, t))
            uncovered.difference_update(covered)
            
        # End while uncovered
        
        # Local Optimization: Remove redundant blocks
        # A block is redundant if all its t-subsets are covered by other blocks.
        # We can check this by temporarily removing a block and seeing if uncovered becomes non-empty.
        # But re-calculating uncovered is expensive.
        # Instead, maintain a count of how many blocks cover each t-subset.
        
        # Let's implement a count-based redundancy check.
        # This is O(N_blocks * C(k, t)).
        # N_blocks ~ 150. C(7,4) = 35. 150 * 35 = 5250 operations. Very fast.
        
        # Rebuild coverage counts from scratch for the current set of blocks
        # This is safer than incremental updates if we do swaps.
        
        # Actually, simple deletion:
        # For each block, check if it is essential.
        # Essential if there is a t-subset in it that is NOT covered by any other block.
        
        # To do this efficiently:
        # 1. Count coverage of each t-subset.
        # 2. A block is redundant if all its t-subsets have count > 1.
        
        # Let's try to remove blocks iteratively.
        # This might be slow if we do it for every block in every run.
        # But 150 blocks * 35 subsets * 150 other blocks = 787,500 checks per pass.
        # 100 runs * 787,500 = 78,750,000 checks. Might be tight in Python.
        
        # Alternative:
        # Just check redundancy once per run.
        
        coverage_count = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] = coverage_count.get(sub, 0) + 1
                
        # Now, try to remove blocks that are fully redundant
        # A block is fully redundant if for all sub in block, coverage_count[sub] > 1
        # But removing one block changes counts for others.
        # So we must iterate.
        
        # To save time, we only do this if the block count is close to best.
        # Or just do it for all runs, it's not that heavy.
        
        # Optimization: Only check redundancy if current block count < best_count
        # If it's already worse, no point in optimizing.
        
        if len(blocks) < best_count:
            # Try to remove redundant blocks
            # We do this in a loop until no more can be removed
            # Max iterations: len(blocks)
            
            changed = True
            while changed:
                changed = False
                # Rebuild counts? Or update incrementally?
                # Rebuilding is O(N * C(k,t)). 150 * 35 = 5250.
                # Doing this inside the while loop might be costly if many removals.
                # But removals are few.
                
                # Let's rebuild counts at the start of each pass.
                coverage_count = {}
                for b in blocks:
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] = coverage_count.get(sub, 0) + 1
                
                # Check each block
                # We need to know if a block has ANY subset with count == 1.
                # If yes, it's essential.
                # If no, it's redundant.
                
                redundant_indices = []
                for i, b in enumerate(blocks):
                    is_essential = False
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        redundant_indices.append(i)
                        
                if redundant_indices:
                    # Remove them
                    # We can remove all at once?
                    # If two blocks are both redundant, removing one makes the other's subsets potentially count 1?
                    # No, if a subset has count 2, and both blocks contain it, removing one makes count 1.
                    # So we must be careful.
                    # However, if a block is redundant, it means ALL its subsets have count > 1.
                    # If we remove one redundant block, the counts of its subsets decrease by 1.
                    # They might become 1.
                    # So other blocks that relied on those subsets having count > 1 might now become essential?
                    # No, essentiality is determined by count == 1.
                    # If a subset count drops from 2 to 1, the blocks containing it (other than the removed one) now have a subset with count 1, so they become essential.
                    # This is safe.
                    
                    # But can we remove multiple redundant blocks at once?
                    # If B1 and B2 are both redundant, and they share a subset S.
                    # Count(S) = 2.
                    # If we remove B1, Count(S) = 1.
                    # B2 still contains S. B2 was redundant before (all subsets count > 1).
                    # Now S has count 1. So B2 is now essential.
                    # So we cannot remove B2 in the same pass if it shares subsets with B1.
                    
                    # So we should remove one at a time, or just mark one and re-evaluate.
                    # Given the small size, let's just remove the first one found and restart the loop.
                    
                    # Actually, just removing all identified redundant blocks in one go is DANGEROUS.
                    # Example: B1={1,2,3,4,5,6,7}, B2={1,2,3,4,8,9,10}. t=4.
                    # Subset {1,2,3,4} is in both. Count=2.
                    # If both are redundant, it means all their other subsets have count > 1.
                    # If we remove both, {1,2,3,4} has count 0. INVALID.
                    
                    # So, we must remove blocks one by one or ensure they are disjoint in their "essential" subsets.
                    # Simplest: Remove one, update counts, repeat.
                    
                    # To speed up, we can just remove one redundant block per outer while iteration.
                    # Pick the first redundant index.
                    idx = redundant_indices[0]
                    blocks.pop(idx)
                    changed = True
                    # Loop restarts, rebuilds counts.
                    
        # End redundancy check
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    # If no blocks found (should not happen), return empty or fallback
    if best_blocks is None:
        # Fallback to simple greedy
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            seed = min(uncovered)
            block = list(seed)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t-1) if tuple(sorted(rest+(x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
        best_count = len(blocks)
        
    return best_blocks

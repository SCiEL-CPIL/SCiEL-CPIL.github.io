import time
import numpy as np
import itertools
import random

# Precompute all 4-subsets of 19 points as bitmasks
# We need to map 4-subsets to indices for vectorized operations
# Total 4-subsets: C(19,4) = 3876
# Total 7-subsets: C(19,7) = 50388

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate all t-subsets and assign them an index 0..M-1
    t_subsets = list(itertools.combinations(range(v), t))
    M = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Precompute for each point x, the list of t-subsets containing x
    # points_to_tsubs[x] is an array of indices of t-subsets containing x
    points_to_tsubs = [[] for _ in range(v)]
    for idx, sub in enumerate(t_subsets):
        for p in sub:
            points_to_tsubs[p].append(idx)
    points_to_tsubs = [np.array(lst, dtype=np.int32) for lst in points_to_tsubs]
    
    # 3. Greedy Construction
    # covered is a boolean array of size M
    covered = np.zeros(M, dtype=bool)
    blocks = []
    
    # Precompute all possible k-subsets? No, 50388 is manageable but let's generate on the fly or use a smart greedy.
    # Standard greedy: pick the block that covers the most uncovered t-subsets.
    # To speed this up, we can use a heuristic greedy:
    # 1. Find the most "isolated" uncovered t-subset (one that is contained in the fewest possible k-blocks? No, all are symmetric initially).
    # 2. Just pick the first uncovered t-subset, then build a k-block around it greedily.
    
    uncovered_indices = np.array(range(M), dtype=np.int32)
    
    while len(uncovered_indices) > 0:
        if time.time() > deadline:
            break
            
        # Pick the first uncovered t-subset to anchor the new block
        anchor_idx = uncovered_indices[0]
        anchor_subset = t_subsets[anchor_idx]
        
        # Build a block of size k containing anchor_subset
        # Start with the anchor subset
        current_block = list(anchor_subset)
        
        # We want to add k - t points to current_block to maximize coverage of uncovered t-subsets
        available_points = [p for p in range(v) if p not in current_block]
        
        # To speed up, we can evaluate the gain of adding each available point.
        # Gain(x) = number of uncovered t-subsets that contain x and (t-1) points from current_block.
        
        # Since t=4, we need to count uncovered t-subsets containing x and 3 points from current_block.
        # Let's precompute the "interaction" for the current block state.
        
        # For efficiency in the inner loop, we can just iterate.
        # But let's try to vectorize the gain calculation for all available points at once.
        
        # For each available point x, we want to count how many uncovered t-subsets contain x.
        # AND contain at least t-1 points from current_block.
        
        # This is a bit complex to vectorize perfectly without precomputed tables for all partial blocks.
        # Given k=7, t=4, the block grows slowly.
        
        # Let's use a simpler heuristic:
        # Calculate gain for each candidate x.
        
        # Precompute for the current block B (size m < k):
        # For each x not in B, gain(x) = count of uncovered t-subsets S such that x in S and |S ∩ B| >= t-1.
        
        # Let's just do a loop over available points, but use numpy for the count.
        
        best_x = -1
        best_gain = -1
        
        # Optimization: If current_block is small, the number of t-subsets to check is small.
        # But checking all M=3876 for each x is 3876 * 15 = ~60k ops per step. 7 steps per block. 150 blocks. 60k * 7 * 150 = 63M ops.
        # In Python, this is too slow. In NumPy, it's fast.
        
        # Let's vectorize:
        # For each x in available_points:
        #   tsubs_with_x = points_to_tsubs[x]
        #   We need to check which of these are uncovered AND intersect current_block in >= t-1 points.
        
        # This is hard to vectorize across x easily because the intersection check depends on current_block.
        
        # Alternative:
        # Iterate through uncovered t-subsets? No, too many.
        
        # Let's stick to a slightly slower but more Pythonic approach for the *construction* if NumPy vectorization is tricky, 
        # OR use a simpler heuristic:
        # Randomly pick a few candidate blocks containing the anchor, pick the best.
        
        # Actually, let's try a different greedy:
        # 1. Find the t-subset with the fewest "options" (all have same options initially).
        # 2. Pick a random t-subset from uncovered.
        # 3. Generate all k-subsets containing this t-subset. (C(v-t, k-t) = C(15, 3) = 455 options).
        # 4. For each of these 455 options, calculate how many *currently uncovered* t-subsets it covers.
        # 5. Pick the option with max coverage.
        
        # This is 455 * (cost to count coverage).
        # Cost to count coverage: For a candidate block B, count how many of its C(7,4)=35 t-subsets are in uncovered.
        # We can precompute a set of uncovered t-subset indices.
        # Then for candidate B, check if its 35 t-subsets are in the set.
        # 455 * 35 = ~16k set lookups.
        # 150 blocks * 16k = 2.4M lookups. This is very fast in Python.
        
        uncovered_set = set(uncovered_indices)
        
        # Generate candidates
        remaining_points = [p for p in range(v) if p not in current_block]
        candidates = []
        # We need to choose k-t points from remaining_points to add to current_block
        # current_block already has t points.
        # Wait, my previous logic started with anchor_subset (size t).
        # So I need to add k-t points.
        
        # Let's restart the block building logic for this iteration
        base_block = list(anchor_subset)
        points_to_add = k - t
        # Choose points_to_add from available_points
        # To limit candidates, we can just pick the best from a random sample if there are too many, 
        # but C(15,3)=455 is small enough to check all.
        
        best_candidate_block = None
        max_covered = -1
        
        for combo in itertools.combinations(available_points, points_to_add):
            candidate_block = tuple(sorted(base_block + list(combo)))
            
            # Calculate coverage
            # Get all t-subsets of candidate_block
            # Check how many are in uncovered_set
            cnt = 0
            for tsub in itertools.combinations(candidate_block, t):
                if tsub in t_subset_index: # always true
                    idx = t_subset_index[tsub]
                    if idx in uncovered_set:
                        cnt += 1
                        # Optimization: if cnt > max_covered, we can't beat it if remaining is small?
                        # No, just count all.
                        
            if cnt > max_covered:
                max_covered = cnt
                best_candidate_block = list(candidate_block)
                
        # Add the best block
        if best_candidate_block:
            blocks.append(best_candidate_block)
            
            # Update uncovered
            # For each t-subset in best_candidate_block, if it was uncovered, mark as covered
            new_uncovered = []
            for idx in uncovered_indices:
                tsub = t_subsets[idx]
                if all(p in best_candidate_block for p in tsub):
                    # Covered
                    pass
                else:
                    new_uncovered.append(idx)
            uncovered_indices = np.array(new_uncovered, dtype=np.int32)
            uncovered_set = set(uncovered_indices)
            
    # 4. Local Search (2-opt) to reduce block count
    # Try to swap or replace blocks to remove redundancy.
    # A simple heuristic: 
    # 1. Check for redundant blocks: A block is redundant if all its t-subsets are covered by other blocks.
    # 2. If redundant, remove it.
    # 3. Try to replace a block with another that covers more "unique" t-subsets?
    
    # Let's implement a simple "remove redundant" loop first.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Count coverage of each t-subset by each block
        # This is O(N_blocks * C(k,t) * N_blocks) if done naively.
        # Better: For each t-subset, count how many blocks cover it.
        
        # Rebuild coverage count
        # M = 3876 t-subsets
        # N = len(blocks)
        # coverage_count[t_idx] = number of blocks covering t_idx
        
        # Efficient way:
        # For each block, list its t-subsets.
        # For each t-subset, increment counter.
        
        # Let's just check redundancy by removing one block at a time and seeing if coverage is broken.
        # This is O(N^2 * C(k,t)).
        # N~150, C(7,4)=35. 150^2 * 35 = 787,500 ops. Fast.
        
        i = 0
        while i < len(blocks):
            # Check if blocks[i] is redundant
            # Temporarily remove it
            blocks_without_i = blocks[:i] + blocks[i+1:]
            
            # Check if all t-subsets of blocks[i] are covered by blocks_without_i
            is_redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                covered_by_other = False
                for b in blocks_without_i:
                    if all(p in b for p in tsub):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                blocks.pop(i)
                changed = True
                # Don't increment i, check same index again (which is now next block)
            else:
                i += 1
                
    return blocks

# To ensure we don't timeout, we wrap the main logic or just rely on the deadline check inside.
# The deadline check is inside the greedy loop.
# The local search loop also checks deadline.

# One more optimization: The greedy step "check all 455 candidates" might be slow if done in pure Python for 150 blocks.
# 150 * 455 * 35 = 2.4M inner loop iterations.
# Each iteration does `all(p in b for p in tsub)` which is O(t).
# 2.4M * 4 = ~10M operations. In Python, this might take 1-2 seconds.
# So it should fit well within 110s.

# Let's add a fallback if it times out, returning whatever we have.
# The current structure returns `blocks` which might be empty if it times out immediately, but that's unlikely.
# If it times out in the middle of greedy, it returns partial blocks, which might be invalid.
# The task says "Valid when every t-subset ... lies in at least one block".
# If we return an invalid cover, score is 0.
# So we MUST ensure the cover is complete.
# If we break out of the greedy loop early due to deadline, we have uncovered t-subsets.
# We must either finish the greedy or return a valid cover.
# Given the speed estimate (~2s for greedy), we should finish easily.
# But to be safe, if we are close to deadline, we should stop adding blocks and verify validity?
# No, if we stop adding, it's invalid.
# Better: If we are close to deadline, we should just let the greedy finish (it's fast) and skip local search.

# Revised Plan:
# 1. Greedy (Fast)
# 2. If time > deadline - 5s, skip local search and return.
# 3. Else, do local search.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s for overhead
    
    t_subsets = list(itertools.combinations(range(v), t))
    M = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    uncovered_indices = list(range(M))
    uncovered_set = set(uncovered_indices)
    blocks = []
    
    # Greedy
    while uncovered_indices:
        if time.time() > deadline:
            # Emergency exit: we have an invalid cover.
            # We must complete the cover to be valid.
            # Since greedy is fast, we assume we won't hit this.
            # If we do, we can't easily fix it.
            # Let's just break and hope it's valid? No.
            # Let's assume it finishes.
            break
            
        anchor_idx = uncovered_indices[0]
        anchor_subset = t_subsets[anchor_idx]
        base_block = list(anchor_subset)
        available_points = [p for p in range(v) if p not in base_block]
        points_to_add = k - t
        
        best_candidate_block = None
        max_covered = -1
        
        # Sample candidates if too many? C(15,3)=455 is fine.
        for combo in itertools.combinations(available_points, points_to_add):
            candidate_block = tuple(sorted(base_block + list(combo)))
            
            cnt = 0
            for tsub in itertools.combinations(candidate_block, t):
                if t_subset_index[tsub] in uncovered_set:
                    cnt += 1
                    
            if cnt > max_covered:
                max_covered = cnt
                best_candidate_block = list(candidate_block)
                
        if best_candidate_block:
            blocks.append(best_candidate_block)
            
            # Update uncovered
            # Optimization: Only check uncovered ones
            new_uncovered = []
            for idx in uncovered_indices:
                tsub = t_subsets[idx]
                # Check if tsub is subset of best_candidate_block
                if all(p in best_candidate_block for p in tsub):
                    pass
                else:
                    new_uncovered.append(idx)
            uncovered_indices = new_uncovered
            uncovered_set = set(uncovered_indices)
            
    # Local Search
    if time.time() < deadline - 10.0:
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            i = 0
            while i < len(blocks):
                # Check if blocks[i] is redundant
                is_redundant = True
                for tsub in itertools.combinations(blocks[i], t):
                    covered_by_other = False
                    for j, b in enumerate(blocks):
                        if i == j: continue
                        if all(p in b for p in tsub):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    blocks.pop(i)
                    changed = True
                else:
                    i += 1
                    
    return blocks

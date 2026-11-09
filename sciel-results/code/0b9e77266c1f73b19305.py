import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a block-replacement local search strategy.
    
    Strategy:
    1. Generate a valid initial covering using a fast deterministic greedy heuristic.
    2. Perform local search by attempting to replace blocks.
       - For each block B, identify its 'unique' contributions (t-subsets covered only by B).
       - Try to find a new block B' that covers all unique contributions of B.
       - If B' also covers other subsets, it might allow removal of other blocks.
       - Accept the replacement if it reduces the total number of blocks or keeps it same 
         but improves a secondary metric (like total coverage count reduction).
    3. Use time budget to perform multiple passes of this replacement search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper: Get all t-subsets of a block
    def get_t_subsets(block):
        return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    # --- Phase 1: Fast Initial Construction ---
    # Use a deterministic greedy to get a valid cover quickly.
    # We pick the block that covers the most uncovered t-subsets.
    uncovered = [1] * num_t_subsets
    blocks = []
    
    # Pre-compute all possible k-blocks? C(19,8) = 75582. This is small enough to precompute
    # their t-subset coverage masks or lists.
    # However, iterating 75k blocks * 126 t-subsets per block might be slow in pure Python if done repeatedly.
    # Let's just do the standard greedy: for each step, scan all candidate blocks?
    # No, standard greedy for covering designs usually picks a random uncovered subset and extends it.
    # Let's stick to the fast random greedy from the parent to get a baseline, 
    # but we will heavily optimize the reduction phase using replacement.
    
    # Actually, let's use a slightly smarter initial construction:
    # Greedy: Pick block with max coverage of uncovered.
    # To do this fast, we need to know which blocks cover which subsets.
    # Since C(19,8) is ~75k, we can precompute the list of t-subset indices for each block.
    
    # Precomputation: Map block (tuple) to list of t-subset indices
    # This takes memory but speeds up the initial greedy significantly.
    # 75582 blocks * 126 ints ~ 9.5M ints. ~76MB. Acceptable.
    
    # But generating all 75k blocks and their subsets takes time.
    # Let's estimate: 75k * 126 operations. ~9.5M ops. < 1s in Python? Maybe 2-3s.
    # Let's try to generate them.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_t_subsets = {}
    for b in all_blocks:
        block_t_subsets[b] = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
    
    # Greedy Construction
    while True:
        # Find block with max coverage of uncovered
        best_block = None
        best_gain = -1
        
        # Optimization: If time is tight, we can't scan all 75k blocks every iteration.
        # But initial construction is only ~300-400 iterations. 400 * 75k = 30M checks.
        # Each check is summing 126 ints. That's too slow in Python (30M * 126 = 3.7B ops).
        
        # Alternative: Use the random greedy from parent to get a valid cover fast,
        # then apply the replacement search.
        pass

    # Re-adopting the fast random greedy for initialization to ensure speed,
    # as the replacement search is the key innovation.
    
    random.seed(42) # Fixed seed for initial construction for reproducibility in first pass?
    # Actually, let's do a few random initializations and pick the best one to start the local search.
    
    best_initial_blocks = None
    best_initial_len = float('inf')
    
    # Do 5 quick random greedy constructions
    for init_seed in range(5):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(init_seed)
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        while uncovered_set:
            target_idx = random.choice(list(uncovered_set))
            target_subset = all_t_subsets[target_idx]
            
            block_points = set(target_subset)
            remaining_points = [x for x in range(v) if x not in block_points]
            
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                
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
            
            if block not in blocks:
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if idx in uncovered_set:
                        uncovered_set.remove(idx)
        
        # Quick reduction: remove redundant blocks
        cover_count = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[t_subset_to_idx[ts]] += 1
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(blocks): continue
                b = blocks[i]
                removable = True
                for ts in itertools.combinations(b, t):
                    if cover_count[t_subset_to_idx[ts]] == 1:
                        removable = False
                        break
                if removable:
                    for ts in itertools.combinations(b, t):
                        cover_count[t_subset_to_idx[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        if len(blocks) < best_initial_len:
            best_initial_len = len(blocks)
            best_initial_blocks = [list(b) for b in blocks]
            
    if best_initial_blocks is None:
        return []
        
    # Start local search with the best initial covering
    current_blocks = best_initial_blocks
    
    # Data structure for local search
    # We need to efficiently find blocks that cover specific sets of t-subsets.
    # Inversion: Map t-subset index -> list of block indices containing it.
    
    def build_inversion(blocks):
        inv = [[] for _ in range(num_t_subsets)]
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                inv[idx].append(i)
        return inv
    
    def compute_cover_counts(blocks):
        counts = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[t_subset_to_idx[ts]] += 1
        return counts
    
    inv = build_inversion(current_blocks)
    counts = compute_cover_counts(current_blocks)
    
    # Local Search: Block Replacement
    # Idea: For each block i, let U_i be the set of t-subsets covered ONLY by block i.
    # If we remove block i, these subsets become uncovered.
    # We must find a new block B_new that covers ALL subsets in U_i.
    # If such a B_new exists and is not already in the set, we can replace i with B_new.
    # This replacement might cover extra subsets, making other blocks redundant.
    
    # Optimization:
    # Precompute for each block i, the list of t-subsets it covers.
    # And identify "critical" subsets (count == 1).
    
    # To find a replacement block B_new for block i:
    # B_new must contain the union of all points in the critical subsets? No.
    # B_new must contain each critical subset.
    # This means B_new must contain the union of the points in each critical subset?
    # No, B_new must contain the specific points of each critical subset.
    # So B_new must contain the intersection of the constraints?
    # Actually, if U_i = {S1, S2, ...}, B_new must contain S1 AND S2 AND ...
    # So B_new must contain Union(S1, S2, ...).
    # If len(Union) > k, no replacement exists.
    # If len(Union) <= k, we can form B_new by taking the Union and adding random points to reach size k.
    
    # This is a very powerful constraint. Let's use it.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Recompute inv and counts if they changed?
        # We'll update them incrementally or recompute if changes are frequent.
        # Given the small size, recomputing after a change is safe.
        
        # Shuffle block indices for random order
        block_indices = list(range(len(current_blocks)))
        random.shuffle(block_indices)
        
        for i in block_indices:
            if time.time() > deadline - 2.0:
                break
            if i >= len(current_blocks):
                continue
                
            b_old = current_blocks[i]
            
            # Find unique subsets of b_old
            unique_subs = []
            for ts in itertools.combinations(b_old, t):
                idx = t_subset_to_idx[ts]
                if counts[idx] == 1:
                    unique_subs.append(ts)
            
            # If no unique subsets, we can just remove the block!
            # (This is the standard reduction, which we already did, but maybe new ones appeared)
            if not unique_subs:
                # Remove block i
                for ts in itertools.combinations(b_old, t):
                    idx = t_subset_to_idx[ts]
                    counts[idx] -= 1
                    inv[idx].remove(i)
                
                # Shift indices for all blocks after i
                # This is O(N). Acceptable for N~300.
                current_blocks.pop(i)
                
                # Re-index inv lists?
                # inv contains indices. If we pop i, all indices > i are now -1.
                # This is expensive to fix.
                # Alternative: Use a list of booleans for active blocks?
                # Or just rebuild inv and counts after each change?
                # Rebuilding is O(N * C(k,t)) = 300 * 126 = 37k ops. Very fast.
                inv = build_inversion(current_blocks)
                counts = compute_cover_counts(current_blocks)
                
                improved = True
                break # Restart scan after change
                
            # If there are unique subsets, we need a replacement
            # The replacement must cover all unique_subs.
            # Let's find the union of points in unique_subs
            union_points = set()
            for ts in unique_subs:
                union_points.update(ts)
            
            if len(union_points) > k:
                continue # No possible replacement can cover all unique subsets
            
            # We need to form a block B_new of size k containing union_points.
            # We want B_new to maximize coverage of OTHER subsets to help remove other blocks.
            
            # Strategy: Try a few random completions of union_points to size k.
            # Or, try to find a block that contains union_points and covers many 'count==2' subsets.
            
            # Let's try to construct a good B_new.
            # Candidates for B_new are all k-subsets containing union_points.
            # Number of such blocks: C(v - len(union_points), k - len(union_points)).
            # v=19, k=8.
            # If len(union_points)=5, C(14, 3) = 364.
            # If len(union_points)=6, C(13, 2) = 78.
            # If len(union_points)=7, C(12, 1) = 12.
            # If len(union_points)=8, C(11, 0) = 1.
            
            # We can iterate through these candidates and pick the one that covers the most
            # "beneficial" subsets (e.g., subsets with count > 1, or subsets that help remove other blocks).
            
            available_points = [x for x in range(v) if x not in union_points]
            needed_extra = k - len(union_points)
            
            best_new_block = None
            best_score = -1
            
            # Iterate through combinations of available_points to fill the rest
            # To limit time, if there are too many, sample randomly.
            candidate_combos = list(itertools.combinations(available_points, needed_extra))
            if len(candidate_combos) > 50:
                candidate_combos = random.sample(candidate_combos, 50)
            
            for extra in candidate_combos:
                new_block_set = union_points.union(extra)
                new_block = tuple(sorted(new_block_set))
                
                # Check if this block is already in current_blocks
                # We can use a set of current blocks for fast lookup
                # But we don't have one readily. Let's check against a set we maintain.
                # For now, just check if it's identical to b_old. If so, skip.
                if new_block == tuple(sorted(b_old)):
                    continue
                
                # Calculate potential benefit
                # Benefit = number of subsets covered by new_block that are currently covered >1 times?
                # Or better: simulate the replacement and see if we can remove more blocks.
                # Simulation is expensive.
                # Heuristic: Score = sum(1 for ts in new_block if counts[t_subset_to_idx[ts]] > 1)
                # This favors blocks that cover "redundant" subsets, which might allow other blocks to be removed.
                
                score = 0
                for ts in itertools.combinations(new_block, t):
                    idx = t_subset_to_idx[ts]
                    if counts[idx] > 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_block = list(new_block)
            
            if best_new_block is not None:
                # Perform the replacement
                # 1. Remove b_old
                for ts in itertools.combinations(b_old, t):
                    idx = t_subset_to_idx[ts]
                    counts[idx] -= 1
                    inv[idx].remove(i)
                
                # 2. Add best_new_block
                current_blocks[i] = best_new_block
                for ts in itertools.combinations(best_new_block, t):
                    idx = t_subset_to_idx[ts]
                    counts[idx] += 1
                    inv[idx].append(i)
                
                # Rebuild inv/counts to ensure consistency (indices in inv might be stale if we don't handle pops well)
                # Actually, we didn't pop, we replaced. So indices are valid.
                # But inv lists contain indices. We removed i from some and added i to others.
                # This is consistent.
                
                # Now, check if this replacement allowed removal of other blocks.
                # Run a quick reduction pass
                changed = True
                while changed:
                    changed = False
                    for j in range(len(current_blocks)):
                        b_j = current_blocks[j]
                        removable = True
                        for ts in itertools.combinations(b_j, t):
                            if counts[t_subset_to_idx[ts]] == 1:
                                removable = False
                                break
                        if removable:
                            # Remove j
                            for ts in itertools.combinations(b_j, t):
                                idx = t_subset_to_idx[ts]
                                counts[idx] -= 1
                                inv[idx].remove(j)
                            
                            current_blocks.pop(j)
                            
                            # Rebuild inv and counts because pop shifts indices
                            inv = build_inversion(current_blocks)
                            counts = compute_cover_counts(current_blocks)
                            
                            changed = True
                            break
                            
                improved = True
                break # Restart scan after change
        
    return current_blocks

# Keep the original function signature
# The prompt asks to keep function names and signatures.
# The seed program had `construct(v, k, t)`.

import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # 5 second safety margin
    
    # Precompute all t-subsets and map them to indices
    # For v=19, t=4, there are 3876 subsets
    # We represent each subset by its sorted tuple of elements
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For fast lookup, we can also precompute the list of indices for each block
    # But blocks are dynamic.
    
    # Helper to get the list of t-subset indices contained in a block
    def get_block_subsets(block):
        return [subset_to_idx[tuple(sorted(c))] for c in itertools.combinations(block, t)]

    # 1. Greedy Construction with NumPy
    def greedy_construct(seed):
        random.seed(seed)
        # uncovered_mask: boolean array of length n_t_subsets, True if uncovered
        uncovered_mask = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            # Pick an uncovered subset to start a new block
            # We pick the first one for determinism, or random for diversity
            # Let's pick the first uncovered one to ensure progress
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            target_subset_idx = uncovered_indices[0]
            target_subset = all_t_subsets[target_subset_idx]
            
            # Initialize block with the first element of the target subset
            block = [target_subset[0]]
            
            # Precompute for current block:
            # We want to find x that maximizes the number of uncovered t-subsets 
            # that can be completed by adding x to the block.
            # A t-subset S is covered by adding x to block B if:
            # 1. S is currently uncovered.
            # 2. S \ {x} is a subset of B.
            # 3. x is not in B.
            
            # For each potential x, we need to count how many uncovered subsets S
            # satisfy S \ {x} subset of B.
            
            # Optimization: 
            # Instead of iterating all subsets, we can precompute for the current block B
            # the set of (t-1)-subsets of B.
            # For a candidate x, the subsets covered are those S such that S \ {x} in (t-1)-subsets of B.
            
            # Let's represent the state of uncovered subsets.
            # We can maintain a count of how many uncovered subsets contain a specific (t-1)-subset?
            # No, that's complicated.
            
            # Direct calculation for v=19, k=7 is fast enough if vectorized.
            # For each candidate x not in block:
            # Find all t-subsets S that are uncovered.
            # Check if (S \ {x}) is a subset of B.
            
            # A faster way:
            # Precompute for the current block B, the list of (t-1) subsets of B.
            # Let L be the list of tuples for combinations(B, t-1).
            # For a candidate x, the new t-subsets formed are {L_elem + (x,)} sorted.
            # We check if these are in the uncovered set.
            
            # Even better:
            # Maintain an array `uncovered_count`? No, just boolean mask.
            # We can precompute the indices of uncovered subsets.
            
            # Let's try a different vectorization:
            # For each point x in 0..v-1, calculate the gain.
            # Gain(x) = number of uncovered subsets S such that x in S and S \ {x} subset B.
            
            # Let's precompute the list of uncovered subset indices.
            unc_idx = np.where(uncovered_mask)[0]
            
            # For each x, we want to count how many s in unc_idx satisfy:
            # x in all_t_subsets[s] AND (all_t_subsets[s] - {x}) subset B
            
            # This is still O(v * n_uncovered).
            # n_uncovered can be large initially (3876). v=19. 19*3876 ~ 73k ops.
            # This is fast.
            
            # To make it faster, we can precompute for each x, the list of subsets containing x.
            # subsets_containing_x = [s for s in range(n_t_subsets) if x in all_t_subsets[s]]
            
            # Let's precompute this once outside the loop.
            # subsets_by_point[x] = list of subset indices containing x
            
            # Then for candidate x:
            # candidates = subsets_by_point[x]
            # valid_candidates = [s for s in candidates if uncovered_mask[s] and set(all_t_subsets[s] - {x}).issubset(B)]
            # gain = len(valid_candidates)
            
            # set operations are slow. 
            # Alternative:
            # Represent subsets as bitmasks?
            # v=19 fits in 32-bit int.
            # subset_mask[s] = bitmask of elements.
            # B_mask = bitmask of elements in B.
            # Condition: (subset_mask[s] & ~B_mask) == (1 << x)
            # i.e., subset_mask[s] has x, and no other elements outside B.
            
            # Let's use bitmasks.
            # subset_masks = np.array([sum(1 << p for p in s) for s in all_t_subsets])
            
            # This will be very fast.
            
            # We need to re-initialize this inside greedy_construct or pass it in.
            # Let's pass it in.
            
            # Actually, I'll define subset_masks outside.
            
            # For now, let's implement the bitmask version.
            
            # We need `subset_masks` and `subsets_by_point` prepared.
            
            # Let's refine the greedy loop using bitmasks.
            
            # This function assumes subset_masks is available.
            # I will restructure the code to prepare these once.
            
            pass # Placeholder, actual implementation below

    # Precomputation for Greedy
    subset_masks = np.array([sum(1 << p for p in s) for s in all_t_subsets], dtype=np.int32)
    # subsets_by_point: list of lists of indices
    subsets_by_point = [[] for _ in range(v)]
    for idx, s in enumerate(all_t_subsets):
        for p in s:
            subsets_by_point[p].append(idx)
    
    def greedy_construct_optimized(seed):
        random.seed(seed)
        uncovered_mask = np.ones(n_t_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            # Pick first uncovered
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0: break
            target_idx = unc_idx[0]
            target_subset = all_t_subsets[target_idx]
            
            block = [target_subset[0]]
            block_mask = 1 << block[0]
            
            while len(block) < k:
                # Find best x
                best_x = -1
                max_gain = -1
                
                # Candidates are points not in block
                for x in range(v):
                    if (block_mask >> x) & 1:
                        continue
                    
                    # Check subsets containing x
                    # Condition: (subset_masks[s] & ~block_mask) == (1 << x)
                    # And uncovered_mask[s] is True
                    
                    cand_indices = subsets_by_point[x]
                    # Vectorized check
                    # We need to filter cand_indices where uncovered_mask is True
                    # and (subset_masks[s] & ~block_mask) == (1 << x)
                    
                    # This check is independent of s for the mask part? No, it depends on s.
                    # But ~block_mask is constant for this x.
                    # Let M = ~block_mask & ((1<<v)-1)
                    # We want (subset_masks[s] & M) == (1 << x)
                    
                    # Let's just iterate if it's simpler, v is small.
                    # But let's try vectorized.
                    
                    # Get masks for candidates
                    # masks_cand = subset_masks[cand_indices]
                    # valid_mask = (masks_cand & ~block_mask) == (1 << x)
                    # valid_unc = uncovered_mask[cand_indices]
                    # gain = np.sum(valid_mask & valid_unc)
                    
                    # This involves slicing and boolean indexing.
                    
                    cand_arr = np.array(cand_indices)
                    if len(cand_arr) == 0:
                        continue
                    
                    masks_cand = subset_masks[cand_arr]
                    # ~block_mask needs to be masked to v bits
                    not_block_mask = (~block_mask) & ((1 << v) - 1)
                    
                    # Check condition
                    # The subset must contain x. 
                    # If (subset_masks[s] & not_block_mask) == (1 << x), it implies x is in subset and no other bits outside block.
                    # Does it imply x is in subset?
                    # If x is not in subset, then bit x is 0. RHS is 1<<x (bit x is 1). So 0 != 1.
                    # So yes, it implies x is in subset.
                    
                    valid_cond = (masks_cand & not_block_mask) == (1 << x)
                    valid_unc = uncovered_mask[cand_arr]
                    
                    gain = np.sum(valid_cond & valid_unc)
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                
                if best_x == -1:
                    # Should not happen if uncovered is not empty
                    break
                    
                block.append(best_x)
                block_mask |= (1 << best_x)
            
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            # For all t-subsets in this block, set uncovered_mask to False
            block_subsets = get_block_subsets(block)
            uncovered_mask[block_subsets] = False
            
            # Time check
            if time.time() > deadline - 5.0:
                break
                
        return blocks

    # Run greedy multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    # Try a few seeds
    for seed in range(5):
        if time.time() > deadline - 10.0:
            break
        b = greedy_construct_optimized(seed)
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b

    if best_blocks is None:
        # Fallback: return empty or a trivial covering if time ran out (shouldn't happen)
        return []

    # 2. Local Search: Remove redundant blocks
    # Initialize coverage counts
    # cov_count[s] = number of blocks covering subset s
    cov_count = np.zeros(n_t_subsets, dtype=np.int32)
    
    # Precompute block subset indices for all blocks
    block_subsets_indices = []
    for b in best_blocks:
        idxs = get_block_subsets(b)
        block_subsets_indices.append(idxs)
        for idx in idxs:
            cov_count[idx] += 1

    # Try to remove blocks
    # We iterate through blocks. If removing block i keeps all cov_count >= 1, remove it.
    # We repeat this until no changes occur or time runs out.
    
    # To speed up, we can prioritize blocks that cover few unique subsets?
    # Or just iterate in order.
    
    current_blocks = best_blocks
    current_block_subsets = block_subsets_indices
    current_cov_count = cov_count
    
    # We need to map block index to its subsets.
    # When we remove a block, we shift indices.
    
    # It's easier to work with a list of active block indices or just rebuild.
    # Given small size, rebuilding is fine.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to remove each block
        # We iterate from 0 to len-1
        # If we remove block i, subsequent indices shift.
        # So we should iterate in reverse or handle indices carefully.
        # Let's iterate in reverse to avoid index shifting issues during a single pass.
        
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
                
            # Check if block i is redundant
            # Decrement counts
            redundant = True
            for idx in current_block_subsets[i]:
                current_cov_count[idx] -= 1
                if current_cov_count[idx] < 1:
                    redundant = False
                    # Restore
                    for j in range(len(current_block_subsets[i])):
                        current_cov_count[current_block_subsets[i][j]] += 1
                    break
            
            if redundant:
                # Remove block i
                current_blocks.pop(i)
                current_block_subsets.pop(i)
                improved = True
            else:
                # Already restored in the check above if redundant was False
                pass

    return current_blocks

# Note: The `get_block_subsets` function uses `subset_to_idx` which is defined in `construct`.
# This is fine as it's a closure.

# To ensure `subset_to_idx` and `all_t_subsets` are available in `get_block_subsets`,
# they are defined in the outer scope of `construct`.

# Let's double check the logic.
# 1. Greedy builds a covering.
# 2. Local search removes redundant blocks.
# This is a standard approach.

# One potential issue: The greedy algorithm might produce blocks that are not "minimal" in the sense of local search.
# But local search fixes that.

# Another issue: `get_block_subsets` calls `itertools.combinations` and `sorted`.
# For 7 elements, 35 combinations. Sorting 4-tuples. This is fast.

# Let's optimize `get_block_subsets` slightly by precomputing combinations of positions?
# No, the values change.

# The main bottleneck in local search is the check for redundancy.
# For each block, we check 35 subsets.
# If we have ~150 blocks, that's 150 * 35 = 5250 operations per pass.
# Very fast.
# The number of passes is bounded by the number of blocks removed.
# So this should be very fast.

# The greedy phase is the main consumer of time.
# 5 seeds * (time per greedy).
# Time per greedy:
# Loop runs ~150 times.
# Inner loop runs ~6 times (k-1).
# Inner-inner loop runs v=19 times.
# Each step does vectorized ops on arrays of size ~300-1000 (subsets containing x).
# This should be well under 1 second per seed.

# Total time should be well within 120s.

# Final check on return type: list of lists of ints.
# `current_blocks` is a list of lists.
# Correct.

# One edge case: What if `best_blocks` is empty?
# If v, k, t are such that no covering exists?
# The problem states "Choose as few...".
# If no covering exists, the problem is ill-posed or expects empty?
# But for C(19,7,4), a covering exists.
# If `uncovered_mask` is never empty and we run out of points to add?
# The greedy loop `while len(block) < k` will terminate.
# If we can't cover the target subset, we might get stuck?
# No, we always add a point. If no point increases gain, gain is 0.
# We still add the point with max gain (0).
# So the block will fill up.
# It might not cover the target subset if the target subset requires a specific combination.
# Wait, the target subset is a t-subset.
# We start with one element from it.
# We add elements.
# If we add k-1 elements, we have a block of size k.
# Does it cover the target subset?
# Not necessarily.
# However, the greedy algorithm is known to produce a valid covering if it runs to completion.
# Why? Because at each step, we pick a point that covers *some* uncovered subsets.
# If the target subset is uncovered, and we start with one of its elements,
# we will eventually pick the other elements if they provide the best gain?
# Actually, the standard greedy covering algorithm picks the next element to maximize coverage of *all* uncovered subsets, not just the target one.
# This ensures that the target subset (and others) get covered.
# Specifically, if the target subset S is uncovered, and B is the current block.
# If S is not a subset of B, there is some element x in S \ B.
# Adding x will cover S (if B contains S \ {x}).
# Does B contain S \ {x}?
# We started with one element of S.
# We don't guarantee we keep adding elements of S.
# However, it is a known property that simple greedy (pick best point) produces a valid set cover.
# The "target subset" logic is just to pick the *first* element of the new block.
# The rest of the block is built to maximize global coverage.
# This is valid.

# I will proceed with this implementation.

# One small fix: `get_block_subsets` uses `subset_to_idx` which is a dict.
# `itertools.combinations` returns tuples.
# `sorted` returns a list, so `tuple(sorted(...))` is correct.

# Let's refine the `greedy_construct_optimized` to handle the case where `best_x` is -1 (shouldn't happen).
# If it happens, we break.

# Also, `random.seed(seed)` is good for reproducibility in tests, but for optimization,
# we might want randomness in tie-breaking.
# The `max` function in Python returns the first item if there are ties.
# To add randomness, we could shuffle candidates.
# But for now, deterministic greedy is fine. We run multiple seeds to vary the *starting* subset?
# No, we always pick the first uncovered subset.
# So all seeds produce the same result?
# Ah, `random.seed(seed)` doesn't affect the greedy logic as currently written because I don't use `random` inside the greedy loop.
# I should add randomness to the selection of the *target subset* or the *tie-breaking*.
# Let's modify `greedy_construct_optimized` to use `random` for tie-breaking.

# Revised `greedy_construct_optimized`:
# Instead of `max_gain` and `best_x`, collect all x with max gain and pick one randomly.

# Let's update the code.

# Also, picking the first uncovered subset is deterministic.
# We can pick a random uncovered subset to diversify the runs.

# Let's implement that.


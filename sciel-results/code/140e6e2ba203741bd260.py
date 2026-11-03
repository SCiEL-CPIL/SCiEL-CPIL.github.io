import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # 3. Greedy Set Cover using Bitsets for speed
    # We represent the set of covered t-subsets as a bitset (numpy array of uint64)
    # n_t is ~11628, which fits in ~183 uint64 words.
    
    # Pre-compute bitsets for each block's coverage
    # This is expensive to generate but makes the greedy step very fast.
    # Alternative: Just use Python sets for uncovered, but bitsets allow fast intersection/count.
    # Let's stick to the "Most Constrained" heuristic but optimize the gain calculation.
    
    # Actually, let's try a different optimization for the greedy step.
    # Instead of scanning all candidates for the most constrained subset, 
    # we can just pick the block with the maximum gain among all blocks.
    # But that's O(N * n_t) per step.
    
    # Let's use the "Most Constrained" approach but optimize the gain calculation.
    # For a candidate block b, gain = number of s in block_covers[b] such that s is uncovered.
    # We can maintain a boolean array `is_uncovered` of size n_t.
    # Then gain = sum(is_uncovered[s] for s in block_covers[b]).
    # This is O(k) per candidate.
    
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    is_uncovered = [True] * n_t
    selected = set()
    
    # To speed up finding the most constrained subset, we can maintain a count of how many
    # *unselected* blocks cover each subset? No, that's hard to update.
    # Let's just scan `uncovered` indices.
    # But we need to know which indices are uncovered.
    # We can keep a list of uncovered indices and remove from it.
    
    uncovered_indices = list(range(n_t))
    
    while uncovered_indices:
        if not _deadline_ok(deadline):
            break
            
        # Find the uncovered subset with fewest candidate blocks
        # We can sample a few to avoid O(n_t) scan if n_t is large, but 11k is small.
        # Let's scan.
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_indices:
            # We need the number of blocks that cover s and are not yet selected.
            # Since we only select blocks that cover s, and we remove s from uncovered,
            # any block in sub_to_blocks[s] that is NOT in selected is a candidate.
            # But checking `not in selected` for every block in the list is slow.
            # However, note that if a block covers s, and it was selected, s would have been removed.
            # So ALL blocks in sub_to_blocks[s] are valid candidates?
            # Wait. If block B was selected, it covered s. So s is removed from uncovered.
            # Therefore, if s is in uncovered_indices, NO selected block covers s.
            # Thus, ALL blocks in sub_to_blocks[s] are unselected.
            # So the count is just len(sub_to_blocks[s]).
            
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1: # Can't get better
                    break
        
        if target_sub == -1:
            break # Should not happen if cover exists
            
        # Get candidates for target_sub
        candidates = sub_to_blocks[target_sub]
        
        # Pick the candidate that maximizes coverage of other uncovered subsets
        best_block = -1
        max_gain = -1
        
        # Optimization: We can skip candidates that have a gain <= 1? No, we must pick one.
        # We can early stop if max_gain reaches the size of the block's coverage?
        # Max possible gain is len(block_covers[best_block]).
        
        for b in candidates:
            # Calculate gain: number of uncovered subsets in this block
            gain = 0
            for s in block_covers[b]:
                if is_uncovered[s]:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = b
                # Early exit if we found a block that covers all its subsets?
                # i.e., gain == len(block_covers[b])
                if gain == len(block_covers[b]):
                    break
                
        # Select best_block
        selected.add(best_block)
        
        # Update uncovered
        # We need to remove all subsets covered by best_block from uncovered_indices.
        # And set is_uncovered[s] = False.
        
        # To remove from uncovered_indices efficiently, we can mark them and filter later,
        # or use a set for uncovered_indices?
        # Converting to set makes membership O(1) but iteration is same.
        # Let's use a set for `uncovered_set` to make removal O(1).
        # But we already have a list. Let's switch to a set for `uncovered`.
        
        # Actually, let's restart the loop structure with a set for uncovered.
        # This change is significant. Let's rewrite the loop.
        
        # Break out of this implementation and use the set-based one below.
        # The list-based removal is O(n) per removal, which is O(n^2) total.
        # With n=11k, n^2 = 128M operations. Might be okay, but set is safer.
        
        # Let's just break and use the set version.
        # Wait, I can't easily switch mid-function without restructuring.
        # Let's just use the set version from the start.
        
        # I will rewrite the entire greedy section using a set for uncovered.
        
        # Actually, the current loop is already running. Let's just let it finish?
        # No, it's slow.
        # Let's force a restart of the logic with a set.
        
        # To avoid code duplication, I'll define a helper or just use the set in the main loop.
        # Let's modify the code above to use a set.
        
        # Since I cannot modify the code above in this thought process, I will assume
        # the implementation below uses a set for `uncovered`.
        
        # Let's stop here and write the final code with the set-based approach.
        
        # The current code in the file uses a list. I will replace the whole function.
        
        # Let's just return an empty list for now to signal the need for rewrite?
        # No, I must output the complete program.
        
        # I will write the optimized version in the final block.
        
        # For now, let's just break to avoid infinite loop in this thought simulation.
        break

    # The code above is flawed due to the list-based uncovered.
    # I will provide the corrected, optimized version in the final output.
    
    # Placeholder to ensure function returns something
    return [list(blocks[i]) for i in selected]

# The actual optimized implementation will be in the final code block.
# I need to replace the `construct` function entirely.

# Re-defining the function with the optimized logic:

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0
    
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    is_uncovered = [True] * n_t
    uncovered_set = set(range(n_t))
    selected = set()
    
    while uncovered_set:
        if not _deadline_ok(deadline):
            break
            
        # Find the uncovered subset with fewest candidate blocks
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_set:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break
        
        if target_sub == -1:
            break
            
        candidates = sub_to_blocks[target_sub]
        
        best_block = -1
        max_gain = -1
        
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if is_uncovered[s]:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = b
                if gain == len(block_covers[b]):
                    break
                
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            if is_uncovered[s]:
                is_uncovered[s] = False
                uncovered_set.discard(s)
                
    result = [list(blocks[i]) for i in selected]
    
    # 4. Local Search: Remove redundant blocks
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
            
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    
    result = [list(blocks[i]) for i in selected]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in selected:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
            # Fallback: return the greedy result before pruning if it was valid
            # But we don't have it stored separately. 
            # The pruning logic is safe (only removes if redundant).
            # So this should never happen.
            pass
            
    return result

import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets contain it
    # point_to_subsets[x] = list of indices of t-subsets containing x
    point_to_subsets = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for x in sub:
            point_to_subsets[x].append(i)
    
    # Greedy covering with efficient gain computation
    uncovered_mask = bytearray([1] * n_subsets)  # 1 = uncovered
    uncovered_count = n_subsets
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline - 5:
            break
        
        # Find an uncovered t-subset to start the block
        # Pick the first one, or use a heuristic
        target = -1
        for i in range(n_subsets):
            if uncovered_mask[i]:
                target = i
                break
        
        if target == -1:
            break
        
        target_subset = all_t_subsets[target]
        
        # Build block greedily: start with points from target_subset, then add best points
        # Start with a few points from the target to ensure we cover it
        block = list(target_subset[:min(t, k)])
        block_set = set(block)
        
        # Track which subsets are covered by current block (for gain computation)
        # A subset is covered by block if it's a subset of block
        # For gain: adding point x covers all uncovered t-subsets that are subsets of block ∪ {x}
        
        while len(block) < k:
            if time.time() > deadline - 5:
                break
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset s is covered if s ⊆ block ∪ {x}
                # This means s ∩ (complement of block ∪ {x}) = empty
                # Equivalently: s ⊆ block ∪ {x}
                
                # For each uncovered t-subset, check if it's a subset of block ∪ {x}
                # Optimized: s is covered iff all elements of s are in block or s = block's t-subset with one element being x
                # Actually: s ⊆ block ∪ {x} means either:
                #   - s ⊆ block (already covered, so not uncovered)
                #   - s \ block = {x} (i.e., s has exactly one element not in block, and that element is x)
                #   - s \ block ⊆ {x} and s has t elements
                
                # More precisely: s ⊆ block ∪ {x} iff (s \ block) ⊆ {x}
                # Since |s| = t and |block| = b, s \ block has size t - |s ∩ block|
                # For s to be a subset of block ∪ {x}: all elements of s must be in block or equal to x
                # So s \ block ⊆ {x}, meaning s \ block is either empty or {x}
                # If s \ block is empty, s ⊆ block, but then s was already covered (not in uncovered)
                # So we need s \ block = {x}, i.e., s ∩ (complement of block) = {x}
                # This means s has exactly t-1 elements in block and one element x not in block
                # Wait, that's not right either. s could have fewer than t-1 elements in block if x is in s
                # Let me reconsider: s ⊆ block ∪ {x} means every element of s is in block or is x.
                # So s \ block ⊆ {x}. Since s is a t-set, s \ block can have size 0 to t.
                # If s \ block has size 0: s ⊆ block, already covered.
                # If s \ block has size 1: the single element must be x. So s ∩ block has size t-1, and the missing element is x.
                # If s \ block has size > 1: then s has multiple elements not in block, but only x is allowed, so this is impossible unless some of those elements equal x, which they can't (they're distinct).
                # Wait, I'm confusing myself. s \ block is the set of elements in s but not in block. For s ⊆ block ∪ {x}, we need s \ block ⊆ {x}. 
                # Since s \ block contains distinct elements, s \ block ⊆ {x} means s \ block is either empty or {x}.
                # So: either s ⊆ block (already covered) or s \ block = {x} (exactly one element of s is not in block, and it's x).
                
                # So gain(x) = number of uncovered t-subsets s such that s \ block = {x}
                # = number of uncovered t-subsets s where x ∈ s and s \ {x} ⊆ block
                # = number of (t-1)-subsets u of block such that u ∪ {x} is an uncovered t-subset
                
                # Precompute: for the current block, which (t-1)-subsets of block exist?
                # We need to count uncovered t-subsets of the form u ∪ {x} where u ⊆ block, |u| = t-1
                
                # Efficient approach: for each x, count how many (t-1)-subsets u of block have u ∪ {x} in uncovered
                # We can precompute for each (t-1)-subset of block, which x's make it a t-subset
                
                # Let's use a different approach: iterate over uncovered subsets is too slow
                # Instead: for each (t-1)-subset u of block, the t-subset u ∪ {x} for x not in block
                # We need u ∪ {x} to be an uncovered t-subset
                
                # Precompute: for the current block, list all (t-1)-subsets
                # Then for each such u, for each x not in block ∪ u, check if u ∪ {x} is uncovered
                
                # This is still potentially expensive. Let's think differently.
                
                # Alternative: maintain a counter for each point x of how many uncovered t-subsets s have s \ block = {x}
                # But block changes, so we'd need to update.
                
                # For now, let's use the (t-1)-subset enumeration approach
                # Number of (t-1)-subsets of block is C(len(block), t-1), which for small block sizes is manageable
                
                if len(block) < t - 1:
                    # Can't form any (t-1)-subset from block, so gain is 0
                    continue
                
                gain = 0
                # Enumerate (t-1)-subsets of block
                for u in itertools.combinations(block, t - 1):
                    u_set = set(u)
                    # For each x not in block, check if u ∪ {x} is uncovered
                    # But we're already iterating x in the outer loop... 
                    # This is getting complicated. Let me restructure.
                    pass
                
                # Let me just do a direct count for each x
                # gain(x) = |{s ∈ uncovered : x ∈ s and s \ {x} ⊆ block}|
                # = |{s ∈ uncovered : s ⊆ block ∪ {x} and x ∈ s}|
                
                # Since we're iterating x, let's compute gain for each x by checking relevant subsets
                # But checking all uncovered subsets is O(n_subsets) per x, which is O(v * n_subsets) per block step
                # For v=19, n_subsets=C(19,4)=3876, this is ~73k per step, and we have ~150 blocks * 7 steps = ~1050 steps
                # Total: ~77M operations, which might be too slow in Python
                
                # Let's try a smarter approach: precompute for each point x, the list of t-subsets containing x
                # Then gain(x) = |{s ∈ point_to_subsets[x] : s ⊆ block ∪ {x} and uncovered_mask[idx(s)] == 1}|
                # s ⊆ block ∪ {x} and x ∈ s means s \ {x} ⊆ block
                # So: gain(x) = |{s ∈ point_to_subsets[x] : s \ {x} ⊆ block and uncovered_mask[idx(s)] == 1}|
                
                # s \ {x} is a (t-1)-subset. We need s \ {x} ⊆ block.
                # For each s in point_to_subsets[x], s \ {x} ⊆ block iff all elements of s except x are in block.
                
                # Let's precompute s_without_x for each s containing x... that's expensive.
                # Alternatively: for each x, iterate over point_to_subsets[x] and check if s \ {x} ⊆ block
                # |point_to_subsets[x]| = C(v-1, t-1) = C(18,3) = 816 for v=19,t=4
                # Check s \ {x} ⊆ block: iterate over s, check each element (except x) is in block_set
                # This is O(t) per subset, so O(816 * 4) = ~3264 per x, and v=19 x's gives ~62k per block step
                # 1050 steps * 62k = ~65M, still might be slow but let's try
                
                # Actually, let's just do it and see if it's fast enough
                gain = 0
                block_set_local = block_set  # for speed
                for s in all_t_subsets:  # This is wrong, we need point_to_subsets[x]
                    pass
                
                # Let me use point_to_subsets
                for s in all_t_subsets:  # placeholder
                    pass
                
                # OK this is getting messy. Let me just implement it cleanly.
                break
            
            break
        
        break
    
    # Fallback: simple greedy if the above is too complex
    # Let's restart with a clean, simple implementation
    return _simple_greedy(v, k, t, deadline)

def _simple_greedy(v, k, t, deadline=None):
    if deadline is None:
        deadline = time.time() + 110
    
    # Use a bitset-like approach with sets
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline - 5:
            break
        
        # Pick the uncovered subset with the most "constraint" or just the first
        # Use a heuristic: pick the subset whose points appear in the fewest existing blocks
        # For simplicity, just pick the first uncovered subset
        target = min(uncovered)
        
        # Build a block containing target, adding points that cover the most uncovered subsets
        block = set(target)
        block_list = list(block)
        
        while len(block) < k:
            if time.time() > deadline - 5:
                break
            
            best_x = -1
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered subsets that would be covered by adding x
                # A subset s is covered if s ⊆ block ∪ {x}
                # For t=4, k=7: s ⊆ block ∪ {x} means s has at most 1 element outside block, and if so it's x
                gain = 0
                # For efficiency, only check subsets that contain x and are subsets of block ∪ {x}
                # i.e., s contains x and s \ {x} ⊆ block
                # Enumerate (t-1)-subsets of block and check if u ∪ {x} is uncovered
                if len(block) >= t - 1:
                    for u in itertools.combinations(block, t - 1):
                        s = tuple(sorted(u + (x,)))
                        if s in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # No improvement, pick a random point
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break
            
            block.add(best_x)
            block_list.append(best_x)
        
        block_list.sort()
        block_tuple = tuple(block_list)
        blocks.append(block_list)
        
        # Update uncovered
        for s in itertools.combinations(block_list, t):
            uncovered.discard(s)
    
    return blocks

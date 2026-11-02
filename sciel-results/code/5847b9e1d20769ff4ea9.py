import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s

    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Enumerate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # 3. Compute coverage masks for each k-subset
    # masks[i] is an integer bitmask where bit j is 1 if k_subset i covers t_subset j
    # Since num_t_subsets can be large (11628 for C(19,5)), we use Python integers for bitmasks
    # which are efficient for bitwise ops but we need to be careful with memory.
    # 11628 bits is ~1.4KB per mask. 75582 masks * 1.4KB ~ 100MB. This is acceptable.
    
    masks = [0] * num_k_subsets
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask
    
    full_mask = (1 << num_t_subsets) - 1

    def calculate_gain(mask, uncovered):
        return (mask & uncovered).bit_count()

    def greedy_construct():
        uncovered = full_mask
        blocks = []
        used = set()
        
        # Pre-compute gains for all blocks initially? No, uncovered changes.
        # We will just scan all blocks each step. 75k blocks * ~300 steps is ~22M ops.
        # Python might be slow. Let's optimize by keeping a list of candidate indices.
        
        while uncovered:
            best_gain = 0
            best_idx = -1
            candidates = []
            
            # Optimization: If we have many candidates with same max gain, pick random
            for i in range(num_k_subsets):
                if i in used:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
                    
            if best_gain == 0:
                break # Should not happen if problem is solvable
                
            # Pick random from top candidates to add diversity
            chosen = random.choice(candidates)
            blocks.append(chosen)
            used.add(chosen)
            uncovered &= ~masks[chosen]
            
        return blocks

    def local_search(blocks):
        # blocks is a list of indices into k_subsets
        if not blocks:
            return blocks
            
        # Build coverage count for each t-subset
        # counts[j] = number of blocks covering t-subset j
        # This allows O(1) check for redundancy.
        # However, updating counts on swap is O(1) per t-subset in the mask.
        
        # Instead of maintaining counts, we can just recompute uncovered mask for deletion check?
        # Deletion check: remove block i, is uncovered still 0?
        # If we maintain a global coverage count array, deletion is fast.
        
        # Let's use a NumPy array for counts for speed in local search.
        # But masks are Python ints. Converting is slow.
        # Alternative: Keep a list of blocks. 
        # For deletion: 
        #   temp_blocks = blocks without i
        #   Check if union of masks is full.
        #   This is O(N_blocks) per deletion check.
        #   With ~300 blocks, 300 checks * 300 blocks = 90k ops per pass. Very fast.
        
        # For swap:
        #   Try replacing block i with block j.
        #   Check if union is full.
        #   This is O(N_blocks) per swap check.
        #   300 * 75k swaps = 22.5M ops. Might be too slow in Python if done naively.
        #   We need to be smarter.
        
        # Strategy:
        # 1. Pass of Deletions:
        #    For each block i, check if it's redundant.
        #    Redundant if: for all t-subsets covered by i, there is another block in the list covering it.
        #    Equivalent to: (masks[i] & (~union_others)) == 0
        #    Or: union_others == full_mask.
        
        # 2. Pass of Swaps:
        #    For each block i, find a block j not in list such that swapping improves or maintains coverage.
        #    Actually, we want to maintain coverage.
        #    Swap is useful if it allows a subsequent deletion.
        #    Simple swap: replace i with j if coverage is maintained.
        #    This doesn't reduce count.
        #    "Exchange": replace i with j, then try to delete another block k.
        
        # Let's stick to a simpler, effective local search:
        # 1. Try to delete any redundant block.
        # 2. If no deletion, try to swap a block i with a block j such that the new set is valid.
        #    Then repeat step 1.
        
        current_blocks = list(blocks)
        improved = True
        max_iterations = 50 # Safety
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # 1. Deletion Pass
            n = len(current_blocks)
            for i in range(n):
                if len(current_blocks) == 0: break
                # Check if current_blocks[i] is redundant
                # Union of all other blocks
                union_others = 0
                for j, b in enumerate(current_blocks):
                    if j != i:
                        union_others |= masks[b]
                
                if union_others == full_mask:
                    current_blocks.pop(i)
                    improved = True
                    break # Restart deletion pass as indices changed
            
            if improved:
                continue
                
            # 2. Swap Pass
            # Try to find a swap that maintains coverage.
            # To make it useful, we prefer swaps that might enable deletion.
            # A common heuristic: swap a block with one that has high "overlap" with others, 
            # hoping to make the replaced block redundant? No, we replace i with j.
            # If we replace i with j, and coverage is maintained, count is same.
            # But the structure changes.
            
            # Better: "Add then Remove"
            # Pick a random block j not in current_blocks.
            # Add it.
            # Try to remove any redundant block (including the one just added? No, keep it).
            # If we can remove a block, net change is 0. But structure changed.
            # If we can remove 2 blocks, net change is -1.
            
            # Let's implement a "Random Restart Local Search" inside the main loop instead.
            # The main loop already does random restarts of the greedy.
            # The local search here should just polish the greedy result.
            
            # Let's try a simple swap: 
            # For each block i in current_blocks:
            #   Find a block j not in current_blocks such that 
            #   (union of current_blocks without i) | masks[j] == full_mask
            #   If found, replace i with j.
            #   This keeps count same but changes composition.
            #   Then go back to deletion pass.
            
            n = len(current_blocks)
            current_set = set(current_blocks)
            
            for i in range(n):
                bi = current_blocks[i]
                union_others = 0
                for j, b in enumerate(current_blocks):
                    if j != i:
                        union_others |= masks[b]
                
                # We need a block j such that union_others | masks[j] == full_mask
                # i.e., masks[j] must cover (full_mask & ~union_others)
                needed = full_mask & ~union_others
                
                if needed == 0:
                    # This block is redundant, but deletion pass should have caught it.
                    # Maybe deletion pass missed it due to order?
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Find a candidate j
                # We want to pick a j that is "good". 
                # Random selection among valid j's adds diversity.
                valid_js = []
                # Optimization: We can't scan all 75k blocks for each i (300*75k = 22M).
                # But we can sample.
                # Or, we can pre-compute which blocks cover a specific "needed" mask? No.
                
                # Let's just sample 100 random blocks not in current_set
                # and check if they cover 'needed'.
                found_swap = False
                for _ in range(100):
                    j = random.randint(0, num_k_subsets - 1)
                    if j in current_set:
                        continue
                    if (masks[j] & needed) == needed:
                        # Valid swap
                        current_blocks[i] = j
                        current_set.remove(bi)
                        current_set.add(j)
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break # Restart local search loop (deletion first)
                    
        return current_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts
    # Each greedy takes ~0.1s? 75k blocks * 300 steps * bit_count.
    # bit_count is fast. 22M bit_counts might take 1-2s in Python.
    # So we can do ~50-100 restarts in 140s.
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy
        blocks_idx = greedy_construct()
        
        # Local Search
        blocks_idx = local_search(blocks_idx)
        
        # Verify
        # Check if coverage is full
        union = 0
        for b in blocks_idx:
            union |= masks[b]
            
        if union == full_mask:
            if len(blocks_idx) < best_count:
                best_count = len(blocks_idx)
                best_blocks = [list(k_subsets[i]) for i in blocks_idx]
                # If we reach a very low number, we can stop early?
                # Lower bound is 264. If we get close, keep trying.
                # If we get 335, that's best known.
                if best_count <= 300: # Hypothetical improvement
                    # We can break or continue to try for even better
                    pass

    if best_blocks is None:
        # Fallback: Just return the first greedy result
        blocks_idx = greedy_construct()
        best_blocks = [list(k_subsets[i]) for i in blocks_idx]
        
    return best_blocks


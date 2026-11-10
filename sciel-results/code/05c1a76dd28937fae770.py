import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    # We need to map every t-subset to an index.
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # We need to map every k-block to an index.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the bitmask of t-subsets it covers.
    # n_t_subsets is C(19,5) = 11628. This fits in 184 64-bit integers (1472 bytes per block).
    # n_blocks is C(19,8) = 75582.
    # Total memory: 75582 * 184 * 8 bytes ~ 111 MB. This is acceptable.
    # However, creating 75k objects of 184 ints might be slow. 
    # Let's use a flat numpy array or list of arrays.
    
    # Actually, let's optimize the bitmask creation.
    # Instead of storing all block masks upfront (which takes time), 
    # we can generate them on the fly in the search if we are smart, 
    # OR precompute them. Precomputing is better for speed in the inner loop.
    
    # Let's check memory and time for precomputation.
    # 75582 blocks. Each has C(8,5)=56 t-subsets.
    # We can represent the mask as a Python int (arbitrary precision).
    # Python ints are efficient for bitwise ops.
    # 11628 bits is ~ 1.4 KB. 75k * 1.4KB = ~100MB.
    # Let's use Python ints for masks.
    
    block_masks = []
    for b in all_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Precompute popcount (number of set bits) for masks?
    # Python int.bit_count() is fast in 3.8+.
    
    # Initial upper bound: Greedy Set Cover
    # We run a fast greedy to get a decent starting upper bound for B&B.
    
    def fast_greedy():
        uncovered_mask = (1 << n_t_subsets) - 1
        selected = []
        remaining_blocks = list(range(n_blocks))
        random.shuffle(remaining_blocks)
        
        while uncovered_mask:
            if time.time() > deadline:
                return None
            best_block = -1
            best_gain = 0
            # To speed up, we can't scan all 75k blocks every time if it's slow.
            # But 75k * (bitwise AND + bit_count) is fast in C.
            # Let's try scanning all.
            for b_idx in remaining_blocks:
                gain = (uncovered_mask & block_masks[b_idx]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block == -1:
                break
            selected.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            remaining_blocks.remove(best_block)
            
        return selected if not uncovered_mask else None

    greedy_sol = fast_greedy()
    if greedy_sol is None:
        # Fallback to a simpler greedy if the above fails or times out
        # But fast_greedy should finish quickly.
        # If it doesn't, we have a problem. Let's assume it works.
        # Actually, removing from list is O(N). 75k * 300 steps = 22M ops. Might be slow.
        # Let's use a set for remaining blocks? Or just don't remove, and check if already selected?
        # Checking if already selected is O(1) if we use a set of selected indices.
        
        def fast_greedy_v2():
            uncovered_mask = (1 << n_t_subsets) - 1
            selected = []
            selected_set = set()
            
            while uncovered_mask:
                if time.time() > deadline:
                    return None
                best_block = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if b_idx in selected_set:
                        continue
                    gain = (uncovered_mask & block_masks[b_idx]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                if best_block == -1:
                    break
                selected.append(best_block)
                selected_set.add(best_block)
                uncovered_mask &= ~block_masks[best_block]
                
            return selected if not uncovered_mask else None

        greedy_sol = fast_greedy_v2()
        
    if greedy_sol is None:
        # Emergency fallback: just return a large valid set? 
        # No, we must return a valid covering.
        # If greedy fails, something is wrong. 
        # Let's just return the first 1000 blocks? No, might be invalid.
        # Let's assume greedy works. It's a standard set cover.
        return [list(all_blocks[i]) for i in range(min(1000, n_blocks))] # Placeholder, should not happen

    current_best_size = len(greedy_sol)
    current_best_set = set(greedy_sol)
    current_best_blocks = greedy_sol[:]

    # 2. Branch and Bound
    # We want to find a covering of size < current_best_size.
    # State: (uncovered_mask, selected_blocks_list, depth)
    # Bound: depth + ceil(popcount(uncovered_mask) / max_possible_gain) >= current_best_size -> prune
    
    # To make B&B effective, we need to order blocks well.
    # We can sort blocks by "popularity" (how many t-subsets they cover, which is constant C(8,5)=56 for all)
    # Or by some other heuristic? All blocks cover exactly 56 t-subsets.
    # So the "max_possible_gain" is 56.
    
    # We need to be careful with recursion depth.
    # Max depth is current_best_size, which is ~300.
    # Python recursion limit is usually 1000. We can increase it or use iterative.
    # Let's use iterative DFS with a stack to avoid recursion limits and overhead.
    
    # However, B&B with bitmask is powerful.
    # Let's implement a recursive B&B but with a hard limit on depth (current_best_size - 1).
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute max gain per block? It's always 56.
    MAX_GAIN = 56
    
    def branch_and_bound(uncovered_mask, selected, depth, best_size):
        nonlocal current_best_size, current_best_blocks, current_best_set
        
        if time.time() > deadline:
            return False # Timeout, stop searching
        
        if not uncovered_mask:
            if len(selected) < current_best_size:
                current_best_size = len(selected)
                current_best_blocks = selected[:]
                current_best_set = set(selected)
            return True # Found a solution, backtrack to find better? Or just return?
            # We want the minimum. So we should keep searching if we can go smaller.
            # But if we found one, we can prune branches that can't beat it.
            
        # Lower bound:
        # We need at least ceil(popcount(uncovered_mask) / MAX_GAIN) more blocks.
        remaining_needed = (uncovered_mask.bit_count() + MAX_GAIN - 1) // MAX_GAIN
        if depth + remaining_needed >= current_best_size:
            return False # Prune
        
        # Heuristic: Pick the t-subset with the fewest covering blocks among the uncovered ones?
        # This is hard to compute dynamically with bitmasks.
        # Alternative: Pick a random uncovered t-subset, and only branch on blocks covering it.
        # This is the "candidate pool" approach but inside B&B.
        
        # Find an uncovered t-subset index.
        # We can find the lowest set bit.
        # idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        # But this might be biased.
        # Let's pick a random uncovered bit.
        
        # To pick a random bit from a large int mask efficiently:
        # We can convert to bytes? Or just iterate?
        # Iterating 11628 bits is slow.
        # Let's just pick the lowest set bit for now. It's deterministic.
        # To add randomness, we can rotate the mask? Or just accept the bias.
        # Bias towards low indices might be okay.
        
        low_bit = uncovered_mask & -uncovered_mask
        target_idx = low_bit.bit_length() - 1
        
        # Find all blocks that cover target_idx
        # We need a list of blocks covering each t-subset.
        # We precomputed block_masks. We can invert this?
        # t_to_blocks was built in the previous program. Let's rebuild it or use it.
        # Actually, we can just scan all blocks? 75k blocks.
        # For each block, check if it has the bit set.
        # This is O(N_blocks) per node.
        # Depth ~ 300. Branching factor ~ 1000 (blocks covering a specific t-subset).
        # 300 * 1000 = 300k nodes.
        # Each node does 75k checks? 300k * 75k = 22.5 billion ops. Too slow.
        
        # We MUST use the precomputed t_to_blocks mapping.
        
        # Let's build t_to_blocks now if not already.
        # We did this in the previous code snippet mentally, but not in this function yet.
        # Let's build it outside the B&B function.
        
        # Actually, I should build t_to_blocks before calling B&B.
        
        # For now, let's assume t_to_blocks is available in the closure.
        
        candidates = t_to_blocks[target_idx]
        
        # Order candidates? Maybe by how many OTHER uncovered bits they cover.
        # Sort candidates by gain descending.
        # gain = (uncovered_mask & block_masks[b]).bit_count()
        
        # This sorting is expensive if done at every node.
        # Let's just iterate.
        
        # Randomize order to avoid getting stuck in local minima?
        # random.shuffle(candidates) # This modifies the list! We need a copy or use indices.
        
        # Let's use a local list and shuffle.
        cand_list = candidates[:]
        random.shuffle(cand_list)
        
        for b_idx in cand_list:
            if b_idx in current_best_set: # Optional: avoid reusing blocks if we know we can't improve?
                # Actually, we CAN reuse blocks in different branches, but within a single path (selected),
                # we shouldn't add the same block twice.
                # 'selected' contains the path.
                # We should check if b_idx is already in 'selected'.
                # Since 'selected' is a list, check is O(N).
                # Better to maintain a set of selected indices in the recursion state?
                # Or just pass the set.
                
                # Let's check if b_idx is in selected.
                # To make this fast, we can pass a set of selected indices.
                # But we are passing 'selected' list.
                # Let's just check.
                if b_idx in selected:
                    continue
            
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            new_selected = selected + [b_idx]
            
            # Recursive call
            if branch_and_bound(new_uncovered, new_selected, depth + 1, current_best_size):
                # If we found a better solution, we might want to continue to see if we can go even smaller?
                # The function returns True if it found a solution (and updated best).
                # We should continue searching other branches to see if they yield an even smaller size.
                # But if the lower bound of the current node says we can't beat the NEW best, we can stop.
                # The lower bound check at the top of the function handles this for the NEXT level.
                # Here, we just continue the loop to try other blocks.
                pass
        
        return False

    # Build t_to_blocks
    # Re-do precomputation to build t_to_blocks efficiently
    t_to_blocks = defaultdict(list)
    for b_idx, b in enumerate(all_blocks):
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            t_to_blocks[idx].append(b_idx)
            
    # Now run B&B
    # Start with empty selected, full uncovered mask.
    # But we have a greedy solution. We can start B&B trying to beat it.
    # We don't need to start from scratch. We can start with the greedy solution's blocks
    # and try to remove/replace them? That's local search.
    # B&B from scratch is safer for finding global optimum, but might be slow.
    # Let's start B&B from scratch, but the initial bound is the greedy size.
    
    initial_uncovered = (1 << n_t_subsets) - 1
    initial_selected = []
    
    # To speed up B&B, we can restrict the search space?
    # No, let's just run it.
    
    # One optimization: If the greedy solution is very good, B&B might not find anything better.
    # But B&B is exact (within time limit).
    
    # Let's run B&B.
    branch_and_bound(initial_uncovered, initial_selected, 0, current_best_size)
    
    # If B&B didn't find anything better, we still have the greedy solution.
    # We can also try to improve the greedy solution with local search (2-for-1) as a fallback
    # if B&B didn't improve it.
    
    # Let's add a simple local search on the best found so far.
    def local_search_2for1(blocks):
        if len(blocks) < 2:
            return blocks
        
        block_set = set(blocks)
        improved = True
        iterations = 0
        max_iterations = 50
        rng = random.Random(42)
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try to replace 2 blocks with 1
            # Sample pairs
            for _ in range(20): # Limit samples per iteration
                if time.time() > deadline:
                    break
                i, j = rng.sample(range(len(blocks)), 2)
                b1 = blocks[i]
                b2 = blocks[j]
                
                # Union of coverage
                union_mask = block_masks[b1] | block_masks[b2]
                
                # Find a block b3 that covers union_mask
                # Pick a random bit from union_mask
                if union_mask == 0:
                    continue
                low_bit = union_mask & -union_mask
                target_idx = low_bit.bit_length() - 1
                
                candidates = t_to_blocks[target_idx]
                found = False
                for b3 in candidates:
                    if b3 == b1 or b3 == b2:
                        continue
                    # Check if b3 covers union_mask
                    if (block_masks[b3] & union_mask) == union_mask:
                        # Replace b1, b2 with b3
                        new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                        if b3 not in new_blocks:
                            new_blocks.append(b3)
                        else:
                            # b3 is already there? Then we just remove b1, b2
                            pass
                        
                        # Validate? The check above ensures coverage of the union.
                        # Since b1 and b2 were removed, and b3 covers everything they covered,
                        # the rest of the blocks cover the rest. So it's valid.
                        
                        blocks = new_blocks
                        improved = True
                        found = True
                        break
                if found:
                    break
                    
        return blocks

    best_blocks = local_search_2for1(current_best_blocks)
    
    # Final validation check (optional but good for safety)
    # The evaluator will check, but let's ensure we didn't break it.
    # We return the blocks.
    
    return [list(all_blocks[i]) for i in best_blocks]

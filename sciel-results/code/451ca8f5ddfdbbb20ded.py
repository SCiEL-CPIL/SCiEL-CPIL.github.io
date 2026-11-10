import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    bit_vals = [1 << i for i in range(num_tsubs)]
    full_mask = (1 << num_tsubs) - 1
    
    # Precompute coverage mask for each block
    block_masks = [0] * num_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            mask |= bit_vals[idx]
        block_masks[b_idx] = mask

    # Helper: greedy set cover
    def greedy_cover():
        covered = 0
        used = []
        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            # Random tie-breaking to diversify
            candidates = list(range(num_blocks))
            random.shuffle(candidates)
            for b_idx in candidates:
                new_bits = block_masks[b_idx] & ~covered
                gain = new_bits.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
            if best_gain <= 0:
                break
            used.append(best_idx)
            covered |= block_masks[best_idx]
        return used

    # Helper: remove redundant blocks
    def prune(blocks):
        if not blocks:
            return blocks
        active = list(blocks)
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            order = list(range(len(active)))
            random.shuffle(order)
            for i in order:
                if i >= len(active):
                    continue
                # Check if active[i] is redundant
                union_others = 0
                mask_i = block_masks[active[i]]
                for j in range(len(active)):
                    if j != i:
                        union_others |= block_masks[active[j]]
                if (mask_i & ~union_others) == 0:
                    del active[i]
                    changed = True
                    break
        return active

    # Helper: local search - try to replace 2 blocks with 1
    def local_search(blocks):
        if len(blocks) < 2:
            return blocks
        active = list(blocks)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Shuffle pairs to try different combinations
            pairs = list(itertools.combinations(range(len(active)), 2))
            random.shuffle(pairs)
            for i, j in pairs:
                if i >= len(active) or j >= len(active) or i == j:
                    continue
                # Union of the two blocks
                union_mask = block_masks[active[i]] | block_masks[active[j]]
                # Find a single block that covers this union
                # We need a block b such that union_mask is a subset of block_masks[b]
                # i.e., union_mask & ~block_masks[b] == 0
                # To speed up, we can check if popcount(union_mask) <= popcount(block_masks[b])
                # Actually, just checking subset is fine, but we can filter by size first.
                union_size = union_mask.bit_count()
                
                # Try to find a replacement
                # We can iterate through all blocks, but that's slow.
                # Optimization: Precompute blocks by size? No, size is fixed k.
                # But we can check if a block exists that contains the union.
                # Since k is fixed, and union size might be large, most blocks won't work.
                
                # Faster check:
                # The union of two k-sets is at most 2k elements.
                # A single k-set can only cover it if the union size is <= k.
                # If union_size > k, it's impossible.
                if union_size > k:
                    continue
                
                # Check if any block covers this specific union
                # We can just iterate all blocks? 34610 checks.
                # Maybe we can be smarter.
                # For now, iterate.
                found = False
                for b_idx in range(num_blocks):
                    if (union_mask & ~block_masks[b_idx]) == 0:
                        # Found a replacement
                        # Remove i and j, add b_idx
                        # Need to be careful with indices if we delete
                        # Let's create a new list
                        new_active = []
                        for idx in active:
                            if idx != active[i] and idx != active[j]:
                                new_active.append(idx)
                        new_active.append(b_idx)
                        
                        # Check if b_idx is already in new_active?
                        # If b_idx was one of the removed ones, we re-added it.
                        # If b_idx was another active block, we have a duplicate.
                        # Let's remove duplicates
                        # Actually, if b_idx was in active and not i or j, it's still there.
                        # So we might have duplicate b_idx.
                        # Set of indices is better?
                        # Let's just use a set for indices to ensure uniqueness
                        # But order matters for reproducibility? Not really.
                        
                        # Let's refine:
                        # Remove active[i] and active[j]
                        # If b_idx is in the remaining set, do nothing (it's already there)
                        # Else add b_idx
                        
                        remaining = [x for x in active if x != active[i] and x != active[j]]
                        if b_idx not in remaining:
                            remaining.append(b_idx)
                        
                        # Validate that remaining covers everything?
                        # The replacement b_idx covers union_mask.
                        # The other blocks cover the rest.
                        # Since i and j were covering union_mask, and b_idx covers it,
                        # and others cover the rest, the total coverage is the same.
                        # So it is valid.
                        
                        active = remaining
                        improved = True
                        found = True
                        break
                if found:
                    break # Restart pair search with new active list
        return active

    # Main Loop: Multiple restarts
    best_solution = None
    best_size = float('inf')
    
    # Run a few greedy restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Greedy
        current = greedy_cover()
        
        # Prune
        current = prune(current)
        
        # Local Search (2-for-1)
        current = local_search(current)
        
        # Prune again after local search
        current = prune(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_solution = current
            # If we hit the theoretical lower bound or close to it, stop?
            # Lower bound is 69.
            if best_size <= 70:
                break

    # Final Prune on the best solution found
    if best_solution:
        best_solution = prune(best_solution)
    
    # If no solution found (should not happen), return empty or fallback
    if not best_solution:
        # Fallback: simple greedy
        best_solution = greedy_cover()
        best_solution = prune(best_solution)

    return [list(all_blocks[i]) for i in best_solution]

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    # Verify
    all_tsubs = set(itertools.combinations(range(24), 3))
    covered = set()
    for b in blocks:
        for t in itertools.combinations(b, 3):
            covered.add(t)
    if all_tsubs == covered:
        print("Valid Cover")
    else:
        print(f"Invalid! Missing {len(all_tsubs - covered)} subsets")

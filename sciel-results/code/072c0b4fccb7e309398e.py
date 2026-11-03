import itertools
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Coverage count array: how many blocks cover each t-subset
    coverage = np.zeros(n_t_subsets, dtype=np.int32)
    
    # List of blocks, each block is a tuple of points
    blocks = []
    
    # Helper: Update coverage for a block
    def update_coverage(block, delta):
        for sub in itertools.combinations(block, t):
            coverage[t_subset_index[sub]] += delta

    # Greedy Initialization
    # We want to cover all t-subsets.
    # Start with an empty set of blocks.
    # While there are uncovered t-subsets, add a block.
    
    # To speed up initialization, we can use a heuristic to pick good blocks.
    # A block is "good" if it covers many uncovered t-subsets.
    
    # Initial random blocks to break symmetry? Or just greedy from scratch.
    # Greedy from scratch is usually decent.
    
    uncovered_mask = np.ones(n_t_subsets, dtype=bool)
    
    while np.any(uncovered_mask):
        if time.time() > deadline:
            break
        
        # Find the block that covers the most uncovered t-subsets
        # This is expensive if done naively. 
        # Heuristic: Pick a random uncovered t-subset, then build a block around it.
        # Or: Try a few random blocks and pick the best one.
        
        # Let's try a "Greedy with Randomization" approach.
        # Pick a random uncovered t-subset.
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered subset to anchor the block
        anchor_idx = random.choice(uncovered_indices)
        anchor_set = all_t_subsets[anchor_idx]
        
        # Build a block containing anchor_set
        # Start with anchor_set, then add k-t points that maximize coverage of other uncovered subsets.
        current_block = list(anchor_set)
        
        while len(current_block) < k:
            best_point = -1
            best_gain = -1
            for p in range(v):
                if p in current_block:
                    continue
                # Calculate gain: how many new uncovered t-subsets would be covered if we add p?
                # A t-subset is covered if it is a subset of current_block + {p}.
                # The new t-subsets covered are those that contain p and t-1 points from current_block.
                # Let S be the set of t-subsets containing p and t-1 points from current_block.
                # We want to maximize |S ∩ uncovered|.
                
                gain = 0
                # Iterate over combinations of t-1 points from current_block
                for comb in itertools.combinations(current_block, t-1):
                    sub = tuple(sorted(comb + (p,)))
                    idx = t_subset_index[sub]
                    if uncovered_mask[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1:
                break
            current_block.append(best_point)
        
        # If we couldn't fill the block, skip
        if len(current_block) < k:
            continue
            
        new_block = tuple(sorted(current_block))
        
        # Check if this block is already in blocks
        if new_block in blocks:
            continue
            
        # Add the block
        blocks.append(new_block)
        update_coverage(new_block, 1)
        
        # Update uncovered mask
        for sub in itertools.combinations(new_block, t):
            idx = t_subset_index[sub]
            if coverage[idx] > 0:
                uncovered_mask[idx] = False

    # Local Search / Simulated Annealing
    # Objective: Minimize len(blocks)
    # Move: 
    # 1. Delete a block if it doesn't cover any unique t-subsets.
    # 2. Swap a point in a block with a point outside, if it maintains coverage and allows deletion of another block?
    #    Simpler: Just try to delete blocks that are redundant.
    #    Then try to "merge" or "replace" blocks to reduce count.
    
    # A common strategy for covering designs is to start with a valid cover and remove redundant blocks.
    # A block is redundant if removing it leaves all t-subsets covered.
    
    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    # We can also try to improve by swapping points to create more redundancy, then deleting.
    
    # Let's implement a simple hill-climbing / local search loop.
    # We will randomly pick a block and try to modify it.
    # Or, simpler: repeatedly scan for redundant blocks and remove them.
    
    # First pass: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            # Check if b is redundant
            # Temporarily remove b
            update_coverage(b, -1)
            if not np.any(coverage == 0):
                # b is redundant, remove it
                blocks.pop(i)
                changed = True
            else:
                # Restore b
                update_coverage(b, 1)
                i += 1 # Skip to next block if we didn't remove
            if not changed:
                # If no block was removed in this inner loop iteration, continue
                pass
        
    # Now try to improve by swapping points.
    # If we swap a point in block B1 with a point in block B2, or just swap within B1,
    # we might create redundancy in other blocks.
    
    # A more effective local search:
    # 1. Pick a random block B.
    # 2. Pick a random point x in B and a random point y not in B.
    # 3. Create B' = (B \ {x}) U {y}.
    # 4. Check if replacing B with B' maintains coverage.
    # 5. If yes, check if this allows removing another block.
    # 6. If we can remove a block, accept the move.
    
    # Let's try a simpler "Greedy Deletion" with random restarts for the local search part.
    
    # Actually, the greedy initialization above is quite strong.
    # Let's just do a few passes of redundant block removal.
    
    # Re-run redundant removal to ensure stability
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            update_coverage(b, -1)
            if not np.any(coverage == 0):
                blocks.pop(i)
                changed = True
            else:
                update_coverage(b, 1)
    
    # If we have time, try to improve by random swaps that maintain coverage
    # and then re-run deletion.
    
    while time.time() < deadline:
        if not blocks:
            break
            
        # Pick a random block
        idx = random.randint(0, len(blocks) - 1)
        b = blocks[idx]
        
        # Pick a random point to swap out
        x = random.choice(b)
        
        # Pick a random point to swap in
        candidates = [p for p in range(v) if p not in b]
        if not candidates:
            continue
        y = random.choice(candidates)
        
        new_b = tuple(sorted((set(b) - {x}) | {y}))
        
        # Check if new_b is valid (distinct points)
        if len(new_b) != k:
            continue
            
        # Check if replacing b with new_b maintains coverage
        # We need to check if all t-subsets in b are covered by other blocks OR by new_b.
        # And all t-subsets in new_b are covered by other blocks OR by b (which is being replaced).
        # Actually, we just need the final set of blocks to be a valid cover.
        
        # Temporarily remove b
        update_coverage(b, -1)
        
        # Add new_b
        # Check if new_b covers any currently uncovered t-subsets? No, we just need the union to cover all.
        # We need to ensure that for any t-subset s in b, if it was only covered by b, it must be covered by new_b or other blocks.
        # Since we removed b, any t-subset s in b that is not in new_b and not in other blocks will be uncovered.
        
        # Let's check coverage after adding new_b.
        update_coverage(new_b, 1)
        
        if not np.any(coverage == 0):
            # The swap is valid.
            blocks[idx] = new_b
            # Now try to remove redundant blocks
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    bi = blocks[i]
                    update_coverage(bi, -1)
                    if not np.any(coverage == 0):
                        blocks.pop(i)
                        changed = True
                    else:
                        update_coverage(bi, 1)
        else:
            # Swap is invalid, revert
            update_coverage(new_b, -1)
            update_coverage(b, 1)

    # Final cleanup
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            update_coverage(b, -1)
            if not np.any(coverage == 0):
                blocks.pop(i)
                changed = True
            else:
                update_coverage(b, 1)

    # Ensure we return a valid cover. If the loop broke early, we might have uncovered subsets.
    # The greedy initialization ensures validity. The local search only accepts valid moves.
    # So the result should be valid.
    
    # Convert tuples to lists as required by the interface
    return [list(b) for b in blocks]

# EVOLVE-BLOCK-END

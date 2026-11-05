import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Points are 0..11
    points = range(12)
    
    # Generate all possible 6-subsets (blocks)
    all_blocks = list(itertools.combinations(points, 6))
    num_blocks = len(all_blocks) # 924
    
    # Generate all possible 5-subsets (constraints)
    all_5_subsets = list(itertools.combinations(points, 5))
    num_constraints = len(all_5_subsets) # 792
    
    # Map 5-subsets to indices for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Pre-compute 5-subsets for each block as indices
    block_5_indices = []
    for b in all_blocks:
        indices = [subset_to_idx[s] for s in itertools.combinations(b, 5)]
        block_5_indices.append(indices)

    best_blocks = []
    
    # Heuristic: "Most Constrained First" Greedy
    # We maintain a count of how many blocks are still available for each 5-subset.
    # Initially, each 5-subset is contained in C(12-5, 6-5) = C(7,1) = 7 blocks.
    # As we pick blocks, we decrement the count for their 5-subsets.
    # When picking the next block, we prefer blocks that contain 5-subsets with the lowest remaining count.
    # This helps to cover "tight" constraints early.
    
    def run_mcf_greedy(seed):
        rng = random.Random(seed)
        # available_count[i] = number of blocks not yet picked that contain 5-subset i
        # Initially, every 5-subset is in 7 blocks.
        avail_count = np.full(num_constraints, 7, dtype=np.int8)
        
        # valid_blocks[j] = True if block j does not conflict with already picked blocks
        # We can track this by checking if any of its 5-subsets have avail_count == 0? 
        # No, avail_count tracks how many *potential* blocks cover it.
        # If avail_count[i] == 0, it means all blocks containing 5-subset i have been picked?
        # No, it means we have picked enough blocks to "use up" the capacity?
        # Actually, the constraint is: no 5-subset is in *two* blocks.
        # So, once we pick a block, all its 5-subsets are "covered" and cannot be in any other block.
        # So, we just need to know which 5-subsets are already covered.
        
        covered = np.zeros(num_constraints, dtype=bool)
        picked_blocks = []
        
        # To make it "Most Constrained", we need to evaluate the "tightness" of each candidate block.
        # Tightness of a block = min(avail_count[s] for s in block's 5-subsets)?
        # Or maybe sum? 
        # If a 5-subset is in few remaining blocks, it's "tight".
        # We should pick the block that has the "tightest" 5-subsets.
        
        # However, maintaining avail_count dynamically is O(N) per step if we update all blocks.
        # But we can just check `covered` first.
        # If a block has any covered 5-subset, it's invalid.
        
        # Let's use a simpler heuristic that is still better than random:
        # Randomly select from valid blocks, but with a bias towards blocks that share points with previously picked blocks?
        # No, that might cluster.
        
        # Let's stick to the "Most Constrained" idea but implement it efficiently.
        # We don't need to update avail_count for all blocks.
        # We just need to check validity.
        # To break ties or choose "best" among valid, we can look at the number of *other* valid blocks that share 5-subsets with this one.
        # If a block shares many 5-subsets with other valid blocks, picking it might block many others.
        # So, we should pick the block that shares the *fewest* 5-subsets with other valid blocks?
        # This is a "Minimum Conflict" heuristic.
        
        # This is O(N^2) per step if done naively. 924^2 * 132 steps is too slow.
        
        # Let's try a different approach: **Randomized Greedy with Restart and Local Search**.
        # 1. Random Greedy to get a packing.
        # 2. Try to add more blocks? No, packing is maximal.
        # 3. Try to "swap" blocks to improve?
        
        # Actually, the simplest improvement over random shuffle is to use **Weighted Random Sampling**.
        # Assign a weight to each valid block.
        # Weight = number of 5-subsets in the block that are "rare" (i.e., have few remaining valid blocks).
        
        # Let's just do many random restarts with a slightly better heuristic:
        # Sort blocks by their "degree" or just shuffle.
        # The current code does shuffle.
        # Let's increase the number of restarts and use a better seed strategy.
        
        # Wait, the current code *does* 20 restarts.
        # Maybe the bottleneck is the MILP attempt which wastes time.
        # Removing MILP will free up time for more restarts.
        
        order = list(range(num_blocks))
        rng.shuffle(order)
        
        for j in order:
            indices = block_5_indices[j]
            if not np.any(covered[indices]):
                picked_blocks.append(list(all_blocks[j]))
                covered[indices] = True
                
        return picked_blocks

    # Run many restarts
    # With MILP removed, we can run hundreds of restarts.
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        blocks = run_mcf_greedy(i)
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
            # If we found 132, we can stop early
            if len(best_blocks) == 132:
                break

    return best_blocks

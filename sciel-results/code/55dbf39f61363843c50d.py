import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def get_elapsed():
        return time.time() - start_time

    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs covered by block i
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Inverse mapping: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # GRASP Construction: Randomized Greedy
    def grasp_construct(alpha=0.5):
        """
        Constructs a solution by greedily selecting blocks.
        At each step, selects a block from the candidate list of blocks
        with the best 'gain' (number of newly covered t-subsets).
        alpha controls the randomness: 
        alpha=0 -> deterministic greedy (best gain)
        alpha=1 -> random from top candidates (or all?)
        Standard GRASP: select from top alpha * max_gain fraction? 
        Or select randomly from blocks with gain > (1-alpha)*max_gain?
        Let's use: select randomly from blocks with gain >= (1 - alpha) * max_gain.
        """
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 5.0:
                return selected_blocks
            
            # Find best gain
            max_gain = 0
            # We need to find the block that covers the most uncovered t-subsets.
            # Efficient way: pick a random uncovered t-subset, look at its blocks, calculate gain.
            # This is the standard "Greedy Set Cover" heuristic.
            
            # Pick a random uncovered t-subset to anchor the search
            # This avoids scanning all blocks every time.
            s_idx = -1
            # Find an uncovered t-subset
            # Optimization: maintain a list of uncovered indices? 
            # For now, random sampling is fast enough if coverage is high.
            # If coverage is low, we might scan.
            
            if uncovered_count < 100:
                uncov_list = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_list:
                    break
                s_idx = random.choice(uncov_list)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = tsub_blocks[s_idx]
            
            # Calculate gains for all candidates covering s_idx
            gains = []
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                gains.append((gain, b_idx))
                
            if not gains:
                break
                
            max_gain_val = max(g[0] for g in gains)
            
            # Determine threshold based on alpha
            # If alpha is small, we are more greedy.
            # Threshold = (1 - alpha) * max_gain_val
            threshold = (1 - alpha) * max_gain_val
            
            # Filter candidates
            valid_candidates = [b_idx for g, b_idx in gains if g >= threshold]
            
            if not valid_candidates:
                valid_candidates = [b_idx for g, b_idx in gains if g == max_gain_val]
                
            # Select randomly from valid candidates
            b_idx = random.choice(valid_candidates)
            
            selected_blocks.append(b_idx)
            
            # Update coverage
            for j in block_tsubs[b_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    # Local Search: Aggressive Reduction
    # Moves: Delete, 1-Swap, 2-Swap
    def local_search(block_indices):
        current_blocks = set(block_indices)
        
        # Precompute cover counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_tsubs[b_idx]] += 1
            
        uncovered_count = np.sum(cover_count == 0)
        
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 2.0:
                break
                
            # 1. Try Deletions
            # Iterate through blocks. If removing a block doesn't increase uncovered_count, remove it.
            # To be efficient, we can check if any t-subset in the block has cover_count == 1.
            # If yes, it's essential. If no, it's removable.
            
            to_remove = []
            for b_idx in list(current_blocks):
                # Check if essential
                essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                for b_idx in to_remove:
                    current_blocks.remove(b_idx)
                    cover_count[block_tsubs[b_idx]] -= 1
                changed = True
                continue # Restart loop after deletions
            
            # 2. Try 1-Swaps
            # Remove one block, add another to reduce size or maintain size while improving "slack"?
            # Since we just deleted all non-essential blocks, the current set is "minimal" in the sense 
            # that no single block can be removed. 
            # A 1-swap (remove A, add B) where B covers all t-subsets that A covered (and maybe more) 
            # allows us to potentially remove other blocks later?
            # Actually, if we swap A for B, and B covers the same set of t-subsets as A (or a superset),
            # the coverage doesn't decrease. But we want to minimize block count.
            # A swap that maintains coverage doesn't reduce count immediately.
            # However, it might enable future deletions.
            
            # Better swap strategy: 
            # Find a block A. Find a block B that covers all t-subsets that A covers (i.e., union of B's tsubs is superset of A's tsubs? No, just that for every tsub in A, cover_count[tsub] >= 2 OR tsub is in B).
            # If we replace A with B, and B covers all "critical" tsubs of A, we can remove A.
            # Then we have the same number of blocks. But maybe B allows removing C?
            
            # Let's try a specific 1-swap: 
            # Pick a random block A.
            # Find a block B such that B covers all t-subsets in A that are currently covered ONLY by A (i.e., cover_count == 1).
            # If such B exists, and B is not in current_blocks, we can swap A -> B.
            # This keeps coverage valid. It might change the "structure" to allow deletions.
            
            swap_found = False
            # Limit swaps to avoid infinite loops or excessive time
            max_swap_attempts = 10
            for _ in range(max_swap_attempts):
                if not current_blocks:
                    break
                A = random.choice(list(current_blocks))
                
                # Find t-subsets in A that are critical (cover_count == 1)
                critical_tsubs = []
                for j in block_tsubs[A]:
                    if cover_count[j] == 1:
                        critical_tsubs.append(j)
                    
                if not critical_tsubs:
                    # A is not essential? But we just did deletion pass. 
                    # This shouldn't happen unless cover_count changed.
                    continue
                
                # Find a block B that covers all critical_tsubs
                # B must cover the intersection of A's critical tsubs.
                # Candidate B's are blocks that cover the first critical tsub.
                candidates_for_B = tsub_blocks[critical_tsubs[0]]
                
                found_B = -1
                for B in candidates_for_B:
                    if B in current_blocks:
                        continue
                    # Check if B covers all critical_tsubs
                    valid = True
                    # Optimization: check if B covers the set of critical tsubs
                    # We can precompute a bitmask or just check membership
                    # For small sets, loop is fine
                    b_tset = set(block_tsubs[B])
                    for ct in critical_tsubs[1:]:
                        if ct not in b_tset:
                            valid = False
                            break
                    if valid:
                        found_B = B
                        break
                
                if found_B != -1:
                    # Perform swap
                    current_blocks.remove(A)
                    cover_count[block_tsubs[A]] -= 1
                    
                    current_blocks.add(found_B)
                    cover_count[block_tsubs[found_B]] += 1
                    
                    changed = True
                    swap_found = True
                    break # Restart loop to check for new deletions
            
            if not swap_found:
                # Try 2-Swaps? 
                # Remove A, B. Add C, D.
                # This is expensive. 
                # Given time constraints, maybe stick to Deletion + 1-Swap.
                # If no move found, we are in a local minimum.
                break
                
        return list(current_blocks)

    # Run GRASP with multiple restarts and different alpha values
    best_blocks = None
    best_len = float('inf')
    
    # Initial deterministic greedy (alpha=0)
    init_blocks = grasp_construct(alpha=0.0)
    optimized = local_search(init_blocks)
    best_len = len(optimized)
    best_blocks = optimized
    
    # Random restarts with varying alpha
    # Alpha range: 0.1 to 0.9
    num_restarts = 0
    while get_elapsed() < deadline - 10.0:
        alpha = random.uniform(0.1, 0.9)
        blocks = grasp_construct(alpha=alpha)
        if not blocks:
            continue
        optimized = local_search(blocks)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_blocks = optimized
            # Optional: if we get very good, break?
            if best_len <= 325:
                break
        num_restarts += 1

    # Convert indices to lists
    return [list(all_blocks[i]) for i in best_blocks]

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct(19, 8, 5)
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(res)}")

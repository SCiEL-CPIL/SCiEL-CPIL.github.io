import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    # Map block tuple to index for quick lookup? 
    # Actually, we just need to generate candidates.
    
    # For fast coverage checking, we represent a block by the set of t-subset indices it covers.
    # Since C(19,7) is ~50k and C(19,4) is ~3876, we can store these as bitsets or lists.
    # Lists of indices are faster to iterate for counting in Python if we use numpy arrays for counts.
    
    block_coverages = []
    for block in all_k_blocks:
        covers = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    best_solution = None
    best_size = float('inf')
    
    # Strategy: Multi-start Local Search
    # We try several different initial constructions.
    # 1. Pure Greedy (baseline)
    # 2. Base-Block Construction: Fix one block (e.g., {0,1,2,3,4,5,6}), then greedily cover remaining.
    #    Then perform local search.
    
    def greedy_cover(initial_blocks=None):
        """
        Greedily add blocks to cover all t-subsets.
        Returns list of block indices.
        """
        if initial_blocks is None:
            initial_blocks = []
        
        # Current coverage counts
        # Using a sparse representation or dense array? Dense is 3876 ints, very fast.
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        current_blocks = list(initial_blocks)
        
        # Update counts for initial blocks
        for b_idx in current_blocks:
            cover_counts[block_coverages[b_idx]] += 1
            
        uncovered = np.sum(cover_counts == 0)
        if uncovered == 0:
            return current_blocks
        
        while uncovered > 0:
            # Find block that covers max number of currently uncovered t-subsets
            # gain[b] = number of uncovered t-subsets in block b
            # We can compute this as: sum(1 for c in block_coverages[b] if cover_counts[c]==0)
            # To speed this up, we can maintain a "potential" array.
            # But simple iteration over 50k blocks * 3876 checks is 200M ops, too slow in Python loop.
            # We need vectorized ops.
            
            # active_mask: 1 if t-subset is uncovered, 0 otherwise
            # We want to maximize dot(product, active_mask) for each block? No, that's not right.
            # We want count of (block_coverage & active_mask).
            
            # Let's use a different approach for speed:
            # Keep a list of uncovered t-subsets.
            # For each candidate block, count how many of its t-subsets are in the uncovered set.
            
            # This is still slow if done naively.
            # Alternative: Randomized Greedy with a limited candidate pool?
            # Or: Use the property that if we fix a base block, the problem reduces.
            
            # Let's try a faster greedy step:
            # Sample a subset of blocks? No, we need optimal greedy for quality.
            # Let's use NumPy to compute gains for all blocks at once?
            # Gains matrix is (num_blocks, num_t_subsets). Too big (50k * 3876 ~ 200M ints).
            
            # Let's stick to a slightly smarter greedy:
            # Sort blocks by size of intersection with uncovered set.
            # We can approximate by picking a random uncovered t-subset, and looking at all blocks containing it.
            # This is the standard "randomized greedy" or "worst-first" heuristic variant.
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(cover_counts == 0)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[np.random.randint(len(uncovered_indices))]
            
            # Find all blocks that cover this target_idx
            # We can precompute: for each t-subset, which blocks cover it?
            # This is the transpose of the incidence matrix.
            # t_sub_to_blocks: list of lists.
            
            # Since we are inside a loop, we should precompute this outside.
            # But we are defining this function inside construct. Let's pass it in or define it outside.
            
            # For now, let's just do a simple greedy that is fast enough for one run,
            # and rely on Local Search to fix the rest.
            # Actually, the current program's greedy is the bottleneck.
            
            # Let's implement a "Greedy with Precomputed Reverse Index"
            # This requires precomputing t_sub_to_blocks.
            
            pass # Placeholder, will implement in main logic below
            
        return current_blocks

    # Precompute reverse index: t_subset_idx -> list of block indices covering it
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverages):
        for t_idx in cov:
            t_sub_to_blocks[t_idx].append(b_idx)
            
    # Now, a faster greedy:
    def fast_greedy(initial_blocks=None):
        if initial_blocks is None:
            initial_blocks = []
            
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        current_blocks = list(initial_blocks)
        
        for b_idx in current_blocks:
            cover_counts[block_coverages[b_idx]] += 1
            
        # Track uncovered
        # Instead of scanning all blocks, pick a random uncovered t-subset
        # and choose the best block among those covering it.
        
        # To be efficient, we maintain a set/list of uncovered t-subsets.
        # Since we update counts, we can just check counts.
        
        while True:
            # Find any uncovered t-subset
            # To avoid O(N) scan every time, we can keep a list and filter, but that's O(N) too.
            # Given N=3876, a scan is fast (~0.1ms).
            uncovered = np.where(cover_counts == 0)[0]
            if len(uncovered) == 0:
                break
                
            # Pick random uncovered
            target = uncovered[np.random.randint(len(uncovered))]
            
            # Candidates are blocks covering target
            candidates = t_sub_to_blocks[target]
            
            # Score each candidate: how many OTHER uncovered t-subsets does it cover?
            best_block = -1
            best_score = -1
            
            # Limit candidates to speed up? No, usually < 100 blocks cover a specific t-subset.
            # C(19-4, 7-4) = C(15,3) = 455 blocks cover a specific 4-set.
            # 455 candidates * 35 t-subsets per block = ~15k ops. Very fast.
            
            for b_idx in candidates:
                cov = block_coverages[b_idx]
                # Count how many of these are currently uncovered
                # score = np.sum(cover_counts[cov] == 0)
                # Vectorized:
                score = np.count_nonzero(cover_counts[cov] == 0)
                if score > best_score:
                    best_score = score
                    best_block = b_idx
                    if score == len(cov): # Perfect coverage of this block's t-subsets
                        break
                        
            if best_block == -1:
                break # Should not happen if problem is solvable
                
            current_blocks.append(best_block)
            cover_counts[block_coverages[best_block]] += 1
            
        return current_blocks

    def local_search(blocks):
        """
        Perform add/remove/swap local search to minimize block count.
        """
        if not blocks:
            return blocks
            
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            cover_counts[block_coverages[b_idx]] += 1
            
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            
            # 1. Deletion: Try to remove any block that is redundant
            # A block is redundant if all its t-subsets are covered by other blocks (count >= 2)
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                cov = block_coverages[b_idx]
                # If min count in cov is >= 2, it's redundant
                if np.min(cover_counts[cov]) >= 2:
                    cover_counts[cov] -= 1
                    blocks.pop(i)
                    changed = True
                    # Continue checking other blocks? Or restart? 
                    # Restarting is safer for finding all redundancies.
                    # But breaking and restarting the while loop is fine.
                    break
            
            if changed:
                continue
                
            # 2. Swap: Try to replace a block with another that covers more uncovered?
            # Actually, in a minimal cover, there are no uncovered.
            # We want to swap block A for block B if B covers all t-subsets of A 
            # that are *only* covered by A, and B allows removal of A?
            # No, standard swap: Remove A, add B. If cover is still valid, and size is same, 
            # we might be stuck. 
            # Better swap: Remove A and B, add C and D? (2-swap)
            # Or: Remove A, add B. If valid, and B "helps" elsewhere?
            # Since we just want to minimize count, a 1-for-1 swap only helps if it enables 
            # subsequent deletions.
            
            # Let's try a "Move" heuristic:
            # Pick a block A. Try to replace it with a block B such that 
            # (Current Cover - A + B) is still a cover.
            # If such a B exists, we might have freed up redundancy in B's neighbors?
            # Actually, if we just swap A->B, the size is constant. 
            # The benefit is if B covers t-subsets that were *only* covered by A, 
            # then A wasn't redundant. 
            # Wait, if A is not redundant, there is some t-subset s covered ONLY by A.
            # If we swap A with B, B MUST cover s.
            # So B must contain s.
            # If we do this, the structure changes.
            
            # Let's implement a simple 1-swap:
            # For each block A in current solution:
            #   Identify t-subsets covered ONLY by A (critical sets).
            #   If none, A is redundant (handled above).
            #   If some, any replacement B must cover all critical sets of A.
            #   Try to find a B (not in solution) that covers all critical sets of A.
            #   If found, swap A->B. Mark changed.
            
            for i, b_idx in enumerate(blocks):
                cov = block_coverages[b_idx]
                # Find critical t-subsets: those with count == 1
                critical_mask = (cover_counts[cov] == 1)
                critical_indices = cov[critical_mask]
                
                if len(critical_indices) == 0:
                    # Should have been deleted
                    continue
                    
                # We need a block B that contains all critical_indices
                # This is a constraint. 
                # If len(critical_indices) == 1, B must contain that t-subset.
                # If len > 1, B must contain the union of these points? No, B must contain each t-subset.
                # Since B is a k-set, it must contain the union of the points in critical_indices?
                # Yes, if B contains t-subset {1,2,3,4} and {1,2,3,5}, B must contain {1,2,3,4,5}.
                # So B must be a superset of the union of points in critical_indices.
                
                # Compute union of points in critical_indices
                # critical_indices are indices into all_t_subsets.
                # We need the actual points.
                
                # This is getting complex. Let's try a simpler "Random Restart" strategy 
                # which is often more effective for these small instances than complex local moves.
                
                # Actually, let's just do a few iterations of "Greedy Repair" if we get stuck?
                # No, we are already in a valid cover.
                
                # Let's try a different local search: 
                # "2-Opt" style: Remove two blocks, add one? No, size increases.
                # Remove one, add one (swap) is the main move.
                
                # To keep it simple and robust, let's just do Deletion and 
                # rely on multiple starts with different random seeds in the greedy phase.
                
                pass # Skip complex swap for now to ensure correctness and speed
                
        return blocks

    # Main Loop: Multiple Starts
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Start 0: Pure Greedy
        # Start 1+: Greedy with a fixed base block (randomly chosen)
        if i == 0:
            initial = []
        else:
            # Pick a random base block
            base_idx = np.random.randint(len(all_k_blocks))
            initial = [base_idx]
            
        # Run Fast Greedy
        sol = fast_greedy(initial)
        
        # Run Local Search (Deletion only for now, as Swap is complex)
        sol = local_search(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol.copy()
            
            # If we reached the lower bound or a very good number, we might stop early?
            # Lower bound is 131. Best known 151.
            # If we get < 150, that's a record.
            if best_size <= 145:
                 # Try to improve further with more time?
                 pass

    if best_solution is None:
        # Fallback: return empty or a single block if something went wrong (should not happen)
        return [list(all_k_blocks[0])]
        
    return [list(all_k_blocks[b_idx]) for b_idx in best_solution]

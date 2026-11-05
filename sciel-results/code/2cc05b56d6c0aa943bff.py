import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    
    # Precompute block coverages as lists of t-subset indices
    # C(19,7) = 50388 blocks. C(19,4) = 3876 t-subsets.
    # 50388 * 35 ints is manageable in memory and fast to access.
    block_coverages = []
    for block in all_k_blocks:
        covers = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Precompute reverse index: t_subset_idx -> list of block indices covering it
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverages):
        for t_idx in cov:
            t_sub_to_blocks[t_idx].append(b_idx)
            
    # Helper: Initialize a valid cover using a fast greedy algorithm
    def get_initial_cover():
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        current_blocks = []
        
        while True:
            uncovered = np.where(cover_counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered t-subset
            target = uncovered[np.random.randint(len(uncovered))]
            candidates = t_sub_to_blocks[target]
            
            # Score candidates by how many OTHER uncovered t-subsets they cover
            best_block = -1
            best_score = -1
            
            # Optimization: Only check a subset of candidates if there are too many?
            # For v=19, k=7, t=4, a t-subset is in C(15,3)=455 blocks.
            # 455 * 35 ops is ~16k ops. Very fast.
            for b_idx in candidates:
                if b_idx in current_blocks: # Avoid duplicates in initial phase? 
                    # Actually, greedy usually doesn't pick duplicates if we track counts, 
                    # but picking a block already in the list adds 0 to new coverage.
                    # We can skip blocks already in the list to save time.
                    # However, 'in' on a list is O(N). 
                    # Let's just compute score. If it's in list, score is lower (or same) 
                    # because it covers fewer *new* things? 
                    # No, score is count of uncovered. 
                    # If a block is already in, its t-subsets are already covered (count >= 1).
                    # So score will be 0 for those. 
                    # We just pick the max score.
                    pass
                
                cov = block_coverages[b_idx]
                score = np.count_nonzero(cover_counts[cov] == 0)
                if score > best_score:
                    best_score = score
                    best_block = b_idx
                    if score == len(cov):
                        break
                        
            if best_block == -1:
                # Should not happen
                break
                
            current_blocks.append(best_block)
            cover_counts[block_coverages[best_block]] += 1
            
        return current_blocks, cover_counts

    # Simulated Annealing
    def simulated_annealing(initial_blocks, initial_counts):
        current_blocks = list(initial_blocks)
        cover_counts = initial_counts.copy()
        current_size = len(current_blocks)
        
        best_blocks = current_blocks.copy()
        best_size = current_size
        
        # Temperature schedule
        T_start = 100.0
        T_end = 0.1
        cooling_rate = 0.999
        
        T = T_start
        
        # Precompute block sets for fast "in" check?
        # current_blocks is a list of indices. 
        # For swap, we need to remove one and add one.
        # We need to know if a candidate block is already in the solution.
        # Using a set is faster, but we need to keep the list for iteration.
        # Let's maintain a set in parallel.
        current_block_set = set(current_blocks)
        
        while time.time() < deadline - 5.0 and T > T_end:
            T *= cooling_rate
            
            # 1. Identify critical blocks
            # A block is critical if it is the ONLY cover for at least one t-subset.
            # We can find these by looking at t-subsets with count == 1.
            # For each such t-subset, the block covering it is critical.
            
            critical_blocks = set()
            # Iterate over all t-subsets? 3876 is small.
            # But we only need to check those with count == 1.
            # np.where(cover_counts == 1)
            single_covered = np.where(cover_counts == 1)[0]
            
            for t_idx in single_covered:
                # The block covering this t-subset is the only one.
                # We need to find which block in current_blocks covers it.
                # t_sub_to_blocks[t_idx] contains ALL blocks covering it.
                # We need the intersection with current_block_set.
                # Since count is 1, there is exactly one block in current_block_set.
                # We can just pick the first one from t_sub_to_blocks[t_idx] that is in current_block_set?
                # No, t_sub_to_blocks[t_idx] has 455 entries. 
                # It's faster to iterate current_blocks and check if t_idx is in their coverage?
                # No, that's 150 * 35.
                
                # Let's use the reverse index.
                # Find the block in t_sub_to_blocks[t_idx] that is in current_block_set.
                # This is O(455) per single-covered t-subset.
                # If there are 1000 single-covered t-subsets, that's 455k ops.
                # This is getting heavy per iteration.
                
                # Alternative: 
                # For each block in current_blocks, check if it has any t-subset with count 1.
                # If yes, it's critical.
                # This is O(N_blocks * 35). 150 * 35 = 5250 ops. Much faster.
                
                pass # We'll do it in the main loop below instead of inside this loop.
            
            # Re-evaluate critical blocks efficiently
            critical_blocks = set()
            for b_idx in current_blocks:
                cov = block_coverages[b_idx]
                # If any t-subset in this block has count 1, this block is critical
                if np.any(cover_counts[cov] == 1):
                    critical_blocks.add(b_idx)
            
            # 2. Generate Move
            # Option A: Remove a non-critical block. (Size decreases)
            # Option B: Swap a critical block with another block. (Size constant)
            
            non_critical_blocks = [b for b in current_blocks if b not in critical_blocks]
            
            if non_critical_blocks and random.random() < 0.3:
                # Try to remove a non-critical block
                # This is a "free" improvement if valid.
                # But in SA, we might want to explore.
                # Actually, removing a non-critical block is always an improvement (size -1).
                # We should always do this if possible? 
                # Yes, greedy deletion is good.
                # Let's do a few greedy deletions first.
                while non_critical_blocks:
                    b_idx = random.choice(non_critical_blocks)
                    # Remove it
                    cover_counts[block_coverages[b_idx]] -= 1
                    current_blocks.remove(b_idx)
                    current_block_set.remove(b_idx)
                    current_size -= 1
                    # Re-evaluate critical blocks
                    critical_blocks = set()
                    for b in current_blocks:
                        cov = block_coverages[b]
                        if np.any(cover_counts[cov] == 1):
                            critical_blocks.add(b)
                    non_critical_blocks = [b for b in current_blocks if b not in critical_blocks]
                    
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = current_blocks.copy()
                    
                # If we removed blocks, continue to next SA step (cooling)
                continue

            # If no non-critical blocks, or we chose swap:
            # Pick a random block to remove (must be critical to be interesting, 
            # but we can pick any and see if we can repair)
            # Actually, standard swap: Pick A, pick B, remove A, add B.
            # If A is critical, B MUST cover the critical t-subsets of A.
            
            # Let's pick a random block A to remove.
            A_idx = random.choice(current_blocks)
            A_cov = block_coverages[A_idx]
            
            # Find t-subsets in A that are ONLY covered by A (count == 1)
            critical_ts = A_cov[cover_counts[A_cov] == 1]
            
            if len(critical_ts) == 0:
                # A is not critical (should have been removed above, but maybe counts changed?)
                # Just remove it?
                # For safety, let's treat it as a swap with a random B.
                pass
            
            # We need to find a block B (not in current) that covers all critical_ts.
            # If critical_ts is empty, any B works.
            # If critical_ts is not empty, B must contain the union of points in critical_ts.
            # Let's find candidates for B.
            
            if len(critical_ts) > 0:
                # The union of points in critical_ts must be in B.
                # critical_ts are indices into all_t_subsets.
                # We need the actual points.
                # Precompute point sets for t-subsets?
                # t_subsets_points = [set(s) for s in all_t_subsets]
                # Union of points:
                union_points = set()
                for t_idx in critical_ts:
                    union_points.update(all_t_subsets[t_idx])
                
                # Candidates for B are blocks that contain union_points.
                # How to find them fast?
                # Iterate all blocks? 50k blocks.
                # Check if union_points is subset of block.
                # This is expensive if done for every SA step.
                
                # Optimization: 
                # If len(critical_ts) == 1, say {1,2,3,4}, then B must contain {1,2,3,4}.
                # t_sub_to_blocks[t_idx] gives all blocks containing {1,2,3,4}.
                # So candidates = t_sub_to_blocks[critical_ts[0]].
                
                # If len(critical_ts) > 1, say {1,2,3,4} and {1,2,3,5}, 
                # union is {1,2,3,4,5}.
                # B must contain {1,2,3,4,5}.
                # We can intersect the candidate lists of each critical t-subset.
                # candidates = intersection(t_sub_to_blocks[t] for t in critical_ts)
                
                # Let's implement intersection for small critical_ts.
                # Usually critical_ts is small (1-3).
                
                candidate_lists = [t_sub_to_blocks[t] for t in critical_ts]
                
                # Find intersection
                # Sort by length and intersect
                candidate_lists.sort(key=len)
                candidates_set = set(candidate_lists[0])
                for cl in candidate_lists[1:]:
                    candidates_set.intersection_update(cl)
                    
                candidates = list(candidates_set)
            else:
                # No critical t-subsets. Any block works.
                # To avoid trivial swaps, pick from a random subset of blocks?
                # Or just pick a random block not in current.
                # Let's pick from all blocks for diversity.
                candidates = list(range(num_blocks))
                
            # Remove current blocks from candidates
            candidates = [c for c in candidates if c not in current_block_set]
            
            if not candidates:
                # No valid swap partner found. 
                # This means we are stuck in a local minimum where no single block can replace A.
                # We can try to "break" the cover? No, we must maintain validity.
                # We can just skip this step.
                continue
                
            # Pick a random candidate B
            B_idx = random.choice(candidates)
            B_cov = block_coverages[B_idx]
            
            # Calculate Delta Energy
            # Energy is related to size. 
            # Swap: Size constant. 
            # But we want to minimize size. 
            # If we swap, size doesn't change. 
            # How does this help? 
            # It might enable future deletions. 
            # In SA, we accept based on "energy". 
            # Let's define Energy = - (number of redundant t-subsets) ? 
            # Or just use Size as the primary metric, and use SA to escape local minima 
            # where no deletion is possible.
            
            # If we swap A->B, the size is the same. 
            # We should accept with some probability to explore.
            # Let's use a simple acceptance: 
            # If the new configuration has more "flexibility" (more non-critical blocks), accept.
            # Or just accept with probability exp(1/T) ?
            
            # Let's calculate the number of non-critical blocks before and after.
            
            # Before: We have critical_blocks set.
            # After: 
            # Remove A, Add B.
            # Update counts.
            
            # Let's simulate the swap on a copy of counts? 
            # Or do it in place and revert if rejected.
            
            old_counts = cover_counts.copy()
            
            # Apply Swap
            cover_counts[A_cov] -= 1
            cover_counts[B_cov] += 1
            
            current_blocks.remove(A_idx)
            current_block_set.remove(A_idx)
            current_blocks.append(B_idx)
            current_block_set.add(B_idx)
            
            # Check if still valid (it should be, by construction of B covering critical_ts)
            # But what if B doesn't cover some other t-subset that was only covered by A?
            # No, B covers all critical_ts of A. 
            # Are there other t-subsets in A that were only covered by A? 
            # critical_ts = A_cov[cover_counts[A_cov] == 1]. 
            # These are ALL t-subsets in A with count 1. 
            # So B covers all of them. 
            # So the cover remains valid.
            
            # Calculate "Flexibility": Number of non-critical blocks.
            # Critical block: has any t-subset with count 1.
            new_critical_count = 0
            for b in current_blocks:
                cov = block_coverages[b]
                if np.any(cover_counts[cov] == 1):
                    new_critical_count += 1
            
            old_critical_count = len(critical_blocks)
            
            # We want to minimize critical_count (more redundancy is better for future deletions)
            delta_flex = old_critical_count - new_critical_count
            
            # Acceptance Probability
            # If delta_flex > 0 (more flexible), accept.
            # If delta_flex < 0 (less flexible), accept with prob exp(delta_flex / T)
            # Note: delta_flex is negative, so exp(negative/T) < 1.
            
            if delta_flex > 0:
                accept = True
            else:
                # delta_flex is <= 0
                # If T is high, accept more.
                accept = random.random() < np.exp(delta_flex / T)
                
            if accept:
                # Keep the swap
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = current_blocks.copy()
            else:
                # Revert
                cover_counts = old_counts
                current_blocks.remove(B_idx)
                current_block_set.remove(B_idx)
                current_blocks.append(A_idx)
                current_block_set.add(A_idx)
                
        return best_blocks

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run a few restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Get initial cover
        init_blocks, init_counts = get_initial_cover()
        
        # Run SA
        sa_blocks = simulated_annealing(init_blocks, init_counts)
        
        if len(sa_blocks) < best_size:
            best_size = len(sa_blocks)
            best_solution = sa_blocks.copy()
            
            # If we get very close to lower bound, we can stop?
            # Lower bound 131. Best known 151.
            # If we get 145, that's great.
            if best_size <= 140:
                break

    if best_solution is None:
        # Fallback
        return [list(all_k_blocks[0])]
        
    return [list(all_k_blocks[b_idx]) for b_idx in best_solution]

import itertools
import time
import random
import heapq
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    def greedy_cover(uncovered_mask, existing_solution=None):
        """
        Greedily add blocks to cover uncovered t-subsets.
        uncovered_mask: list or array where 1 means uncovered.
        existing_solution: list of block indices already selected.
        Returns the updated list of block indices.
        """
        if existing_solution is None:
            solution = []
        else:
            solution = list(existing_solution)
            
        # Calculate initial uncovered counts for each block
        # We only care about blocks that cover at least one uncovered t-subset
        # uncovered_count_per_block[b] = number of uncovered t-subsets in block b
        uncovered_count_per_block = [0] * n_blocks
        # We need to know which t-subsets are uncovered
        # Instead of a full mask, we can just iterate over uncovered t-subsets
        # But we need to update counts for all blocks covering a t-subset.
        
        # Let's maintain a list of currently uncovered t-subsets
        # Initially, if existing_solution is provided, we need to compute what is covered.
        # This is expensive. Let's assume we start from scratch or have a cover_count array.
        
        # For efficiency in ILS, we will maintain `cover_count` globally.
        # This function is a helper for the initial phase or re-greedy after perturbation.
        
        # Let's just implement a simple greedy that takes a set of uncovered t-subsets
        # and a set of available blocks (all blocks not in solution? or all blocks?).
        # Actually, standard greedy set cover:
        # 1. Identify uncovered t-subsets.
        # 2. For each block, count how many uncovered t-subsets it covers.
        # 3. Pick block with max count.
        # 4. Update uncovered set.
        
        # To make this fast, we use a heap.
        # But updating counts for all blocks when we pick one is O(n_blocks) in worst case if we do it naively.
        # With tsub_to_blocks, we can update only affected blocks.
        
        # Let's use a dictionary or array for uncovered status.
        # Since we are rewriting, let's do a specialized greedy for the initial step
        # and a local search loop.
        
        pass

    # --- Phase 1: Initial Greedy Construction ---
    # uncovered[s] = 1 if t-subset s is not covered
    uncovered = [1] * n_t_subs
    total_uncovered = n_t_subs
    solution_blocks = []
    solution_set = set()
    
    # uncovered_count_per_block[b] = number of t-subsets in block b that are currently uncovered
    uncovered_count_per_block = [len(block_to_tsubs[b]) for b in range(n_blocks)]
    
    # Use a heap for max-heap behavior (negate values)
    # ( -count, block_index )
    heap = [(-uncovered_count_per_block[b], b) for b in range(n_blocks)]
    heapq.heapify(heap)
    
    while total_uncovered > 0 and time.time() < deadline:
        # Pop until we find a valid block
        valid_block_idx = -1
        valid_count = 0
        while heap:
            neg_c, b_idx = heapq.heappop(heap)
            c = -neg_c
            # Check if this count is still valid
            if c == uncovered_count_per_block[b_idx] and c > 0:
                valid_block_idx = b_idx
                valid_count = c
                break
            # If stale, we don't push it back unless we updated it? 
            # Actually, we update counts by decrementing. 
            # The heap contains old states. We just discard stale entries.
            # But we need to push updated entries when counts change.
        
        if valid_block_idx == -1:
            # This shouldn't happen if total_uncovered > 0
            break
            
        solution_blocks.append(valid_block_idx)
        solution_set.add(valid_block_idx)
        
        # Update uncovered status and counts
        for s in block_to_tsubs[valid_block_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Update uncovered_count_per_block for all blocks covering s
                for b_idx in tsub_to_blocks[s]:
                    old_count = uncovered_count_per_block[b_idx]
                    new_count = old_count - 1
                    uncovered_count_per_block[b_idx] = new_count
                    # Push new state to heap
                    if new_count > 0:
                        heapq.heappush(heap, (-new_count, b_idx))

    # --- Phase 2: Local Search (ILS) ---
    # cover_count[s] = number of blocks in solution covering t-subset s
    cover_count = [0] * n_t_subs
    for b_idx in solution_blocks:
        for s in block_to_tsubs[b_idx]:
            cover_count[s] += 1
            
    best_solution = list(solution_blocks)
    best_len = len(solution_blocks)
    
    no_improve_count = 0
    max_no_improve = 5000  # Threshold for perturbation
    
    while time.time() < deadline - 5:
        improved = False
        
        # Pass 1: Try to remove redundant blocks
        # A block is redundant if all its t-subsets have cover_count > 1
        for i in range(len(solution_blocks)):
            b_idx = solution_blocks[i]
            redundant = True
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in block_to_tsubs[b_idx]:
                    cover_count[s] -= 1
                solution_blocks.pop(i)
                solution_set.remove(b_idx)
                improved = True
                no_improve_count = 0
                if len(solution_blocks) < best_len:
                    best_len = len(solution_blocks)
                    best_solution = list(solution_blocks)
                break # Restart loop as indices changed
                
        # Pass 2: If no removal, try to swap
        if not improved and time.time() < deadline - 5:
            # Find a block with critical subsets
            # To speed up, we can sample or iterate.
            # Let's iterate through blocks.
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                critical_subsets = []
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        critical_subsets.append(s)
                        
                if not critical_subsets:
                    # Should have been removed, but handle just in case
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    no_improve_count = 0
                    if len(solution_blocks) < best_len:
                        best_len = len(solution_blocks)
                        best_solution = list(solution_blocks)
                    break
                    
                # Find candidate block that covers all critical subsets
                # Intersection of tsub_to_blocks for critical subsets
                if critical_subsets:
                    # Start with smallest list to reduce intersection size
                    critical_subsets.sort(key=lambda x: len(tsub_to_blocks[x]))
                    candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                            
                    if candidate_set:
                        # Pick a candidate not in current solution
                        # Heuristic: pick one that covers most other subsets?
                        # Or just pick random?
                        # Let's pick the one that minimizes "damage" or maximizes "gain".
                        # Gain: covers subsets that are currently covered by only 1 block (other than this one)?
                        # Actually, we just need a valid swap.
                        # Let's pick a random candidate to explore space.
                        candidates = [c for c in candidate_set if c not in solution_set]
                        if candidates:
                            # Pick one with max coverage of non-critical subsets?
                            # Simple: pick first one.
                            best_candidate = candidates[0]
                            
                            # Perform swap
                            # Decrement cover_count for b_idx
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            # Increment cover_count for best_candidate
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            # Replace
                            solution_blocks[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            
                            improved = True
                            no_improve_count = 0
                            if len(solution_blocks) < best_len:
                                best_len = len(solution_blocks)
                                best_solution = list(solution_blocks)
                            break
                            
        if not improved:
            no_improve_count += 1
            # Perturbation
            if no_improve_count > max_no_improve:
                # Randomly remove a few blocks and re-greedy fill
                # Or just random swap
                if len(solution_blocks) > 10:
                    # Remove 2-5 random blocks
                    num_remove = random.randint(2, 5)
                    for _ in range(num_remove):
                        if not solution_blocks: break
                        idx = random.randint(0, len(solution_blocks)-1)
                        b_idx = solution_blocks[idx]
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_blocks.pop(idx)
                        solution_set.remove(b_idx)
                        
                    # Re-greedy fill
                    # Identify uncovered t-subsets
                    uncovered_list = [s for s in range(n_t_subs) if cover_count[s] == 0]
                    
                    if uncovered_list:
                        # Simple greedy on the uncovered set
                        # We need to efficiently find blocks covering these.
                        # This is a mini set cover.
                        # For speed, we just iterate and pick best.
                        # Since num_remove is small, uncovered_list is small.
                        
                        # Create a set of uncovered indices for fast lookup
                        uncovered_set = set(uncovered_list)
                        
                        # Calculate gain for each block
                        # gain[b] = number of uncovered subsets in block b
                        # We only care about blocks that cover at least one uncovered subset.
                        
                        # To avoid O(n_blocks) scan every time, we can just do it.
                        # n_blocks is ~77k. 77k * 11 (avg tsubs in block? C(7,4)=35) is large.
                        # Let's optimize:
                        # Only consider blocks that intersect with uncovered_set.
                        
                        # Actually, a simpler perturbation is random swap.
                        # Let's stick to random swap to avoid complex re-greedy.
                        # Revert the removals and do random swaps.
                        for _ in range(num_remove):
                             # Pick random block from best_solution? No, from current.
                             # Just pick random indices to swap.
                             if len(solution_blocks) >= 2:
                                 i = random.randint(0, len(solution_blocks)-1)
                                 b_idx = solution_blocks[i]
                                 # Find a random other block
                                 # This is getting complicated.
                                 
                # Simpler Perturbation: Random Swap
                if len(solution_blocks) >= 2:
                    i = random.randint(0, len(solution_blocks)-1)
                    b_idx = solution_blocks[i]
                    critical_subsets = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    if critical_subsets:
                        # Find any block covering them
                        # Use intersection
                        critical_subsets.sort(key=lambda x: len(tsub_to_blocks[x]))
                        candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                        for s in critical_subsets[1:]:
                            candidate_set.intersection_update(tsub_to_blocks[s])
                            if not candidate_set: break
                            
                        candidates = [c for c in candidate_set if c not in solution_set]
                        if candidates:
                            new_b = random.choice(candidates)
                            # Swap
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[new_b]:
                                cover_count[s] += 1
                            solution_blocks[i] = new_b
                            solution_set.remove(b_idx)
                            solution_set.add(new_b)
                            
                no_improve_count = 0

    # Return best solution
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
    return final_blocks

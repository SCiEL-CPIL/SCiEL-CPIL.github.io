import itertools
import random
import time
import sys

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the set of t-subset indices it covers
    block_t_indices = [set() for _ in range(num_k_subsets)]
    for b_idx, block in enumerate(all_k_subsets):
        for s in itertools.combinations(block, t):
            block_t_indices[b_idx].add(t_subset_to_idx[s])
            
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [set() for _ in range(num_t_subsets)]
    for b_idx in range(num_k_subsets):
        for t_idx in block_t_indices[b_idx]:
            t_subset_blocks[t_idx].add(b_idx)
            
    # 2. Phase 1: Greedy Independent Set (Core)
    # We want a set of blocks that don't share any t-subsets.
    # This maximizes unique coverage per block.
    
    pool = set(range(num_k_subsets))
    core_blocks = set()
    covered_by_core = set()
    
    # Pre-compute "potential" coverage for each block? 
    # Just iterate.
    
    # Optimization: Sort blocks by size of coverage? All same size C(k,t).
    # So we just pick any, then remove conflicts.
    # To get a *large* independent set, we should pick blocks that have many neighbors?
    # Or just randomize to escape local minima?
    # The current code picks the one covering the most *uncovered* items.
    # Since in Phase 1 everything is uncovered initially, it's just max degree?
    # No, it's max coverage of *uncovered*. Initially all are uncovered.
    # So it picks a block, removes its neighbors (conflicts).
    
    # Let's use a randomized greedy to allow multiple runs if needed, 
    # but for now, deterministic greedy is fine.
    
    # To make it faster, we can just pick a random block from the pool, 
    # add it to core, and remove all its neighbors from the pool.
    # This is O(N * E) roughly.
    
    # A better heuristic: Pick the block with the fewest neighbors (least constrained) 
    # to keep the pool large? Or most neighbors (most constrained) to prune fast?
    # For Independent Set, picking high degree nodes is often good for MWIS heuristics.
    
    while pool:
        # Pick a block. 
        # Heuristic: Pick the one that appears in the most t_subset_blocks lists? 
        # i.e., high degree in conflict graph.
        # Or just random.
        # Let's stick to the current logic: Max coverage of uncovered.
        # But we can speed it up.
        
        # If we just want a valid independent set, random is fast.
        # If we want a *large* one, we need a good heuristic.
        
        # Let's try: Pick block with max number of uncovered t-subsets.
        # Since initially all are uncovered, this is just C(k,t) for all.
        # So it's arbitrary. 
        # Once we pick one, we remove its t-subsets from "uncovered".
        # Then the next block's "value" is reduced by overlap with covered.
        # But in an Independent Set, overlap is 0!
        # So if we strictly enforce independence, the "value" of any block in the pool 
        # is always C(k,t) because it shares no t-subsets with the core.
        # Therefore, ANY block in the pool is equally good.
        # So we can just pick the first one or a random one.
        
        # Wait, is that true?
        # If B1 and B2 are in the pool, they don't share t-subsets with the CORE.
        # Do they share t-subsets with each other?
        # If they do, they are connected in the conflict graph.
        # If we pick B1, we must remove B2 from the pool.
        # So the choice of B1 affects which blocks remain in the pool.
        # To maximize the size of the independent set, we should pick blocks 
        # that have many neighbors (to prune the pool aggressively?) 
        # No, we want to keep the pool "rich".
        # Actually, for Max Independent Set, a common heuristic is "Min Degree" 
        # (pick the node with fewest connections to the rest of the graph) 
        # or "Max Degree" (pick the hub).
        
        # Let's just use a random block from the pool for speed and simplicity,
        # or the one with the smallest index.
        # To improve quality, we can run this greedy multiple times with different seeds 
        # and keep the best core.
        
        # Let's implement a fast greedy:
        # Pick random b from pool.
        # Add to core.
        # Remove b and all neighbors of b from pool.
        
        # To find neighbors of b:
        # Neighbors are blocks sharing any t-subset with b.
        # We have t_subset_blocks.
        # neighbors = union of t_subset_blocks[t_idx] for t_idx in block_t_indices[b]
        
        # This is fast.
        
        b_idx = random.choice(list(pool)) # O(N) to convert to list, O(1) to pick.
        # Better: keep pool as a list? No, set is good for removal.
        # Converting set to list every time is O(N). N=50k. 50k * 50k steps? No.
        # The pool shrinks.
        
        core_blocks.add(b_idx)
        
        # Find neighbors to remove
        to_remove = set()
        for t_idx in block_t_indices[b_idx]:
            to_remove.update(t_subset_blocks[t_idx])
            
        pool.difference_update(to_remove)
        
        # Update covered_by_core? 
        # Not needed for the greedy selection if we assume independence guarantees no overlap.
        # But we need to know what is covered for Phase 2.
        for t_idx in block_t_indices[b_idx]:
            covered_by_core.add(t_idx)

    # 3. Phase 2: Branch and Bound for Residual Set Cover
    
    residual_uncovered = set(range(num_t_subsets)) - covered_by_core
    
    if not residual_uncovered:
        result = [list(all_k_subsets[b]) for b in core_blocks]
        return result

    # Precompute for B&B:
    # For each block, which residual_uncovered does it cover?
    # We can filter blocks to those that cover at least one residual_uncovered.
    
    # To speed up B&B, we can sort blocks by coverage of residual_uncovered.
    
    # B&B State:
    # current_blocks: set of indices
    # uncovered: set of t_idx
    
    best_solution = None
    best_size = len(core_blocks) + 1000 # Upper bound
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s for final polish
    
    # Heuristic for B&B:
    # 1. Find the uncovered t-subset with the fewest covering blocks (in the entire universe, 
    #    restricted to those not already in core? No, any block can be added).
    #    Actually, we can add ANY block.
    # 2. Branch on each block that covers this t-subset.
    
    # Optimization: 
    # If a block covers a subset of another block's coverage, it's dominated.
    # We can pre-filter dominated blocks for the residual problem.
    
    # Let's implement a simple B&B.
    
    def solve_residual(uncovered, current_blocks, depth):
        nonlocal best_solution, best_size
        
        if time.time() - start_time > time_limit:
            return
        
        if not uncovered:
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_solution = set(current_blocks)
            return
            
        # Lower Bound
        # Max coverage of any single block for the current uncovered set
        # We can approximate this by the max number of uncovered items a block *could* cover.
        # Precomputed max is C(k,t) = 35.
        # But some blocks might cover fewer *uncovered* items.
        # For a tight bound, we need max_{b} |Cover(b) \cap uncovered|
        # This is expensive to compute every node.
        # Use a static upper bound: C(k,t) = 35.
        # LB = ceil(len(uncovered) / 35)
        
        lb = (len(uncovered) + 34) // 35
        
        if len(current_blocks) + lb >= best_size:
            return
            
        # Variable Selection: Most constrained uncovered item
        # Find t_idx in uncovered with min len(t_subset_blocks[t_idx])
        # Note: t_subset_blocks[t_idx] contains ALL blocks.
        # We should only consider blocks that are NOT in current_blocks? 
        # Yes, adding a block already in current_blocks does nothing.
        
        min_count = float('inf')
        target_t_idx = -1
        for t_idx in uncovered:
            # Count how many blocks cover t_idx and are NOT in current_blocks
            # This is O(N_blocks) per t_idx. Too slow.
            # Approximation: Use len(t_subset_blocks[t_idx]).
            # If a block is in current_blocks, it's already "used", but it still "covers" the item.
            # Wait, if a block is in current_blocks, it has already been selected.
            # We are looking for *additional* blocks.
            # The constraint is that the *new* blocks must cover the *uncovered* items.
            # The existing blocks in current_blocks do NOT cover items in `uncovered` 
            # (by definition of `uncovered` in the recursion? 
            # No, `uncovered` is the set of items not covered by `current_blocks`).
            # So any block in `current_blocks` covers NO items in `uncovered`.
            # So we can just use `len(t_subset_blocks[t_idx])` as a proxy for branching factor.
            
            if len(t_subset_blocks[t_idx]) < min_count:
                min_count = len(t_subset_blocks[t_idx])
                target_t_idx = t_idx
                
        # Branch on blocks covering target_t_idx
        # To improve, sort candidates by "gain" (how many other uncovered items they cover)
        
        candidates = []
        for b_idx in t_subset_blocks[target_t_idx]:
            if b_idx in current_blocks:
                continue
            # Calculate gain
            gain = 0
            for t_idx in block_t_indices[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            candidates.append((gain, b_idx))
            
        # Sort by gain descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for gain, b_idx in candidates:
            if gain == 0:
                continue
                
            new_uncovered = uncovered - block_t_indices[b_idx]
            new_current_blocks = current_blocks | {b_idx}
            
            solve_residual(new_uncovered, new_current_blocks, depth + 1)
            
            if time.time() - start_time > time_limit:
                return

    # Initialize B&B
    # We start with the core blocks.
    # uncovered is residual_uncovered.
    # current_blocks is core_blocks.
    
    # To help B&B, we can run a greedy first to get a good initial upper bound.
    # Greedy Set Cover on residual.
    
    greedy_blocks = set(core_blocks)
    greedy_uncovered = set(residual_uncovered)
    
    while greedy_uncovered:
        best_b = -1
        max_gain = 0
        # Sample some blocks to speed up?
        # For now, iterate all.
        for b_idx in range(num_k_subsets):
            if b_idx in greedy_blocks:
                continue
            gain = 0
            for t_idx in block_t_indices[b_idx]:
                if t_idx in greedy_uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_b = b_idx
                
        if best_b == -1:
            break
            
        greedy_blocks.add(best_b)
        for t_idx in block_t_indices[best_b]:
            if t_idx in greedy_uncovered:
                greedy_uncovered.remove(t_idx)
                
    best_size = len(greedy_blocks)
    best_solution = set(greedy_blocks)
    
    # Now run B&B to try to beat the greedy solution.
    solve_residual(residual_uncovered, core_blocks, 0)
    
    if best_solution is None:
        best_solution = greedy_blocks
        
    # 4. Local Search / Minimalization
    
    active_blocks = best_solution
    cover_count = [0] * num_t_subsets
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    # Swap search
    start_time_ls = time.time()
    time_limit_ls = 110.0 - (time.time() - start_time) # Remaining time
    if time_limit_ls < 0:
        time_limit_ls = 5.0
        
    while time.time() - start_time_ls < time_limit_ls:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Find t-subsets uniquely covered by B
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if not uncovered_by_B:
                continue

            # Sort by number of covering blocks
            uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
            
            candidates = None
            for t_idx in uncovered_by_B:
                blocks_list = t_subset_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_list)
                else:
                    candidates.intersection_update(blocks_list)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter candidates: must cover all uniquely covered by B, and not be B
            valid_candidates = []
            for c in candidates:
                if c != B and c not in active_blocks:
                    # Check if c covers all uncovered_by_B
                    # Since c is in the intersection of blocks covering each t_idx in uncovered_by_B,
                    # it covers all of them.
                    valid_candidates.append(c)
                    
            if not valid_candidates:
                continue
            
            B_new = random.choice(valid_candidates)
            
            active_blocks.remove(B)
            active_blocks.add(B_new)
            
            for t_idx in block_t_indices[B]:
                cover_count[t_idx] -= 1
            for t_idx in block_t_indices[B_new]:
                cover_count[t_idx] += 1
            
            # Minimize
            active_blocks, cover_count = minimize(active_blocks, cover_count)
            improved = True
            
        if not improved:
            break
            
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result

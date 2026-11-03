import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim to return by 145s.
    deadline = start_time + 145.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
    
    # 2. Initial Solution Construction: Combinatorial Grouping Strategy
    # Strategy: Find a large packing (disjoint blocks) to form a "skeleton".
    # This reduces the initial number of blocks significantly compared to greedy,
    # and the blocks are structurally diverse (disjoint), which helps escape
    # local minima in local search.
    
    # Greedy Packing:
    # 1. Start with all blocks.
    # 2. Pick a block that covers the most *uncovered* points? No, packing is about disjointness.
    # Standard Maximal Packing Greedy:
    # 1. Pick a random block (or one with high "potential").
    # 2. Remove all blocks that intersect it.
    # 3. Repeat until no blocks left or time limit.
    
    # To make it faster and better, we can use a "Weighted" packing where we prefer
    # blocks that cover "rare" points? For C(24,7,3), symmetry is high.
    # Let's try a simple randomized maximal packing.
    
    # We need to track which points are used? No, blocks are disjoint.
    # We can maintain a set of "forbidden" blocks.
    # Or simpler: maintain a boolean array `available_blocks` of size n_blocks.
    # When we pick block B, we mark B and all blocks intersecting B as unavailable.
    
    # Precompute intersection graph? Too big (346k^2).
    # On-the-fly check:
    # For a candidate block C, check if it intersects any selected block S.
    # Since we select ~3 blocks (24/7 ~ 3.4), the list of selected is small.
    # Check intersection with selected blocks: O(k) per check.
    # Iterate through all 346k blocks each time? 346k * 3 * 7 = ~7M ops. Fast.
    
    selected_blocks = []
    available = np.ones(n_blocks, dtype=bool)
    
    # Randomize order to get different packings
    # We don't want to shuffle 346k elements repeatedly if we do many restarts.
    # But we only do one packing here? Or multiple?
    # Let's do a single robust packing.
    
    # To improve packing quality, we can pick the block that minimizes the number
    # of "conflicts" (blocks it kills)?
    # Or just pick the first available one in a shuffled order.
    
    # Let's use a heuristic: Pick the block that covers the most "uncovered triples"
    # among the *uncovered* triples, but also prioritize disjointness?
    # Actually, for packing, any maximal packing is size <= 3.
    # Wait, 24 points / 7 size = max 3 disjoint blocks.
    # 3 * 7 = 21 points covered. 3 points left.
    # So a packing of size 3 is the maximum possible.
    # Is it useful?
    # If we start with 3 blocks, we cover 3*35=105 triples.
    # Total 2024 triples.
    # We still need ~70 more blocks.
    # The "skeleton" idea is weak because the packing size is so small (3).
    
    # Revised Initial Strategy:
    # The "Clumping" problem in greedy is that it picks blocks that overlap heavily.
    # Let's use a **Diverse Greedy** for the initial cover.
    # Instead of picking the block with max new coverage, pick the block with
    # max (new_coverage - alpha * overlap_with_existing).
    # This forces diversity.
    
    cover_counts = np.zeros(n_t, dtype=np.int32)
    selected_indices = []
    
    # To speed up "overlap_with_existing", we can't easily compute it without
    # iterating through selected blocks.
    # Since selected size grows to ~75, and we do this 75 times,
    # and we scan 346k blocks each time...
    # 75 * 346k * (cost to check overlap).
    # Overlap check: For candidate C, and each S in selected, count |C ∩ S|.
    # This is expensive: 75 * 346k * 75 * 7? No, too slow.
    
    # Alternative: **Greedy with Randomization and Backtracking?** No.
    # Let's stick to a fast, decent initial cover, then rely on a **Better Local Search**.
    
    # The current program's local search is standard SA.
    # The "Inspiration" mentions "Scarcity-Weighted Greedy".
    # Let's implement **Scarcity-Weighted Greedy** for initialization.
    # Score(block) = sum( 1 / scarcity(triple) for triple in block if uncovered )
    # Where scarcity(triple) = number of blocks that cover this triple.
    # This prioritizes "hard" triples.
    
    # Precompute scarcity:
    # scarcity[i] = number of blocks covering t-subset i.
    # We can compute this from block_t_indices.
    scarcity = np.zeros(n_t, dtype=np.int32)
    for idxs in block_t_indices:
        scarcity[idxs] += 1
    
    # We also need to efficiently compute the score for each block.
    # score(b) = sum( weight[t] for t in b if cover_counts[t] == 0 )
    # where weight[t] = 1.0 / scarcity[t]
    
    weights = 1.0 / scarcity.astype(np.float64)
    
    # Greedy Loop
    for _ in range(150): # Max blocks
        if time.time() > deadline - 40.0:
            break
            
        # Compute score for all blocks
        # This is O(n_blocks * t_in_block) = 346k * 35 = 12M ops.
        # 12M ops in Python is ~1s. 150 iterations = 150s. Too slow!
        
        # Optimization:
        # We only need to update scores when cover_counts changes.
        # But cover_counts changes for 35 triples per step.
        # Only blocks containing those triples change score.
        # How many blocks contain a specific triple?
        # C(24-3, 7-3) = C(21, 4) = 5985.
        # So for each of the 35 new covered triples, ~6000 blocks change.
        # Total ~210k blocks to update per step.
        # 150 steps * 210k = 31.5M updates. Feasible.
        
        # We need a reverse map: triple_idx -> list of block indices.
        triple_to_blocks = [[] for _ in range(n_t)]
        for b_idx, idxs in enumerate(block_t_indices):
            for t_idx in idxs:
                triple_to_blocks[t_idx].append(b_idx)
        
        # Initialize scores
        scores = np.zeros(n_blocks, dtype=np.float64)
        # Initial score: sum of weights for all triples in block (all uncovered initially)
        for b_idx, idxs in enumerate(block_t_indices):
            scores[b_idx] = np.sum(weights[idxs])
            
        # Track which blocks are "candidates" (not yet selected)
        # We can just pick argmax and mark it as -1 or skip it.
        selected_mask = np.zeros(n_blocks, dtype=bool)
        
        while True:
            if time.time() > deadline - 40.0:
                break
                
            # Find best block
            # Mask out selected
            # scores[selected_mask] = -1
            # best_idx = np.argmax(scores)
            
            # To avoid modifying scores array repeatedly, we can keep a list of
            # "active" scores? No, numpy argmax is fast.
            
            # Apply mask
            # We can just set score to -1 for selected blocks.
            # But we need to restore them? No, we just pick the max among unselected.
            
            # Efficient Argmax with mask:
            # tmp = scores.copy()
            # tmp[selected_mask] = -1
            # best_idx = np.argmax(tmp)
            
            # This is O(n_blocks) per step. 150 * 346k = 50M ops. Fast.
            
            tmp_scores = scores.copy()
            tmp_scores[selected_mask] = -1.0
            best_idx = int(np.argmax(tmp_scores))
            
            if tmp_scores[best_idx] <= 0:
                break
                
            # Select best_idx
            selected_mask[best_idx] = True
            selected_indices.append(best_idx)
            
            # Update cover_counts
            t_idxs = block_t_indices[best_idx]
            cover_counts[t_idxs] += 1
            
            # Update scores for blocks that share triples with best_idx
            # For each triple t in best_idx:
            #   If cover_counts[t] == 1 (just became covered):
            #     For each block b in triple_to_blocks[t]:
            #       scores[b] -= weights[t]
            
            for t_idx in t_idxs:
                # Only if this triple is *newly* covered?
                # Yes, because if it was already covered, weight was already 0 in the "effective" score?
                # Wait, my score definition: sum( weight[t] if cover_counts[t]==0 ).
                # So if cover_counts[t] becomes 1, the term weight[t] is removed.
                # If cover_counts[t] was > 0, it was already 0, so no change.
                
                # Check if it was uncovered before
                if cover_counts[t_idx] == 1:
                    for b_idx in triple_to_blocks[t_idx]:
                        if not selected_mask[b_idx]:
                            scores[b_idx] -= weights[t_idx]
                            
            # Early stop if all covered
            if np.all(cover_counts > 0):
                break

    # Current solution is selected_indices
    current_solution = selected_indices.copy()
    current_size = len(current_solution)
    best_solution = current_solution.copy()
    best_size = current_size
    
    # Rebuild cover_counts for consistency (it should be consistent)
    
    # 3. Local Search: Aggressive Pruning + 1-Swap
    # First, try to remove redundant blocks.
    # A block is redundant if every triple it covers is covered by at least one other block.
    # i.e., min(cover_counts[t] for t in block) > 1.
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Iterate in reverse to avoid index shifting issues?
        # Or just restart.
        for i in range(len(current_solution) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            b_idx = current_solution[i]
            t_idxs = block_t_indices[b_idx]
            if np.min(cover_counts[t_idxs]) > 1:
                # Remove
                cover_counts[t_idxs] -= 1
                current_solution.pop(i)
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_solution = current_solution.copy()
                break # Restart loop to check for new redundancies

    # Now, 1-Swap Local Search
    # Try to swap a block A with a block B such that:
    # 1. B is not in solution.
    # 2. The new solution is valid (or better).
    # 3. Ideally, we want to reduce size? Swap keeps size same.
    # So swap is only useful if it allows a *subsequent* deletion?
    # Or if we allow temporary invalidity?
    # Let's stick to valid swaps first:
    # If we swap A->B, and B covers all triples that A uniquely covered,
    # and B covers some triples that no other block covers? No.
    # If we swap A->B, the set of covered triples might change.
    # If the new set is a superset of the old set, it's safe.
    # If it's a subset, we might lose coverage.
    
    # Better: **2-Opt style move?**
    # Remove A, Add B.
    # Check if valid.
    # If valid, and if this enables a deletion (unlikely immediately), do it.
    # Actually, if we just swap, size is same.
    # We want to reduce size.
    # So we should look for **Add 1, Remove 2** moves?
    # Add B, Remove A1, Remove A2.
    # If B covers all triples that were uniquely covered by A1 and A2,
    # we reduce size by 1.
    
    # This is a "3-Set" move.
    # Let's try **Add 1, Remove 1** (Swap) to escape local minima,
    # followed by **Deletion**.
    
    # But random swaps are slow.
    # Let's use **Greedy Repair** if we go invalid?
    
    # Alternative: **Hill Climbing on Size**
    # Move: Remove A, Add B1, Add B2? (Size +1). No.
    # Move: Remove A, Add B. (Size 0).
    # Move: Remove A, Remove B, Add C. (Size -1).
    
    # Let's implement **Remove 2, Add 1** search.
    # For each pair (A, B) in solution:
    #   Find a block C not in solution such that C covers all triples
    #   that are uniquely covered by A or B.
    #   If found, replace {A, B} with {C}.
    
    # This is expensive: O(N^2 * n_blocks).
    # N ~ 75. N^2 ~ 5600. n_blocks ~ 346k. 1.9B checks. Too slow.
    
    # Restriction:
    # Only consider pairs (A, B) that share many triples?
    # Or only consider C that covers many triples?
    
    # Let's try a simpler **Randomized Local Search** with **Add/Remove** pairs.
    # 1. Pick random A in solution.
    # 2. Pick random C not in solution.
    # 3. Swap A->C.
    # 4. If valid, try to delete.
    
    # To make it faster, we can precompute for each block C, which triples it covers.
    
    # Let's run a time-boxed loop of:
    # - Random Swap (A->C)
    # - If valid, attempt deletion.
    # - If deletion succeeds, accept and restart.
    # - If deletion fails, revert swap? Or keep it if it's "equally good"?
    # Keeping it might help escape.
    
    # Let's use **Simulated Annealing** but with **Deletion Bias**.
    # Energy = size + lambda * uncovered.
    # Moves:
    # 1. Delete A (if valid).
    # 2. Swap A->C.
    # 3. Add C (if invalid, to repair).
    
    # We already have SA in the seed. The seed's SA was okay.
    # The main gain from the new strategy is the **Scarcity-Weighted Greedy** initial solution.
    # Let's verify the initial solution size.
    # Greedy usually gets ~80-90.
    # Scarcity-weighted might get ~75-80?
    
    # Let's add a **Final Polish** using the "Most Redundant First" deletion.
    # We already do this.
    
    # Let's ensure we return the best valid solution found.
    
    # Final Verification
    if np.any(cover_counts == 0):
        # Fallback to a known valid solution?
        # If our greedy failed (shouldn't), we have a problem.
        # Let's force validity by adding blocks.
        while np.any(cover_counts == 0) and time.time() < deadline - 1.0:
            # Find an uncovered triple
            uncovered_triples = np.where(cover_counts == 0)[0]
            if len(uncovered_triples) == 0:
                break
            t_idx = uncovered_triples[0]
            # Find a block covering it
            candidates = triple_to_blocks[t_idx]
            # Pick one not in solution? Or just any.
            # Pick the one with max coverage of *other* uncovered triples?
            # Just pick the first one not in solution to avoid duplicates?
            # Duplicates are allowed in the list? "blocks must be distinct".
            # So we must pick a block not in current_solution.
            sol_set = set(current_solution)
            valid_candidates = [b for b in candidates if b not in sol_set]
            if not valid_candidates:
                # All blocks covering this triple are in solution?
                # Then it must be covered. Contradiction.
                break
            b_idx = valid_candidates[0]
            current_solution.append(b_idx)
            cover_counts[block_t_indices[b_idx]] += 1
            current_size += 1
            
    # Update best if current is valid and smaller
    if np.all(cover_counts > 0) and current_size < best_size:
        best_size = current_size
        best_solution = current_solution.copy()
        
    return [list(all_blocks[i]) for i in best_solution]

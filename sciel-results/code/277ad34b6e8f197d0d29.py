import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    # Fallback for non-standard parameters (should not happen for this specific task, but safe)
    if v != 19 or k != 7 or t != 4:
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            for block in itertools.combinations(range(v), k):
                count = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_count == 0: break
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                if ts in uncovered:
                    uncovered.remove(ts)
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    # 1. Precomputation
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs) # 3876
    
    all_blocks = list(itertools.combinations(points, k)) # 50388
    num_blocks = len(all_blocks)
    
    block_masks = []
    block_tsubs = []
    
    # Precompute masks and tsub ids for all blocks
    for block_tuple in all_blocks:
        mask = 0
        tsub_ids = []
        for tsub in itertools.combinations(block_tuple, t):
            tid = tsub_to_id[tsub]
            mask |= (1 << tid)
            tsub_ids.append(tid)
        block_masks.append(mask)
        block_tsubs.append(tsub_ids)
        
    # Precompute inverse mapping: for each tsub, which blocks cover it?
    # This is useful for repair or specific swap checks, but masks are often faster for "covers all U" checks.
    # However, for finding *any* block covering a specific tsub, a list is good.
    # Let's stick to masks for the main logic as they are very fast for intersection checks.
    
    # 2. Greedy Initialization
    # We want to start with a valid covering.
    # Standard greedy: pick block with max gain.
    # Gain = number of uncovered tsubs covered by block.
    # Using bitsets: gain = popcount(block_mask & ~covered_mask)
    
    covered_mask = 0
    current_indices = []
    used_blocks = set()
    
    # To speed up greedy, we can maintain a list of candidate blocks.
    # But 50k blocks is small enough that a linear scan with bit_count is fast in Python 3.10+?
    # Let's check time. 50k * 150 iterations (if we need 150 blocks) = 7.5M bit_counts.
    # Python int bit_count is implemented in C, so it's very fast.
    
    # Optimization: Randomize tie-breaking to avoid getting stuck in a specific greedy trap?
    # Or just standard greedy. Standard greedy is usually good enough for a start.
    
    while time.time() < deadline:
        if covered_mask == (1 << num_tsubs) - 1:
            break
            
        best_block_idx = -1
        best_gain = 0
        
        for i in range(num_blocks):
            if i in used_blocks:
                continue
            gain = (block_masks[i] & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        current_indices.append(best_block_idx)
        used_blocks.add(best_block_idx)
        covered_mask |= block_masks[best_block_idx]
        
        if time.time() > deadline - 20: # Reserve time for local search
            break

    # 3. Local Search / Improvement
    # We have a valid covering. We want to minimize the number of blocks.
    # Strategy:
    # 1. Greedy Deletion: Remove any block that is redundant (all its tsubs covered by others).
    # 2. Swap: Try to swap a block A with a block B not in the set, such that the set remains valid.
    #    If we remove A, some tsubs might become uncovered (count 1 -> 0).
    #    B must cover all those tsubs.
    #    If such B exists, we replace A with B. This doesn't reduce count, but might allow further deletions?
    #    Actually, if we just swap, count is same.
    #    Wait, the goal is to reduce count.
    #    Deletion reduces count.
    #    If we are stuck (no deletions possible), we need to change the structure.
    #    A "2-move" could be: Remove A, Remove B, Add C, Add D? No, that's complex.
    #    
    #    Better approach for reduction:
    #    Try to remove a block A. If it fails (some tsubs uncovered), find a block B that covers those tsubs.
    #    If we add B, we have +1 block. Not helpful for reduction directly.
    #    
    #    However, sometimes removing A and adding B allows removing C later.
    #    So, we can accept swaps that don't change the count, as long as they enable further deletions.
    #    This is effectively a local search where the state is the set of blocks, and the objective is size.
    #    Moves:
    #    - Delete A (if valid) -> Accept (Size -1)
    #    - Swap A for B (if valid) -> Accept (Size 0) only if it leads to a different local minimum?
    #      Or just accept with some probability (Simulated Annealing) to escape local minima.
    
    # Let's implement a simple Simulated Annealing or Hill Climbing with restarts?
    # Given time constraints, a robust "Greedy Deletion + Randomized Swap" loop is good.
    
    # State representation:
    # current_indices: list of block indices
    # cover_count: np.array of size num_tsubs, counts how many current blocks cover each tsub.
    
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    for idx in current_indices:
        for tid in block_tsubs[idx]:
            cover_count[tid] += 1
            
    num_uncovered = np.sum(cover_count == 0)
    # Ensure validity
    if num_uncovered > 0:
        # Greedy didn't finish? Fill with random blocks until valid?
        # Should not happen if greedy ran to completion.
        # If it timed out, we need to ensure validity.
        # Let's just greedily add blocks to cover remaining.
        while num_uncovered > 0:
            # Find block covering most uncovered
            best_idx = -1
            best_gain = 0
            for i in range(num_blocks):
                if i in used_blocks: continue
                gain = 0
                for tid in block_tsubs[i]:
                    if cover_count[tid] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1: break
            current_indices.append(best_idx)
            used_blocks.add(best_idx)
            for tid in block_tsubs[best_idx]:
                cover_count[tid] += 1
            num_uncovered -= best_gain # Approximation, recompute if needed
            
    rng = random.Random(42)
    
    # Helper to check if a block is redundant
    def is_redundant(idx):
        for tid in block_tsubs[idx]:
            if cover_count[tid] == 1:
                return False
        return True
        
    # Phase 1: Greedy Deletion (Pass until stable)
    while time.time() < deadline - 5:
        changed = False
        # Randomize order to avoid bias
        indices_list = list(current_indices)
        rng.shuffle(indices_list)
        
        for idx in indices_list:
            if time.time() > deadline - 5: break
            if idx not in current_indices: continue # Might have been removed? 
            # Actually, we are iterating a copy, so idx is valid in the list.
            # But if we remove items, the list changes.
            # Let's use a different approach: iterate through a copy, and if redundant, remove from original.
            
            if is_redundant(idx):
                # Remove
                for tid in block_tsubs[idx]:
                    cover_count[tid] -= 1
                current_indices.remove(idx)
                changed = True
                
        if not changed:
            break
            
    # Phase 2: Local Search with Swaps
    # We are at a local minimum for deletion.
    # Try to swap a block A with a block B (not in set) to potentially allow further deletions.
    # We accept a swap if:
    # 1. The set remains valid.
    # 2. We use Simulated Annealing to accept worse states (same size, different structure) occasionally.
    #    Or simply: accept any valid swap with probability p.
    
    # To speed up finding B:
    # When removing A, identify uncovered tsubs U.
    # We need B such that B covers U.
    # Precompute: for each tsub, list of blocks covering it.
    # Then intersect lists for tsubs in U.
    
    # Building this inverse map is O(num_blocks * C(k,t)) = 50k * 35 = 1.7M operations. Fast.
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tids in enumerate(block_tsubs):
        for tid in tids:
            tsub_to_blocks[tid].append(i)
            
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We want to minimize len(current_indices).
    # Since we only do valid moves (deletions and valid swaps), len never increases.
    # Deletions decrease it. Swaps keep it same.
    # So we just need to find a sequence of swaps that enables a deletion.
    # We can run a loop:
    # 1. Try deletions. If any, do it.
    # 2. If no deletions, try a random swap.
    #    - Pick random A in current.
    #    - Compute U (tsubs only covered by A).
    #    - If U is empty, A is redundant (should have been caught).
    #    - Find candidate B's that cover U.
    #    - Pick random B from candidates.
    #    - If B is not in current, perform swap.
    #    - This might enable a deletion of some other block C.
    
    # To avoid infinite loops of swaps without progress, we can track the best size seen and restart if stuck for long.
    # Or just run until time.
    
    best_size = len(current_indices)
    stuck_count = 0
    
    while time.time() < deadline - 2:
        # 1. Try Deletions
        deleted = False
        indices_list = list(current_indices)
        rng.shuffle(indices_list)
        
        for idx in indices_list:
            if idx not in current_indices: continue
            if is_redundant(idx):
                for tid in block_tsubs[idx]:
                    cover_count[tid] -= 1
                current_indices.remove(idx)
                deleted = True
                best_size = len(current_indices)
                stuck_count = 0
                break # Restart deletion pass? Or continue?
                # Continuing is fine, as counts updated.
                
        if deleted:
            continue # Go back to top of loop to try more deletions
            
        # 2. No deletions. Try a Swap.
        if not current_indices:
            break
            
        # Pick random block A
        idx_a = rng.choice(current_indices)
        
        # Find uncovered tsubs if A removed
        U = []
        for tid in block_tsubs[idx_a]:
            if cover_count[tid] == 1:
                U.append(tid)
                
        if not U:
            # Shouldn't happen if deletion pass was thorough, but just in case
            # Actually if U is empty, A is redundant. Why wasn't it deleted?
            # Maybe race condition or logic error. Delete it.
            for tid in block_tsubs[idx_a]:
                cover_count[tid] -= 1
            current_indices.remove(idx_a)
            continue
            
        # Find candidates B that cover all tids in U
        # Intersection of tsub_to_blocks[tid] for tid in U
        # Start with the smallest list
        U_sorted = sorted(U, key=lambda x: len(tsub_to_blocks[x]))
        candidates = set(tsub_to_blocks[U_sorted[0]])
        for tid in U_sorted[1:]:
            candidates.intersection_update(tsub_to_blocks[tid])
            
        # Remove A from candidates (B != A)
        candidates.discard(idx_a)
        
        # Remove blocks already in current set?
        # If B is in current set, it's already there. 
        # But wait, if B is in current set, and B covers U, then the tids in U would be covered by B (and A).
        # So count would be >= 2.
        # But we found count == 1 for tids in U.
        # This implies NO other block in the current set covers those tids.
        # So B (if in current set) cannot cover them?
        # Contradiction?
        # If B is in current set, and B covers tid, then count[tid] >= 2 (A and B).
        # But we are in the branch where count[tid] == 1.
        # So B cannot be in the current set.
        # So we don't strictly need to filter current set, but it's good practice.
        # However, checking `if b in current_set` is O(N).
        # Since we proved B cannot be in current set if it covers U (where U are tids with count 1),
        # we can skip this check for efficiency, or use a set for O(1).
        current_set = set(current_indices)
        
        valid_candidates = [b for b in candidates if b not in current_set]
        
        if not valid_candidates:
            stuck_count += 1
            if stuck_count > 100:
                # Restart with a different random seed or break?
                # Let's just break out of this swap attempt and try another A.
                stuck_count = 0
            continue
            
        # Pick a random candidate B
        idx_b = rng.choice(valid_candidates)
        
        # Perform Swap: Remove A, Add B
        # Update counts for A
        for tid in block_tsubs[idx_a]:
            cover_count[tid] -= 1
        # Update counts for B
        for tid in block_tsubs[idx_b]:
            cover_count[tid] += 1
            
        # Update list
        current_indices.remove(idx_a)
        current_indices.append(idx_b)
        
        # Check if this swap enabled a deletion?
        # We can immediately check for redundancies.
        # Or just let the next loop iteration handle it.
        # Let's do a quick check to see if we can delete something now.
        # This is an optimization to accept "good" swaps faster.
        
        # (Optional) Quick deletion check
        # We'll just let the main loop handle it.
        
    # Final Cleanup: One more pass of deletions
    while True:
        changed = False
        indices_list = list(current_indices)
        for idx in indices_list:
            if idx not in current_indices: continue
            if is_redundant(idx):
                for tid in block_tsubs[idx]:
                    cover_count[tid] -= 1
                current_indices.remove(idx)
                changed = True
        if not changed:
            break
            
    # Convert to final output
    final_blocks = [list(all_blocks[i]) for i in current_indices]
    
    return final_blocks

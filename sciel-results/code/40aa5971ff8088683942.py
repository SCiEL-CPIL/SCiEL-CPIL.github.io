import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    all_points = list(range(v))
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [t_sub_to_idx[tsub] for tsub in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # For each t-subset, precompute the list of blocks that cover it
    tsub_blocks = defaultdict(list)
    for i, b in enumerate(all_blocks):
        for tsub_idx in block_tsubs[i]:
            tsub_blocks[tsub_idx].append(i)
            
    def time_ok():
        return time.time() < deadline

    def greedy_construct():
        """
        Greedy construction using a lazy max-heap.
        """
        covered_count = [0] * n_t
        blocks_used = []
        used_block_ids = set()
        
        # Initialize heap with (gain, block_id)
        # gain is number of uncovered t-subsets in the block
        heap = []
        for i in range(n_blocks):
            gain = len(block_tsubs[i]) # Initially all are uncovered
            heapq.heappush(heap, (-gain, i))
            
        while True:
            if not time_ok():
                break
            
            # Check if all t-subsets are covered
            # Efficient check: maintain a count of uncovered subsets
            # For simplicity in this rewrite, we check min(covered_count) > 0? No, O(n_t).
            # Better: track uncovered_count.
            # Let's add uncovered_count tracking.
            pass # We will implement this in the actual function body below
            
        return blocks_used

    # --- Actual Implementation ---
    
    # We will run a few iterations of (Greedy + Local Search) with different random seeds/tie-breaks
    # But since the problem is small, a single strong run might suffice.
    # Let's do one robust run.
    
    # State variables
    covered_count = [0] * n_t
    uncovered_count = n_t
    blocks_in_solution = []
    used_block_ids = set()
    
    # Heap of (-gain, block_id)
    heap = []
    for i in range(n_blocks):
        # Gain is number of uncovered t-subsets covered by block i
        # Initially all t-subsets are uncovered, so gain is just len(block_tsubs[i]) which is C(k,t)
        gain = len(block_tsubs[i])
        heapq.heappush(heap, (-gain, i))
        
    # Greedy Phase
    while uncovered_count > 0 and time_ok():
        # Get best block
        valid_block = None
        while heap:
            neg_gain, b_id = heapq.heappop(heap)
            # Check if this gain is still valid
            # Recalculate gain for b_id
            current_gain = 0
            for tsub_idx in block_tsubs[b_id]:
                if covered_count[tsub_idx] == 0:
                    current_gain += 1
            if current_gain == -neg_gain:
                valid_block = b_id
                break
            else:
                # Stale entry, push updated value if gain > 0
                if current_gain > 0:
                    heapq.heappush(heap, (-current_gain, b_id))
        
        if valid_block is None:
            # No block with positive gain? This shouldn't happen if uncovered_count > 0
            # unless all blocks are already used? No, we can reuse blocks? 
            # "blocks must be distinct". 
            # If we run out of blocks with gain > 0, we have a problem.
            # But every t-subset is in some block. If it's uncovered, that block has gain >= 1.
            # So valid_block should not be None.
            break
            
        # Add block to solution
        blocks_in_solution.append(valid_block)
        used_block_ids.add(valid_block)
        
        # Update coverage
        for tsub_idx in block_tsubs[valid_block]:
            if covered_count[tsub_idx] == 0:
                uncovered_count -= 1
            covered_count[tsub_idx] += 1
            
        # We don't update the heap for other blocks immediately (Lazy approach)
        # The next pop will handle stale entries.
        
    # Local Search Phase
    # Goal: Reduce number of blocks
    
    # Helper: Check if block b_id is redundant given current solution
    # A block is redundant if all its t-subsets are covered by at least one OTHER block in the solution.
    
    # To speed up redundancy checks, we maintain cover_map: tsub_idx -> list of block_ids in solution covering it
    # But updating this map on every change is O(C(k,t)).
    
    # Let's implement a simpler local search:
    # 1. Remove redundant blocks.
    # 2. Try swapping blocks.
    
    # We will use a list of block IDs in the solution.
    sol_blocks = blocks_in_solution[:]
    
    # Build cover_map for current solution
    cover_map = defaultdict(list)
    for b_id in sol_blocks:
        for tsub_idx in block_tsubs[b_id]:
            cover_map[tsub_idx].append(b_id)
            
    def is_redundant(b_id):
        for tsub_idx in block_tsubs[b_id]:
            # If this tsub is covered only by b_id, b_id is not redundant
            if len(cover_map[tsub_idx]) == 1:
                return False
        return True
        
    def remove_block(b_id):
        # Remove b_id from sol_blocks
        sol_blocks.remove(b_id)
        # Update cover_map
        for tsub_idx in block_tsubs[b_id]:
            cover_map[tsub_idx].remove(b_id)
            if not cover_map[tsub_idx]:
                del cover_map[tsub_idx]
                
    def add_block(b_id):
        sol_blocks.append(b_id)
        for tsub_idx in block_tsubs[b_id]:
            if tsub_idx not in cover_map:
                cover_map[tsub_idx] = []
            cover_map[tsub_idx].append(b_id)

    # Pass 1: Remove all redundant blocks
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(sol_blocks) - 1, -1, -1):
            if is_redundant(sol_blocks[i]):
                remove_block(sol_blocks[i])
                changed = True
                # Restart loop as indices changed? No, we iterate backwards, so it's safe.
                # But removing one might make another redundant.
                # The while loop handles that.
                
    # Pass 2: Try to replace blocks to enable further reduction
    # Strategy: For each block B_i, find its unique t-subsets U_i.
    # If U_i is not empty, find a new block B_new that covers U_i.
    # If B_new exists and is not in sol_blocks, replace B_i with B_new.
    # Then run Pass 1 again.
    
    # To find B_new:
    # The union of points in U_i must be subset of B_new.
    # If len(union) > k, impossible.
    # If len(union) <= k, we can find all blocks containing this union.
    # We precomputed tsub_blocks, but we need block candidates based on point sets.
    # This is a bit complex to precompute for all possible unions.
    # Alternative: Just iterate through all blocks to find one that covers U_i?
    # n_blocks is 50k. U_i size is small.
    # Check: all(tsub_idx in block_tsubs[b_id] for tsub_idx in U_i)
    # This is fast if U_i is small.
    
    improved = True
    while improved and time_ok():
        improved = False
        for i in range(len(sol_blocks)):
            if not time_ok(): break
            b_id = sol_blocks[i]
            # Find unique t-subsets for b_id
            unique_tsubs = []
            for tsub_idx in block_tsubs[b_id]:
                if len(cover_map[tsub_idx]) == 1:
                    unique_tsubs.append(tsub_idx)
                    
            if not unique_tsubs:
                # Should have been removed in Pass 1, but maybe became redundant
                # after a swap?
                # Actually, if it's redundant, we can just remove it.
                remove_block(b_id)
                improved = True
                break # Restart
            
            # Try to find a replacement
            # The replacement must cover all unique_tsubs
            # Let's pick the first block (other than b_id) that covers all unique_tsubs
            # To be more effective, we might want to pick one that covers the most
            # "shared" t-subsets to help remove other blocks.
            
            best_new_block = -1
            best_score = -1
            
            # Optimization: Only check blocks that contain the points of the first unique tsub?
            # No, it must contain ALL points of ALL unique tsubs.
            # Let's just iterate all blocks. 50k * (len(unique_tsubs) checks) is fast.
            
            for new_b_id in range(n_blocks):
                if new_b_id == b_id:
                    continue
                if new_b_id in sol_blocks: # Must be distinct blocks? 
                    # "blocks must be distinct"
                    # Yes, so we can't use a block already in the solution unless we replace it with itself (no-op)
                    # But we are replacing b_id, so new_b_id cannot be in sol_blocks \ {b_id}
                    # Wait, if new_b_id is in sol_blocks, it's already there. 
                    # If we replace b_id with new_b_id, and new_b_id is already there, we just remove b_id.
                    # That's equivalent to removing b_id if b_id is redundant.
                    # But b_id is NOT redundant (it has unique tsubs).
                    # So new_b_id must NOT be in sol_blocks.
                    continue
                
                # Check if new_b_id covers all unique_tsubs
                valid = True
                for tsub_idx in unique_tsubs:
                    if tsub_idx not in block_tsubs[new_b_id]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score the new block: How many t-subsets does it cover that are currently covered by ONLY b_id?
                # We already know it covers unique_tsubs.
                # Maybe score by how many OTHER blocks' unique t-subsets it covers?
                # Or just pick the first one.
                
                # Let's try to pick one that minimizes the "cost" or maximizes overlap with others
                # Simple heuristic: Pick the first one found.
                best_new_block = new_b_id
                break # Just take the first valid one to save time
                
            if best_new_block != -1:
                # Perform replacement
                # Remove old
                remove_block(b_id)
                # Add new
                add_block(best_new_block)
                improved = True
                break # Restart loop to check for new redundancies
    
    # Final validation (optional, but good for safety)
    # Check if all t-subsets are covered
    # Rebuild cover_map from scratch to be sure
    final_cover = [0] * n_t
    for b_id in sol_blocks:
        for tsub_idx in block_tsubs[b_id]:
            final_cover[tsub_idx] += 1
            
    if any(c == 0 for c in final_cover):
        # Fallback: return the greedy result if local search broke it (shouldn't happen)
        # Or just return what we have if we are confident.
        # For safety, if invalid, return the initial greedy blocks_in_solution
        return [list(all_blocks[i]) for i in blocks_in_solution]
        
    return [list(all_blocks[i]) for i in sol_blocks]

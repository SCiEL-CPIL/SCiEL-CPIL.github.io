import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage structure
    # block_triple_indices[b][i] = triple index
    block_triple_indices = [[] for _ in range(num_blocks)]
    # triple_to_blocks[idx] = list of block indices covering this triple
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    # For fast bitset operations
    block_bitsets = [0] * num_blocks
    
    for i, b in enumerate(all_blocks):
        tris_in_block = []
        bs = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris_in_block.append(idx)
            triple_to_blocks[idx].append(i)
            bs |= (1 << idx)
        block_triple_indices[i] = tris_in_block
        block_bitsets[i] = bs

    # Helper: Fast greedy set cover
    # Returns a list of block indices that cover all triples
    def fast_greedy_cover(uncovered_bitset):
        selected = []
        current_uncovered = uncovered_bitset
        while current_uncovered:
            best_gain = -1
            best_idx = -1
            # Scan all blocks for max gain
            # Optimization: We can pre-filter blocks that intersect current_uncovered
            # But scanning 346k blocks with bit_count is fast enough in C via Python int ops?
            # 346k * (1 bit_count) ~ 0.1s? Maybe. Let's try.
            # Actually, we can just iterate.
            
            # To speed up, we can maintain a list of "active" blocks?
            # For now, full scan.
            for i in range(num_blocks):
                gain = (block_bitsets[i] & current_uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == 0: # No block covers anything new? Break? No, maybe later ones do.
                        # But if best_gain is 0, we are stuck.
                        # In a valid covering problem, if current_uncovered is not 0,
                        # there must be a block covering it (since we are building a cover).
                        # However, if we start from a partial cover, it might not be possible.
                        # But we assume we can always cover.
                        pass
            
            if best_gain <= 0:
                break # Should not happen for valid inputs if we have full block list
                
            selected.append(best_idx)
            current_uncovered &= ~block_bitsets[best_idx]
            
        return selected

    # Helper: Check if a set of blocks is a valid cover
    def is_valid_cover(block_indices):
        covered = 0
        for b in block_indices:
            covered |= block_bitsets[b]
        return covered == (1 << num_triples) - 1

    # Helper: Count uncovered triples for a set of blocks
    def count_uncovered(block_indices):
        covered = 0
        for b in block_indices:
            covered |= block_bitsets[b]
        return ((1 << num_triples) - 1) ^ covered

    # Initial Solution: Strong Greedy
    initial_sol = fast_greedy_cover((1 << num_triples) - 1)
    
    if not initial_sol:
        return [all_blocks[i] for i in range(100)] # Fallback, likely invalid but prevents crash

    current_sol = list(initial_sol)
    best_sol = list(current_sol)
    best_size = len(current_sol)
    
    # ILS: Iterated Local Search with k-exchange
    # Strategy:
    # 1. Take current solution.
    # 2. Randomly remove k blocks.
    # 3. Run fast_greedy_cover on the remaining uncovered triples.
    # 4. If the new solution is smaller, accept.
    # 5. Optionally, try to remove redundant blocks from the new solution.
    
    # We will iterate this process.
    # To avoid getting stuck, we can also try "shake" moves:
    # Replace a random block with a random other block, then repair.
    
    # Let's implement a robust ILS.
    
    # Pre-compute "redundancy" check speedup?
    # For a solution S, a block b is redundant if removing it doesn't uncover any triple.
    # i.e., for all triples in b, count in S > 1.
    
    def try_remove_redundant(sol):
        # Returns new sol with redundant blocks removed
        # We need to know coverage counts.
        # Since sol size is small (< 100), we can compute counts.
        covered_counts = [0] * num_triples
        for b in sol:
            for tri_idx in block_triple_indices[b]:
                covered_counts[tri_idx] += 1
        
        new_sol = []
        for b in sol:
            # Check if b is essential
            is_essential = False
            for tri_idx in block_triple_indices[b]:
                if covered_counts[tri_idx] == 1:
                    is_essential = True
                    break
            if not is_essential:
                # Redundant, do not add to new_sol
                # Decrement counts for its triples
                for tri_idx in block_triple_indices[b]:
                    covered_counts[tri_idx] -= 1
            else:
                new_sol.append(b)
        return new_sol

    # Main ILS Loop
    T = 50.0 # Temperature for simulated annealing style acceptance?
    # Or just deterministic improvement?
    # Let's use a simple hill climbing with random restarts/shakes.
    
    num_restarts = 0
    max_restarts = 500 # Limit restarts to avoid infinite loop if stuck
    
    while time.time() < deadline - 5:
        # 1. Local Improvement: Remove redundant blocks
        current_sol = try_remove_redundant(current_sol)
        
        if len(current_sol) < best_size:
            best_sol = list(current_sol)
            best_size = len(current_sol)
            
        # 2. Check time
        if time.time() > deadline - 10:
            break
            
        # 3. Generate a perturbation (Shake)
        # Strategy: Remove k random blocks, then repair.
        # k depends on current size.
        # Let's try k=1, 2, 3.
        
        # Pick a random k
        k_remove = random.choice([1, 1, 2, 3])
        
        if len(current_sol) <= k_remove:
            # If solution is small, just restart from scratch?
            # Or just break.
            # Let's try a full restart from a new random greedy?
            # Random greedy is slow.
            # Let's just break and try a different initial?
            # For now, break.
            break
            
        # Select k blocks to remove
        # To make it interesting, remove blocks that are "least essential"?
        # Or random?
        # Random is good for diversity.
        
        # We need to calculate the uncovered bitset after removal.
        # covered_bitset = OR of all blocks in current_sol
        # new_uncovered = (~covered_bitset) | (OR of removed blocks' bits)
        # Wait, if we remove blocks, the triples they covered might become uncovered
        # IF they were not covered by other blocks.
        
        # Efficient way:
        # 1. Compute total covered bitset of current_sol.
        # 2. Compute covered bitset of (current_sol - removed).
        # 3. new_uncovered = total_covered ^ covered_remaining? No.
        # new_uncovered = (~covered_remaining) & total_covered? No.
        # A triple is uncovered in the new state if it was covered in old state
        # and is NOT covered in remaining state.
        
        # Let's just compute covered_remaining.
        # Since current_sol is small, this is fast.
        
        removed_blocks = random.sample(current_sol, k_remove)
        remaining_blocks = [b for b in current_sol if b not in removed_blocks]
        
        covered_remaining = 0
        for b in remaining_blocks:
            covered_remaining |= block_bitsets[b]
            
        # The triples that are now "at risk" or "uncovered" are those not in covered_remaining.
        # But we only care about those that were covered before?
        # Actually, fast_greedy_cover expects a bitset of triples that MUST be covered.
        # The triples already covered by remaining_blocks don't need to be covered again.
        # The triples NOT covered by remaining_blocks MUST be covered by the new blocks we add.
        
        all_triples_bitset = (1 << num_triples) - 1
        needed_cover = all_triples_bitset & ~covered_remaining
        
        # Repair: Find min blocks to cover needed_cover
        # We can use fast_greedy_cover, but it scans ALL blocks.
        # We can optimize by only considering blocks that intersect needed_cover.
        # But for simplicity, use fast_greedy_cover.
        
        repair_blocks = fast_greedy_cover(needed_cover)
        
        new_sol = remaining_blocks + repair_blocks
        
        # Remove duplicates?
        # fast_greedy_cover might pick a block already in remaining_blocks?
        # No, because fast_greedy_cover picks blocks that cover needed_cover.
        # If a block in remaining_blocks covers a triple in needed_cover,
        # that triple would be in covered_remaining, so it wouldn't be in needed_cover.
        # So repair_blocks are disjoint from remaining_blocks in terms of "essential coverage".
        # However, they might be the same block index?
        # If a block is in remaining_blocks, its bits are in covered_remaining.
        # So it covers no triple in needed_cover.
        # So fast_greedy_cover will not pick it (gain 0).
        # So new_sol should have distinct blocks.
        
        # Apply local improvement to new_sol
        new_sol = try_remove_redundant(new_sol)
        
        # Acceptance Criterion
        # If new_sol is smaller, accept.
        # If new_sol is same size, accept with some probability (to escape local optima)?
        # Or just accept if it's different?
        # Let's accept if size is smaller.
        # If size is same, we might get stuck.
        # Let's try to accept if size <= best_size? No, that's not helpful.
        # Let's use a simple random walk on the size.
        
        if len(new_sol) < len(current_sol):
            current_sol = new_sol
            num_restarts = 0 # Reset stagnation counter
        else:
            # Stagnation
            num_restarts += 1
            if num_restarts > 50:
                # Try a bigger shake? Or restart from scratch?
                # Let's try a bigger shake
                k_remove = random.choice([3, 4, 5])
                if len(current_sol) > k_remove:
                    removed_blocks = random.sample(current_sol, k_remove)
                    remaining_blocks = [b for b in current_sol if b not in removed_blocks]
                    covered_remaining = 0
                    for b in remaining_blocks:
                        covered_remaining |= block_bitsets[b]
                    needed_cover = all_triples_bitset & ~covered_remaining
                    repair_blocks = fast_greedy_cover(needed_cover)
                    new_sol = remaining_blocks + repair_blocks
                    new_sol = try_remove_redundant(new_sol)
                    
                    if len(new_sol) < len(current_sol):
                        current_sol = new_sol
                        num_restarts = 0
                    else:
                        # Give up on this path, try a completely new random initial?
                        # Generating a new random initial is expensive.
                        # Let's just break and hope best_sol is good.
                        pass
                else:
                    pass
                    
    # Final Polish on best_sol
    if time.time() < deadline - 2:
        best_sol = try_remove_redundant(best_sol)
        
    # Ensure validity
    if not is_valid_cover(best_sol):
        # Fallback to initial if it was valid
        if is_valid_cover(initial_sol):
            best_sol = initial_sol
        else:
            # Last resort: return something valid?
            # If initial is invalid, we have a problem.
            # But initial was constructed by greedy to cover all.
            # So it should be valid.
            best_sol = initial_sol
            
    result_blocks = [all_blocks[i] for i in best_sol]
    
    return result_blocks

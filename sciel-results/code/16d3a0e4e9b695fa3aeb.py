import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Pre-compute triples for every possible k-subset (block)
    # v=24, k=7, C(24,7)=346104. This is feasible in Python if optimized.
    # However, storing lists of triples for 346k blocks might be memory heavy (~346k * 35 triples * 3 ints).
    # Let's try to generate on the fly or pre-compute carefully.
    # Actually, 346k * 35 = ~12 million triples. Each tuple is small. 
    # Let's pre-compute to speed up the swap phase significantly.
    
    # Memory check: 346104 blocks. Each has 35 triples.
    # Total triples stored: ~12M.
    # In Python, a tuple of 3 ints is ~64 bytes. 12M * 64 = 768MB. Too much.
    
    # Alternative: Pre-compute for each point, the list of triples containing it?
    # Or just calculate gain on the fly but optimize the inner loop.
    
    # Let's try a hybrid: 
    # 1. Greedy construction (fast, gets us close to optimal)
    # 2. Swap local search (slow per step, but improves count)
    
    def greedy_construct():
        uncovered = set(all_triples)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start
            start_tri = random.choice(list(uncovered))
            block = list(start_tri)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                best_gain = -1
                best_cand = None
                # Optimization: Instead of checking all candidates, 
                # we can just pick the one that covers the most uncovered triples.
                # To speed this up, we can sample or use a heuristic.
                # For v=24, k=7, this is fast enough.
                for x in candidates:
                    gain = 0
                    # Triples formed by x and 2 elements from block
                    for p1, p2 in itertools.combinations(block, t-1):
                        tri = tuple(sorted((p1, p2, x)))
                        if tri in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = x
                
                if best_cand is not None:
                    block.append(best_cand)
                else:
                    # Fallback (should not happen if v>k)
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
                    
        return blocks

    def count_triples(blocks):
        counts = [0] * n_triples
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        return counts

    def local_search_swap(blocks):
        # We want to reduce the number of blocks.
        # Strategy: Try to replace block B_i with a new block B_new such that:
        # 1. B_new covers at least one triple that was ONLY covered by B_i (or generally improves coverage efficiency).
        # 2. Actually, a simpler heuristic: 
        #    - Remove B_i.
        #    - Identify triples that are now uncovered (count went from 1 to 0).
        #    - Find a new block B_new that covers the maximum number of these "lost" triples.
        #    - If B_new covers ALL lost triples, we can swap B_i -> B_new? No, that keeps count same.
        #    - To REDUCE count, we need to find a B_new that covers all triples of B_i that were unique to B_i, 
        #      AND B_new doesn't introduce any new uncovered triples? 
        #      Wait, if we replace B_i with B_new, the count stays the same. 
        #      The goal is to eventually make a block redundant so we can remove it.
        #      
        #      Better Strategy for Reduction:
        #      1. Try to remove a block B_i.
        #      2. If removal causes uncovered triples U, try to find a SINGLE block B_new that covers all U.
        #      3. If such B_new exists, we can swap B_i -> B_new. This doesn't reduce count immediately,
        #         but it might change the structure so that another block becomes redundant.
        #      4. Alternatively, if we find B_new that covers U, and B_new is already in the set? No.
        #      
        #      Let's use a "Repair" approach:
        #      1. Randomly remove a block.
        #      2. Find all uncovered triples.
        #      3. Greedily add blocks to cover them? That increases count.
        #      
        #      Standard approach for Covering Design optimization:
        #      - Start with a valid cover.
        #      - Try to remove a block. If invalid, try to "fix" it by swapping one point in the removed block 
        #        or swapping the removed block with another one that better covers the "weak" spots.
        #      
        #      Let's try: 
        #      For each block B_i:
        #         Let U_i be the set of triples covered ONLY by B_i (count == 1).
        #         If U_i is empty, B_i is redundant -> Remove it.
        #         If U_i is not empty:
        #             Find a block B_j (j != i) or a new block B_new that covers the most triples in U_i.
        #             If we swap B_i with B_new, and B_new covers all U_i, the cover remains valid.
        #             This doesn't reduce count, but it might make B_j redundant later?
        #             
        #      Actually, the most effective local search for covering numbers is often:
        #      1. Remove a block.
        #      2. If uncovered triples exist, try to add a block that covers the MAXIMUM number of uncovered triples.
        #      3. If the number of uncovered triples is 0, we reduced count by 1.
        #      4. If not, we revert. 
        #      This is "Add-Remove" or "Perturbation".
        #      
        #      Let's implement a "Perturbation" loop:
        #      1. Take current blocks.
        #      2. Randomly remove a block B_r.
        #      3. Calculate uncovered triples.
        #      4. If none, we have a smaller cover! Return it.
        #      5. If some, try to find ONE block that covers ALL uncovered triples.
        #         - If found, replace B_r with this new block. (Count same, but structure changed).
        #         - If not found, revert.
        #      6. Also try: Find a block that covers the MAX number of uncovered triples.
        #         - If it covers all, swap.
        #         - If it covers most, maybe swap if it also doesn't break other things? 
        #           But we already removed B_r, so we just need to cover the holes.
        #      
        #      To find a block covering a specific set of triples T:
        #      We need a 7-set that contains all triples in T.
        #      This is a constraint satisfaction problem.
        #      For v=24, k=7, we can iterate through all 346k blocks and check if they contain T.
        #      This is fast if T is small.
        
        best_blocks = list(blocks)
        best_count = len(best_blocks)
        
        counts = count_triples(best_blocks)
        
        # Pre-compute for each triple, the list of blocks that cover it?
        # No, we have the blocks list.
        
        # To speed up "find block covering set T":
        # We can pre-compute a mapping: for each triple, which blocks cover it?
        # triple_to_blocks = {tri: [indices of blocks covering tri]}
        # Memory: 2024 triples * ~15 blocks avg = 30k entries. Very small.
        
        triple_to_blocks = {i: [] for i in range(n_triples)}
        for idx, b in enumerate(best_blocks):
            for tri in itertools.combinations(b, t):
                triple_to_blocks[triple_to_idx[tri]].append(idx)
        
        # Helper: Find a block (from a pool or all possible) that covers all triples in T
        # We will search among all possible 7-subsets of 24 points.
        # There are 346,104 such subsets.
        # Checking if a subset covers T: 
        #   For each tri in T, check if tri is subset of block.
        #   If T is small (e.g., 1-3 triples), this is fast.
        
        def find_block_covering(T):
            if not T:
                return None # No block needed
            # T is a list of triple indices
            # We need a 7-set containing all these triples.
            # Let's iterate through all 346k blocks.
            # To make this faster, we can pre-compute the "signature" of each block?
            # Or just check.
            # Optimization: 
            #   The union of points in all triples in T must be a subset of the block.
            #   Let U_points = union of points in T.
            #   If len(U_points) > k, no block can cover T.
            #   If len(U_points) <= k, we only need to check blocks that contain U_points.
            
            U_points = set()
            for tri_idx in T:
                tri = all_triples[tri_idx]
                U_points.update(tri)
            
            if len(U_points) > k:
                return None
            
            # Now we need to find a 7-set that contains U_points.
            # Remaining points to choose: k - len(U_points)
            # Available points: v - len(U_points)
            # We can iterate through combinations of the remaining points.
            remaining_pool = [x for x in range(v) if x not in U_points]
            num_to_pick = k - len(U_points)
            
            for combo in itertools.combinations(remaining_pool, num_to_pick):
                candidate_block = tuple(sorted(list(U_points) + list(combo)))
                # Verify it covers all T (it should by construction of U_points, but let's be safe? 
                # Actually, if U_points is the union, and we add more points, it still contains U_points.
                # Since each tri in T is a subset of U_points, the block contains all tris in T.)
                return list(candidate_block)
            
            return None

        # Main Local Search Loop
        while time.time() < deadline - 5:
            current_blocks = best_blocks
            current_count = len(current_blocks)
            counts = count_triples(current_blocks)
            
            # Rebuild triple_to_blocks for current best
            triple_to_blocks = {i: [] for i in range(n_triples)}
            for idx, b in enumerate(current_blocks):
                for tri in itertools.combinations(b, t):
                    triple_to_blocks[triple_to_idx[tri]].append(idx)
            
            improved = False
            # Try to remove each block
            for i in range(current_count):
                if time.time() > deadline - 5:
                    break
                
                # Simulate removing block i
                # Find triples that become uncovered (count == 1 and covered by i)
                lost_triples = []
                for tri in itertools.combinations(current_blocks[i], t):
                    tri_idx = triple_to_idx[tri]
                    if counts[tri_idx] == 1:
                        # Only block i covers it
                        lost_triples.append(tri_idx)
                
                if not lost_triples:
                    # Block i is redundant!
                    new_blocks = [b for j, b in enumerate(current_blocks) if j != i]
                    best_blocks = new_blocks
                    best_count = len(new_blocks)
                    improved = True
                    break # Restart loop with new best
                else:
                    # Try to find a replacement block that covers all lost_triples
                    # AND is not already in the set (to avoid duplicates, though duplicates don't help)
                    # Actually, we can just find ANY block that covers them.
                    # If we find one, we swap block i with this new block.
                    # This doesn't reduce count, but might help future reductions.
                    
                    # Optimization: Only try to repair if the number of lost triples is small?
                    # If lost_triples is large, it's unlikely to be covered by one block.
                    if len(lost_triples) > 10:
                        continue
                    
                    replacement = find_block_covering(lost_triples)
                    if replacement:
                        # Check if this replacement is already in the set
                        rep_tuple = tuple(sorted(replacement))
                        already_exists = False
                        for b in current_blocks:
                            if tuple(sorted(b)) == rep_tuple:
                                already_exists = True
                                break
                        
                        if not already_exists:
                            # Perform swap
                            new_blocks = list(current_blocks)
                            new_blocks[i] = replacement
                            best_blocks = new_blocks
                            # Count remains the same, but structure changed.
                            # We don't break immediately, we might find a removal in the next iteration.
                            # But to avoid infinite loops of swaps, we should limit swaps or use a time check.
                            # Let's just update and break to recalculate counts.
                            improved = True
                            break
                            
            if not improved:
                # No immediate improvement. Try a random perturbation?
                # Or just stop if we've hit a local minimum?
                # For now, if we can't improve, we might get stuck.
                # Let's try a random swap to escape local minima.
                if current_count > 1:
                    i = random.randint(0, current_count - 1)
                    # Pick a random new block
                    new_block = list(random.sample(range(v), k))
                    new_block.sort()
                    
                    # Check if it's valid (distinct from existing)
                    nb_tuple = tuple(new_block)
                    exists = any(tuple(sorted(b)) == nb_tuple for b in current_blocks)
                    if not exists:
                        # Tentatively swap
                        test_blocks = list(current_blocks)
                        test_blocks[i] = new_block
                        test_counts = count_triples(test_blocks)
                        
                        # Check if this swap maintains validity (it should, as we are just swapping blocks)
                        # But does it help? We can't tell immediately.
                        # To avoid random walk, only accept if it reduces "waste"?
                        # Let's just accept it with some probability (Simulated Annealing style)?
                        # No, let's keep it simple: only accept if it allows a removal in the next step?
                        # Too complex. Let's just stop if no improvement found in a full pass.
                break

        return best_blocks

    # 1. Initial Construction
    blocks = greedy_construct()
    
    # 2. Local Search
    if time.time() < deadline - 10:
        blocks = local_search_swap(blocks)
        
    # 3. Final Cleanup (remove any remaining redundant blocks)
    # Just in case the local search left some
    counts = count_triples(blocks)
    final_blocks = []
    for b in blocks:
        # Check if b is redundant given the OTHERS?
        # This is O(N^2 * 35). N is ~70. 70^2 * 35 = 171k. Fast.
        redundant = True
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tri]
            # Count how many blocks in final_blocks cover this tri
            # This is inefficient. 
            # Better: Just do a full check at the end.
            pass
        
    # Simpler final cleanup:
    # Iterate and remove if redundant
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        counts = count_triples(blocks)
        for i in range(len(blocks)):
            if time.time() > deadline - 2:
                break
            b = blocks[i]
            can_remove = True
            for tri in itertools.combinations(b, t):
                if counts[triple_to_idx[tri]] == 1:
                    can_remove = False
                    break
            if can_remove:
                blocks.pop(i)
                changed = True
                break # Restart count calculation

    return blocks

# Run once to test? No, the evaluator calls construct.

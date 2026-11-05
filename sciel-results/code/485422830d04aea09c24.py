import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # 120s limit, leave 5s margin
    deadline = start_time + 115
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Map blocks to the triples they cover
    # block_triples[i] is a list of triple indices covered by block i
    block_triples = [[] for _ in range(num_blocks)]
    # triple_blocks[j] is a list of block indices covering triple j
    triple_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_blocks[idx].append(i)
        block_triples[i] = tris

    def greedy_set_cover(seed):
        """
        Standard greedy set cover algorithm.
        At each step, pick the block that covers the most uncovered triples.
        """
        rng = random.Random(seed)
        uncovered_count = [1] * num_triples
        selected = []
        selected_set = set()
        
        # Optimization: Instead of scanning all blocks every time (O(N_blocks * N_triples)),
        # we can just scan. With N_blocks ~ 346k and N_triples ~ 2024, 
        # one pass is ~700M ops. Too slow in Python if done repeatedly.
        # However, we can optimize by only checking blocks that cover at least one uncovered triple.
        # Or, simpler: just run it a few times if it's fast enough? 
        # Actually, pure Python loop over 346k blocks * 2024 triples is too slow.
        # We need a faster greedy.
        
        # Alternative: Randomized Greedy with limited candidates?
        # Or use NumPy?
        
        # Let's try a NumPy-based gain calculation for the first few steps, 
        # but maintaining state in Python lists is tricky.
        
        # Given the time constraint, a pure Python greedy might be too slow for many iterations.
        # Let's stick to a faster heuristic initialization: 
        # 1. Randomly select blocks until coverage is high? No.
        # 2. Use the "conflict" based approach directly on a random initial set?
        
        # Let's try a "Greedy with Batching" or just a fast random restart if greedy is too slow.
        # Actually, for v=24, k=7, t=3, the number of blocks is C(24,7) = 346104.
        # Iterating over all of them in Python is ~0.1s per full pass? 
        # 346k iterations * (inner loop 35 triples) = 12M ops. ~0.5s in Python.
        # If we need ~76 blocks, that's 76 * 0.5s = 38s. That's tight but doable for 1-2 restarts.
        
        # Let's implement a slightly optimized greedy:
        # Maintain a list of active blocks.
        
        active_blocks = list(range(num_blocks))
        
        while any(uncovered_count) and time.time() < deadline - 5:
            best_gain = 0
            best_block = -1
            # Sample a subset of blocks to speed up? 
            # No, greedy needs exact max for good performance.
            # But 346k blocks is large.
            
            # Optimization: Only consider blocks that intersect with uncovered triples.
            # We can track which blocks are "useful".
            
            # For simplicity and speed in this specific instance, 
            # let's just do a single pass and pick the best.
            # If this is too slow, we fall back to random.
            
            for i in active_blocks:
                if i in selected_set:
                    continue
                gain = 0
                for tri_idx in block_triples[i]:
                    if uncovered_count[tri_idx] > 0:
                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            # Early break if we can't beat best? No, we need exact max.
                            # But if gain reaches theoretical max (35), we can break?
                            # Max gain is 35.
                            if gain == 35: 
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            
            if best_block == -1 or best_gain == 0:
                break
            
            selected.append(best_block)
            selected_set.add(best_block)
            
            # Update uncovered counts
            for tri_idx in block_triples[best_block]:
                uncovered_count[tri_idx] -= 1
            
            # Remove selected from active? 
            # We keep it in active but skip via selected_set.
            # This is O(N) check. 
            # Better: maintain a list of candidates that still have gain > 0.
            # But calculating gain for all is the bottleneck.
            
        return selected

    def min_conflict_search(initial_sol, seed, time_limit):
        """
        Local search minimizing the number of blocks while maintaining coverage.
        Energy = number of uncovered triples (must be 0) + lambda * number of blocks.
        Actually, we want to minimize blocks subject to coverage.
        So Energy = num_blocks if covered, else num_blocks + large_penalty * uncovered.
        
        Moves:
        1. Remove a block (if it doesn't uncover any triple).
        2. Swap a block with another block that covers the same "critical" triples.
        3. Add a block (if we are stuck with uncovered triples, though we start covered).
        
        Since we start with a valid covering, we only need to remove blocks.
        The "Min-Conflict" idea: 
        If we remove a block, some triples might become uncovered.
        We want to find a block to remove that minimizes the number of newly uncovered triples.
        Ideally 0.
        
        This is equivalent to finding a minimal dominating set in the hypergraph.
        
        Algorithm:
        1. Calculate coverage counts for all triples.
        2. Loop:
           a. Find a block that is "redundant" (all its triples are covered by other blocks).
              If found, remove it.
           b. If no redundant block found, we are at a local minimum.
              To escape, we can perform a "swap":
              - Pick a block B to remove.
              - Identify triples T_uncovered that would become uncovered.
              - Find a block B' (not in set) that covers all T_uncovered.
              - If found, swap B with B'.
              - If not, try a "2-swap" or accept a temporary increase in size? 
                No, we want to minimize size.
                
        Actually, standard local search for Set Cover minimization:
        - Start with a valid cover.
        - Try to remove blocks. If a block removal causes coverage loss, 
          look for another block that can cover the lost triples.
        """
        rng = random.Random(seed)
        sol = list(initial_sol)
        sol_set = set(sol)
        
        # counts[tri_idx] = number of blocks in sol covering tri_idx
        counts = np.zeros(num_triples, dtype=np.int32)
        for idx in sol:
            for tri_idx in block_triples[idx]:
                counts[tri_idx] += 1
        
        current_size = len(sol)
        best_sol = list(sol)
        best_size = current_size
        
        # Precompute for each block, the list of triples it covers (already have block_triples)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Remove redundant blocks
            i = 0
            while i < len(sol):
                idx = sol[i]
                # Check if any triple covered by idx has count == 1
                # If yes, idx is essential.
                # If no, idx is redundant.
                is_essential = False
                for tri_idx in block_triples[idx]:
                    if counts[tri_idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove it
                    sol.pop(i)
                    sol_set.discard(idx)
                    for tri_idx in block_triples[idx]:
                        counts[tri_idx] -= 1
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_sol = list(sol)
                else:
                    i += 1
            
            # 2. If no simple removal, try swaps to enable future removals
            # This is harder. 
            # Strategy: Pick a random block B.
            # If B is essential, find triples T that are only covered by B.
            # Find a block B' that covers T.
            # If B' is in sol, it's a no-op or bad.
            # If B' is not in sol, add B' and remove B.
            # This keeps size same, but might make other blocks redundant.
            
            if time.time() < time_limit - 5:
                # Try a few random swaps
                num_swap_attempts = 10
                for _ in range(num_swap_attempts):
                    if not sol:
                        break
                    idx = rng.choice(sol)
                    
                    # Find triples that become uncovered if we remove idx
                    critical_tris = []
                    for tri_idx in block_triples[idx]:
                        if counts[tri_idx] == 1:
                            critical_tris.append(tri_idx)
                    
                    if not critical_tris:
                        continue # Should have been removed in step 1, but maybe race condition?
                    
                    # Find a block B' that covers all critical_tris
                    # We can search in triple_blocks[critical_tris[0]]
                    candidates = triple_blocks[critical_tris[0]]
                    
                    found_swap = False
                    for cand in candidates:
                        if cand in sol_set:
                            continue
                        # Check if cand covers all critical_tris
                        # Optimize: Check intersection
                        covers_all = True
                        cand_tris = block_triples[cand]
                        # Simple check
                        for ct in critical_tris:
                            if ct not in cand_tris:
                                covers_all = False
                                break
                        if covers_all:
                            # Perform swap
                            # Remove idx
                            sol.remove(idx)
                            sol_set.discard(idx)
                            for tri_idx in block_triples[idx]:
                                counts[tri_idx] -= 1
                            
                            # Add cand
                            sol.append(cand)
                            sol_set.add(cand)
                            for tri_idx in block_triples[cand]:
                                counts[tri_idx] += 1
                            
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break # Restart the removal loop after a successful swap
        
        return best_sol

    best_sol = None
    best_size = num_blocks
    
    # Run multiple restarts
    # 1. Greedy Set Cover (might be slow, limit time)
    # 2. Min-Conflict Search
    
    # Since Greedy might be slow, let's try a faster initialization:
    # Random blocks until covered? No, too many.
    # Let's just run Greedy with a tight time limit per restart.
    
    num_restarts = 10 # We will break early if time runs out
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialization: Greedy
        # To speed up greedy, we can use a "randomized" greedy where we pick from top-k gains?
        # Or just standard greedy.
        
        # Given the size of the problem, standard greedy in Python might take ~1s per block.
        # 76 blocks * 1s = 76s. That's too slow for multiple restarts.
        
        # Alternative Initialization: 
        # Use a precomputed "good" structure? No, must be generic.
        
        # Let's try a "Fast Greedy" using NumPy for the gain calculation.
        # Represent uncovered as a binary vector.
        # Gain = sum(uncovered[block_triples[i]]).
        
        def fast_greedy_init(seed):
            rng = random.Random(seed)
            uncovered = np.ones(num_triples, dtype=np.int8)
            selected = []
            selected_set = set()
            
            # Precompute block_triples as a list of arrays for faster indexing?
            # block_triples is list of lists.
            
            while np.any(uncovered) and time.time() < deadline - 5:
                # Calculate gains for all blocks
                # This is a matrix multiplication or sum reduction.
                # Gains[i] = sum(uncovered[block_triples[i]])
                
                # Since block_triples are lists of varying length (always 35 for k=7, t=3),
                # we can pad them? Or just loop.
                
                # Loop in Python is slow.
                # Let's try to vectorize.
                # Create a matrix M of shape (num_blocks, num_triples) with 1s where block covers triple?
                # Memory: 346k * 2024 * 1 byte = 700MB. Too much.
                
                # So we must loop.
                # To make it faster, we can skip blocks that are already selected.
                # And we can break early if we find a block with max possible gain?
                # Max gain is 35.
                
                best_gain = 0
                best_block = -1
                
                # Optimization: Only iterate over blocks that cover at least one uncovered triple.
                # We can maintain a list of "active" blocks.
                # Initially all.
                # When we pick a block, we don't remove others, but we can filter.
                
                # Actually, for the first few steps, all blocks are active.
                # Let's just do the loop.
                
                for i in range(num_blocks):
                    if i in selected_set:
                        continue
                    gain = 0
                    for tri_idx in block_triples[i]:
                        if uncovered[tri_idx]:
                            gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                if best_gain == 35: # Max possible
                                    break
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                
                if best_block == -1 or best_gain == 0:
                    break
                
                selected.append(best_block)
                selected_set.add(best_block)
                
                for tri_idx in block_triples[best_block]:
                    uncovered[tri_idx] = 0
                
            return selected

        # Run fast greedy
        g_sol = fast_greedy_init(100 + i)
        
        # Run Min-Conflict
        mc_sol = min_conflict_search(g_sol, 200 + i, deadline - 5)
        
        if len(mc_sol) < best_size:
            # Verify
            counts = [0] * num_triples
            for idx in mc_sol:
                for tri_idx in block_triples[idx]:
                    counts[tri_idx] += 1
            if all(c > 0 for c in counts):
                best_size = len(mc_sol)
                best_sol = mc_sol

    if best_sol is None:
        # Fallback: Just return a valid covering, even if large
        # Greedy should have produced one.
        best_sol = fast_greedy_init(1)
    
    return [all_blocks[i] for i in best_sol]

if __name__ == "__main__":
    # Test with small parameters
    # v=8, k=3, t=2
    # Blocks of 3 from 8 points.
    # C(8,3) = 56 blocks.
    # Triples C(8,2) = 28.
    # Lower bound: 28 / C(3,2) = 28/3 = 9.33 -> 10.
    
    # v=24, k=7, t=3 is the target.
    pass

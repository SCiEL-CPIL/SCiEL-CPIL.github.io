import itertools
import random
import time
import heapq
import math
import numpy as np

def construct(v, k, t):
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Precompute triples for a block as a list of indices
    # To speed up, we precompute for a specific block on the fly but cache if needed?
    # For v=24, k=7, t=3, binom(7,3)=35 triples per block.
    # We can represent a block by its frozenset of points.
    
    def get_block_triple_indices(block):
        return [sub_to_idx[tr] for tr in itertools.combinations(block, t)]

    # Phase 1: Generate a Pool of Candidate Blocks
    # We want a pool of size ~200-300 to give the optimizer enough freedom.
    
    def generate_pool(num_blocks, seed):
        random.seed(seed)
        pool = set()
        # Track coverage to prioritize "hard" triples
        cover_counts = [0] * n_subs
        uncovered = set(range(n_subs))
        
        for _ in range(num_blocks):
            if not uncovered:
                break
            # Pick a random uncovered triple to anchor the block
            # Or pick a triple that is "hardest" (low coverage potential)?
            # Since all triples have same potential initially, random is fine.
            # But as we add blocks, some triples get covered more.
            # Let's pick a random uncovered triple.
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            
            # Build a block around this triple greedily
            # We want to maximize the number of *new* uncovered triples covered.
            block = list(triple)
            
            # Candidates for remaining slots
            candidates = [p for p in range(v) if p not in block]
            
            while len(block) < k:
                best_x = -1
                max_gain = -1
                # Shuffle candidates to break ties randomly
                random.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Triples formed: (t-1) from block + x
                    for rest in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_triple]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    break
                block.append(best_x)
                candidates.remove(best_x)
            
            block.sort()
            block_tuple = tuple(block)
            
            if block_tuple not in pool:
                pool.add(block_tuple)
                # Update uncovered for the next block generation
                for tr in itertools.combinations(block, t):
                    idx = sub_to_idx[tr]
                    cover_counts[idx] += 1
                    if cover_counts[idx] == 1:
                        uncovered.discard(idx)
                        
        return list(pool)

    # Phase 2: Optimize Selection from Pool
    # We treat this as a Minimum Set Cover problem on the pool.
    
    def optimize_selection(pool, time_limit, seed):
        random.seed(seed)
        
        # Map pool blocks to their triple indices
        pool_triples = [get_block_triple_indices(b) for b in pool]
        n_pool = len(pool)
        
        # Initial Selection: Greedy Set Cover on the pool
        # We want to select a subset of pool blocks such that all n_subs are covered.
        
        # State:
        # cover_counts: array of size n_subs, count of how many selected blocks cover each triple
        # selected: boolean array of size n_pool
        # block_ids: list of currently selected block indices
        
        cover_counts = [0] * n_subs
        selected = [False] * n_pool
        block_ids = []
        uncovered = set(range(n_subs))
        
        # Greedy initialization
        # To speed up, we can't check all pool blocks against all uncovered triples every time.
        # We'll just do a standard greedy: pick block with max gain.
        # Since pool is ~200, and n_subs=2024, this is 200*2024 ~ 400k ops per step.
        # Max 100 steps? 40M ops. Might be slow in Python.
        
        # Optimization: For each pool block, precompute its triples.
        # We can maintain a "gain" array for each block?
        # Gain of block b = number of uncovered triples in pool_triples[b].
        
        gains = [0] * n_pool
        for i, trs in enumerate(pool_triples):
            gain = 0
            for tr in trs:
                if tr in uncovered:
                    gain += 1
            gains[i] = gain
            
        while uncovered:
            # Find block with max gain
            max_gain = -1
            best_block = -1
            for i in range(n_pool):
                if not selected[i] and gains[i] > max_gain:
                    max_gain = gains[i]
                    best_block = i
                    
            if best_block == -1:
                # Should not happen if pool is good enough
                break
                
            # Select block
            selected[best_block] = True
            block_ids.append(best_block)
            
            # Update cover_counts and uncovered
            for tr in pool_triples[best_block]:
                cover_counts[tr] += 1
                if cover_counts[tr] == 1:
                    uncovered.discard(tr)
                # Update gains for other blocks?
                # If a triple becomes covered, any block covering it loses 1 gain.
                # This is expensive to update globally.
                # However, for greedy initialization, we can just recalculate gains for remaining blocks
                # or accept that gains might be slightly stale (overestimating if triple covered, underestimating? No, if covered, it's no longer in uncovered, so gain should decrease).
                # Actually, if we don't update gains, we might pick a block that covers already covered triples.
                # But since we remove from `uncovered`, the condition `if tr in uncovered` in the initial calc is gone.
                # We need to update gains.
                
            # Update gains for all unselected blocks
            # This is O(n_pool * binom(k,t)) per step.
            # 200 * 35 = 7000 ops. 100 steps = 700k. Acceptable.
            for j in range(n_pool):
                if not selected[j]:
                    # Recalculate gain? Or just subtract if the new block covered some of its triples?
                    # Simpler: just recalculate from scratch for unselected blocks?
                    # No, that's O(n_pool * 35).
                    # Let's just recalculate gain for all unselected blocks.
                    pass # We'll do a full recalculation of gains every step for simplicity and correctness, 
                         # as n_pool is small.
            
            # Recalculate gains for all unselected blocks
            for j in range(n_pool):
                if not selected[j]:
                    gain = 0
                    for tr in pool_triples[j]:
                        if tr in uncovered:
                            gain += 1
                    gains[j] = gain

        # Now we have an initial valid selection `block_ids`.
        # Perform Local Search to reduce size.
        
        start_time = time.time()
        
        # Data structures for local search
        # For each selected block, we want to know if it's redundant.
        # Redundant if all its triples have cover_count > 1.
        
        # We will iterate:
        # 1. Try to remove any redundant block.
        # 2. If no removal, try to swap a block with another pool block to enable removal?
        #    Or just try to swap to reduce "criticality".
        
        # Let's define a metric: "critical_count" = number of triples with cover_count == 1.
        # We want to minimize this to enable removals.
        
        critical_count = 0
        for c in cover_counts:
            if c == 1:
                critical_count += 1
                
        best_block_ids = block_ids[:]
        best_size = len(block_ids)
        
        # Precompute for each pool block, which triples it covers.
        # And for each selected block, which triples are "critical" (count==1) for it?
        # A triple is critical for block B if cover_count[triple]==1 and B covers it.
        
        # To speed up swap evaluation:
        # If we remove block i and add block j:
        # New cover_counts = old - triples(i) + triples(j)
        # We need to ensure no cover_count becomes 0.
        # This means for every triple in triples(i), it must be covered by some other block OR by block j.
        # i.e., for all tr in triples(i): (cover_counts[tr] - 1 + (1 if tr in triples(j) else 0)) > 0
        
        # Let's implement a simple hill climbing on the selection set.
        
        while time.time() - start_time < time_limit:
            
            # Step 1: Remove redundant blocks
            removed_any = False
            for i in range(len(block_ids)):
                b_idx = block_ids[i]
                is_redundant = True
                for tr in pool_triples[b_idx]:
                    if cover_counts[tr] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    for tr in pool_triples[b_idx]:
                        cover_counts[tr] -= 1
                        if cover_counts[tr] == 1:
                            critical_count -= 1
                        elif cover_counts[tr] == 2:
                            critical_count += 1
                    selected[b_idx] = False
                    block_ids.pop(i)
                    removed_any = True
                    break # Restart loop as indices changed
                    
            if removed_any:
                if len(block_ids) < best_size:
                    best_size = len(block_ids)
                    best_block_ids = block_ids[:]
                continue # Continue to check for more removals
                
            # Step 2: Try Swaps
            # Pick a random selected block to swap out
            if not block_ids:
                break
                
            i = random.randint(0, len(block_ids) - 1)
            b_out_idx = block_ids[i]
            tr_out = pool_triples[b_out_idx]
            
            # Find a candidate block to swap in
            # It must cover all "critical" triples of b_out_idx
            # Critical triples of b_out_idx are those tr in tr_out where cover_counts[tr] == 1.
            
            critical_triples = [tr for tr in tr_out if cover_counts[tr] == 1]
            
            if not critical_triples:
                # Should have been removed in Step 1
                continue
            
            # We need a block b_in such that critical_triples is a subset of pool_triples[b_in]
            # And b_in is not currently selected.
            
            # To be efficient, we can filter candidates.
            # For each critical triple, the candidate MUST cover it.
            
            # Let's just iterate through unselected blocks and check.
            # n_pool is ~200. This is fast.
            
            best_swap_gain = -1
            best_b_in = -1
            
            for b_in_idx in range(n_pool):
                if selected[b_in_idx]:
                    continue
                    
                tr_in = pool_triples[b_in_idx]
                
                # Check if b_in covers all critical triples
                valid = True
                for tr in critical_triples:
                    if tr not in tr_in: # 'in' on list is O(N). N=35. Acceptable.
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate "gain" or "quality" of this swap.
                # We want to increase redundancy of other blocks.
                # Metric: How many triples does b_in cover that are currently count==2 (making them count==3)?
                # Or how many triples does b_in cover that are count==1 (making them count==2)?
                # Actually, we want to create new redundancies.
                # A triple becomes redundant if its count goes from 1 to 2.
                # So we want b_in to cover as many "critical" triples (count==1) as possible?
                # No, it MUST cover the critical ones of b_out.
                # If it covers OTHER critical triples (from other blocks), those become count==2,
                # potentially making those other blocks redundant!
                
                # Count how many "other" critical triples b_in covers.
                # "Other" critical triples are those with count==1, not in tr_out (or even in tr_out but not critical for b_out? No, if in tr_out and count==1, it's critical for b_out).
                
                # Let's just count total critical triples covered by b_in.
                # If b_in covers a critical triple, that triple's count becomes 2 (since b_out is removed, -1, b_in added, +1? No.
                # If tr is in b_out and b_in:
                #   Old count: C.
                #   New count: C - 1 + 1 = C. No change.
                # If tr is in b_in but NOT in b_out:
                #   Old count: C.
                #   New count: C + 1.
                #   If C==1, it becomes 2. This is GOOD (reduces critical count).
                
                # So, gain = number of triples tr such that:
                # 1. tr is in b_in
                # 2. tr is NOT in b_out
                # 3. cover_counts[tr] == 1
                
                gain = 0
                # To check "NOT in b_out" efficiently, we can use a set for tr_out?
                # tr_out is small (35).
                set_out = set(tr_out)
                
                for tr in tr_in:
                    if tr not in set_out and cover_counts[tr] == 1:
                        gain += 1
                        
                # Also, we might want to prefer blocks that are "similar" to b_out? No.
                # Just maximize gain.
                
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_b_in = b_in_idx
                    
            if best_b_in != -1:
                # Perform Swap
                # Remove b_out
                for tr in tr_out:
                    cover_counts[tr] -= 1
                    if cover_counts[tr] == 1:
                        critical_count -= 1
                    elif cover_counts[tr] == 2:
                        critical_count += 1
                selected[b_out_idx] = False
                
                # Add b_in
                tr_in = pool_triples[best_b_in]
                for tr in tr_in:
                    cover_counts[tr] += 1
                    if cover_counts[tr] == 1:
                        critical_count -= 1
                    elif cover_counts[tr] == 2:
                        critical_count += 1
                selected[best_b_in] = True
                
                # Update block_ids
                block_ids[i] = best_b_in
                
                # If this swap increased redundancy (gain > 0), we might find removals in next step.
                # We don't update best_size here because size is same.
                
            # If no good swap found, we might need to add a new block to the pool?
            # For now, we stick to the fixed pool.
            
        return best_block_ids

    # Main Execution
    
    start_time = time.time()
    total_time_limit = 110
    
    best_blocks = None
    best_count = float('inf')
    
    # Generate Pool
    # We need a pool large enough to contain the optimal solution.
    # Best known is 76. Let's aim for a pool of 150-200.
    pool_size = 200
    
    # Run multiple iterations with different seeds for pool generation and optimization
    seed = 42
    while time.time() - start_time < total_time_limit:
        
        # Generate Pool
        pool = generate_pool(pool_size, seed)
        
        # Optimize Selection
        remaining_time = total_time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        # Allocate time: 80% to optimization, 20% to next iteration?
        # Or just run optimization until time limit?
        # Let's run optimization for a fixed chunk, say 20s, then try next seed.
        opt_time = min(20, remaining_time)
        
        selected_ids = optimize_selection(pool, opt_time, seed + 1000)
        
        if selected_ids:
            current_count = len(selected_ids)
            if current_count < best_count:
                best_count = current_count
                best_blocks = [pool[i] for i in selected_ids]
                
        seed += 1
        
        if best_count <= 70:
            break
            
    if best_blocks is None:
        # Fallback: return greedy from a single run
        pool = generate_pool(100, seed)
        selected_ids = optimize_selection(pool, 10, seed + 2000)
        best_blocks = [pool[i] for i in selected_ids] if selected_ids else []
        
    return best_blocks

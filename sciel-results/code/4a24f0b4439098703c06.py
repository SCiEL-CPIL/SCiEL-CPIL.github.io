import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    # Deadline: 115 seconds to be safe within 120s limit
    deadline = start + 115.0
    
    # 1. Precomputation
    # Map all t-subsets to unique IDs
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)
        
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)
            
    # 2. Greedy Set Cover with Coverage Counts
    # coverage[ti] = number of selected blocks covering tset ti
    coverage = np.zeros(n_tsub, dtype=np.int32)
    selected = []
    
    # Initial gains: how many tsets each block covers (all are uncovered initially)
    # We can just use len(block_tsets[bi])
    # But we need to update gains as we add blocks.
    # gain[bi] = number of currently uncovered tsets covered by block bi
    
    # To speed up greedy, we can just pick max gain.
    # Updating gain for all blocks when adding one block is O(n_blocks * tsets_in_block).
    # n_blocks ~ 75k, tsets_in_block ~ 56. 75k * 56 ~ 4M ops per step.
    # Steps ~ 300. Total ~ 1.2B ops. Might be too slow in Python.
    
    # Alternative: Lazy evaluation or just recompute best block each time?
    # Recomputing best block: Iterate all blocks, count uncovered tsets.
    # 75k blocks * 56 tsets = 4.2M ops per step.
    # 300 steps * 4.2M = 1.26B ops. Still potentially slow in pure Python.
    
    # Let's try the standard greedy with gain array updates.
    # We need to be careful with performance.
    
    # Actually, for v=19, k=8, t=5, the numbers are small enough that
    # a well-optimized Python loop might pass, but let's try to optimize.
    
    # Optimization: Use numpy for gain?
    # gain[bi] = sum(1 for ti in block_tsets[bi] if coverage[ti] == 0)
    # This is hard to vectorize easily because block_tsets are lists of varying lengths (though all same length here).
    
    # Let's stick to a simple greedy but optimize the inner loop.
    # We will maintain a list of uncovered tsets.
    # When we add a block, we update coverage for its tsets.
    # We don't need to update gain for ALL blocks. We only need the best block.
    # So, in each step, we scan all blocks to find the one with max gain.
    # This is O(n_blocks * len(block_tsets)).
    
    # Let's estimate time.
    # 75582 blocks. 56 tsets each.
    # Loop: for bi in range(n_blocks): count = 0; for ti in block_tsets[bi]: if coverage[ti]==0: count+=1.
    # This is ~4 million simple operations. In Python, ~0.1-0.2s.
    # 300 iterations -> 30-60s. This is too slow for the 115s limit if we also do local search.
    
    # We need a faster greedy.
    # Idea: Maintain a list of "active" blocks?
    # Or, use a randomized greedy?
    # Or, pre-compute something?
    
    # Let's try a different approach for greedy:
    # Start with all tsets uncovered.
    # Pick a random uncovered tset.
    # Pick a random block covering it.
    # This is the "randomized greedy" or "heuristic".
    # It's not optimal but fast.
    # Then we do local search to fix it.
    
    # Let's try Randomized Greedy:
    # 1. uncovered_set = set(all tset indices)
    # 2. while uncovered_set:
    #    pick random ti in uncovered_set
    #    pick random bi from tset_to_blocks[ti]
    #    add bi to selected
    #    for tj in block_tsets[bi]:
    #        if tj in uncovered_set:
    #            uncovered_set.remove(tj)
    #            coverage[tj] += 1 # and increment for others?
    # 
    # This is O(n_selected * k_tsets). Very fast.
    # The quality will be lower, but local search should improve it significantly.
    
    # Let's implement Randomized Greedy.
    
    random.seed(42) # For reproducibility in testing, but evaluator might use different?
    # Actually, let's use a fixed seed for the first run to ensure it works, 
    # but maybe allow randomness for diversity? The task says "score is seed/value".
    # So deterministic is better if we know it works.
    
    selected = []
    coverage = np.zeros(n_tsub, dtype=np.int32)
    uncovered_indices = set(range(n_tsub))
    
    # To make it slightly better than pure random, we can pick the block
    # that covers the most uncovered tsets among those covering the random ti.
    # But that's O(k_tsets) per candidate.
    
    # Let's just do pure random first to get a valid cover fast.
    while uncovered_indices:
        if time.time() > deadline - 30.0:
            break
        # Pick a random uncovered tset
        ti = random.choice(list(uncovered_indices))
        # Pick a random block covering it
        candidates = tset_to_blocks[ti]
        bi = random.choice(candidates)
        
        # Add block
        selected.append(bi)
        
        # Update coverage and uncovered set
        for tj in block_tsets[bi]:
            coverage[tj] += 1
            if coverage[tj] == 1:
                # It was uncovered, now covered
                if tj in uncovered_indices:
                    uncovered_indices.remove(tj)
                    
    # 3. Deletion Pass (Reduce)
    def reduce_solution(sel, cov):
        # cov is a numpy array of counts
        # sel is a list of block indices
        changed = True
        while changed:
            changed = False
            # Shuffle to break symmetry
            random.shuffle(sel)
            for i in range(len(sel)):
                bi = sel[i]
                # Check if bi is redundant
                redundant = True
                for ti in block_tsets[bi]:
                    if cov[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove bi
                    for ti in block_tsets[bi]:
                        cov[ti] -= 1
                    sel.pop(i)
                    changed = True
                    break # Restart loop with new list
        return sel, cov

    selected, coverage = reduce_solution(selected, coverage)
    best_selected = list(selected)
    best_coverage = coverage.copy()
    best_size = len(selected)
    
    # 4. Local Search (Kick and Repair)
    # We will iterate until time runs out.
    # In each iteration:
    # 1. Save state
    # 2. Kick: remove a random block
    # 3. Repair: greedily cover the newly uncovered tsets
    # 4. Reduce: remove redundant blocks
    # 5. If size < best_size, update best
    
    # To speed up repair, we need to find the best block to add.
    # "Best" = covers most currently uncovered tsets.
    # Currently uncovered tsets are those with coverage[ti] == 0.
    
    while time.time() < deadline - 5.0:
        if not selected:
            break
            
        # Save state
        sel_copy = list(selected)
        cov_copy = coverage.copy()
        
        # Kick
        idx = random.randint(0, len(selected) - 1)
        bi_kick = selected[idx]
        selected.pop(idx)
        
        # Update coverage
        uncovered_tsets = []
        for ti in block_tsets[bi_kick]:
            coverage[ti] -= 1
            if coverage[ti] == 0:
                uncovered_tsets.append(ti)
                
        if not uncovered_tsets:
            # Block was redundant, we removed it. Good.
            # No repair needed.
            # Just reduce to be sure
            selected, coverage = reduce_solution(selected, coverage)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                best_coverage = coverage.copy()
            continue
            
        # Repair
        # We need to cover uncovered_tsets.
        # We will greedily pick blocks.
        # To find the best block, we check all blocks that cover any ti in uncovered_tsets.
        
        # Optimization: If uncovered_tsets is small, we can just check candidates.
        # Let's maintain a set of uncovered tsets for fast lookup.
        unc_set = set(uncovered_tsets)
        
        while unc_set:
            if time.time() > deadline - 5.0:
                break
                
            # Find best block
            best_bi = -1
            max_gain = 0
            
            # Get candidates
            candidates = set()
            for ti in unc_set:
                candidates.update(tset_to_blocks[ti])
                
            # Evaluate candidates
            # To avoid checking the same block multiple times, we use the set.
            # For each candidate, count how many tsets in unc_set it covers.
            for bj in candidates:
                # Count intersection
                gain = 0
                for ti in block_tsets[bj]:
                    if ti in unc_set:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_bi = bj
                    
            if best_bi == -1:
                # Should not happen if cover is possible
                break
                
            # Add best_bi
            selected.append(best_bi)
            for ti in block_tsets[best_bi]:
                coverage[ti] += 1
                if coverage[ti] == 1 and ti in unc_set:
                    unc_set.remove(ti)
                    
        # After repair, reduce again
        selected, coverage = reduce_solution(selected, coverage)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            best_coverage = coverage.copy()
        else:
            # Revert
            selected = sel_copy
            coverage = cov_copy
            
    # Return best solution
    return [list(all_blocks[bi]) for bi in best_selected]

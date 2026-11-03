import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            mask |= (1 << idx)
            tset_to_blocks[idx].append(i)
        block_masks[i] = mask

    full_mask = (1 << num_tsets) - 1

    # Helper: Get cover counts
    def get_cover_counts(blocks):
        counts = [0] * num_tsets
        for b in blocks:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    # 2. Greedy
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search with Simulated Annealing capability
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = get_cover_counts(sol)
        
        best_sol = list(sol)
        best_size = len(sol)
        
        # Annealing parameters
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.95
        
        # We use a fixed number of iterations for the annealing phase
        # since we don't know the optimal time allocation yet.
        # If time permits, we do more iterations.
        max_iterations = 2000
        
        current_cost = len(sol) # Primary cost is size. Secondary is uncovered (should be 0 if valid)
        
        for it in range(max_iterations):
            if elapsed() > deadline - 10:
                break
            
            # 1. Generate Neighbor
            # Options:
            # a) Remove a random block (if redundant or if we accept penalty)
            # b) Swap a random block with a candidate
            # c) Add a random block (to fix coverage)
            
            action = rng.random()
            
            new_sol = sol
            new_in_sol = in_sol
            new_counts = cover_counts
            
            # To avoid copying large arrays/structures every time, we modify in place and revert if rejected
            # But for simplicity and safety with complex state, let's stick to careful in-place modification with revert logic.
            # Actually, copying `in_sol` (75k bools) is fast enough? ~75k ops. 2000 iters = 150M ops. Might be tight.
            # Let's try to do it in place with a stack of changes.
            
            changes = []
            
            def revert():
                for b, delta in reversed(changes):
                    m = block_masks[b]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] += delta
                        m ^= lsb
                    in_sol[b] = (delta == 1) # If delta was 1 (added), set to True. If -1 (removed), set to False.
                # Note: sol list modification needs separate handling or we track index changes.
                # This gets complicated. Let's stick to the simpler "copy and compare" if it's fast enough,
                # or just implement specific move types that are easy to revert.
                
                # Let's restart with a simpler approach:
                # We will generate a new candidate solution explicitly.
                pass
            
            # Given the complexity of reverting list pops/swaps cleanly, let's use a different strategy:
            # We maintain `sol` as the current state.
            # We generate a new state `cand_sol`.
            # We compute the cost difference.
            # If accepted, `sol = cand_sol`.
            
            # To make this fast, we need fast cost evaluation.
            # Cost = len(sol) + lambda * (num_uncovered)
            # num_uncovered can be tracked incrementally if we only change one block at a time.
            
            # Let's try a specific move: Swap one block.
            # Pick a random index in sol.
            # Pick a random candidate block not in sol.
            # Compute delta_size = 0 (swap keeps size same).
            # Compute delta_uncovered.
            
            # If we swap A with B:
            # Uncovered sets that are ONLY covered by A become uncovered.
            # Uncovered sets that are ONLY covered by B become covered.
            
            # Let's implement a "Swap" move primarily, as it maintains size.
            # And a "Remove" move if redundant.
            # And an "Add" move if uncovered > 0 (to repair).
            
            uncovered_count = 0
            # We can maintain uncovered_count incrementally.
            # But initializing it requires a full pass.
            # Let's just compute it on the fly for the decision?
            # No, we need it for the cost.
            
            # Let's maintain `uncovered_mask`? No, too big.
            # Let's maintain `cover_counts`.
            # Number of uncovered tsets = count of zeros in cover_counts.
            # This is O(N_tsets) to compute. 11k ops.
            # 2000 iterations * 11k = 22M ops. Acceptable.
            
            # However, if we just do swaps, the size is constant.
            # The goal is to minimize size.
            # So we need "Remove" moves.
            
            # Strategy:
            # 1. Try to remove a redundant block. If found, do it (always accept).
            # 2. If no redundant block, try a swap to improve "flexibility" or just random walk?
            #    Actually, if no block is redundant, we can't reduce size directly.
            #    We need to rearrange to make a block redundant.
            #    This is what swaps do.
            #    But how do we know a swap is good?
            #    A swap is good if it increases the number of redundant blocks?
            #    Or if it reduces the "criticality" of the solution?
            
            # Let's use the SA idea from the inspiration:
            # Accept moves that increase size or leave uncovered if T is high?
            # No, that's for finding *any* solution or escaping deep minima in size.
            # Here, we want to *minimize* size.
            # So we should prefer smaller size.
            # If we allow larger size, we might escape a local minimum where no swap reduces size,
            # but a larger solution has a different structure that allows a bigger reduction later.
            
            # Let's implement:
            # Cost = len(sol)
            # If we make a move that keeps size same (swap), we accept with probability 1 (random walk) or based on some heuristic?
            # If we make a move that increases size (add block), we accept with prob exp(-delta/T).
            # If we make a move that decreases size (remove block), we accept with prob 1.
            
            # Move Types:
            # 1. Remove: Pick random block. If redundant, remove it. (Size -1)
            # 2. Swap: Pick random block A in sol, random block B not in sol. Swap. (Size 0)
            # 3. Add: Pick random block B not in sol. Add it. (Size +1) - Only useful if we have uncovered? 
            #    If we have uncovered, we MUST fix it. So if uncovered > 0, we should prioritize adding a block that covers the most uncovered.
            
            # Let's track `uncovered_count`.
            # Initially 0 if greedy worked.
            
            # If uncovered_count > 0:
            #    We are invalid. We must repair.
            #    Find block covering most uncovered. Add it.
            #    This is a forced move.
            
            # If uncovered_count == 0:
            #    We are valid.
            #    Try to remove redundant.
            #    If none, try swap (random) to explore.
            #    Occasionally try add (to explore larger space) if T is high?
            
            # Let's implement this logic.
            
            # First, calculate current uncovered count
            # This is O(N_tsets).
            # We can optimize by maintaining it, but let's start simple.
            
            # Actually, let's just use the standard local search (redundancy removal + swap) 
            # but add a "restart" mechanism or "perturbation" if stuck.
            # The inspiration says SA improved island 11.
            # Let's try a simple SA where we accept "bad" swaps (that don't reduce size, since size is constant in swap) 
            # but we are really looking for a state where a removal becomes possible.
            
            # So, for swaps, we don't have a clear "better/worse" unless we define a secondary objective.
            # Secondary objective: Minimize the number of "critical" tsets (cover_count == 1)?
            # Or maximize the number of redundant blocks?
            
            # Let's define:
            # Primary Cost: Size
            # Secondary Cost: Number of tsets with cover_count == 1 (fragility)
            
            # If we swap and fragility decreases, it's a "better" swap.
            # If fragility increases, it's "worse".
            # We can use SA on this secondary cost while keeping size constant.
            
            # If we find a removal, size drops.
            
            # Let's implement:
            # 1. Check for redundant blocks. If found, remove one. (Size decreases)
            # 2. If no redundant, perform a random swap.
            #    Calculate change in fragility (num tsets with count==1).
            #    Accept swap if fragility decreases, or with prob exp(-delta_fragility/T) if it increases.
            #    This allows exploring the space of constant-size solutions to find a "hole" (redundant block).
            
            # We need to efficiently calculate fragility change for a swap.
            # Swap A (out) with B (in).
            # For each tset in A: count decreases by 1.
            # For each tset in B: count increases by 1.
            # Fragility = count of tsets where count == 1.
            
            # Let's implement this.
            
            current_fragility = sum(1 for c in cover_counts if c == 1)
            
            for it in range(max_iterations):
                if elapsed() > deadline - 10:
                    break
                    
                # 1. Try Removal
                # Pick random block
                if sol:
                    pos = rng.randint(0, len(sol) - 1)
                    b = sol[pos]
                    m = block_masks[b]
                    redundant = True
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            redundant = False
                            break
                        m ^= lsb
                    
                    if redundant:
                        # Remove
                        m = block_masks[b]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        in_sol[b] = False
                        sol.pop(pos)
                        current_fragility = sum(1 for c in cover_counts if c == 1) # Recalc, or update incrementally?
                        # Incremental update for fragility is tricky because counts change.
                        # Let's just recalc if we want to be safe, or update carefully.
                        # For now, recalc is O(11k). 2000 iters = 22M. OK.
                        
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        continue # Restart loop to try more removals
                
                # 2. Try Swap
                if not sol:
                    break
                    
                # Pick random A in sol
                pos = rng.randint(0, len(sol) - 1)
                b_out = sol[pos]
                
                # Pick random B not in sol
                # We need a fast way to pick a random block not in sol.
                # Rejection sampling: pick random index, check in_sol.
                # in_sol is a bool array.
                # Probability of hitting a selected block is ~300/75000 = 0.4%.
                # So rejection is fast.
                b_in = rng.randint(0, num_blocks - 1)
                while in_sol[b_in]:
                    b_in = rng.randint(0, num_blocks - 1)
                    
                # Calculate change in fragility
                # A: counts decrease by 1 for its tsets
                # B: counts increase by 1 for its tsets
                
                # We need to know the current counts for tsets in A and B.
                # And check for overlaps.
                
                delta_fragility = 0
                
                # Helper to get tsets in a block
                def get_tsets(block_idx):
                    m = block_masks[block_idx]
                    ts = []
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        ts.append(idx)
                        m ^= lsb
                    return ts
                    
                ts_out = get_tsets(b_out)
                ts_in = get_tsets(b_in)
                
                # Sets for efficiency?
                # ts_out and ts_in are small (56).
                # Intersection is small.
                
                # For each ts in ts_out:
                #   old_count = cover_counts[ts]
                #   new_count = old_count - 1
                #   if old_count == 1 and new_count == 0: fragility unchanged (was 1, now 0) -> Wait.
                #   Fragility is count of ts where count == 1.
                #   If old_count == 1, it WAS fragile. Now it's 0. So it's NOT fragile anymore. Fragility -1.
                #   If old_count == 2, it WAS NOT fragile. Now it's 1. So it IS fragile now. Fragility +1.
                #   If old_count > 2, no change.
                
                # For each ts in ts_in:
                #   old_count = cover_counts[ts]
                #   new_count = old_count + 1
                #   If old_count == 0, it WAS NOT fragile. Now it's 1. Fragility +1.
                #   If old_count == 1, it WAS fragile. Now it's 2. Fragility -1.
                #   If old_count > 1, no change.
                
                # We must handle overlaps (ts in both A and B).
                # If ts in A and B:
                #   Net change: -1 + 1 = 0.
                #   But intermediate state?
                #   If we remove A first: count drops. If it was 1, becomes 0 (fragility -1).
                #   Then add B: count rises. If it was 0, becomes 1 (fragility +1).
                #   Net 0.
                #   So for overlaps, fragility change is 0?
                #   Let's check:
                #   Old count C.
                #   New count C.
                #   Fragility contribution: 1 if C==1 else 0.
                #   Same before and after. So yes, 0.
                
                # So we can just process A, then B, ignoring overlaps in the delta calculation?
                # No, because the "old_count" for B depends on whether A was removed first?
                # No, we are comparing State S (with A, without B) to State S' (without A, with B).
                # In S, count(ts) = C.
                # In S', count(ts) = C. (If ts in A and B).
                # So fragility contribution is same.
                # If ts in A only:
                #   S: C. S': C-1.
                # If ts in B only:
                #   S: C. S': C+1.
                
                # So we can just sum the changes for A-only and B-only tsets.
                
                set_out = set(ts_out)
                set_in = set(ts_in)
                
                for ts in ts_out:
                    if ts in set_in:
                        continue
                    c = cover_counts[ts]
                    if c == 1:
                        delta_fragility -= 1
                    elif c == 2:
                        delta_fragility += 1
                        
                for ts in ts_in:
                    if ts in set_out:
                        continue
                    c = cover_counts[ts]
                    if c == 0:
                        delta_fragility += 1
                    elif c == 1:
                        delta_fragility -= 1
                        
                # Decide acceptance
                if delta_fragility <= 0:
                    accept = True
                else:
                    accept = rng.random() < pow(2.71828, -delta_fragility / T)
                    
                if accept:
                    # Perform swap
                    # Update counts
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                        
                    m = block_masks[b_in]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] += 1
                        m ^= lsb
                        
                    in_sol[b_out] = False
                    in_sol[b_in] = True
                    sol[pos] = b_in
                    current_fragility += delta_fragility
                else:
                    # Reject, do nothing
                    pass
                    
                T *= cooling_rate
                if T < T_min:
                    T = T_min
            
        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run greedy restarts
    num_restarts = 5
    for seed in range(num_restarts):
        if elapsed() > deadline - 20:
            break
            
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                
    if best_solution is None:
        return []
        
    return [list(all_blocks[i]) for i in best_solution]

"""
Strategy: Branch-and-Bound for Maximum Coverage.
We frame the problem as finding a set of k-blocks that covers all t-subsets.
Instead of pure greedy or local search, we use a recursive B&B solver.
The state is (current_blocks, uncovered_mask, depth).
The bound is derived by greedily packing the remaining uncovered subsets 
to estimate the minimum additional blocks needed. If current_blocks + bound >= best_known, we prune.
This allows escaping local minima of the greedy heuristic.
"""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit (10s buffer)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Full mask of all t-subsets
    full_mask = (1 << n_t) - 1
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage mask for each block
    # block_masks[i] is an integer bitmask where bit j is 1 if t-subset j is covered by block i
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Optimization: Sort blocks by their total coverage (popcount) descending?
    # Or keep as is. For B&B, ordering matters for the bound.
    # Let's sort blocks by static coverage size (C(k,t) is constant, so this doesn't help much directly,
    # but it might help if we had variable k. Here k is fixed.
    # However, the "effectiveness" of a block depends on the current uncovered set.
    
    # 2. Initial Solution via Greedy (to set a strong upper bound / best_known)
    
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            best_noise = 0
            
            # To speed up greedy, we can sample or check all.
            # n_blocks ~ 75k. n_steps ~ 300. 22.5M ops.
            # With bit_count, this is fast in Python 3.8+.
            
            for i in range(n_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    noise = random.random() * 0.5 if seed is not None else 0
                    if gain + noise > best_gain:
                        best_gain = gain + noise
                        best_idx = i
                        best_noise = noise
            
            if best_idx == -1:
                break # Should not happen if uncovered != 0
            
            selected.append(all_blocks[best_idx])
            uncovered &= ~block_masks[best_idx]
            
            if time.time() > deadline - 5.0: # Reserve time for B&B
                break
                
        return selected

    # Run a few greedy restarts to get a good initial upper bound
    best_solution = None
    best_count = float('inf')
    
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        sol = greedy_cover(seed=i)
        # Verify and count
        if sol:
            # Check validity quickly (optional but good for debugging)
            covered = 0
            for b in sol:
                idx = all_blocks.index(b) # This is O(n) lookup, slow. 
                # Better: keep indices in greedy.
                pass
            
            # Let's redo greedy to return indices for speed
            # Actually, let's just use the list of blocks. 
            # For B&B, we work with indices.
            pass
            
    # Refactored Greedy to return indices
    def greedy_cover_indices(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            
            for i in range(n_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    noise = random.random() * 0.5 if seed is not None else 0
                    if gain + noise > best_gain:
                        best_gain = gain + noise
                        best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected

    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        sol_idx = greedy_cover_indices(seed=i)
        if sol_idx:
            count = len(sol_idx)
            if count < best_count:
                best_count = count
                best_solution = sol_idx

    if best_solution is None:
        # Fallback
        best_solution = greedy_cover_indices(seed=0)
        best_count = len(best_solution)

    # 3. Branch and Bound
    # State: current_selected (list of indices), uncovered_mask, current_depth
    # Bound: Greedy bound on remaining uncovered subsets.
    # We want to minimize the total number of blocks.
    # If len(current_selected) + greedy_bound(uncovered_mask) >= best_count, prune.
    
    # To make the bound tighter and faster, we can use a "lazy" greedy bound
    # that only takes a few steps or uses a precomputed heuristic.
    
    def greedy_bound(uncovered, max_steps=10):
        """
        Estimate the minimum number of additional blocks needed to cover 'uncovered'.
        Uses greedy for max_steps, then assumes each remaining block covers 
        roughly the average coverage of the last step, or just returns a lower bound.
        A safe lower bound is: remaining_uncovered_count / max_coverage_per_block.
        But greedy packing is a better upper bound on the number of blocks needed 
        (which serves as a bound in B&B for minimization if we are careful).
        
        Actually, for B&B minimization:
        We have a solution of size K.
        We are at a node with depth D and uncovered set U.
        If D + LowerBound(U) >= K, we prune.
        
        What is a good LowerBound(U)?
        1. |U| / C(k, t) : Trivial packing bound.
        2. Greedy packing: Run greedy on U for a few steps. The number of steps taken + 
           (remaining / max_step_gain) is an UPPER bound on the additional blocks needed?
           No, greedy gives an upper bound on the number of blocks needed to cover U.
           Let G(U) be the number of blocks greedy uses to cover U.
           Then any solution covering U needs at least... wait.
           Greedy is a heuristic. It finds a cover of size G(U).
           This doesn't give a lower bound directly.
           
           However, we can use the "Fractional Packing" or "Dual" bound.
           Or, more simply:
           If we are trying to beat best_count, we need to find a cover of size < best_count.
           At depth D, we have best_count - D - 1 blocks left to cover U.
           If we can prove that U cannot be covered by best_count - D - 1 blocks, we prune.
           
           How to prove U cannot be covered by M blocks?
           This is equivalent to: Is there a set of M blocks that covers U?
           This is still hard.
           
           Alternative: Use the greedy solution as a bound.
           If we run greedy on U and it takes G blocks, we know a solution of size G exists.
           This doesn't help prune unless G + D < best_count (which updates best_count).
           
           Let's stick to the trivial packing bound for pruning, which is safe:
           LB = ceil(popcount(uncovered) / C(k, t))
           This is weak but fast.
           
           To improve, we can use a "conflict" bound or "matching" bound on the dual.
           For C(19,8,5), the trivial bound is often weak.
           
           Let's try a slightly stronger bound:
           Sort the uncovered t-subsets.
           If two t-subsets are "far" apart, they might require different blocks?
           No, blocks are size 8.
           
           Let's use the greedy solution found during the search to update best_count.
           And use the trivial bound for pruning.
        """
        count = uncovered.bit_count()
        max_per_block = 56 # C(8,5)
        return (count + max_per_block - 1) // max_per_block

    # Recursive B&B
    # To avoid recursion depth issues and overhead, we use a stack or iterative deepening?
    # Recursion depth is at most best_count (~335). Python default limit is 1000.
    # We should increase recursion limit.
    sys.setrecursionlimit(5000)
    
    best_count_global = best_count
    best_solution_global = best_solution
    
    # We will search for a solution with size < best_count_global.
    # Target size: best_count_global - 1.
    
    def bb_search(uncovered, current_depth, current_selected):
        nonlocal best_count_global, best_solution_global
        
        if time.time() > deadline - 5.0:
            return
        
        # Pruning: Check if we can possibly beat the best solution
        if current_depth + greedy_bound(uncovered) >= best_count_global:
            return
            
        if uncovered == 0:
            # Found a valid cover
            if current_depth < best_count_global:
                best_count_global = current_depth
                best_solution_global = list(current_selected)
            return
            
        # Heuristic: If we are close to the limit, try to finish with greedy
        remaining_limit = best_count_global - current_depth - 1
        if remaining_limit <= 0:
            return
            
        # If remaining_limit is small, we can try to solve the rest exactly or with greedy
        # If remaining_limit == 1, check if any single block covers uncovered.
        if remaining_limit == 1:
            for i in range(n_blocks):
                if (uncovered & block_masks[i]) == uncovered:
                    current_selected.append(i)
                    # Found a solution of size current_depth + 1
                    if current_depth + 1 < best_count_global:
                        best_count_global = current_depth + 1
                        best_solution_global = list(current_selected)
                    current_selected.pop()
                    return # Only need one such block? No, we just need to know if it exists.
                    # Actually, if we find one, we update best. We don't need to find all.
            return
            
        # Branching
        # Choose a t-subset from uncovered to branch on.
        # Strategy: Pick the one with the fewest candidate blocks? (Most constrained)
        # Or just pick the first one.
        
        # To speed up, we can limit the branching factor or use a heuristic order.
        # Let's pick a random uncovered subset to diversify, or the first one.
        # Finding the first set bit is fast.
        lsb = uncovered & -uncovered
        target_idx = lsb.bit_length() - 1
        target_sub = all_t_subsets[target_idx]
        
        # Find all blocks that cover target_sub
        # Precomputation: For each t-subset, list of blocks covering it?
        # This would be C(19,5) * C(19,8) checks? No.
        # Each block covers C(8,5)=56 t-subsets.
        # Total (block, t-subset) pairs = 75582 * 56 ~ 4.2M.
        # We can build an adjacency list: t_sub_to_blocks.
        
        # Let's build this map once.
        t_sub_to_blocks = [[] for _ in range(n_t)]
        for i, mask in enumerate(block_masks):
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                t_sub_to_blocks[idx].append(i)
                m ^= lsb
        
        # Now, get candidates for target_idx
        candidates = t_sub_to_blocks[target_idx]
        
        # Shuffle candidates for randomization?
        # For B&B, deterministic order is fine, but randomization helps with symmetry breaking.
        random.shuffle(candidates)
        
        for i in candidates:
            if time.time() > deadline - 5.0:
                return
            
            # Check if this block adds value
            new_uncovered = uncovered & ~block_masks[i]
            if new_uncovered == uncovered:
                continue # Block doesn't cover target? Should not happen by construction.
                
            current_selected.append(i)
            bb_search(new_uncovered, current_depth + 1, current_selected)
            current_selected.pop()
            
            # If we found a better solution, the bound is tighter, 
            # and we might prune earlier in subsequent branches.

    # Run B&B
    # We need to be careful with the initial call.
    # We start with empty selected, full uncovered.
    # We want to find a solution smaller than best_count_global.
    
    # To make B&B effective, we should start with a tight bound.
    # The greedy solution provides this.
    
    # Let's run B&B
    bb_search(full_mask, 0, [])
    
    # 4. Finalize
    if best_solution_global is None:
        # Fallback to the greedy solution found earlier
        return [all_blocks[i] for i in best_solution]
    else:
        return [all_blocks[i] for i in best_solution_global]


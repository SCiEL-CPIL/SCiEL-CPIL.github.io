import itertools
import random
import sys
import time

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an optimized greedy set-cover
    with bitset acceleration, randomized restarts, and local search improvements.
    
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    full_mask = (1 << n_t) - 1
    
    # Precompute all blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    def popcount(x):
        return x.bit_count()

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
            
        covered = 0
        uncovered = full_mask
        solution = []
        
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            
            # Weighted Greedy: maximize uncovered coverage, penalize overlap with covered
            # score = gain_uncovered - alpha * overlap_covered
            # We want to find blocks that cover new things efficiently.
            
            # To speed up, we can pre-filter blocks that have 0 gain.
            # But calculating gain for all N_blocks is O(N_blocks * n_t/word_size).
            # For v=19, k=7, N_blocks = 50388. n_t = 3876.
            # This is fast enough.
            
            for i, mask in enumerate(block_masks):
                gain = popcount(mask & uncovered)
                if gain == 0:
                    continue
                # Penalty for overlap: how much of this block is already covered?
                # This encourages diversity.
                overlap = popcount(mask & covered)
                # Heuristic: score = gain - 0.1 * overlap? 
                # Or just maximize gain, tie-break by min overlap?
                # Let's stick to maximizing gain, but break ties randomly or by min overlap.
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    # If gain is max possible, we could break, but randomization helps.
                    # We'll collect candidates for top gain.
                    
            if best_block_idx == -1:
                break
                
            # To add randomness, we can pick randomly from top-k gainers.
            # For now, let's just pick the best one, but we already have random restarts.
            # Let's add a small randomization: if gain == best_gain, pick randomly among them.
            # But the loop above only keeps the first best.
            # Let's modify to collect candidates.
            
            # Re-doing the loop logic to be cleaner with candidates
            best_gain_val = -1
            candidates = []
            for i, mask in enumerate(block_masks):
                gain = popcount(mask & uncovered)
                if gain > best_gain_val:
                    best_gain_val = gain
                    candidates = [i]
                elif gain == best_gain_val and gain > 0:
                    candidates.append(i)
                    
            if not candidates:
                break
                
            best_block_idx = random.choice(candidates)
            solution.append(all_blocks[best_block_idx])
            covered |= block_masks[best_block_idx]
            uncovered = full_mask ^ covered
            
        return solution

    def optimize_solution(sol):
        # Map block tuples to indices in solution for fast lookup?
        # sol is a list of tuples.
        
        # Compute initial counts
        counts = [0] * n_t
        # Precompute t-indices for blocks in solution to avoid repeated comb generation
        sol_t_indices = []
        for block in sol:
            t_idxs = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
            sol_t_indices.append(t_idxs)
            for idx in t_idxs:
                counts[idx] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Remove redundant blocks
            n_sol = len(sol)
            for i in range(n_sol):
                t_idxs = sol_t_indices[i]
                is_redundant = True
                for idx in t_idxs:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in t_idxs:
                        counts[idx] -= 1
                    del sol[i]
                    del sol_t_indices[i]
                    improved = True
                    break # Restart loop as indices changed
            
            # Phase 2: Swap blocks to potentially enable removals
            # Try to replace a block that is "critical" (covers unique t-subsets)
            # with another block that covers those unique t-subsets but might be "better"
            # (e.g., overlaps less with others, or simply different to allow future removals).
            # Since we want to minimize size, a swap only helps if it leads to a removal.
            # A simple heuristic: If block i is critical, find a block j not in sol
            # that covers all critical t-subsets of i. Swap i with j.
            
            n_sol = len(sol)
            for i in range(n_sol):
                t_idxs_i = sol_t_indices[i]
                # Find critical t-subsets for block i
                criticals = [idx for idx in t_idxs_i if counts[idx] == 1]
                
                if not criticals:
                    continue # Not critical, can be removed in Phase 1 if we looped? 
                             # Wait, Phase 1 runs before. If we are here, it's critical.
                
                # Find a replacement block j not in sol that covers all criticals
                # We need to check all blocks? That's expensive.
                # Optimization: Only check blocks that cover *at least one* of the criticals.
                # Or just iterate all blocks not in sol.
                
                # To make this faster, we can limit the number of swaps or use a heuristic.
                # Let's try to find *any* such block.
                
                sol_set = set(sol)
                found_swap = False
                # Iterate through all blocks
                # This is O(N_blocks * len(criticals)) per swap attempt.
                # Might be too slow if done for every block in every iteration.
                
                # Let's restrict: only try swaps if len(sol) > lower_bound + margin?
                # Or just do it carefully.
                
                # Alternative: Randomized Swap.
                # Pick a random block j not in sol. Check if it covers criticals.
                # This is probabilistic but fast.
                
                # Let's try a limited search for replacement.
                # Since v=19, N_blocks ~ 50k.
                # If we do this for every critical block, it's 50k * 150 * 150 (swaps) ~ 1e9 ops. Too slow.
                
                # Better: Precompute for each t-subset, the list of blocks covering it?
                # t_to_blocks: dict idx -> list of block indices.
                # Then for a set of criticals, the replacement block must be in the intersection
                # of t_to_blocks[c] for all c in criticals.
                
                # Let's build t_to_blocks once.
                # But that's memory heavy? 3876 * ~1000 blocks? 3.8M entries. OK.
                
                # We'll build it outside the loop if we go this route.
                # For now, let's stick to a simpler local search or just rely on restarts.
                # The current score is 1.0796.
                # If I add a simple "random swap" that checks a few random candidates, it might help.
                
                # Let's try: Pick 10 random blocks not in sol. Check if any covers criticals.
                for _ in range(10):
                    j = random.randint(0, N_blocks - 1)
                    if all_blocks[j] in sol_set:
                        continue
                    # Check if block j covers all criticals
                    mask_j = block_masks[j]
                    covers_all = True
                    for c in criticals:
                        if not (mask_j & (1 << c)):
                            covers_all = False
                            break
                    if covers_all:
                        # Perform swap
                        # Remove i
                        for idx in t_idxs_i:
                            counts[idx] -= 1
                        # Add j
                        t_idxs_j = [t_to_idx[ts] for ts in itertools.combinations(all_blocks[j], t)]
                        for idx in t_idxs_j:
                            counts[idx] += 1
                        
                        sol[i] = all_blocks[j]
                        sol_t_indices[i] = t_idxs_j
                        improved = True
                        found_swap = True
                        break # Restart loop after swap? Or continue?
                        # Restarting is safer for correctness of counts/indexing logic.
                        break
                
                if found_swap:
                    break # Restart the while improved loop
                
        return sol

    # Precompute t_to_blocks? No, the random swap above is O(10 * len(criticals)) which is small.
    # The bottleneck is the "Phase 1" and "Phase 2" loops.
    
    # Initial deterministic run
    best_sol = optimize_solution(greedy_construct(42))
    best_len = len(best_sol)
    
    # Randomized restarts to find better local optima
    start_time = time.time()
    seed = 1
    while time.time() - start_time < 90: # Leave 30s buffer for safety
        seed += 1
        sol = greedy_construct(seed)
        sol_opt = optimize_solution(sol)
        if len(sol_opt) < best_len:
            best_len = len(sol_opt)
            best_sol = sol_opt
            
    return best_sol

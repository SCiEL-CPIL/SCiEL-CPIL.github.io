import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 125  # Safety margin
    rng = random.Random(42)
    
    # Precompute all t-subsets and k-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Build incidence matrix: block_tsubs[i] is array of t-subset indices covered by block i
    block_tsubs = []
    for block in all_ksubs:
        idxs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                idxs.append(idx)
        block_tsubs.append(np.array(idxs, dtype=np.int32))
        
    # Convert to a list of lists for faster iteration in some contexts, or keep as arrays
    # For SA, we need fast add/remove.
    # Let's use a list of lists of ints for block_tsubs to allow easy numpy ops on the whole array if needed,
    # but actually, iterating over the small list (size 35) is fast in Python if we avoid numpy overhead per op.
    # However, updating the global coverage vector (size 3876) is the bottleneck.
    # We can use numpy for the global vector update.
    
    # Initial Greedy Construction
    # We want a good starting point.
    # Use a simple greedy: pick block covering most uncovered.
    # To make it fast, we'll use a heap or just scan a subset.
    # Given 50k blocks, scanning all 50k * 35 is ~1.7M ops per step. 
    # If we need ~150 blocks, that's 250M ops. Too slow in pure Python loops.
    
    # Optimized Greedy:
    # Maintain an array 'uncovered_count' or just 'covered'.
    # Actually, let's use the "dynamic gain" idea from inspiration 3 but simplified.
    # Or just a faster greedy:
    # 1. Randomly select a subset of blocks? No, we need global view.
    # 2. Use NumPy to compute gains?
    #    Let M be n_ksubs x n_tsubs incidence matrix (sparse).
    #    We can't build full dense (50k x 3.8k is huge).
    
    # Let's stick to a Python loop but optimize it.
    # Greedy step:
    # 1. Identify uncovered t-subsets.
    # 2. For each candidate block, count how many uncovered t-subsets it covers.
    # 3. Pick best.
    
    # To speed up Greedy:
    # We can pre-compute for each t-subset, the list of blocks covering it.
    tsub_blocks = defaultdict(list)
    for b_idx, tsub_arr in enumerate(block_tsubs):
        for t_idx in tsub_arr:
            tsub_blocks[t_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access
    tsub_blocks_arr = [np.array(lst, dtype=np.int32) for lst in tsub_blocks.values()]
    
    def greedy_initial():
        selected = []
        covered = np.zeros(n_tsubs, dtype=bool)
        uncovered_count = n_tsubs
        
        while uncovered_count > 0:
            # Find the block that covers the most currently uncovered t-subsets
            best_gain = -1
            best_block = -1
            
            # Optimization: Instead of checking all 50k blocks, 
            # we can check blocks that cover at least one uncovered t-subset.
            # But finding those is also expensive.
            
            # Alternative: Just sample a few thousand blocks? 
            # No, greedy needs global optimum step.
            
            # Let's try to speed up the inner loop.
            # We maintain a list of uncovered t-subsets.
            # For each block, gain = sum(covered[t] == False for t in block_tsubs[b])
            
            # Vectorized approach?
            # If we had a dense matrix, we could do it.
            # Since we don't, we loop.
            
            # Heuristic: Only consider blocks that cover a random uncovered t-subset?
            # This is randomized greedy.
            
            # Let's pick a random uncovered t-subset, and consider all blocks covering it.
            # This is a valid heuristic for covering problems (Greedy Set Cover variant).
            # It reduces candidates from 50k to ~50k * (35/3876) ~ 450 blocks.
            
            # Find an uncovered t-subset
            # We need to efficiently find an uncovered one.
            # covered is a bool array.
            # np.where(~covered)[0] gives all uncovered.
            # We can pick a random one from there.
            
            unc_idx = np.where(~covered)[0]
            if len(unc_idx) == 0:
                break
                
            # Pick a random uncovered t-subset
            target_t = unc_idx[rng.randint(0, len(unc_idx)-1)]
            
            # Get all blocks covering this t-subset
            candidates = tsub_blocks_arr[target_t]
            
            # Evaluate gain for each candidate
            for b_idx in candidates:
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if not covered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                elif gain == best_gain and best_block != -1:
                    # Tie breaking: random
                    if rng.random() < 0.5:
                        best_block = b_idx
            
            if best_block != -1:
                selected.append(best_block)
                # Update covered
                for t_idx in block_tsubs[best_block]:
                    if not covered[t_idx]:
                        covered[t_idx] = True
                        uncovered_count -= 1
            else:
                # Should not happen if problem is solvable
                break
                
        return selected, covered

    init_selected, init_covered = greedy_initial()
    
    # Convert covered to int array for SA
    cov = init_covered.astype(np.int32)
    # Wait, SA needs counts to know if removal breaks feasibility.
    # Greedy above only tracks boolean. 
    # Let's re-run greedy to get counts? Or just rebuild counts.
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in init_selected:
        for t_idx in block_tsubs[b_idx]:
            cov[t_idx] += 1
            
    best_selected = set(init_selected)
    best_count = len(best_selected)
    
    # Simulated Annealing
    # State: set of selected blocks
    # Objective: Minimize count, subject to all cov >= 1.
    # If infeasible, penalty = sum(cov[t] < 1 for t) * large_weight
    
    # Parameters
    T = 100.0
    T_min = 1.0
    alpha = 0.999
    max_iter = 50000
    time_limit = deadline - 10
    
    current_selected = set(init_selected)
    current_cov = cov.copy()
    current_count = len(current_selected)
    current_penalty = np.sum(current_cov < 1)
    
    best_selected_sa = set(current_selected)
    best_count_sa = current_count
    
    # Helper to calculate objective
    def get_obj(count, penalty):
        return count + penalty * 1000
    
    current_obj = get_obj(current_count, current_penalty)
    best_obj = current_obj
    
    # Precompute for fast swaps?
    # We need to update cov efficiently.
    
    while time.time() < time_limit and T > T_min:
        # Move 1: Remove a random block
        if current_selected and rng.random() < 0.5:
            b_idx = rng.choice(list(current_selected))
            # Calculate delta
            # Removing b_idx decreases cov for its t-subsets.
            # If any cov becomes 0, penalty increases.
            # Count stays same - 1.
            
            new_cov = current_cov.copy()
            for t_idx in block_tsubs[b_idx]:
                new_cov[t_idx] -= 1
                if new_cov[t_idx] == 0:
                    # Was 1, now 0. Penalty +1
                    pass
                elif new_cov[t_idx] < 0:
                    # Should not happen if cov was correct
                    pass
            
            new_penalty = np.sum(new_cov < 1)
            new_count = current_count - 1
            new_obj = get_obj(new_count, new_penalty)
            
            delta = new_obj - current_obj
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                current_selected.remove(b_idx)
                current_cov = new_cov
                current_count = new_count
                current_penalty = new_penalty
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_selected_sa = set(current_selected)
                    best_count_sa = current_count
        else:
            # Move 2: Swap a random selected block with a random unselected block
            if not current_selected:
                # Add a random block
                b_idx = rng.randint(0, n_ksubs - 1)
                if b_idx not in current_selected:
                    new_cov = current_cov.copy()
                    for t_idx in block_tsubs[b_idx]:
                        new_cov[t_idx] += 1
                    new_penalty = np.sum(new_cov < 1)
                    new_count = current_count + 1
                    new_obj = get_obj(new_count, new_penalty)
                    
                    delta = new_obj - current_obj
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_selected.add(b_idx)
                        current_cov = new_cov
                        current_count = new_count
                        current_penalty = new_penalty
                        current_obj = new_obj
                        
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_selected_sa = set(current_selected)
                            best_count_sa = current_count
            else:
                # Pick random selected to remove
                b_remove = rng.choice(list(current_selected))
                # Pick random unselected to add
                # To avoid picking selected ones repeatedly, we can just pick random index and check
                # Or maintain a list of unselected? Expensive to update.
                # Just pick random index.
                b_add = rng.randint(0, n_ksubs - 1)
                if b_add in current_selected:
                    continue # Skip this step
                
                new_cov = current_cov.copy()
                # Remove b_remove
                for t_idx in block_tsubs[b_remove]:
                    new_cov[t_idx] -= 1
                # Add b_add
                for t_idx in block_tsubs[b_add]:
                    new_cov[t_idx] += 1
                    
                new_penalty = np.sum(new_cov < 1)
                new_count = current_count # Swap: count unchanged
                new_obj = get_obj(new_count, new_penalty)
                
                delta = new_obj - current_obj
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_selected.remove(b_remove)
                    current_selected.add(b_add)
                    current_cov = new_cov
                    # current_count unchanged
                    current_penalty = new_penalty
                    current_obj = new_obj
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_selected_sa = set(current_selected)
                        best_count_sa = current_count
                        
        T *= alpha
        
    # Final cleanup: Greedy removal from best solution
    final_selected = set(best_selected_sa)
    final_cov = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in final_selected:
        for t_idx in block_tsubs[b_idx]:
            final_cov[t_idx] += 1
            
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        sel_list = list(final_selected)
        rng.shuffle(sel_list)
        for b_idx in sel_list:
            # Try remove
            new_cov = final_cov.copy()
            for t_idx in block_tsubs[b_idx]:
                new_cov[t_idx] -= 1
            if np.min(new_cov) >= 1:
                final_selected.remove(b_idx)
                final_cov = new_cov
                improved = True
                break # Restart loop with new set
    
    # Convert to list of lists
    blocks = [list(all_ksubs[i]) for i in final_selected]
    return blocks

if __name__ == "__main__":
    # Test with small parameters to ensure it runs
    try:
        res = construct(19, 7, 4)
        print(f"Found covering with {len(res)} blocks")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

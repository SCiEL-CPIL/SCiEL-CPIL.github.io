import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # ---------- Precomputation ----------
    # C(19, 8) = 75582 blocks
    # C(19, 5) = 11628 t-subsets
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each block, list of t-subset indices it covers
    # C(8, 5) = 56 t-subsets per block
    block_ts = []
    for b in all_k_blocks:
        block_ts.append([ts_index[ts] for ts in itertools.combinations(b, t)])
        
    # For each t-subset, list of block indices that cover it
    # This is expensive to build naively, but we can do it efficiently
    ts_blocks = [[] for _ in range(num_t)]
    for i, b in enumerate(all_k_blocks):
        for ts_idx in block_ts[i]:
            ts_blocks[ts_idx].append(i)

    def C(n, r):
        if r < 0 or r > n:
            return 0
        r = min(r, n - r)
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    def fast_greedy(seed):
        """
        Fast vectorized-ish greedy to get a valid initial solution.
        Returns list of block indices.
        """
        random.seed(seed)
        # Gain: number of uncovered t-subsets covered by block
        # Initially all C(k,t)
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        uncovered = np.full(num_t, True, dtype=bool)
        selected = np.full(num_k, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            # Find candidates with max gain
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            # Random pick from top candidates to break symmetry
            # Use a small random sample to avoid bias
            if len(cands) > 10:
                b_idx = cands[random.randint(0, len(cands)-1)]
            else:
                b_idx = cands[random.randint(0, len(cands)-1)]
                
            sol.append(int(b_idx))
            selected[b_idx] = True
            gain[b_idx] = -1
            
            # Update gains for neighbors
            for ts_idx in block_ts[b_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for nb in ts_blocks[ts_idx]:
                        if not selected[nb]:
                            gain[nb] -= 1
                            
        return sol

    def count_covered(sol_indices):
        """
        Returns cover_count array where cover_count[i] is number of blocks in sol covering t-subset i.
        """
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in sol_indices:
            for ts_idx in block_ts[b_idx]:
                cover_count[ts_idx] += 1
        return cover_count

    def is_valid(cover_count):
        return np.all(cover_count > 0)

    def remove_redundant(sol_indices, cover_count):
        """
        Remove blocks that are fully redundant (all their t-subsets are covered by other blocks).
        Returns new list of indices and new cover_count.
        """
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            for i in range(len(sol_indices) - 1, -1, -1):
                b_idx = sol_indices[i]
                # Check if any t-subset of this block has cover_count == 1
                redundant = True
                for ts_idx in block_ts[b_idx]:
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for ts_idx in block_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    sol_indices.pop(i)
                    changed = True
                    break
        return sol_indices, cover_count

    # ---------- Main Search ----------
    
    # Phase 1: Generate a good initial solution
    # Try a few greedy runs with different seeds to find a good starting point
    best_sol = None
    best_len = float('inf')
    
    for seed in range(5):
        if time.time() > deadline - 60.0:
            break
        sol = fast_greedy(seed)
        if sol is None:
            continue
        # Remove redundancies
        cover_count = count_covered(sol)
        sol, cover_count = remove_redundant(sol, cover_count)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol.copy()
            best_cover = cover_count.copy()
            
    if best_sol is None:
        # Fallback: just return a greedy solution if we ran out of time
        sol = fast_greedy(0)
        if sol:
            return [list(all_k_blocks[i]) for i in sol]
        return []

    # Phase 2: Simulated Annealing
    # State: list of block indices
    # Move: Swap one block in the solution with a random block not in the solution
    # Cost: Number of blocks (we want to minimize). 
    # But we must maintain validity. 
    # Standard SA allows invalid states with a penalty.
    # Penalty = number of uncovered t-subsets.
    # Objective = len(sol) + lambda * uncovered_count
    
    current_sol = best_sol.copy()
    current_cover = best_cover.copy()
    current_uncovered = np.sum(current_cover == 0)
    current_cost = len(current_sol) + 10 * current_uncovered  # High penalty for invalidity
    
    best_sa_sol = current_sol.copy()
    best_sa_cover = current_cover.copy()
    best_sa_cost = current_cost
    best_sa_len = len(current_sol)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # We will run SA for a fixed number of steps or until time runs out
    # Let's define a step limit based on time
    steps_per_sec = 50000  # Estimate
    time_left = deadline - 10.0 - time.time()
    max_steps = int(time_left * steps_per_sec)
    
    if max_steps < 1000:
        max_steps = 1000
        
    # Create a set of current blocks for fast lookup
    current_set = set(current_sol)
    
    # Precompute random block indices for sampling
    # We can just use random.randint(0, num_k-1)
    
    step = 0
    while step < max_steps and time.time() < deadline - 5.0:
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Choose a random block to remove
        if len(current_sol) == 0:
            break
        remove_idx = random.randint(0, len(current_sol) - 1)
        old_block = current_sol[remove_idx]
        
        # Choose a random block to add
        new_block = random.randint(0, num_k - 1)
        
        # If same block, skip
        if new_block == old_block:
            step += 1
            continue
            
        # Check if new_block is already in solution
        if new_block in current_set:
            step += 1
            continue

        # Calculate delta in uncovered count
        # Remove old_block:
        # For each ts in old_block, decrement cover. If it becomes 0, uncovered++.
        # Add new_block:
        # For each ts in new_block, increment cover. If it was 0, uncovered--.
        
        # We need to be careful with the order. 
        # Let's calculate the change in uncovered count directly.
        
        # 1. Tentatively remove old_block
        # We need to know which t-subsets become uncovered.
        # A t-subset becomes uncovered if its cover count was 1 and it is in old_block.
        
        # 2. Tentatively add new_block
        # A t-subset becomes covered if its cover count was 0 (after removal) and it is in new_block.
        
        # To avoid modifying the array and rolling back, we can compute the net change.
        
        # Let U_old be the set of t-subsets in old_block that have cover_count == 1.
        # Let U_new be the set of t-subsets in new_block that have cover_count == 0.
        # Note: If a t-subset is in both old_block and new_block, and cover_count == 1:
        #   Remove: becomes 0 (uncovered)
        #   Add: becomes 1 (covered)
        #   Net change: 0.
        
        # If a t-subset is in old_block (cover==1) and NOT in new_block:
        #   Remove: becomes 0 (uncovered)
        #   Net change: +1 uncovered.
        
        # If a t-subset is in new_block (cover==0) and NOT in old_block:
        #   Add: becomes 1 (covered)
        #   Net change: -1 uncovered.
        
        # If a t-subset is in new_block (cover==0) and in old_block (cover==1):
        #   This is the "both" case above. Net 0.
        
        # So:
        # delta_uncovered = (count of ts in old_block with cover==1 AND not in new_block) 
        #                  - (count of ts in new_block with cover==0 AND not in old_block)
        
        # We can compute this efficiently if we have a way to check membership.
        # But iterating through all 56 t-subsets of old_block and 56 of new_block is fast enough.
        
        delta_uncovered = 0
        
        # To check "not in new_block", we can just iterate.
        # To check "not in old_block", we can just iterate.
        
        # Let's create a set of t-subsets in new_block for fast lookup? 
        # Or just nested loops? 56*56 = 3136 ops per step. 50000 steps = 150M ops. Might be slow in Python.
        
        # Optimization: Precompute t-subset sets? 
        # block_ts is a list of lists.
        
        # Let's use a boolean array for "in_new_block" for the specific new_block?
        # Or just a set.
        
        new_ts_set = set(block_ts[new_block])
        old_ts_set = set(block_ts[old_block])
        
        # Count ts in old_block with cover==1 and not in new_block
        for ts_idx in block_ts[old_block]:
            if current_cover[ts_idx] == 1 and ts_idx not in new_ts_set:
                delta_uncovered += 1
                
        # Count ts in new_block with cover==0 and not in old_block
        for ts_idx in block_ts[new_block]:
            if current_cover[ts_idx] == 0 and ts_idx not in old_ts_set:
                delta_uncovered -= 1
                
        # New uncovered count
        new_uncovered = current_uncovered + delta_uncovered
        
        # New cost
        # Length stays the same (swap)
        new_cost = len(current_sol) + 10 * new_uncovered
        
        # Acceptance probability
        if new_cost < current_cost:
            accept = True
        else:
            delta_cost = new_cost - current_cost
            if delta_cost == 0:
                accept = True
            else:
                # Simulated Annealing acceptance
                # p = exp(-delta_cost / T)
                # To avoid underflow/overflow, use log
                # if log(random) < -delta_cost / T
                r = random.random()
                if r < np.exp(-delta_cost / T):
                    accept = True
                else:
                    accept = False
                    
        if accept:
            # Apply move
            # Update cover counts
            for ts_idx in block_ts[old_block]:
                current_cover[ts_idx] -= 1
            for ts_idx in block_ts[new_block]:
                current_cover[ts_idx] += 1
                
            current_uncovered = new_uncovered
            current_cost = new_cost
            
            # Update solution list
            current_sol[remove_idx] = new_block
            current_set.remove(old_block)
            current_set.add(new_block)
            
            # Update best
            if current_cost < best_sa_cost:
                best_sa_cost = current_cost
                best_sa_sol = current_sol.copy()
                best_sa_cover = current_cover.copy()
                best_sa_len = len(current_sol)
                
        step += 1
        
        # Periodic cleanup: remove redundant blocks
        # This is expensive, so do it rarely
        if step % 10000 == 0:
            current_sol, current_cover = remove_redundant(current_sol, current_cover)
            current_uncovered = np.sum(current_cover == 0)
            current_cost = len(current_sol) + 10 * current_uncovered
            current_set = set(current_sol)
            
            if current_cost < best_sa_cost:
                best_sa_cost = current_cost
                best_sa_sol = current_sol.copy()
                best_sa_cover = current_cover.copy()
                best_sa_len = len(current_sol)

    # Final cleanup
    best_sa_sol, best_sa_cover = remove_redundant(best_sa_sol, best_sa_cover)
    
    # Verify validity
    if not is_valid(best_sa_cover):
        # Fallback to initial best
        return [list(all_k_blocks[i]) for i in best_sol]
        
    return [list(all_k_blocks[i]) for i in best_sa_sol]


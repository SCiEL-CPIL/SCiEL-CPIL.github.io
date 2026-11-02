import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # Convert to numpy arrays for faster updates? 
    # block_to_ts is a list of lists. We can keep it as is for iteration.
    
    # State
    is_selected = np.zeros(num_k_blocks, dtype=bool)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    uncovered_count = num_t_subsets
    
    solution_blocks_idx = []
    
    # Greedy Phase
    # Gain: number of new t-subsets covered
    gain = np.array([len(bts) for bts in block_to_ts], dtype=np.int32)
    # Actually, gain should be the number of UNCOVERED t-subsets covered by the block.
    # We need to update gain dynamically.
    
    # Initialize gain as total t-subsets in block
    current_gain = gain.copy()
    
    while uncovered_count > 0:
        if time.time() > deadline - 20.0:
            break
            
        # Find block with max current_gain
        # To speed this up, we can just pick the max.
        # Updating gain for all blocks when one is picked is O(num_k_blocks * C(k,t)) which is expensive.
        # Alternative: Calculate gain on the fly for a sample? Or just use the static gain?
        # Static gain (total t-subsets) is a good heuristic (greedy set cover).
        
        # Let's use a slightly smarter greedy: 
        # Score = number of uncovered t-subsets covered.
        # We can approximate this by sampling or just using the max static gain initially, 
        # then refining.
        
        # For speed, let's just pick the block with the highest number of uncovered t-subsets it covers.
        # We can calculate this for all blocks? 
        # num_k_blocks is C(19,8) = 75582.
        # C(8,5) = 56.
        # 75582 * 56 ~ 4.2M operations per iteration.
        # We need ~300 blocks. 300 * 4.2M = 1.2B operations. Too slow in Python.
        
        # Optimization: 
        # Maintain a list of uncovered t-subsets.
        # For each block, gain = sum(1 for ts in block_to_ts[b] if cover_count[ts] == 0)
        
        # Let's use a randomized greedy with sampling to find a good block quickly.
        best_block = -1
        best_gain = -1
        
        # Sample some blocks? Or just iterate?
        # If we iterate all 75k blocks, and for each check 56 t-subsets, it's ~4M checks.
        # 4M checks in Python is ~0.1s.
        # If we do this 300 times, it's 30s. Acceptable.
        
        for b in range(num_k_blocks):
            if is_selected[b]:
                continue
            g = 0
            for ts_idx in block_to_ts[b]:
                if cover_count[ts_idx] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_block = b
            if g == C(k, t): # Perfect block, can't get better
                break
        
        if best_block == -1:
            break
            
        is_selected[best_block] = True
        solution_blocks_idx.append(best_block)
        
        for ts_idx in block_to_ts[best_block]:
            if cover_count[ts_idx] == 0:
                uncovered_count -= 1
            cover_count[ts_idx] += 1
            
    best_solution_idx = solution_blocks_idx[:]
    best_cost = len(best_solution_idx)
    
    # Simulated Annealing / Local Search
    # We will use a move-based local search that allows temporary invalid states.
    # State: is_selected, cover_count, uncovered_count
    # Cost: sum(is_selected) + lambda * uncovered_count
    
    # Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    lambda_penalty = 10.0 # High penalty for uncovered sets
    
    current_x = is_selected
    current_cc = cover_count
    current_uncov = uncovered_count
    current_cost = len(solution_blocks_idx)
    
    best_x = current_x.copy()
    best_cc = current_cc.copy()
    best_uncov = current_uncov
    best_cost = current_cost
    
    active_blocks = np.where(current_x)[0].tolist()
    inactive_blocks = np.where(~current_x)[0].tolist()
    
    # Precompute block_to_ts as numpy arrays for faster updates?
    # block_to_ts_np = [np.array(bts, dtype=np.int32) for bts in block_to_ts]
    
    while time.time() < deadline - 5.0 and T > min_T:
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a block
            if not active_blocks:
                continue
            b1 = random.choice(active_blocks)
            
            # Calculate delta
            # Removing b1:
            # new_cost = current_cost - 1
            # new_uncov = current_uncov + (number of ts in b1 with count 1)
            
            delta_uncov = 0
            for ts_idx in block_to_ts[b1]:
                if current_cc[ts_idx] == 1:
                    delta_uncov += 1
            
            delta_cost = -1
            delta_penalty = lambda_penalty * delta_uncov
            delta_E = delta_cost + delta_penalty
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Apply move
                current_x[b1] = False
                for ts_idx in block_to_ts[b1]:
                    current_cc[ts_idx] -= 1
                    if current_cc[ts_idx] == 0:
                        current_uncov += 1
                current_cost -= 1
                active_blocks.remove(b1)
                inactive_blocks.append(b1)
                
                if current_uncov == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_x = current_x.copy()
                    best_cc = current_cc.copy()
                    
        elif move_type < 0.8:
            # Add a block
            if not inactive_blocks:
                continue
            b1 = random.choice(inactive_blocks)
            
            # Calculate delta
            # Adding b1:
            # new_cost = current_cost + 1
            # new_uncov = current_uncov - (number of ts in b1 with count 0)
            
            delta_uncov = 0
            for ts_idx in block_to_ts[b1]:
                if current_cc[ts_idx] == 0:
                    delta_uncov -= 1 # Uncov decreases
            
            delta_cost = 1
            delta_penalty = lambda_penalty * delta_uncov
            delta_E = delta_cost + delta_penalty
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Apply move
                current_x[b1] = True
                for ts_idx in block_to_ts[b1]:
                    if current_cc[ts_idx] == 0:
                        current_uncov -= 1
                    current_cc[ts_idx] += 1
                current_cost += 1
                inactive_blocks.remove(b1)
                active_blocks.append(b1)
                
                if current_uncov == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_x = current_x.copy()
                    best_cc = current_cc.copy()
                    
        else:
            # Swap
            if not active_blocks or not inactive_blocks:
                continue
            b1 = random.choice(active_blocks)
            b2 = random.choice(inactive_blocks)
            
            # Calculate delta
            # Remove b1, Add b2
            # delta_cost = 0
            # delta_uncov = (ts in b1 with count 1) - (ts in b2 with count 0)
            
            delta_uncov_remove = 0
            for ts_idx in block_to_ts[b1]:
                if current_cc[ts_idx] == 1:
                    delta_uncov_remove += 1
            
            delta_uncov_add = 0
            for ts_idx in block_to_ts[b2]:
                if current_cc[ts_idx] == 0:
                    delta_uncov_add -= 1
            
            delta_uncov = delta_uncov_remove + delta_uncov_add
            delta_cost = 0
            delta_penalty = lambda_penalty * delta_uncov
            delta_E = delta_cost + delta_penalty
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Apply move
                # Remove b1
                for ts_idx in block_to_ts[b1]:
                    current_cc[ts_idx] -= 1
                    if current_cc[ts_idx] == 0:
                        current_uncov += 1
                current_x[b1] = False
                active_blocks.remove(b1)
                inactive_blocks.append(b1)
                
                # Add b2
                for ts_idx in block_to_ts[b2]:
                    if current_cc[ts_idx] == 0:
                        current_uncov -= 1
                    current_cc[ts_idx] += 1
                current_x[b2] = True
                inactive_blocks.remove(b2)
                active_blocks.append(b2)
                
                if current_uncov == 0 and current_cost < best_cost:
                    best_cost = current_cost
                    best_x = current_x.copy()
                    best_cc = current_cc.copy()
                    
        T *= cooling_rate
        
        # Occasionally, if we are stuck, do a "kick" or reset to best?
        # Or just let it cool down.
        
    # Final cleanup: Hill climb on the best solution found
    # We can try to remove any redundant blocks from best_x
    final_x = best_x.copy()
    final_cc = best_cc.copy()
    final_uncov = best_uncov
    
    # Ensure it's valid
    if final_uncov != 0:
        # Fallback to greedy if SA failed to maintain validity?
        # But we track best valid state.
        # If best state is invalid, we need to fix it.
        # However, we only update best_x if current_uncov == 0.
        # So best_x should be valid.
        pass
        
    # Remove redundant blocks from final_x
    active_final = np.where(final_x)[0].tolist()
    improved = True
    while improved:
        improved = False
        random.shuffle(active_final)
        for b in active_final:
            # Check if b is redundant
            redundant = True
            for ts_idx in block_to_ts[b]:
                if final_cc[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                final_x[b] = False
                for ts_idx in block_to_ts[b]:
                    final_cc[ts_idx] -= 1
                active_final.remove(b)
                improved = True
                break # Restart loop since list changed
                
    final_blocks = [list(all_k_blocks[j]) for j in np.where(final_x)[0]]
    
    return final_blocks

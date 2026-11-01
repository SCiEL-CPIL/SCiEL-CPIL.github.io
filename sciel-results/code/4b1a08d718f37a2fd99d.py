import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Greedy initialization
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks_idx = []
    
    while True:
        if time.time() > deadline - 40.0:
            break
            
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        solution_blocks_idx.append(max_idx)
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1

    best_solution_idx = solution_blocks_idx[:]
    best_len = len(solution_blocks_idx)
    
    # Simulated Annealing with 2-exchange moves
    def simulated_annealing(initial_idx, max_iters=50000):
        x = np.zeros(num_k_blocks, dtype=bool)
        for j in initial_idx:
            x[j] = True
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for j in initial_idx:
            for ts_idx in block_to_ts[j]:
                cover_count[ts_idx] += 1
        
        current_cost = len(initial_idx)
        best_cost = current_cost
        best_x = x.copy()
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        no_improvement = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Choose a move: either remove a block or 2-exchange
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to remove a block
                selected_indices = np.where(x)[0]
                if len(selected_indices) == 0:
                    break
                j = random.choice(selected_indices)
                
                # Check if removable
                removable = True
                for ts_idx in block_to_ts[j]:
                    if cover_count[ts_idx] == 1:
                        removable = False
                        break
                
                if not removable:
                    no_improvement += 1
                    T *= cooling
                    continue
                
                # Apply removal
                for ts_idx in block_to_ts[j]:
                    cover_count[ts_idx] -= 1
                x[j] = False
                new_cost = current_cost - 1
                
                # Acceptance
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_x = x.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                current_cost = new_cost
                T *= cooling
                
            else:
                # 2-exchange: remove block j, add block j'
                selected_indices = np.where(x)[0]
                unselected_indices = np.where(~x)[0]
                
                if len(selected_indices) == 0 or len(unselected_indices) == 0:
                    break
                
                j = random.choice(selected_indices)
                j_prime = random.choice(unselected_indices)
                
                # Check if this swap keeps coverage valid
                # First, what t-subsets would be uncovered if we remove j?
                would_uncover = []
                for ts_idx in block_to_ts[j]:
                    if cover_count[ts_idx] == 1:
                        would_uncover.append(ts_idx)
                
                # If removing j doesn't uncover anything, it's a pure improvement
                # Otherwise, j' must cover all of would_uncover
                if len(would_uncover) == 0:
                    # Just remove j (this is a 1-exchange, which is better)
                    for ts_idx in block_to_ts[j]:
                        cover_count[ts_idx] -= 1
                    x[j] = False
                    new_cost = current_cost - 1
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_x = x.copy()
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    
                    current_cost = new_cost
                    T *= cooling
                    continue
                
                # Check if j' covers all would_uncover
                covers_all = True
                for ts_idx in would_uncover:
                    if j_prime not in ts_to_blocks[ts_idx]:
                        covers_all = False
                        break
                
                if not covers_all:
                    no_improvement += 1
                    T *= cooling
                    continue
                
                # Apply the swap
                for ts_idx in block_to_ts[j]:
                    cover_count[ts_idx] -= 1
                x[j] = False
                
                for ts_idx in block_to_ts[j_prime]:
                    cover_count[ts_idx] += 1
                x[j_prime] = True
                
                new_cost = current_cost  # Cost unchanged
                delta = 0
                
                # Acceptance: always accept if cost doesn't increase (it's equal here)
                # But we should check if this enables future improvements
                # For SA, we accept with probability exp(-delta/T) where delta is cost change
                # Since delta = 0, we always accept
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_x = x.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                current_cost = new_cost
                T *= cooling
            
            # Early termination if we hit the lower bound
            lb = (C(v, t) + C(k, t) - 1) // C(k, t)
            if best_cost == lb:
                break
        
        return best_x, best_cost

    # Run simulated annealing
    initial_idx = best_solution_idx
    sa_x, sa_cost = simulated_annealing(initial_idx)
    
    if sa_cost < best_len:
        best_len = sa_cost
        best_solution_idx = np.where(sa_x)[0].tolist()
    
    # Final cleanup: remove redundant blocks
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for j in best_solution_idx:
        for ts_idx in block_to_ts[j]:
            cover_count[ts_idx] += 1
    
    # Greedy removal pass
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        selected_indices = np.where(final_x)[0].copy()
        random.shuffle(selected_indices)
        for j in selected_indices:
            is_critical = False
            for ts_idx in block_to_ts[j]:
                if cover_count[ts_idx] == 1:
                    is_critical = True
                    break
            if not is_critical:
                final_x[j] = False
                for ts_idx in block_to_ts[j]:
                    cover_count[ts_idx] -= 1
                improved = True
                break
    
    final_cost = np.sum(final_x)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()

    final_blocks = [list(all_k_blocks[j]) for j in best_solution_idx]
    
    return final_blocks

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

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
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    # Build incidence structures
    # block_to_ts[j]: list of t-subset indices in block j
    # ts_to_blocks[i]: list of block indices containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # 1. Initial Greedy Solution
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    solution_blocks_idx = []

    while True:
        if time.time() > deadline - 20.0:
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

    best_solution_idx = set(solution_blocks_idx)
    best_len = len(best_solution_idx)

    # Helper to get list of blocks from indices
    def indices_to_blocks(indices):
        return [list(all_k_blocks[i]) for i in indices]

    # 2. Simulated Annealing / WalkSAT Local Search
    # State: selected (bool array), cover_count (int array)
    # Objective: minimize num_selected + penalty * num_uncovered
    
    def local_search_sa(initial_indices, time_limit):
        selected = np.zeros(num_k_blocks, dtype=bool)
        for i in initial_indices:
            selected[i] = True
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for i in initial_indices:
            for ts_idx in block_to_ts[i]:
                cover_count[ts_idx] += 1
        
        num_selected = np.sum(selected)
        num_uncovered = np.sum(cover_count == 0)
        
        # Track best solution
        best_selected = selected.copy()
        best_len = num_selected
        best_uncovered = num_uncovered
        
        # Temperature
        T = 10.0
        cooling = 0.995
        penalty = 1000 # High penalty for uncovered
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T < 1e-6:
                T = 1.0 # Restart temperature occasionally if stuck? Or just keep going.
                
            # Calculate current cost
            current_cost = num_selected + penalty * num_uncovered
            
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to remove a selected block
                selected_indices = np.where(selected)[0]
                if len(selected_indices) == 0:
                    continue
                b_idx = selected_indices[random.randint(0, len(selected_indices) - 1)]
                
                # Check impact of removal
                # new_uncovered = number of ts in block b_idx that have cover_count == 1
                new_uncovered_delta = 0
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        new_uncovered_delta += 1
                
                # Delta cost: -1 (block removed) + penalty * new_uncovered_delta
                delta_cost = -1 + penalty * new_uncovered_delta
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected[b_idx] = False
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    num_selected -= 1
                    num_uncovered += new_uncovered_delta
                    
                    if num_uncovered == 0 and num_selected < best_len:
                        best_len = num_selected
                        best_selected = selected.copy()
                    elif num_uncovered < best_uncovered or (num_uncovered == best_uncovered and num_selected < best_len):
                         # We prioritize valid solutions (uncovered=0), but keep track of best invalid too?
                         # Actually, we only care about valid solutions for the final answer.
                         # But intermediate invalid states help escape.
                         pass
            else:
                # Try to add a block
                # Strategy: Pick a random uncovered t-subset, then a random block covering it
                if num_uncovered > 0:
                    uncovered_indices = np.where(cover_count == 0)[0]
                    if len(uncovered_indices) > 0:
                        u_ts_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                        candidate_blocks = ts_to_blocks[u_ts_idx]
                        if len(candidate_blocks) > 0:
                            b_idx = candidate_blocks[random.randint(0, len(candidate_blocks) - 1)]
                            if not selected[b_idx]:
                                # Check impact of addition
                                # new_covered = number of ts in block b_idx that have cover_count == 0
                                new_covered_delta = 0
                                for ts_idx in block_to_ts[b_idx]:
                                    if cover_count[ts_idx] == 0:
                                        new_covered_delta += 1
                                
                                # Delta cost: +1 (block added) - penalty * new_covered_delta
                                delta_cost = 1 - penalty * new_covered_delta
                                
                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    # Accept
                                    selected[b_idx] = True
                                    for ts_idx in block_to_ts[b_idx]:
                                        cover_count[ts_idx] += 1
                                    num_selected += 1
                                    num_uncovered -= new_covered_delta
                                    
                                    if num_uncovered == 0 and num_selected < best_len:
                                        best_len = num_selected
                                        best_selected = selected.copy()
                else:
                    # If no uncovered, try adding a random unselected block to explore
                    # This is less likely to help if we are already valid, but might lead to better structure?
                    # Usually, we want to remove blocks when valid.
                    # So if num_uncovered == 0, we should bias heavily towards removal.
                    # The 0.5 split is okay, but if valid, removal is preferred.
                    pass

            T *= cooling
            
        return best_selected

    # Run SA
    # We have a budget. Let's use 80% of remaining time for SA.
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed - 5.0 # 5s safety
    if remaining_time > 10:
        sa_selected = local_search_sa(list(best_solution_idx), remaining_time * 0.8)
        sa_indices = np.where(sa_selected)[0]
        # Verify and clean up
        # The SA might end in an invalid state if it didn't find a valid one with lower cost?
        # No, it tracks best_selected which is updated only if valid (num_uncovered==0) or better invalid?
        # My code above: `if num_uncovered == 0 and num_selected < best_len`.
        # It does NOT update best_selected for invalid states.
        # So best_selected is always a valid covering (or the initial one if none found).
        # However, if the initial was valid, best_selected is at least as good as initial.
        
        if len(sa_indices) < best_len:
            best_len = len(sa_indices)
            best_solution_idx = set(sa_indices)
        else:
            # If SA didn't improve, stick with greedy
            pass

    # 3. Final Polish: Remove redundancies
    # Convert indices to list of blocks for the final output
    final_indices = list(best_solution_idx)
    
    # Build cover count for final check
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for i in final_indices:
        for ts_idx in block_to_ts[i]:
            cover_count[ts_idx] += 1
    
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(final_indices)):
            if time.time() > deadline - 1.0:
                break
            b_idx = final_indices[i]
            redundant = True
            for ts_idx in block_to_ts[b_idx]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                final_indices.pop(i)
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] -= 1
                changed = True
                break
        if time.time() > deadline - 1.0:
            break

    # Convert to lists of points
    result_blocks = [list(all_k_blocks[i]) for i in final_indices]
    
    return result_blocks

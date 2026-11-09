import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert to arrays for fast access
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    def run_greedy_indices(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            
            if randomize:
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                best_block_idx = np.argmax(block_gain)
                
            selected_indices.append(best_block_idx)
            
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    block_gain[tsub_blocks_np[si]] -= 1
                    
        return selected_indices

    def anneal_solution(initial_indices, time_limit=5.0):
        """
        Simulated Annealing on the set of blocks.
        Cost = len(blocks) + M * uncovered_count
        """
        if not initial_indices:
            return initial_indices
            
        M = 1000 # Penalty for uncovered t-subsets
        T = 1.0
        alpha = 0.95
        T_min = 1e-4
        
        # State
        selected = list(initial_indices)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        
        for b_idx in selected:
            for si in block_tsubs[b_idx]:
                cov[si] += 1
                
        uncovered_count = np.sum(cov == 0)
        current_cost = len(selected) + M * uncovered_count
        
        best_cost = current_cost
        best_selected = list(selected)
        
        start = time.time()
        
        while T > T_min and (time.time() - start) < time_limit and time.time() < DEADLINE:
            # Generate Move
            # 50% Remove, 50% Add
            if random.random() < 0.5 and selected:
                # Remove
                b_idx = random.choice(selected)
                # Calculate Delta Cost
                # Uncovered increase by count of si in block_tsubs[b_idx] where cov[si] == 1
                delta_uncovered = 0
                for si in block_tsubs[b_idx]:
                    if cov[si] == 1:
                        delta_uncovered += 1
                delta_cost = -1 + M * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    for si in block_tsubs[b_idx]:
                        cov[si] -= 1
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
            else:
                # Add
                # Pick a random block not in selected
                # To be efficient, pick from blocks that cover uncovered subsets?
                # Or just random from all_blocks?
                # Random from all_blocks is O(1) but might be useless.
                # Let's pick a random block from all_blocks.
                b_idx = random.randint(0, N_blocks - 1)
                if b_idx in selected_set:
                    continue # Try again? Or skip. Skip is fine.
                
                # Calculate Delta Cost
                # Uncovered decrease by count of si in block_tsubs[b_idx] where cov[si] == 0
                delta_uncovered = 0
                for si in block_tsubs[b_idx]:
                    if cov[si] == 0:
                        delta_uncovered -= 1 # It becomes covered, so uncovered count decreases
                # Wait, delta_uncovered should be the change in uncovered count.
                # If cov[si] was 0, it becomes 1. Uncovered count decreases by 1.
                # So change is -1.
                # Let's just count how many become covered.
                newly_covered = 0
                for si in block_tsubs[b_idx]:
                    if cov[si] == 0:
                        newly_covered += 1
                
                delta_cost = 1 - M * newly_covered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected.append(b_idx)
                    selected_set.add(b_idx)
                    for si in block_tsubs[b_idx]:
                        cov[si] += 1
                    uncovered_count -= newly_covered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
            
            T *= alpha
        
        # Repair: Ensure validity
        # Greedily add blocks to cover any remaining uncovered t-subsets
        uncovered_tsubs = [i for i in range(N_t) if cov[i] == 0]
        while uncovered_tsubs:
            # Find block covering most uncovered
            best_b = -1
            max_cov = 0
            # Sample some blocks if list is too long?
            # For now, check all? Might be slow.
            # But N_blocks is 75k. N_t is 11k.
            # This repair phase should be fast if SA left few uncovered.
            # Optimization: Only check blocks that cover at least one uncovered t-subset.
            candidate_blocks = set()
            for si in uncovered_tsubs[:10]: # Check top 10 uncovered
                candidate_blocks.update(tsub_blocks[si])
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                count = 0
                for si in block_tsubs[b_idx]:
                    if cov[si] == 0:
                        count += 1
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx
            
            if best_b == -1:
                break # Should not happen if problem is solvable
                
            selected.append(best_b)
            selected_set.add(best_b)
            for si in block_tsubs[best_b]:
                if cov[si] == 0:
                    cov[si] = 1
                    uncovered_tsubs.remove(si) # Inefficient, but ok for small lists
            # Re-evaluate uncovered list?
            # Simpler: just re-compute uncovered_tsubs if list is small.
            # Actually, `uncovered_tsubs` is a list of indices.
            # If we remove one, we should update.
            # For efficiency, just break and re-calculate if needed.
            # But we are in a while loop.
            # Let's just re-calculate uncovered_tsubs from cov array.
            uncovered_tsubs = [i for i in range(N_t) if cov[i] == 0]
            
        # Final Local Search: Reduction
        # Try to remove any redundant blocks
        improved = True
        while improved:
            improved = False
            for b_idx in list(selected):
                can_remove = True
                for si in block_tsubs[b_idx]:
                    if cov[si] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    for si in block_tsubs[b_idx]:
                        cov[si] -= 1
                    improved = True
                    break # Restart loop
        
        return selected

    # Main Loop
    best_sol_idx = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = anneal_solution(sol_idx, time_limit=10.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)
    
    # 2. Randomized Restarts
    restart_count = 0
    while time.time() < DEADLINE - 15:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        # Limit time per restart
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break
        time_limit = min(5.0, remaining - 10.0) # Leave 10s for repair/final checks
        if time_limit < 0.5:
            break
            
        cand_idx = anneal_solution(cand_idx, time_limit=time_limit)
        
        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)
            
        restart_count += 1

    return [all_blocks[i] for i in best_sol_idx]

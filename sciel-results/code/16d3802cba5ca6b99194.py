import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

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
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures:
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # State for Greedy initialization
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks_idx = []
    
    # Greedy Phase (Deterministic) to get a valid initial solution
    while True:
        if time.time() > deadline - 30.0:
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
                    # If already selected, no change to gain needed for this logic, 
                    # but gain was already set to -1.
                    
    best_solution_idx = solution_blocks_idx[:]
    best_len = len(solution_blocks_idx)
    
    # WalkSAT / MaxSAT Local Search
    # Variables: x[j] = 1 if block j is in the solution.
    # Constraints: For each t-subset i, OR(x[j] for j in ts_to_blocks[i]) must be True.
    # Objective: Minimize sum(x[j]).
    # We maximize satisfied "cost" clauses: Clause j is satisfied if x[j] == 0.
    
    # Precompute for fast WalkSAT
    # For each block j, we need to know which t-subsets it covers (block_to_ts[j])
    # For each t-subset i, we need to know which blocks cover it (ts_to_blocks[i])
    
    # We will maintain:
    # x: boolean array of selected blocks
    # cover_count[i]: number of selected blocks covering t-subset i
    # cost_unsat: number of blocks that are selected (we want to minimize this, 
    #             so in MaxSAT terms, we maximize the number of blocks that are NOT selected)
    #             Let's stick to minimizing selected blocks directly.
    
    def get_cover_counts(x):
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for j in range(num_k_blocks):
            if x[j]:
                for ts_idx in block_to_ts[j]:
                    counts[ts_idx] += 1
        return counts
    
    # Initial state from Greedy
    x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        x[j] = True
        
    cover_count = get_cover_counts(x)
    # Check validity (should be valid from Greedy)
    # min_cover = np.min(cover_count)
    # assert min_cover >= 1, "Initial solution invalid"
    
    # WalkSAT Parameters
    p_random = 0.5
    max_flips = 5000
    restarts = 20
    
    # We want to minimize the number of True variables (blocks).
    # Let's define the "score" as the number of blocks.
    # A flip is beneficial if it reduces the number of blocks while keeping all constraints satisfied.
    # Or if it reduces the number of unsatisfied constraints (if we allow temporary invalidity).
    
    # Standard WalkSAT minimizes unsatisfied clauses.
    # Here, we have two types of "clauses":
    # 1. Coverage constraints: MUST be satisfied. (Hard)
    # 2. Cost constraints: Prefer x[j]=0. (Soft)
    
    # We can treat this as a Weighted MaxSAT:
    # Minimize: sum(x[j]) * 1  (Soft cost)
    # Subject to: cover_count[i] >= 1 for all i (Hard)
    
    # A simple local search:
    # 1. Try to remove a block (flip 1->0) if it doesn't break coverage.
    # 2. If stuck, try to swap blocks or move to a nearby valid solution.
    
    # Let's implement a more robust local search:
    # State: x (boolean array)
    # Objective: Minimize sum(x)
    # Move: Flip a bit.
    #   - If flip 1->0 keeps all cover_count >= 1: Accept (improvement).
    #   - If flip 0->1: Only accept if it allows a subsequent 1->0 that results in net improvement? 
    #     This is complex. 
    # 
    # Alternative: Simulated Annealing or simple Hill Climbing with restarts.
    # Let's try Hill Climbing with "Add" and "Remove" moves.
    
    def local_search_hill_climb(x, cover_count, max_iters=10000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = np.sum(x)
        best_cost = current_cost
        best_x = x.copy()
        
        no_improvement = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            
            # Phase 1: Try to remove blocks (1->0)
            # Identify blocks that are "removable" (their removal doesn't leave any t-subset uncovered)
            # A block j is removable if for all ts in block_to_ts[j], cover_count[ts] > 1.
            
            # To speed this up, we can check candidates randomly.
            selected_indices = np.where(x)[0]
            if len(selected_indices) == 0:
                break
                
            # Shuffle to avoid bias
            random.shuffle(selected_indices)
            
            for j in selected_indices:
                # Check if block j is critical for any t-subset
                is_critical = False
                for ts_idx in block_to_ts[j]:
                    if cover_count[ts_idx] == 1:
                        is_critical = True
                        break
                        
                if not is_critical:
                    # Remove block j
                    x[j] = False
                    for ts_idx in block_to_ts[j]:
                        cover_count[ts_idx] -= 1
                    current_cost -= 1
                    improved = True
                    break # Restart scan after a change
                    
            if improved:
                no_improvement = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == C(v, k) // C(k, t) * C(t, t): # Theoretical lower bound? No, just check for valid solution
                         # Lower bound is ceil(C(v,t) / C(k,t))
                         lb = (C(v, t) + C(k, t) - 1) // C(k, t)
                         if best_cost == lb:
                             return best_x, best_cost
            else:
                no_improvement += 1
                if no_improvement > 100:
                    break
                    
        return best_x, best_cost

    # Run Hill Climbing
    x_init = x
    cc_init = cover_count
    
    # We might get stuck. Let's add a perturbation phase.
    # If we can't remove any blocks, try swapping a block with another to enable removal.
    
    def local_search_with_swap(x, cover_count, max_iters=5000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = np.sum(x)
        best_cost = current_cost
        best_x = x.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Try removals
            removed_any = False
            selected_indices = np.where(x)[0].copy()
            random.shuffle(selected_indices)
            
            for j in selected_indices:
                is_critical = False
                for ts_idx in block_to_ts[j]:
                    if cover_count[ts_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    x[j] = False
                    for ts_idx in block_to_ts[j]:
                        cover_count[ts_idx] -= 1
                    current_cost -= 1
                    removed_any = True
                    break
                    
            if removed_any:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == (C(v, t) + C(k, t) - 1) // C(k, t):
                        return best_x, best_cost
                continue
                
            # 2. If no removals possible, try a swap:
            # Pick a critical block j (blocks whose removal would break coverage).
            # Find a block j' not in solution that covers the same critical t-subsets,
            # such that swapping j -> j' allows at least one other block to be removed?
            # This is hard to check directly.
            
            # Simpler Swap:
            # Pick a random block j in solution.
            # Find a block j' not in solution.
            # Swap j and j'.
            # Check if the new solution allows immediate removals.
            
            if len(selected_indices) == 0:
                break
                
            j = random.choice(selected_indices)
            
            # Find candidates for j'
            # j' should cover the t-subsets that j uniquely covers?
            # Let's just pick a random j' not in solution.
            # To be smarter, pick j' that shares many t-subsets with j.
            
            # For now, random swap to escape local minimum
            unselected_indices = np.where(~x)[0]
            if len(unselected_indices) == 0:
                break
                
            j_prime = random.choice(unselected_indices)
            
            # Perform swap
            # Remove j
            for ts_idx in block_to_ts[j]:
                cover_count[ts_idx] -= 1
            x[j] = False
            
            # Add j'
            for ts_idx in block_to_ts[j_prime]:
                cover_count[ts_idx] += 1
            x[j_prime] = True
            
            # Check validity: Are all cover_count >= 1?
            if np.min(cover_count) < 1:
                # Invalid swap, revert
                for ts_idx in block_to_ts[j_prime]:
                    cover_count[ts_idx] -= 1
                x[j_prime] = False
                for ts_idx in block_to_ts[j]:
                    cover_count[ts_idx] += 1
                x[j] = True
                continue
                
            # Now try to remove blocks from the new state
            # (This is effectively restarting the removal phase with a perturbed state)
            # We don't explicitly loop here, the next iteration of the for loop will try removals.
            # But we should update best if cost decreased? Cost is same.
            # The hope is that the swap enables future removals.

        return best_x, best_cost

    # Run the search
    current_x = x_init
    current_cc = cc_init
    
    # First pass: Pure hill climbing (removals)
    current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=5000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        
    # Second pass: Hill climbing with swaps (perturbation)
    # Re-initialize cover counts for the current best
    current_cc = get_cover_counts(current_x)
    
    # Run multiple times with different random seeds/perturbations
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart)
        # Perturb the current best solution slightly?
        # Or just run the swap-based search on the current best.
        new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=2000)
        
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(current_x)
            
    # Final cleanup: Remove any redundant blocks again
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    final_cc = get_cover_counts(final_x)
    
    # One more pass of pure removal to ensure minimality
    final_x, final_cost = local_search_hill_climb(final_x, final_cc, max_iters=1000)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[j]) for j in best_solution_idx]
    
    return final_blocks

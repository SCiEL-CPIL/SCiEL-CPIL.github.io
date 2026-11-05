import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    t0 = time.time()
    TIMEOUT = 135.0
    def elapsed():
        return time.time() - t0

    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # block_subsets[i] = list of subset indices covered by block i
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # blocks_covering_subset[s] = list of block indices covering subset s
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    def time_exceeded():
        return time.time() - t0 > TIMEOUT

    # 2. Vectorized Greedy
    def run_greedy(randomize_ties=False, seed=None):
        if elapsed() > TIMEOUT - 5:
            return None
        if seed is not None:
            random.seed(seed)
            
        # potential[i] = number of uncovered subsets covered by block i
        potential = np.array([len(s) for s in block_subsets], dtype=np.int32)
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        selected_blocks = []
        uncovered_count = N_subsets
        
        while uncovered_count > 0:
            if elapsed() > TIMEOUT - 2:
                return None
                
            max_pot = np.max(potential)
            if max_pot == 0:
                break
                
            if randomize_ties:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = np.argmax(potential)
                
            selected_blocks.append(best_idx)
            
            # Update potentials: for each subset covered by best_idx,
            # if it was uncovered (count 0), decrement potential for all blocks covering it
            for s in block_subsets[best_idx]:
                if cover_count[s] == 0:
                    uncovered_count -= 1
                    # Decrement potential for all blocks covering s
                    # Vectorized update: potential[blocks_covering_subset[s]] -= 1
                    # But we need to be careful with int32 overflow? No, max is 1287.
                    # However, doing this in a loop over blocks_covering_subset[s] is fast enough?
                    # len(blocks_covering_subset[s]) = C(v-t, k-t) = C(14, 3) = 364.
                    # 1287 subsets * 364 ops = ~468k ops per step.
                    # ~300 steps -> ~140M ops. Python loop might be slow.
                    # Let's use numpy.
                    potential[blocks_covering_subset[s]] -= 1
                cover_count[s] += 1
        return selected_blocks

    # 3. Local Search: Reduction and Swap
    def local_search(selected_blocks):
        if elapsed() > TIMEOUT - 5:
            return selected_blocks
            
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            cover_count[block_subsets[b_idx]] += 1
            
        selected_set = set(selected_blocks)
        
        # Phase 1: Reduction
        changed = True
        while changed and elapsed() < TIMEOUT - 10:
            changed = False
            # Iterate in reverse
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                # Check if any subset covered by b_idx has count 1
                # Vectorized check: np.min(cover_count[block_subsets[b_idx]]) == 1
                if np.min(cover_count[block_subsets[b_idx]]) == 1:
                    is_redundant = False
                else:
                    # Remove
                    cover_count[block_subsets[b_idx]] -= 1
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
                    
        # Phase 2: Swap Search (1-for-1)
        # Try to swap blocks to enable further reductions
        # We limit the number of swap iterations
        max_swap_passes = 20
        for _ in range(max_swap_passes):
            if elapsed() > TIMEOUT - 10:
                break
                
            improved = False
            current_list = list(selected_blocks)
            random.shuffle(current_list)
            
            for b_idx in current_list:
                if b_idx not in selected_set:
                    continue
                # Tentatively remove b_idx
                # Find subsets that become uncovered (count 1)
                subs_b = block_subsets[b_idx]
                counts_b = cover_count[subs_b]
                uncovered_subs = subs_b[counts_b == 1]
                
                if len(uncovered_subs) == 0:
                    # Redundant
                    cover_count[subs_b] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.remove(b_idx)
                    improved = True
                    break
                    
                # Find candidate blocks that cover all uncovered_subs
                # Intersection of blocks_covering_subset[s] for s in uncovered_subs
                # Start with the first one
                candidate_blocks = set(blocks_covering_subset[uncovered_subs[0]])
                for s in uncovered_subs[1:]:
                    candidate_blocks &= set(blocks_covering_subset[s])
                    
                candidate_blocks -= selected_set
                
                if candidate_blocks:
                    # Pick a candidate
                    # Heuristic: Minimize "loss" or just random
                    # Let's pick one that maximizes potential gain?
                    # Or just random for now.
                    j_idx = random.choice(list(candidate_blocks))
                    
                    # Perform swap
                    # Remove b_idx
                    cover_count[subs_b] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.remove(b_idx)
                    
                    # Add j_idx
                    subs_j = block_subsets[j_idx]
                    cover_count[subs_j] += 1
                    selected_blocks.append(j_idx)
                    selected_set.add(j_idx)
                    
                    improved = True
                    break # Restart reduction to see if new block makes others redundant
                    
            if not improved:
                break
                
            # Re-run reduction
            changed = True
            while changed and elapsed() < TIMEOUT - 10:
                changed = False
                for i in range(len(selected_blocks) - 1, -1, -1):
                    b_idx = selected_blocks[i]
                    if np.min(cover_count[block_subsets[b_idx]]) == 1:
                        continue
                    else:
                        cover_count[block_subsets[b_idx]] -= 1
                        selected_blocks.pop(i)
                        selected_set.remove(b_idx)
                        changed = True
                        
        return selected_blocks

    # 4. Main Execution
    best_blocks = None
    best_len = float('inf')
    
    # Run deterministic greedy
    res = run_greedy(randomize_ties=False)
    if res:
        res = local_search(res)
        if res:
            best_blocks = res
            best_len = len(res)
            
    # Run randomized restarts
    # Estimate time taken so far
    t_elapsed = time.time() - t0
    if t_elapsed < TIMEOUT - 5:
        # Do restarts
        # Each restart takes ~1s greedy + ~1-2s local search?
        # Let's try up to 20 restarts or until time runs out
        for i in range(20):
            if time.time() - t0 > TIMEOUT - 2:
                break
            res = run_greedy(randomize_ties=True, seed=i)
            if res:
                res = local_search(res)
                if res and len(res) < best_len:
                    best_len = len(res)
                    best_blocks = res

    if best_blocks is None:
        return []

    # 5. Final Polish: ILP for reduction
    # This is a high-precision polish pass.
    # We try to solve a small ILP to find redundant blocks.
    # Variables: x_i = 1 if block i is kept, 0 if removed.
    # Objective: Minimize sum(x_i)
    # Constraints: For each subset s, sum(x_i for i in blocks_covering_subset[s]) >= 1
    # But we only consider blocks currently in selected_set.
    # If the ILP finds a solution with fewer blocks, we update.
    
    if elapsed() < TIMEOUT - 10 and len(selected_blocks) > 50:
        # Build the ILP
        # Only consider blocks in best_blocks
        block_indices = list(best_blocks)
        n_vars = len(block_indices)
        
        # A matrix: N_subsets x n_vars
        # A[s, j] = 1 if block_indices[j] covers subset s
        # We can build this efficiently
        # Initialize A as zeros
        # For each block j, for each subset s in block_subsets[block_indices[j]], A[s, j] = 1
        # This is sparse.
        
        # Let's use scipy.sparse
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for j, b_idx in enumerate(block_indices):
            for s in block_subsets[b_idx]:
                rows.append(s)
                cols.append(j)
                data.append(1)
                
        A = sp.csr_matrix((data, (rows, cols)), shape=(N_subsets, n_vars))
        
        # Objective: c = 1 for all
        c = np.ones(n_vars)
        
        # Constraints: A @ x >= 1
        # LinearConstraint(A, lb=np.ones(N_subsets), ub=np.inf)
        constraints = LinearConstraint(A, lb=np.ones(N_subsets), ub=np.inf)
        integrality = np.ones(n_vars)
        bounds = Bounds(0, 1)
        
        try:
            # Solve ILP
            # This might be slow if n_vars is large, but we limit it.
            # If n_vars > 500, skip? Or try.
            if n_vars <= 500:
                res_milp = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
                if res_milp.status == 0: # Optimal
                    # Extract solution
                    x = res_milp.x
                    new_blocks = [block_indices[j] for j in range(n_vars) if x[j] > 0.5]
                    if len(new_blocks) < len(best_blocks):
                        # Verify validity (just in case)
                        # Check cover count
                        cover_count = np.zeros(N_subsets, dtype=np.int32)
                        for b_idx in new_blocks:
                            cover_count[block_subsets[b_idx]] += 1
                        if np.min(cover_count) >= 1:
                            best_blocks = new_blocks
                            best_len = len(best_blocks)
        except Exception:
            pass

    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks

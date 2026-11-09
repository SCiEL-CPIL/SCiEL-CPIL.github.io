import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5s safety margin
    
    points = list(range(v))
    
    # Precompute all blocks and t-subsets
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    
    # Map t-subsets to indices
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Build incidence: for each block, which t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
    
    # Convert to numpy arrays for faster access
    block_tsubs_np = [np.array(subs, dtype=np.int32) for subs in block_tsubs]
    
    # Build reverse index: for each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs_np):
        for si in subs:
            tsub_blocks[si].append(i)
    
    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        gains = np.array([len(subs) for subs in block_tsubs_np], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.bool_)
        selected_blocks = []
        uncovered_count = N_t
        
        while uncovered_count > 0:
            if time.time() > DEADLINE:
                break
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1 and random_seed is not None:
                best_idx = random.choice(candidates.tolist())
            else:
                best_idx = candidates[0]
                
            selected_blocks.append(all_blocks[best_idx])
            
            for si in block_tsubs_np[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        """Remove redundant blocks (those with no unique t-subsets)."""
        if not blocks:
            return blocks
            
        n = len(blocks)
        cov_count = np.zeros(N_t, dtype=np.int32)
        block_tsubs_list = []
        for b in blocks:
            subs = []
            for s in itertools.combinations(b, t):
                subs.append(t_idx[s])
            block_tsubs_list.append(subs)
            for si in subs:
                cov_count[si] += 1
                
        removed = [False] * n
        changed = True
        while changed:
            if time.time() > DEADLINE:
                break
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                has_unique = False
                for si in block_tsubs_list[i]:
                    if cov_count[si] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    removed[i] = True
                    changed = True
                    for si in block_tsubs_list[i]:
                        cov_count[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    def try_milp_refinement(blocks, time_budget=30.0):
        """
        Use MILP to find a smaller subset of blocks from the candidate pool
        that still covers all t-subsets.
        """
        if not blocks or time.time() > DEADLINE:
            return blocks
            
        n = len(blocks)
        if n < 2:
            return blocks
        
        # Build the candidate pool: all blocks in 'blocks' plus some additional candidates
        # For efficiency, we'll work with the blocks we already have
        # and try to minimize their selection.
        
        # Build incidence matrix: A[i, j] = 1 if block i covers t-subset j
        # This is sparse. We'll use scipy sparse.
        from scipy.sparse import csr_matrix
        
        row_indices = []
        col_indices = []
        for i, b in enumerate(blocks):
            for si in block_tsubs_np[all_blocks.index(b)]:
                row_indices.append(si)
                col_indices.append(i)
        
        A = csr_matrix((np.ones(len(row_indices), dtype=np.int8), 
                        (row_indices, col_indices)), 
                       shape=(N_t, n))
        
        # Objective: minimize sum of x_i (select as few blocks as possible)
        c = np.ones(n, dtype=np.float64)
        
        # Constraints: for each t-subset j, sum_i A[i,j] * x_i >= 1
        # i.e., A @ x >= 1
        lower_bound = np.ones(N_t, dtype=np.float64)
        upper_bound = np.full(N_t, np.inf, dtype=np.float64)
        
        constraint = LinearConstraint(A, lower_bound, upper_bound)
        
        # x_i in {0, 1}
        integrality = np.ones(n, dtype=np.int8)
        bounds = Bounds(lb=0, ub=1)
        
        # Set time limit
        # scipy.optimize.milp doesn't have a direct time limit parameter in all versions
        # We'll use a callback to check time, but milp doesn't support callbacks directly.
        # Instead, we'll just run it and hope it finishes in time, or use a simpler approach.
        
        # Actually, let's use a different approach: we'll use the HiGHS solver via scipy
        # which should handle this reasonably well for small instances.
        
        try:
            # Run MILP with a timeout by running in a thread
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def solve_milp():
                try:
                    res = milp(
                        c=c,
                        constraints=constraint,
                        integrality=integrality,
                        bounds=bounds
                    )
                    result_queue.put(res)
                except Exception as e:
                    result_queue.put(e)
            
            solver_thread = threading.Thread(target=solve_milp)
            solver_thread.start()
            solver_thread.join(timeout=time_budget)
            
            if solver_thread.is_alive():
                # Thread didn't finish in time; we can't easily kill it
                # Just return the original blocks
                return blocks
                
            res = result_queue.get()
            if isinstance(res, Exception):
                return blocks
            if not res.success:
                return blocks
            
            # Extract solution
            x = res.x
            if x is None:
                return blocks
                
            selected = [blocks[i] for i in range(n) if x[i] > 0.5]
            
            # Verify the solution is valid
            if len(selected) == 0:
                return blocks
                
            # Check coverage
            covered = set()
            for b in selected:
                for s in itertools.combinations(b, t):
                    covered.add(s)
                    
            if len(covered) == N_t:
                return selected
            else:
                return blocks
                
        except Exception:
            return blocks

    best_solution = []
    best_size = float('inf')
    
    # Run multiple greedy restarts
    for i in range(15):
        if time.time() > DEADLINE:
            break
            
        if i == 0:
            sol = run_greedy(random_seed=None)
        else:
            sol = run_greedy(random_seed=i)
            
        if not sol:
            continue
            
        sol = try_reduce(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
    
    # If we have time, try MILP refinement on the best solution
    if best_solution and time.time() < DEADLINE - 35.0:
        time_budget = min(30.0, DEADLINE - time.time() - 10.0)
        if time_budget > 5.0:
            refined = try_milp_refinement(best_solution, time_budget=time_budget)
            if refined and len(refined) < best_size:
                best_solution = refined
                best_size = len(refined)
    
    return best_solution

import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets (targets)
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    # block_ts[b] is a list of indices into all_ts
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = [[] for _ in range(n_ts)]
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # --- 1. Greedy Initialization to get an Upper Bound ---
    def greedy_init(seed):
        local_rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        
        # gain[b] = number of currently uncovered t-subsets covered by block b
        gains = np.array([len(ts) for ts in block_ts], dtype=np.int32)
        
        selected = []
        
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 5.0:
                break
                
            # Find block with max gain
            # Optimization: Instead of scanning all 50k blocks every time, 
            # we could use a heap, but simple scan is O(N_blocks) ~ 50k.
            # 150 steps * 50k = 7.5M ops. This is fast in numpy if vectorized, 
            # but we need argmax. 
            best_b = np.argmax(gains)
            best_gain = gains[best_b]
            
            if best_gain <= 0:
                break
            
            selected.append(best_b)
            
            # Update uncovered and gains
            # For each t-subset covered by best_b:
            #   if it was uncovered, mark covered, decrement uncovered_count
            #   for all blocks covering that t-subset, decrement gain
            for s_idx in block_ts[best_b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    uncovered_count -= 1
                    # Decrement gains of all blocks covering this t-subset
                    for other_b in ts_to_blocks[s_idx]:
                        gains[other_b] -= 1
                        if gains[other_b] < 0:
                            gains[other_b] = 0
        
        return selected

    # --- 2. Local Search: Deletion (Polish) ---
    def polish(blocks):
        # Remove redundant blocks
        # coverage[s] = number of blocks covering s
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
                
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 5.0:
            changed = False
            # Iterate backwards to safely remove
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # Check if any set is covered ONLY by b
                redundant = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks.pop(i)
                    changed = True
                    # Restart loop to check for new redundancies
                    break
        return blocks

    # --- 3. ILP Formulation ---
    def solve_ilp(initial_ub, time_limit):
        # Variables: x_b for b in 0..n_blocks-1
        # Objective: min sum(x_b)
        # Constraints: for each s in 0..n_ts-1, sum(x_b for b covering s) >= 1
        
        # c = objective coefficients
        c = np.ones(n_blocks)
        
        # A = constraint matrix
        # Rows: n_ts (one for each 4-subset)
        # Cols: n_blocks (one for each 7-subset)
        # A[i, j] = 1 if block j covers subset i
        
        A = lil_matrix((n_ts, n_blocks), dtype=np.int8)
        for b_idx, ts_list in enumerate(block_ts):
            for s_idx in ts_list:
                A[s_idx, b_idx] = 1
        A = A.tocsr()
        
        # Constraints: A @ x >= 1
        # milp requires: lb <= A @ x <= ub
        lb = np.ones(n_ts)
        ub = np.inf * np.ones(n_ts)
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Bounds for x: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
        
        # Integrality: all variables are integers
        integrality = np.ones(n_blocks)
        
        # Run solver
        # time_limit is in seconds
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={'time_limit': time_limit, 'mip_rel_gap': 0.01})
            
            if res.status == 0 or res.status == 1: # 0: optimal, 1: feasible found
                x = res.x
                # Extract blocks
                selected = [i for i in range(n_blocks) if x[i] > 0.5]
                return selected
            else:
                return None
        except Exception as e:
            print(f"ILP Error: {e}")
            return None

    # --- Main Execution ---
    
    # 1. Get a quick feasible solution (Upper Bound)
    initial_blocks = greedy_init(42)
    initial_blocks = polish(initial_blocks)
    current_best = len(initial_blocks)
    best_solution = initial_blocks
    
    if time.time() - _start_time > _DEADLINE - 10.0:
        return [list(all_blocks[b]) for b in best_solution]

    # 2. Try ILP
    # Allocate most of the remaining time to ILP
    ilp_time = _remaining() - 5.0
    
    if ilp_time > 10.0:
        ilp_solution = solve_ilp(current_best, ilp_time)
        if ilp_solution is not None:
            ilp_solution = polish(ilp_solution)
            if len(ilp_solution) < current_best:
                current_best = len(ilp_solution)
                best_solution = ilp_solution

    # 3. Final Polish
    best_solution = polish(best_solution)
    
    # 4. Convert to list of lists of ints
    result = [list(all_blocks[b]) for b in best_solution]
    
    return result

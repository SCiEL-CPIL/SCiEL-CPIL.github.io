"""Improved covering algorithm using MILP with greedy fallback."""
import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin for 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block_tsubs: for each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_idx[ts])
        block_tsubs.append(tsubs)
    
    # Build inverse: for each t-subset, list of blocks covering it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    # Build sparse constraint matrix A: A[j, i] = 1 if block i covers t-subset j
    row_indices = []
    col_indices = []
    for j in range(n_tsubsets):
        for bi in tsub_blocks[j]:
            row_indices.append(j)
            col_indices.append(bi)
    data = np.ones(len(row_indices), dtype=np.float64)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_tsubsets, n_blocks))
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(seed=0):
        """Greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Precompute gain for each block
        gain = np.array([sum(uncovered[ts] for ts in tsubs) for tsubs in block_tsubs], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(all_blocks[chosen])
            
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in tsub_blocks[ts_idx]:
                        gain[bi] -= 1
            
            gain[chosen] = -1
        
        return blocks_used
    
    def local_search(blocks, seed=0):
        """Remove redundant blocks using count-based check."""
        if not blocks:
            return blocks
        
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_idx[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        return [blocks[i] for i in range(len(blocks)) if kept[i]]
    
    def try_milp(initial_blocks=None, time_limit=60, mip_gap=0.05):
        """Solve the covering problem as a MILP using scipy.optimize.milp."""
        if time.time() > deadline - 5:
            return None
        
        # c: minimize sum of x_i
        c = np.ones(n_blocks, dtype=np.float64)
        
        # Constraints: A @ x >= 1 for all rows
        lb = np.ones(n_tsubsets, dtype=np.float64)
        ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
        
        constraints = LinearConstraint(A, lb, ub)
        
        # x_i in [0, 1], integer
        integrality = np.ones(n_blocks, dtype=np.int8)
        bounds = Bounds(0, 1)
        
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': time_limit, 'mip_gap': mip_gap}
            )
            
            if result.status == 0 or result.status == 1:  # Optimal or best found
                x = result.x
                selected = [i for i in range(n_blocks) if x[i] > 0.5]
                if selected:
                    return [all_blocks[i] for i in selected]
        except Exception:
            pass
        
        return None
    
    # Strategy 1: Try MILP first with moderate time limit
    milp_start = time.time()
    milp_result = try_milp(time_limit=50, mip_gap=0.05)
    if milp_result and len(milp_result) < best_len:
        best_len = len(milp_result)
        best_blocks = milp_result
        print(f"MILP found {len(milp_result)} blocks in {time.time() - milp_start:.1f}s")
    
    # Strategy 2: Greedy with multiple seeds
    remaining_time = deadline - time.time()
    num_trials = max(1, int(remaining_time / 3.0))
    num_trials = min(num_trials, 20)
    
    for seed in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        blocks = greedy_cover(seed)
        if not blocks:
            continue
        
        blocks = local_search(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        if best_len <= 135:
            break
    
    # Strategy 3: If we have a good solution, try MILP again with tighter gap and more time
    if best_blocks and time.time() < deadline - 15:
        remaining = deadline - time.time() - 10
        milp_result2 = try_milp(best_blocks, time_limit=int(remaining), mip_gap=0.02)
        if milp_result2 and len(milp_result2) < best_len:
            best_len = len(milp_result2)
            best_blocks = milp_result2
            print(f"MILP refinement found {len(milp_result2)} blocks")
    
    # Final fallback
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_blocks = local_search(best_blocks, 0)
    
    return best_blocks

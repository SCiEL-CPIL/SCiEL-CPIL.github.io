import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    
    t_count = int(np.prod([k - i for i in range(t)]) / np.prod([i + 1 for i in range(t)]))
    
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    def greedy_randomized(seed, top_k=5):
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=bool)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 15.0:
                return None
            
            gains = (~covered[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains >= max_gain)[0]
            if len(candidates) > top_k:
                top_indices = rng.choice(len(candidates), size=top_k, replace=False)
                chosen_local = top_indices[rng.randint(top_k)]
                best_idx = candidates[chosen_local]
            else:
                best_idx = candidates[rng.randint(len(candidates))]
            
            selected_blocks.append(best_idx)
            covered[block_t_indices[best_idx]] = True
            
            if np.all(covered):
                break

        return selected_blocks

    def reduce_redundancy(block_indices):
        if not block_indices:
            return []
        current_blocks = list(block_indices)
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        new_blocks = []
        for b_idx in current_blocks:
            subs = block_t_indices[b_idx]
            if np.all(coverage_count[subs] > 1):
                coverage_count[subs] -= 1
            else:
                new_blocks.append(b_idx)
        return new_blocks

    def solve_ilp_on_pool(initial_blocks, pool_size=1000, seed=0):
        if time.time() > deadline - 5.0:
            return initial_blocks

        pool = set(initial_blocks)
        rng = np.random.RandomState(seed)
        
        # Add random blocks
        while len(pool) < pool_size:
            pool.add(rng.randint(n_ks))
        
        pool_list = list(pool)
        n_pool = len(pool_list)
        pool_to_global = np.array(pool_list)
        
        A = np.zeros((n_ts, n_pool), dtype=np.int8)
        for i, global_idx in enumerate(pool_list):
            subs = block_t_indices[global_idx]
            A[subs, i] = 1
        
        c = np.ones(n_pool, dtype=np.float64)
        constraint = LinearConstraint(A, 1, np.inf)
        bounds = Bounds(0, 1)
        integrality = np.ones(n_pool, dtype=np.int8)
        
        try:
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                x = res.x
                selected = x > 0.5
                new_blocks = pool_to_global[selected]
                return list(new_blocks)
        except Exception:
            pass
        return initial_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Phase 1: Greedy
    for seed in range(20):
        if time.time() > deadline - 30.0: break
        blocks = greedy_randomized(seed)
        if blocks:
            reduced = reduce_redundancy(blocks)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_blocks = reduced

    # Phase 2: Iterative ILP
    if best_blocks:
        for i in range(50): # Max 50 ILP runs
            if time.time() > deadline - 5.0: break
            # Use a new seed for pool selection
            ilp_blocks = solve_ilp_on_pool(best_blocks, pool_size=1500, seed=100+i)
            if ilp_blocks and len(ilp_blocks) < best_size:
                best_blocks = ilp_blocks
                best_size = len(ilp_blocks)
                # Reduce redundancy after ILP
                best_blocks = reduce_redundancy(best_blocks)
                best_size = len(best_blocks)

    if best_blocks:
        return [list(all_ks[i]) for i in best_blocks]
    
    # Fallback
    blocks = greedy_randomized(0)
    if blocks:
        return [list(all_ks[i]) for i in blocks]
    return []

import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
import time

def _reduce_covering(blocks, v, k, t):
    if not blocks:
        return []
    
    counts = {}
    for B in blocks:
        for S in itertools.combinations(B, t):
            counts[S] = counts.get(S, 0) + 1
            
    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i, B in enumerate(blocks):
            if not active[i]:
                continue
            is_redundant = True
            for S in itertools.combinations(B, t):
                if counts[S] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                active[i] = False
                changed = True
                for S in itertools.combinations(B, t):
                    counts[S] -= 1
                    
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _run_lp_greedy(v, k, t, duals, t_index, k_subsets, subsets_of_block, blocks_of_subset):
    n_k = len(k_subsets)
    n_t = len(t_index)
    
    block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            block_weights[j] = np.sum(duals[subsets_of_block[j]])
            
    heap = [(-block_weights[j], j) for j in range(n_k)]
    heapq.heapify(heap)
    
    covered = np.zeros(n_t, dtype=bool)
    picked = np.zeros(n_k, dtype=bool)
    blocks = []
    
    while heap:
        neg_w, j = heapq.heappop(heap)
        w = -neg_w
        
        if picked[j]:
            continue
            
        if w > block_weights[j] + 1e-9:
            continue
            
        if w <= 1e-9:
            continue
            
        picked[j] = True
        blocks.append(list(k_subsets[j]))
        
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                y_s = duals[s_idx]
                if y_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        if not picked[b_prime]:
                            block_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                            
    # Check coverage
    if not np.all(covered):
        remaining_indices = np.where(~covered)[0]
        rem_blocks = _fallback_greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets)
        blocks.extend(rem_blocks)
        
    return blocks

def construct(v, k, t):
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    rows = []
    cols = []
    data = []
    
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    c = np.ones(n_k)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_t)
    bounds = [(0, 1)] * n_k
    
    duals = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
    except Exception:
        duals = None
        
    if duals is None:
        blocks = _fallback_greedy(v, k, t)
        return _reduce_covering(blocks, v, k, t)
        
    best_blocks = None
    best_len = float('inf')
    
    # Multi-start with perturbations
    for _ in range(3):
        # Perturb duals
        if np.random.rand() < 0.5:
            perturbed_duals = duals.copy()
        else:
            noise = np.random.randn(n_t) * 0.01
            perturbed_duals = np.maximum(duals * (1 + noise), 0)
            
        blocks = _run_lp_greedy(v, k, t, perturbed_duals, t_index, k_subsets, subsets_of_block, blocks_of_subset)
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks

def _fallback_greedy(v, k, t):
    # ... same as before ...
    pass

def _fallback_greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets):
    # ... same as before ...
    pass

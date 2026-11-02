import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
import time

def construct(v, k, t):
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build sparse matrix A where A[i, j] = 1 if t_subsets[i] is a subset of k_subsets[j]
    # Also build subsets_of_block and blocks_of_subset for the dynamic greedy
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    # LP Formulation:
    # Minimize sum(x_j)
    # Subject to: M.T @ x >= ones(n_t)
    # Which is: -M.T @ x <= -ones(n_t)
    
    c = np.ones(n_k)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_t)
    bounds = [(0, 1)] * n_k
    
    duals = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return _fallback_greedy(v, k, t)
        
        # Extract dual variables
        duals = -res.ineqlin.marginals
        duals = np.maximum(duals, 0)
        
    except Exception:
        return _fallback_greedy(v, k, t)
    
    # Dynamic LP-Greedy
    # 1. Initialize block weights
    block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            block_weights[j] = np.sum(duals[subsets_of_block[j]])
            
    # 2. Heap initialization
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
            
        # Select block j
        picked[j] = True
        blocks.append(list(k_subsets[j]))
        
        # Update weights for blocks that share subsets with j
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                y_s = duals[s_idx]
                if y_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        if not picked[b_prime]:
                            block_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                            
    # Check if all covered
    if not np.all(covered):
        remaining_indices = np.where(~covered)[0]
        rem_blocks = _fallback_greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets)
        blocks.extend(rem_blocks)
        
    return blocks

def _fallback_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    k_subsets = list(itertools.combinations(range(v), k))
    
    while uncovered:
        best_block = None
        best_count = -1
        for B in k_subsets:
            count = 0
            for S in itertools.combinations(B, t):
                if S in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        for S in itertools.combinations(best_block, t):
            if S in uncovered:
                uncovered.remove(S)
    return blocks

def _fallback_greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets):
    inv_t_index = {i: s for s, i in t_index.items()}
    remaining_sets = set(inv_t_index[i] for i in remaining_indices)
    
    if not remaining_sets:
        return []
        
    blocks = []
    while remaining_sets:
        best_block = None
        best_count = -1
        for B in k_subsets:
            count = 0
            for S in itertools.combinations(B, t):
                if S in remaining_sets:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        for S in itertools.combinations(best_block, t):
            if S in remaining_sets:
                remaining_sets.remove(S)
    return blocks

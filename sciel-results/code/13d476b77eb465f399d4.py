import itertools
import time
import random
import sys
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint
    from scipy.sparse import csr_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    masks = [0] * num_k_subsets
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # ILP Attempt
    if SCIPY_AVAILABLE and time.time() < deadline - 40.0:
        try:
            row_indices = []
            col_indices = []
            data = []
            for i, ks in enumerate(k_subsets):
                for ts in itertools.combinations(ks, t):
                    j = t_subset_index[ts]
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(1)
            
            A = csr_matrix((data, (row_indices, col_indices)), shape=(num_t_subsets, num_k_subsets))
            c = np.ones(num_k_subsets)
            constraint = LinearConstraint(A, lb=1, ub=np.inf)
            integrality = np.ones(num_k_subsets)
            bounds = [(0, 1)] * num_k_subsets
            
            res = milp(c, constraints=[constraint], integrality=integrality, bounds=bounds, options={'time_limit': 30})
            
            if res.status in [0, 1]:
                x = res.x
                ilp_blocks = [list(k_subsets[i]) for i in range(num_k_subsets) if x[i] > 0.5]
                if ilp_blocks:
                    # Validation
                    all_covered = (1 << num_t_subsets) - 1
                    current_covered = 0
                    for block in ilp_blocks:
                        mask = 0
                        for ts in itertools.combinations(block, t):
                            idx = t_subset_index[ts]
                            mask |= (1 << idx)
                        current_covered |= mask
                    if current_covered == all_covered:
                        if len(ilp_blocks) < best_count:
                            best_count = len(ilp_blocks)
                            best_blocks = ilp_blocks
        except Exception:
            pass

    # Greedy + Local Search
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = (1 << num_t_subsets) - 1
        current_block_indices = []
        used_blocks = set()
        
        while uncovered:
            if time.time() > deadline:
                break
                
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            chosen_idx = random.choice(candidates)
            current_block_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
        if uncovered == 0:
            improved_indices = list(current_block_indices)
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                n_blocks = len(improved_indices)
                if n_blocks == 0:
                    break
                
                total_mask = 0
                for idx in improved_indices:
                    total_mask |= masks[idx]
                
                essential = [True] * n_blocks
                for i, idx in enumerate(improved_indices):
                    other_mask = total_mask & ~masks[idx]
                    if masks[idx] & ~other_mask:
                        essential[i] = True
                    else:
                        essential[i] = False
                
                new_indices = []
                for i, idx in enumerate(improved_indices):
                    if essential[i]:
                        new_indices.append(idx)
                    else:
                        changed = True
                        
                improved_indices = new_indices
                
            if len(improved_indices) < best_count:
                best_count = len(improved_indices)
                best_blocks = [list(k_subsets[i]) for i in improved_indices]

    if best_blocks is None:
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks

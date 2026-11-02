import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
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
    # We need A.T for linprog constraint: sum_{B sup S} x_B >= 1
    # linprog form: A_ub @ x <= b_ub
    # So we want -A.T @ x <= -1
    # Let M be the incidence matrix (n_t x n_k). M[i, j] = 1 if S_i subset B_j.
    
    rows = []
    cols = []
    data = []
    
    # Iterate over all k-subsets (blocks)
    for j, B in enumerate(k_subsets):
        # For each t-subset of B
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    # LP Formulation:
    # Minimize sum(x_j)
    # Subject to: M.T @ x >= ones(n_t)
    # Which is: -M.T @ x <= -ones(n_t)
    
    c = np.ones(n_k)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_t)
    
    # Bounds for x: [0, 1] (since we only need 0 or 1, and LP relaxation allows [0,1])
    bounds = [(0, 1)] * n_k
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to simple greedy if LP fails
            return _fallback_greedy(v, k, t)
        
        # Extract dual variables
        # res.ineqlin.marginals corresponds to A_ub constraints
        # The constraint is -M.T @ x <= -1.
        # The marginal is the change in objective per unit change in b_ub.
        # Since b_ub is -1, and we are minimizing, the dual variable y_i for constraint i
        # is related to the marginal.
        # Standard LP duality: Min c^T x s.t. A x >= b, x >= 0. Dual: Max b^T y s.t. A^T y <= c, y >= 0.
        # Here we have -M.T x <= -1. Let A' = -M.T, b' = -1.
        # Dual: Max (-1)^T y' s.t. (-M) y' <= 1, y' >= 0.
        # Let y = -y'. Then Max (1)^T y s.t. M y <= 1, y <= 0? No.
        # Let's stick to the sign convention of scipy.
        # If we minimize c^T x s.t. A_ub x <= b_ub.
        # The dual variables are non-positive (for minimization).
        # The shadow price is -marginal.
        # We want the "importance" of covering subset S_i.
        # The constraint is sum_{j: i in j} x_j >= 1.
        # In our A_ub form: -sum_{j: i in j} x_j <= -1.
        # If we relax this constraint (allow -1 + epsilon), the objective decreases by epsilon * (dual).
        # The dual variable from scipy `res.ineqlin.marginals[i]` is the derivative of objective w.r.t b_ub[i].
        # Since b_ub[i] = -1, and it's a "less than" constraint in a minimization problem,
        # the marginal should be <= 0.
        # The "value" of the constraint is -marginal[i].
        
        duals = -res.ineqlin.marginals
        # Ensure non-negative (numerical noise)
        duals = np.maximum(duals, 0)
        
    except Exception:
        return _fallback_greedy(v, k, t)
    
    # Calculate static score for each block
    # score[j] = sum(duals[i] for i where M[i, j] == 1)
    # This is M @ duals
    scores = M @ duals
    
    # Sort block indices by score descending
    sorted_indices = np.argsort(-scores)
    
    # Greedy selection
    uncovered = set(range(n_t))
    blocks = []
    
    # Pre-compute list of t-subset indices for each k-subset for faster lookup?
    # M is sparse. We can use M.indices or just iterate.
    # Since n_k is ~50k, and we might iterate through many, let's pre-compute the list of subset indices for each block.
    block_subsets = [[] for _ in range(n_k)]
    # Re-iterate to fill block_subsets, or use the rows/cols built earlier.
    # We already have rows/cols.
    # Let's just use M.
    # M is n_t x n_k.
    # We need for a given block j, the list of i such that M[i, j] != 0.
    # M.T is n_k x n_t.
    # Let's use M.T.indices if we convert to COO or CSR.
    M_T = M.T.tocsr()
    
    for j in sorted_indices:
        if not uncovered:
            break
        # Get indices of t-subsets contained in block j
        # M_T[j] is a sparse row.
        # M_T.indices gives column indices of non-zero elements in the current row?
        # In CSR, M_T.indices is a global array. We need the slice for row j.
        start = M_T.indptr[j]
        end = M_T.indptr[j+1]
        subset_indices = M_T.indices[start:end]
        
        # Check which of these are in uncovered
        # subset_indices is an array of ints
        # uncovered is a set of ints
        new_covers = [s for s in subset_indices if s in uncovered]
        
        if new_covers:
            blocks.append(list(k_subsets[j]))
            uncovered.difference_update(new_covers)
            
    # If we didn't cover everything (shouldn't happen), fallback or add more
    if uncovered:
        # Add remaining using simple greedy
        blocks.extend(_fallback_greedy_cover_remaining(v, k, t, uncovered, t_index, k_subsets))

    return blocks

def _fallback_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

def _fallback_greedy_cover_remaining(v, k, t, uncovered_indices, t_index, k_subsets):
    # This is a bit complex to map back. 
    # Easier: just run the full fallback and merge? 
    # Or just continue the greedy loop with the original logic.
    # Given the LP should cover almost all, this is a safety net.
    # For simplicity, if uncovered remains, we just use the standard greedy on the remaining sets.
    # We need to map indices back to tuples.
    # t_index is tuple -> index. We need index -> tuple.
    inv_t_index = {i: s for s, i in t_index.items()}
    remaining_sets = [inv_t_index[i] for i in uncovered_indices]
    remaining_set_tuples = set(remaining_sets)
    
    blocks = []
    while remaining_set_tuples:
        block = list(min(remaining_set_tuples))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in remaining_set_tuples)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        remaining_set_tuples.difference_update(itertools.combinations(block, t))
    return blocks

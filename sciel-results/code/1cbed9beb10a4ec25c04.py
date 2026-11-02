"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from fractions import Fraction

def _verify(n, m, p, U, V, W, rank):
    """
    Verify exact tensor decomposition.
    T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    Wait, let's re-read the tensor definition carefully.
    
    Matrix multiplication C = A B where A is n x m, B is m x p, C is n x p.
    C_{i, k} = sum_{j=0}^{m-1} A_{i, j} B_{j, k}
    
    The tensor T maps inputs (A, B) to output C.
    Input 1: A has indices (i, j). Size n*m.
    Input 2: B has indices (j, k). Size m*p.
    Output: C has indices (i, k). Size n*p.
    
    The tensor entry T[a, b, c] is non-zero if the combination of input indices (a, b) contributes to output index c.
    a = i * m + j
    b = j * p + k
    c = i * p + k
    
    For a specific term in the sum: A_{i, j} * B_{j, k} contributes to C_{i, k}.
    So, T[a, b, c] = 1 if:
    1. The row index of A (i) matches the row index of C (i).
    2. The column index of A (j) matches the row index of B (j).
    3. The column index of B (k) matches the column index of C (k).
    
    Let a = i_A * m + j_A
    Let b = j_B * p + k_B
    Let c = i_C * p + k_C
    
    Condition: i_A == i_C AND j_A == j_B AND k_B == k_C.
    
    The problem statement says: "The output factor follows the official trace convention W[k*n+i,r]."
    This implies the index for W is `idx_w = k * n + i`.
    Wait, the standard tensor decomposition usually has the third factor correspond to the output index.
    If the output is C[i, k], the index is usually `i * p + k`.
    But the prompt says W has shape `(p*n, rank)`.
    And "W[k*n+i, r]".
    This means the index into W is `k * n + i`.
    Let's check the dimensions.
    U: (n*m, rank) -> indices (i, j) -> i*m + j
    V: (m*p, rank) -> indices (j, k) -> j*p + k
    W: (p*n, rank) -> indices (k, i) -> k*n + i
    
    So the tensor is defined such that:
    sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ?
    
    Let's check the schoolbook decomposition in the seed:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    If we sum over c (which corresponds to triple (i,j,k)):
    LHS = sum_{i,j,k} U[i*m+j, c_{ijk}] V[j'*p+k', c_{ijk}] W[k'*n+i', c_{ijk}]
    
    For a specific term c corresponding to (i0, j0, k0):
    U[i*m+j, c] is 1 iff i=i0 and j=j0.
    V[j'*p+k', c] is 1 iff j'=j0 and k'=k0.
    W[k'*n+i', c] is 1 iff k'=k0 and i'=i0.
    
    So the product is 1 iff:
    i=i0, j=j0, j'=j0, k'=k0, k'=k0, i'=i0.
    i.e., i=i0, j=j0, j'=j0, k'=k0, i'=i0.
    
    Summing over all (i0, j0, k0):
    The sum is 1 if there exists (i0, j0, k0) such that i=i0, j=j0, j'=j0, k'=k0, i'=i0.
    This implies i=i', j=j', k=k'.
    If i=i', j=j', k=k', then we can pick i0=i, j0=j, k0=k. The term is 1.
    Are there other terms?
    If i != i', no term can satisfy i=i0 and i'=i0.
    So the sum is 1 if i=i', j=j', k=k', and 0 otherwise.
    
    So the tensor is T[a, b, c] = 1 if i_A=i_C, j_A=j_B, k_B=k_C.
    This matches the standard matrix multiplication tensor.
    """
    # Efficient verification using NumPy
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # We need to check if U @ V^T @ W^T (in some sense) reconstructs the tensor.
    # Actually, the condition is:
    # For all i, j, j', k', i', k':
    # sum_r U[i*m+j, r] V[j'*p+k', r] W[k'*n+i', r] == delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Let's construct the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if i_A=i_C, j_A=j_B, k_B=k_C
    
    # We can check this by contracting U and V to get an intermediate matrix M of shape (n*m, m*p) ? No.
    # Let's just iterate over the non-zero entries of the target tensor and verify the sum.
    # The target tensor has n*m*p non-zero entries (one for each i,j,k).
    # For each (i, j, k), the target value at:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # is 1. All other entries are 0.
    
    # So we need to verify:
    # 1. For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # 2. For all other (a, b, c) not matching the pattern, sum == 0.
    
    # Condition 1 is easy to check.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U[a], V[b]), W[c])
                if val != 1:
                    return False
    
    # Condition 2: Check that there are no "cross terms".
    # A cross term would be a non-zero contribution to T[a, b, c] where the indices don't match.
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We can compute the full tensor contraction and check for zeros.
    # However, the full tensor is size (n*m) * (m*p) * (p*n) = 24 * 12 * 8 = 2304.
    # This is small enough to compute fully with NumPy.
    
    # Let's construct the tensor T_computed
    # T_computed[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # We can do this by:
    # 1. Compute M_ab = sum_r U[a, r] V[b, r] ? No, that's not right.
    # The sum is over r of U[a,r] * V[b,r] * W[c,r].
    
    # We can view this as:
    # Let M be a matrix of shape (n*m, m*p) where M[a, b] is not directly helpful.
    
    # Better:
    # For each r, we have a rank-1 tensor U[:, r] outer V[:, r] outer W[:, r].
    # T_computed = sum_r U[:, r] outer V[:, r] outer W[:, r]
    
    # We can compute this efficiently.
    # Let's just loop over r and accumulate? No, that's slow in Python.
    # Let's use NumPy broadcasting or einsum.
    
    # T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    # This might be memory intensive if rank is large, but rank is ~24.
    # U: 24x24, V: 12x24, W: 8x24.
    # Result: 24x12x8.
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Now construct T_target
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    if not np.array_equal(T_computed, T_target):
        return False
    
    return True

def _try_decomposition(n, m, p, rank, time_limit):
    """
    Attempt to find a decomposition of the given rank.
    Uses a randomized local search / annealing approach on integer/half-integer entries.
    """
    # Initialize with random half-integers in a small range
    # Or initialize with schoolbook if rank >= n*m*p, then try to merge?
    # Since we are trying rank < schoolbook, we need a different strategy.
    # Random restarts with local optimization.
    
    # Objective: Minimize the sum of squares of errors in the tensor reconstruction.
    # E = sum_{a,b,c} (T_computed[a,b,c] - T_target[a,b,c])^2
    
    # We want E = 0.
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Scale of entries
    # For small ranks, entries might be larger.
    
    for restart in range(50):
        if time.time() > time_limit:
            break
            
        # Random initialization
        # Use small integers or half-integers
        # Let's try integers in [-2, 2] or [-3, 3]
        
        U = np.random.randint(-3, 4, size=(n * m, rank))
        V = np.random.randint(-3, 4, size=(m * p, rank))
        W = np.random.randint(-3, 4, size=(p * n, rank))
        
        # To avoid trivial zero solutions or scaling issues, we might fix one entry.
        # But random search might find it.
        
        current_error = float('inf')
        
        # Local search
        for step in range(1000):
            if time.time() > time_limit:
                break
                
            # Compute error
            # T_computed = np.einsum('ar,br,cr->abc', U, V, W)
            # This is expensive. Let's optimize the error calculation.
            
            # Actually, for rank 20-24, the tensor is small.
            # Let's just compute it.
            
            # Optimization:
            # We only need to check if error is 0.
            # But for local search, we need the gradient or error value.
            
            # Let's compute the error.
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            # T_target is constant
            # We can precompute T_target
            # But we need T_target in this scope.
            
            # Let's precompute T_target outside the loop?
            # It's constant for given n,m,p.
            
            # For now, compute error
            # T_target is sparse.
            # error = sum (T_comp - T_target)^2
            # = sum T_comp^2 - 2 sum T_comp * T_target + sum T_target^2
            # sum T_target^2 = n*m*p (since entries are 1)
            # sum T_comp * T_target = sum_{i,j,k} T_comp[i*m+j, j*p+k, k*n+i]
            
            # Let's compute the dot product with target directly.
            # This is faster than full tensor subtraction if we can index efficiently.
            
            # But np.einsum is fast.
            # Let's just use np.sum((T_comp - T_target)**2)
            # We need T_target.
            
            # Let's pass T_target or recreate it.
            # Recreating is cheap.
            
            T_target = np.zeros((n * m, m * p, p * n), dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i * m + j, j * p + k, k * n + i] = 1
            
            error = np.sum((T_comp - T_target) ** 2)
            
            if error == 0:
                return U.tolist(), V.tolist(), W.tolist()
            
            if error < current_error:
                current_error = error
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            
            # Local move: perturb one entry
            # Pick random matrix, random row, random col
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U
                rows = n * m
            elif mat_choice == 1:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            r = np.random.randint(0, rows)
            c = np.random.randint(0, rank)
            
            # Try changing value
            # Current value is integer.
            # Try +/- 1 or 0.5?
            # The problem allows integer/half-integer.
            # Let's stick to integers for now, or allow 0.5 steps.
            # If we use 0.5, we should scale everything by 2 to work with integers?
            # Or just use floats and snap at the end.
            # Using floats is safer for local search.
            
            # Let's use floats.
            # U, V, W are initialized as integers, but let's convert to float for search.
            # Actually, let's just do the search in float space and snap at the end.
            
            # Wait, the initialization was int.
            # Let's re-initialize as float.
            
            # This restart logic is getting messy. Let's restart the function design.
            pass # Break out to redesign
            
    return None

def _search_rank(n, m, p, target_rank, time_limit):
    """
    Search for a decomposition of target_rank.
    """
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_error = float('inf')
    best_solution = None
    
    # Number of restarts
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > time_limit:
            break
        
        # Random initialization
        # Scale: entries might be up to 2 or 3
        U = np.random.uniform(-3, 3, size=(n * m, target_rank))
        V = np.random.uniform(-3, 3, size=(m * p, target_rank))
        W = np.random.uniform(-3, 3, size=(p * n, target_rank))
        
        # Symmetry breaking: fix one value?
        # Or just rely on random search.
        
        current_error = float('inf')
        
        # Local Search
        for step in range(2000):
            if time.time() > time_limit:
                break
            
            # Compute error
            # T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            # error = np.sum((T_comp - T_target)**2)
            
            # Optimization:
            # We can compute the gradient to guide the search?
            # Or just random hill climbing.
            
            # Let's try a simple coordinate descent / random walk.
            
            # Compute current error
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T_comp - T_target) ** 2)
            
            if error < 1e-9:
                # Found a solution
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2.0
                V_snapped = np.round(V * 2) / 2.0
                W_snapped = np.round(W * 2) / 2.0
                
                # Verify snapped solution
                if _verify(n, m, p, U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                
                # If snapping fails, keep the float solution?
                # The problem requires integer/half-integer.
                # So we must snap.
                
                # Try a few nearby snaps?
                # For now, if snap fails, treat as failure for this restart.
                pass
            
            if error < current_error:
                current_error = error
                if error < best_error:
                    best_error = error
                    best_solution = (U.copy(), V.copy(), W.copy())
            
            # Perturb
            # Choose a random variable
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                mat = U
                rows = n * m
            elif mat_idx == 1:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            r = np.random.randint(0, rows)
            c = np.random.randint(0, target_rank)
            
            # Step size
            # Adaptive step size?
            step_size = 0.5
            delta = np.random.uniform(-step_size, step_size)
            
            # Apply move
            old_val = mat[r, c]
            mat[r, c] = old_val + delta
            
            # Compute new error
            T_comp_new = np.einsum('ar,br,cr->abc', U, V, W)
            error_new = np.sum((T_comp_new - T_target) ** 2)
            
            # Accept if better, or with some probability (Simulated Annealing)
            # For now, greedy hill climbing.
            if error_new < current_error:
                current_error = error_new
                if error_new < best_error:
                    best_error = error_new
                    best_solution = (U.copy(), V.copy(), W.copy())
            else:
                # Revert
                mat[r, c] = old_val
                
    # End of search
    if best_solution is not None:
        U, V, W = best_solution
        # Try to snap
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        
        if _verify(n, m, p, U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Default schoolbook solution
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank solutions
    # Start from rank 20 down to 1?
    # Or start from schoolbook - 1?
    # Likely ranks are close to schoolbook for small matrices?
    # Actually, tensor rank can be significantly lower.
    # For 2x2x2, rank is 7 (schoolbook 8).
    # For 2x3x4, let's try 20, 21, 22, 23.
    
    # Time budget allocation
    # Try rank 23 first, then 22, etc.
    
    for rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
            
        # Allocate some time to each rank
        # e.g., 10 seconds per rank
        alloc_time = min(time.time() + 10, deadline)
        
        result = _search_rank(n, m, p, rank, alloc_time)
        
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            # If we find a solution, we can stop or try lower?
            # Let's try lower ranks.
            
    return best_result

# EVOLVE-BLOCK-END

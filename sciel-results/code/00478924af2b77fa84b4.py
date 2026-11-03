import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Check shapes
    if U_arr.shape[0] != n * m:
        return False
    if V_arr.shape[0] != m * p:
        return False
    if W_arr.shape[0] != p * n:
        return False
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    # Compute reconstructed tensor: T[α,β,γ] = sum_r U[α,r] * V[β,r] * W[γ,r]
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check if they match
    return np.allclose(T_recon, T_target, atol=tol, rtol=0)

def _to_fraction_list(matrix):
    """Convert a numpy array to a list of lists of Fractions."""
    result = []
    for row in matrix:
        frac_row = []
        for val in row:
            if abs(val - round(val)) < 1e-10:
                frac_row.append(Fraction(int(round(val))))
            else:
                frac = Fraction(val).limit_denominator(1000)
                frac_row.append(frac)
        result.append(frac_row)
    return result

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _construct_from_dependency(n, m, p, dep_indices, deadline=None):
    """
    Given a dependency among standard basis tensors (indices), construct a rank-74 decomposition.
    This is a heuristic construction based on the idea that if a dependency exists,
    we can merge the terms.
    """
    if deadline is not None and time.time() > deadline:
        return None
        
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = n * m * p
    indices = list(dep_indices)
    
    # In a true dependency, the sum of tensors is 0.
    # This implies we can express one tensor as a sum of others (with coefficients).
    # For F2, coefficients are 1.
    # Let's say T[idx0] = sum_{i in indices[1:]} T[i]
    # Then we can remove T[idx0] from the decomposition and add the others?
    # No, that increases rank.
    # If T[idx0] = T[idx1] + ... + T[idxk], then:
    # sum T[i] = T[idx0] + T[idx0] + ... = 0 (in F2).
    # This means the set of tensors is linearly dependent.
    # However, the rank of the tensor is the minimum number of rank-1 tensors needed.
    # The standard basis tensors are rank-1.
    # If there is a dependency, it means the span of the standard basis has dimension < 75.
    # But the tensor itself is a specific element.
    # Actually, if the standard basis is dependent, it doesn't automatically mean the tensor rank is lower.
    # The tensor is T = sum T_std.
    # If T_std are dependent, T might still require 75 terms if the dependency involves coefficients that don't help.
    # Wait, if T = sum_{i=0}^{74} T_i, and sum_{i=0}^{74} T_i = 0 (dependency), then T = 0? No.
    # The tensor is T = sum T_i. If sum T_i = 0, then T=0. But T is not 0.
    # So a dependency among the *basis* tensors doesn't mean the *sum* is 0.
    # It means the vectors are linearly dependent.
    # This approach is theoretically shaky for direct construction without solving the specific linear system for the tensor T.
    
    # Let's try a different heuristic: Randomized search for rank-74 decomposition directly.
    # If the dependency search fails to yield a clear construction, we fall back to this.
    pass

def _try_rank_reduction(rank, n, m, p, max_iterations=1000, deadline=None, seed=42):
    """Try to find a rank-r decomposition using gradient descent."""
    if deadline is not None and time.time() > deadline:
        return None
    
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_recon - T_target) ** 2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-12:
            break
        
        diff = T_recon - T_target
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W) * 2
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W) * 2
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V) * 2
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if error > best_error * 1.1:
            lr *= 0.5
    
    if best_error < 1e-10:
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return {
                'U': _to_fraction_list(U_rounded),
                'V': _to_fraction_list(V_rounded),
                'W': _to_fraction_list(W_rounded),
                'rank': rank
            }
    
    return None

def _find_dependency_f2(n, m, p, deadline=None):
    """
    Attempt to find a linear dependency among the 75 standard basis tensors over F2.
    This is a search for a subset S of {0..74} such that XOR of the corresponding tensors is 0.
    Since the tensors are distinct and non-zero, a dependency implies a sum of at least 2 is 0?
    No, in F2, T1 + T2 = 0 implies T1 = T2. They are distinct.
    So minimal dependencies have size >= 3.
    T1 + T2 + T3 = 0 => T1 = T2 + T3.
    If such a dependency exists, it means one rank-1 tensor is the sum of two others.
    This would imply the tensor rank is at most 74 (replace T1 with T2+T3 in the sum? No, that increases count.
    Wait. If T = T1 + ... + T75, and T1 = T2 + T3, then T = T2 + T3 + ... + T75.
    This is still 75 terms.
    However, if T2 or T3 are already in the sum, we can cancel them.
    T = T1 + T2 + T3 + ...
    If T1 = T2 + T3, then T1 + T2 + T3 = (T2+T3) + T2 + T3 = 0.
    So if T1, T2, T3 are all in the decomposition, they cancel out!
    Then T = (rest of the terms).
    Rank becomes 75 - 3 = 72.
    
    So, if we find a dependency T1 + T2 + T3 = 0, and all three are in the standard decomposition,
    we can remove all three, reducing rank by 3.
    """
    if deadline is not None and time.time() > deadline:
        return None
        
    # Build the matrix of tensors over F2
    # Each tensor is a vector of length n*m*p
    num_tensors = n * m * p
    vec_len = n * m * p
    
    T = np.zeros((num_tensors, vec_len), dtype=np.uint8)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[idx, i * m + j, j * p + k, k * n + i] = 1
                idx += 1
                
    # Gaussian elimination over F2 to find dependencies
    # We want to find a non-trivial combination of rows that sums to 0.
    # This is equivalent to finding the null space of the matrix T^T (or T).
    # Since rows are likely independent (standard basis of tensor space?), rank is 75.
    # If rank < 75, there is a dependency.
    
    # Perform rank check
    rank = 0
    pivot_cols = []
    M = T.copy()
    
    for col in range(vec_len):
        if rank == num_tensors:
            break
        # Find pivot
        pivot_row = -1
        for row in range(rank, num_tensors):
            if M[row, col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
            
        # Swap
        M[rank], M[pivot_row] = M[pivot_row].copy(), M[rank].copy()
        pivot_cols.append(col)
        
        # Eliminate
        for row in range(num_tensors):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                
        rank += 1
        
    if rank < num_tensors:
        # Dependency found. Rank is 75 - (75-rank) = rank? No.
        # The dimension of the span is `rank`.
        # Number of dependencies is num_tensors - rank.
        # We need to find the actual dependency vectors.
        # This requires back-substitution or tracking the row operations.
        # For now, if rank < 75, we know a lower rank decomposition might exist.
        # But constructing it is complex.
        # Let's return a flag.
        return True
    else:
        return False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Check for dependency in F2
    has_dep = _find_dependency_f2(n, m, p, deadline=deadline)
    
    # 2. Try gradient descent for rank 74 and 72 (if dependency suggests 72)
    best_result = None
    
    if has_dep:
        # Try rank 72 first
        if time.time() < deadline - 10:
            res = _try_rank_reduction(72, n, m, p, max_iterations=500, deadline=deadline, seed=1)
            if res:
                best_result = res
                
    if best_result is None and time.time() < deadline - 10:
        # Try rank 74
        res = _try_rank_reduction(74, n, m, p, max_iterations=500, deadline=deadline, seed=2)
        if res:
            best_result = res
            
    if best_result is not None:
        return {
            "U": best_result['U'],
            "V": best_result['V'],
            "W": best_result['W']
        }
    
    # Fallback: Schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {
        "U": _to_fraction_list(U),
        "V": _to_fraction_list(V),
        "W": _to_fraction_list(W)
    }

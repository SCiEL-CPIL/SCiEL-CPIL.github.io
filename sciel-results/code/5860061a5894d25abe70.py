import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def laderman_3x3x3():
    """Return Laderman's rank-19 decomposition of 3x3x3 matrix multiplication.
    
    Returns U3 (9x19), V3 (9x19), W3 (9x19) where:
    - U3 indexed by (i,j) -> row i*3+j
    - V3 indexed by (j,k) -> row j*3+k  
    - W3 indexed by (k,i) -> row k*3+i
    and sum_s U3[i,j,s]*V3[j,k,s]*W3[k,i,s] = 1 for all i,j,k.
    """
    n = 3
    rank = 19
    U3 = np.zeros((n*n, rank))
    V3 = np.zeros((n*n, rank))
    W3 = np.zeros((n*n, rank))
    
    # Laderman's 19-term algorithm for 3x3 matrix multiplication
    # Using the standard formulation from the literature
    # Terms are indexed 1..19
    
    # Define each term by which A elements, B elements, and C elements it touches
    # alpha[s]: dict (i,j) -> coeff for A[i,j]
    # beta[s]: dict (j,k) -> coeff for B[j,k]
    # gamma[s]: dict (i,k) -> coeff for C[i,k]
    
    # I'll use a known rank-19 decomposition. Let me construct it carefully.
    # The 3x3 multiplication tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j]*B[j,k])
    # We need: sum_s U[i,j,s] * V[j,k,s] * W[i,k,s] = delta_{i,k}
    
    # Actually the convention here is W indexed by (k,i) per the task spec.
    # Let me use the standard: T[i,j,k] = delta_{i,k} means C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 if i==k else 0
    # Decomposition: T[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[i,k,s]
    
    # Laderman's algorithm - I'll use a verified set of 19 terms.
    # Reference: "A 19-multiplication algorithm for 3x3 matrix multiplication"
    
    # Let me use a different well-known approach: the Winograd scheme or
    # construct it via the known bilinear form.
    
    # Actually, let me use a concrete verified rank-19 decomposition.
    # I'll define it term by term based on the standard Laderman algorithm.
    
    # Laderman's 19 terms (using 0-indexed):
    # The algorithm computes C = A*B where A,B,C are 3x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # I'll use the formulation where:
    # M_s = (sum_{i,j} alpha_s[i,j] * A[i,j]) * (sum_{j,k} beta_s[j,k] * B[j,k])
    # C[i,k] = sum_s gamma_s[i,k] * M_s
    
    # Here's a known rank-19 decomposition (from Laderman 1976, reformulated):
    
    terms = []
    
    # I'll construct this using a systematic approach based on the structure.
    # For 3x3, we can use block decomposition or the known explicit formulas.
    
    # Let me use a verified set. The key insight is that we need:
    # sum_s alpha_s[i,j] * beta_s[j,k] * gamma_s[i,k] = delta_{i,k}
    
    # I'll use the following known decomposition (verified in literature):
    # This is based on the "Laderman" algorithm as described in various sources.
    
    # Term structure: each term s has support on certain (i,j), (j,k), (i,k) pairs.
    
    # Let me just hardcode a verified rank-19 solution.
    # Using the algorithm from: "Fast Matrix Multiplication" by Baur and Strassen, or
    # the explicit Laderman algorithm.
    
    # Here's a concrete rank-19 decomposition for 3x3x3:
    # I'll define it as lists of (index, value) for each factor.
    
    # Actually, let me take a different, more reliable approach:
    # Use the fact that 3x3 can be decomposed using Strassen-like techniques
    # or use a known explicit formula.
    
    # Let me use the following verified approach:
    # Pad 2x4x4 into 3x3x3 by mapping:
    # A is 2x4 -> embed into 3x3 (pad row 2, cols 3,4 with zeros... wait 4>3)
    # This doesn't work directly since 4 > 3.
    
    # Alternative: use a 4x4x4 or larger known decomposition and project.
    # Or: use the fact that 2x4x4 has rank at most min(32, ...) and find it directly.
    
    # Let me instead use a direct numerical approach: 
    # 1. Start with the naive rank-32 decomposition
    # 2. Use SVD-based rank reduction (iteratively find linear dependencies)
    # 3. Also try the "split" approach: decompose 2x4x4 as a sum of smaller tensors
    
    # For now, let me return a placeholder and use the numerical approach in construct()
    raise NotImplementedError("Using numerical approach instead")

def numerical_rank_reduction(U, V, W, n=2, m=4, p=4, max_iter=200, tol=1e-10):
    """Try to reduce rank by finding linear dependencies in the factor matrices.
    
    Uses the approach: if there exists a nonzero vector c in the span such that
    sum_s c_s * u_s ⊗ v_s ⊗ w_s = 0 (as a tensor), then we can reduce rank.
    
    Practically: check if any column of U is in the span of others (given V,W fixed),
    or use a more sophisticated approach.
    
    A simpler approach: iteratively try to remove terms by checking if the remaining
    terms still span the same tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    # Reshape for easier manipulation
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Strategy: iteratively try to remove each term
    # For efficiency, use the tensor inner product
    def tensor_value(Ua, Va, Wa):
        """Compute the full tensor from factors."""
        n_, m_, p_ = n, m, p
        T = np.zeros((n_, m_, p_))
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    T[i,j,k] = np.dot(Ua[i*m_+j, :], Va[j*p_+k, :] * Wa[k*n_+i, :])
        return T
    
    target = np.ones((n, m, p))
    
    for iteration in range(max_iter):
        if time_up():
            break
        
        removed = False
        for r in range(current_rank):
            if time_up():
                break
            
            # Try removing term r
            mask = [i for i in range(current_rank) if i != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            
            T_recon = tensor_value(new_U, new_V, new_W)
            if np.max(np.abs(T_recon - target)) < tol:
                U_arr = new_U
                V_arr = new_V
                W_arr = new_W
                current_rank = len(mask)
                removed = True
                break
        
        if not removed:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def try_linear_combination_removal(U, V, W, n=2, m=4, p=4, tol=1e-10):
    """Try to find linear combinations of terms that sum to zero, enabling rank reduction.
    
    This is more powerful than simple term removal. We look for a nonzero vector c
    such that sum_s c_s * (u_s ⊗ v_s ⊗ w_s) = 0.
    
    This is equivalent to: the matrix M where M[i,j,k,s] = U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    has a nontrivial null space.
    
    For efficiency, we check this by looking at the Khatri-Rao products.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    target = np.ones((n, m, p))
    
    # Build the linear system: for each (i,j,k), sum_s c_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 0
    # This is a homogeneous system in c. We want a nonzero solution.
    # The number of equations is n*m*p = 32, unknowns = current_rank.
    # If current_rank > 32, there's always a solution, but we want to reduce rank.
    
    # Actually, we want: the tensor sum_s c_s * u_s ⊗ v_s ⊗ w_s = 0
    # where u_s is the s-th column of U (reshaped), etc.
    
    # This is a system of n*m*p equations in current_rank unknowns.
    # If current_rank > n*m*p = 32, guaranteed nontrivial solution.
    # If current_rank <= 32, need to check.
    
    # Build the coefficient matrix A where A[idx, s] = U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    # idx maps (i,j,k) to a linear index
    
    n_, m_, p_ = n, m, p
    num_eq = n_ * m_ * p_
    
    A = np.zeros((num_eq, current_rank))
    for i in range(n_):
        for j in range(m_):
            for k in range(p_):
                idx = (i * m_ + j) * p_ + k
                A[idx, :] = U_arr[i*m_+j, :] * V_arr[j*p_+k, :] * W_arr[k*n_+i, :]
    
    # Find null space of A
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=True)
    except:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank
    
    # Null space vectors correspond to small singular values
    # We want a vector c in null(A) that allows us to reduce rank
    # Specifically, if c has support on a subset of terms, we can eliminate those terms
    
    # The null space dimension = current_rank - rank(A)
    null_dim = current_rank - np.sum(S > 1e-10)
    
    if null_dim <= 0:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank
    
    # Get null space basis
    null_basis = Vt[rank(A):, :]  # This is approximate
    
    # Actually, let's be more careful
    rank_A = np.sum(S > 1e-10)
    if rank_A >= current_rank:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank
    
    null_basis = Vt[rank_A:, :]  # (null_dim, current_rank)
    
    # For each null vector, find the minimal support
    # If a null vector has support of size k, we can potentially remove k terms
    # (by expressing one term as a combination of others)
    
    best_reduction = 0
    best_mask = None
    
    for c in null_basis:
        # Find support (nonzero entries)
        support = np.where(np.abs(c) > 1e-10)[0]
        if len(support) < 2:
            continue
        
        # Try to eliminate the term with smallest coefficient in support
        # Actually, we can eliminate any one term in the support
        # by expressing it as a linear combination of the others
        
        # For each term in support, try removing it
        for r in support:
            if time_up():
                break
            # Check if removing term r (and adjusting others) works
            # This is complex; simpler to just try direct removal first
            pass
    
    # For now, fall back to simple removal if linear combination doesn't help easily
    # Try simple removal again
    for r in range(current_rank):
        if time_up():
            break
        mask = [i for i in range(current_rank) if i != r]
        new_U = U_arr[:, mask]
        new_V = V_arr[:, mask]
        new_W = W_arr[:, mask]
        
        T_recon = np.zeros((n_, m_, p_))
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    T_recon[i,j,k] = np.dot(new_U[i*m_+j, :], new_V[j*p_+k, :] * new_W[k*n_+i, :])
        
        if np.max(np.abs(T_recon - target)) < tol:
            U_arr = new_U
            V_arr = new_V
            W_arr = new_W
            current_rank = len(mask)
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def try_rank_reduction(U, V, W, n=2, m=4, p=4):
    """Try to reduce rank by greedily removing redundant terms."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    # Multiple passes of greedy removal
    for pass_num in range(10):
        if time_up():
            break
        removed_any = False
        for r in range(current_rank):
            if time_up():
                break
            mask = [i for i in range(current_rank) if i != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            
            T_recon = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] = np.sum(new_U[i*m+j, :] * new_V[j*p+k, :] * new_W[k*n+i, :])
            
            if np.max(np.abs(T_recon - 1.0)) < 1e-9:
                U_arr = new_U
                V_arr = new_V
                W_arr = new_W
                current_rank = len(mask)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), current_rank

def random_search(rank, n=2, m=4, p=4, iterations=1000):
    """Random search for a decomposition of given rank."""
    for it in range(iterations):
        if time_up():
            break
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        
        if np.max(np.abs(T_recon - 1.0)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist(), rank
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank with greedy removal
    if not time_up():
        U2, V2, W2, rank2 = try_rank_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try linear combination removal
    if not time_up() and best_rank > 24:
        U3, V3, W3, rank3 = try_linear_combination_removal(U2, V2, W2, n, m, p)
        if rank3 < best_rank:
            valid, err = verify_decomposition(U3, V3, W3, n, m, p)
            if valid:
                best_rank = rank3
                best_result = {"U": U3, "V": V3, "W": W3}
    
    # Try random search for lower ranks
    if not time_up() and best_rank > 20:
        for r in range(best_rank - 1, 18, -1):
            if time_up():
                break
            result = random_search(r, n, m, p, iterations=200)
            if result is not None:
                U4, V4, W4, rank4 = result
                valid, err = verify_decomposition(U4, V4, W4, n, m, p)
                if valid and rank4 < best_rank:
                    best_rank = rank4
                    best_result = {"U": U4, "V": V4, "W": W4}
                break
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")

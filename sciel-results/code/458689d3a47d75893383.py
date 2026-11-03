"""Matrix multiplication 2x4 by 4x5 using block-wise Strassen decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Start with the rank-36 Strassen decomposition.
    - Attempt to reduce rank by finding linear dependencies among the rank-1 tensors.
    - If a dependency is found, return a lower-rank decomposition.
    
    Total rank: 36 (or lower if a dependency is found).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Deadline guard
    _start = time.time()
    _deadline = _start + 140.0
    
    # Build the rank-36 decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    rank = U[0].__len__()
    
    # Try to find linear dependencies to reduce rank
    # This is a heuristic search for border rank reduction
    try:
        U_reduced, V_reduced, W_reduced, new_rank = _try_reduce_rank(U, V, W, n, m, p, _start, _deadline)
        if new_rank < rank:
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_decomposition(n, m, p):
    """Build the rank-36 Strassen decomposition."""
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        c = col_start
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        
        col = c
        return c
    
    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        nonlocal col
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c
    
    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, start_time, deadline):
    """
    Try to find a linear dependency among the rank-1 tensors to reduce rank.
    
    Returns (U_reduced, V_reduced, W_reduced, new_rank) if successful,
    otherwise raises an exception.
    """
    rank = U[0].__len__()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check for exact linear dependencies among the rank-1 tensors
    # A rank-1 tensor T_r = u_r ⊗ v_r ⊗ w_r
    # We're looking for coefficients c_r such that sum_r c_r * T_r = 0
    # This is equivalent to finding a non-trivial solution to:
    # sum_r c_r * (u_r ⊗ v_r) ⊗ w_r = 0
    
    # Flatten each rank-1 tensor into a vector
    # T_r vectorized has length n*m*p = 40
    tensor_vectors = []
    for r in range(rank):
        u_r = U_arr[:, r]  # n*m
        v_r = V_arr[:, r]  # m*p
        w_r = W_arr[:, r]  # p*n
        
        # Create the full tensor and flatten
        # T[i,j,k] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        tensor_vectors.append(T.flatten())
    
    tensor_matrix = np.array(tensor_vectors).T  # shape (40, rank)
    
    # Find the rank of this matrix
    # If rank < rank, there's a linear dependency
    try:
        # Use SVD to find the null space
        U_svd, s, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        
        # Find singular values that are close to zero
        tol = 1e-10
        null_dim = sum(1 for sv in s if sv < tol)
        
        if null_dim > 0:
            # We found a dependency!
            # The null space vectors give us the coefficients
            # Vt[rank - null_dim:] gives the null space basis
            
            # For simplicity, we'll just check if we can drop one tensor
            # This is a heuristic - in practice we'd need to be more careful
            
            # Try dropping each tensor and checking if the decomposition still works
            for r_drop in range(rank):
                if time.time() > deadline - 10:
                    break
                
                # Create reduced matrices
                U_red = U_arr[:, [r for r in range(rank) if r != r_drop]]
                V_red = V_arr[:, [r for r in range(rank) if r != r_drop]]
                W_red = W_arr[:, [r for r in range(rank) if r != r_drop]]
                
                # Verify the decomposition
                if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                    # Convert back to lists
                    U_out = U_red.tolist()
                    V_out = V_red.tolist()
                    W_out = W_red.tolist()
                    return U_out, V_out, W_out, rank - 1
            
    except Exception:
        pass
    
    raise ValueError("No rank reduction found")

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factor matrices correctly decompose the matrix multiplication tensor."""
    rank = U[0].__len__() if U[0] else 0
    
    # For random test matrices, verify that the decomposition gives correct results
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    # C[i,k] = sum_r U[i*m + ?] * V[? * p + k] * W[k*n + i]
    # Wait, the indices are:
    # U has shape (n*m, rank) - indexed by (i, j) where i in [0,n), j in [0,m)
    # V has shape (m*p, rank) - indexed by (j, k) where j in [0,m), k in [0,p)
    # W has shape (p*n, rank) - indexed by (k, i) where k in [0,p), i in [0,n)
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for r in range(rank):
                    # U[i*m + j] * V[j*p + k] * W[k*n + i]
                    C_computed[i, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

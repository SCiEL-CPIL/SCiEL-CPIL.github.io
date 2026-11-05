"""Matrix multiplication 3x3 by 3x7 - rank minimization via tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The condition is that for all (i, j, k), the tensor T[i,j,k] = 1 is represented as:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  ... wait, that's not right.
    
    Actually, the bilinear algorithm condition is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The decomposition says:
    C[i,k] = sum_c (sum_{j'} U[i*m+j', c] A[i,j']) * (sum_{k'} V[j*p+k', c] B[j,k']) * W[k*n+i, c]
    
    Wait, that's still not right. Let me re-read the seed code.
    
    In the seed code, for each (i,j,k), we have one column c with:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    This is the standard rank-1 tensor decomposition where each term is a rank-1 tensor.
    The tensor T for matrix multiplication has T[i,j,k] = 1 for all i,j,k (it's the identity in the j-index).
    
    Actually, the matrix multiplication tensor is defined as:
    T(i,j,k) = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    
    The bilinear algorithm uses r multiplications M_c = L_c(A) * R_c(B)
    where L_c(A) = sum_{i,j} U[i*m+j, c] A[i,j]
    and R_c(B) = sum_{j,k} V[j*p+k, c] B[j,k]
    
    Then C[i,k] = sum_c W[k*n+i, c] M_c
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_c W[k*n+i, c] * (sum_{j'} U[i*m+j', c] A[i,j']) * (sum_{k'} V[j*p+k', c] B[j,k'])
    = sum_j A[i,j] B[j,k]
    
    This must hold for all A and B, so the coefficient of A[i,j] B[j,k] on the left must be 1,
    and the coefficient of A[i,j'] B[j'',k] for (j',j'') != (j,j) must be 0.
    
    The coefficient of A[i,j] B[j,k] is:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    
    So the verification condition is:
    For all (i, j, k): sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
    For all (i, j, j', k) with j != j': sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = 0
    For all (i, i', j, k) with i != i': sum_c W[k*n+i, c] * U[i'*m+j, c] * V[j*p+k, c] = 0
    For all (i, j, k, k') with k != k': sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k', c] = 0
    
    Actually, a simpler way to verify is to check that the tensor defined by the decomposition
    equals the matrix multiplication tensor. The tensor T[i,j,k] is defined as:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And we need T[i,j,k] = 1 for all i,j,k.
    
    Wait, that's not right either. Let me think again.
    
    The bilinear algorithm computes C = A * B using r scalar multiplications.
    M_c = L_c(A) * R_c(B)
    L_c(A) = sum_{i,j} U[i*m+j, c] A[i,j]
    R_c(B) = sum_{j,k} V[j*p+k, c] B[j,k]
    C[i,k] = sum_c W[k*n+i, c] M_c
    
    For this to equal the standard matrix product, we need:
    sum_c W[k*n+i, c] * (sum_{j'} U[i*m+j', c] A[i,j']) * (sum_{k'} V[j*p+k', c] B[j,k'])
    = sum_j A[i,j] B[j,k]
    
    This is a bilinear identity in A and B. The coefficient of A[i,j] B[j,k] must be 1,
    and all other coefficients must be 0.
    
    The coefficient of A[i,j] B[j,k] is:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    
    So we need: for all i, j, k: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
    
    And for all (i, j, j', k) with j' != j: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = 0
    And for all (i, i', j, k) with i' != i: sum_c W[k*n+i, c] * U[i'*m+j, c] * V[j*p+k, c] = 0
    And for all (i, j, k, k') with k' != k: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k', c] = 0
    
    Actually, the simplest way to verify is to construct the tensor T from the decomposition
    and check that it equals the matrix multiplication tensor.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (in the sense that the
    bilinear form is sum_j A[i,j] B[j,k]).
    
    Actually, I think the correct verification is:
    For all (i, j, k), the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    
    Let me just verify by checking the bilinear identity directly.
    """
    r = len(U[0])
    # Check the main diagonal terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Check off-diagonal terms (should be 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j2*p+k][c] * W[k*n+i][c]
                    if abs(s) > tol:
                        return False
                for i2 in range(n):
                    if i2 == i:
                        continue
                    s = 0.0
                    for c in range(r):
                        s += U[i2*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s) > tol:
                        return False
                for k2 in range(p):
                    if k2 == k:
                        continue
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k2][c] * W[k*n+i][c]
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 140.0  # Leave safety margin before the 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    school = _schoolbook(n, m, p)
    best_U = [row[:] for row in school["U"]]
    best_V = [row[:] for row in school["V"]]
    best_W = [row[:] for row in school["W"]]
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using random search
    # We'll try to construct a decomposition with rank < 63 by randomly initializing
    # and then checking validity.
    
    # Strategy: Try different ranks and random initializations
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() - start_time > deadline:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(10):
            if time.time() - start_time > deadline:
                break
            
            # Initialize with random values in {-1, 0, 1}
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Randomly assign values
            for i in range(n * m):
                for c in range(target_rank):
                    if np.random.random() < 0.3:
                        U[i][c] = np.random.choice([-1, 1])
            for j in range(m * p):
                for c in range(target_rank):
                    if np.random.random() < 0.3:
                        V[j][c] = np.random.choice([-1, 1])
            for k in range(p * n):
                for c in range(target_rank):
                    if np.random.random() < 0.3:
                        W[k][c] = np.random.choice([-1, 1])
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
        
        if best_rank < n * m * p:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

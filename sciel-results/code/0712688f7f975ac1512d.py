"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] for output (i,k)."""
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    
    diff = np.abs(T_recon - T_true)
    return np.max(diff) < tol, np.max(diff)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7) each with integer entries.
    """
    U = np.zeros((4, 7), dtype=float)
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    V = np.zeros((4, 7), dtype=float)
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[2, 2] = 1; V[3, 2] = -1
    V[1, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    W = np.zeros((4, 7), dtype=float)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _hybrid_decomposition(n, m, p):
    """
    Construct a hybrid Strassen/Schoolbook decomposition for n=2, m=3, p=7.
    
    Strategy:
    - For each inner dimension j in {0, 1, 2}, we have terms A[i,j]*B[j,k]
    - Group the 7 output columns into pairs: (0,1), (2,3), (4,5), and single (6)
    - For each pair of output columns (k1, k2) and inner index j, we have a 2x1x2 subproblem
      which is just schoolbook (2 terms)
    
    Better approach: Use the structure more cleverly.
    
    For 2x3x7, we can use:
    - 3 Strassen-like blocks for the first 6 columns (2 columns each)
    - Schoolbook for the last column
    
    Actually, let's use a known construction:
    Rank = 7 (for j=0, columns 0-1) + 7 (for j=1, columns 2-3) + 7 (for j=2, columns 4-5) + 3 (schoolbook for column 6)
    = 24 + 3 = 27? That doesn't seem right.
    
    Let me think again. The key insight is that we can combine terms across different j values
    for the same output columns.
    
    For output columns (k1, k2), we need:
    C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
    C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
    C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2] + A[0,2]*B[2,k2]
    C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2] + A[1,2]*B[2,k2]
    
    This is a 2x3x2 subproblem. The rank of 2x3x2 is at most 6 (schoolbook).
    
    So for 3 pairs of columns + 1 single column:
    - 3 * 6 = 18 for the pairs
    - 3 for the single column
    Total = 21
    
    But we can do better by using Strassen on the 2x2 parts.
    
    Let's use a different grouping. For each pair of output columns, use a 2x2x2 Strassen
    on the (i, k) dimensions, but we have 3 inner dimensions.
    
    Actually, the known best rank for 2x3x7 is 39 according to the cell findings.
    Let me construct it as:
    - Use Strassen for the first 2 columns of output (2x2x2 subproblem with inner dim 2)
    - But we have inner dim 3, so we need to handle that.
    
    Let me just implement a valid decomposition with rank 39 and verify it.
    
    Construction:
    For j=0: use 7 Strassen terms for columns 0-1 (2x2x2 with A[:,0] and B[0,:])
    For j=1: use 7 Strassen terms for columns 2-3 (2x2x2 with A[:,1] and B[1,:])  
    For j=2: use 7 Strassen terms for columns 4-5 (2x2x2 with A[:,2] and B[2,:])
    For column 6: use 3 schoolbook terms
    
    Total = 7 + 7 + 7 + 3 = 24
    
    Wait, that's not 39. Let me reconsider.
    
    The issue is that Strassen for 2x2x2 uses 7 multiplications, but the subproblem here
    is 2x1x2 (2 rows of A, 1 column of B for a fixed j, 2 columns of output).
    The rank of 2x1x2 is 2, not 7.
    
    So the hybrid approach doesn't directly apply Strassen.
    
    Let me try a different known construction. One approach for 2x3x7:
    
    Use the following identity-based decomposition. We'll create rank-1 terms that
    combine multiple (i,j,k) triples.
    
    For now, let's return the schoolbook and let the search improve it.
    """
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, time_budget=10.0):
    """Try to find a decomposition of the given rank using gradient descent."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    n_schoolbook = n * m * p
    if rank > n_schoolbook:
        return None
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Initialize with schoolbook pattern if possible
    if rank >= n_schoolbook:
        for col in range(rank):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
    else:
        # Random initialization with structure
        for col in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3:
                            U[i * m + j, col] += random.gauss(0, 0.5)
                            V[j * p + k, col] += random.gauss(0, 0.5)
                            W[k * n + i, col] += random.gauss(0, 0.5)
    
    best_diff = np.inf
    lr = 0.01
    
    for iteration in range(500):
        if time.time() - start_time > time_budget:
            break
        
        # Compute reconstruction
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = T_recon - T
        diff = np.max(np.abs(error))
        
        if diff < best_diff:
            best_diff = diff
            if best_diff < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
        
        # Compute gradients
        # dL/dU[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
        grad_U = np.einsum('abc,br,cr->ar', error, V, W)
        grad_V = np.einsum('abc,ar,cr->br', error, U, W)
        grad_W = np.einsum('abc,ar,br->cr', error, U, V)
        
        # Gradient descent step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if iteration % 50 == 0 and iteration > 0:
            lr *= 0.5
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    diff = np.max(np.abs(T_recon - T))
    if diff < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _polish(U, V, W, n, m, p, tol=1e-12, max_iter=100):
    """Polish a near-exact decomposition to exact using high-precision steps."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    
    for _ in range(max_iter):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = T_recon - T
        diff = np.max(np.abs(error))
        
        if diff < tol:
            break
        
        grad_U = np.einsum('abc,br,cr->ar', error, V, W)
        grad_V = np.einsum('abc,ar,cr->br', error, U, W)
        grad_W = np.einsum('abc,ar,br->cr', error, U, V)
        
        # Small step
        U -= 0.001 * grad_U
        V -= 0.001 * grad_V
        W -= 0.001 * grad_W
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    diff = np.max(np.abs(T_recon - T))
    if diff < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication of sizes 2x3 and 3x7.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Get the schoolbook solution as a fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try the hybrid decomposition
    U_hybrid, V_hybrid, W_hybrid = _hybrid_decomposition(n, m, p)
    if U_hybrid:
        valid, diff = _verify(U_hybrid, V_hybrid, W_hybrid, n, m, p)
        if valid:
            hybrid_rank = len(U_hybrid[0])
            if hybrid_rank < best_rank:
                best_rank = hybrid_rank
                best_result = (U_hybrid, V_hybrid, W_hybrid)
    
    # Try to find even lower rank decompositions
    lower_bound = max(n * m, m * p, p * n)
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, time_budget=10.0)
            if result is not None:
                U, V, W = result
                valid, diff = _verify(U, V, W, n, m, p)
                if valid:
                    # Try to polish
                    polished = _polish(U, V, W, n, m, p)
                    if polished:
                        best_rank = rank
                        best_result = polished
                    else:
                        best_rank = rank
                        best_result = (U, V, W)
                    break
    
    U, V, W = best_result
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END

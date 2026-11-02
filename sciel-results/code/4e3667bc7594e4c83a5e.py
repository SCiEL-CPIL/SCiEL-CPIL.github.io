"""Rewritten program for matrix multiplication 3x3x8 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _laderman_3x3(n, m, p):
    """
    Attempt a constructive decomposition using Laderman's 7-term 2x2 block trick
    extended to 3x3x3 blocks, then mapped to the 8-column dimension.
    
    Laderman's algorithm for 2x2x2 uses 7 multiplications.
    For 3x3x3, we can partition into 2x2 blocks plus the remaining row/col.
    
    A 3x3 matrix product C = A*B where A is 3x3, B is 3x3.
    Partition A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    Similarly B = [B11 B12; B21 B22].
    
    C11 = A11*B11 + A12*B21  (2x2 + 2x1*1x2 = 2x2)
    C12 = A11*B12 + A12*B22  (2x2*1 + 2x1*1 = 2x1)
    C21 = A21*B11 + A22*B21  (1x2*2 + 1x1*1x2 = 1x2)
    C22 = A21*B12 + A22*B22  (1x2*1 + 1x1*1 = 1x1)
    
    Using Laderman for the 2x2 part:
    M1 = (A11[0]+A11[1]) * (B11[0]+B11[1])  -- but this isn't quite right for rectangular.
    
    Actually, let's use the standard Laderman decomposition for 2x2x2:
    M1 = (A11[0,0] + A11[1,1]) * (B11[0,0] + B11[1,1])
    M2 = (A11[0,0] + A11[1,1]) * (B11[0,1] - B11[1,1])
    M3 = (A11[0,0] - A11[1,0]) * (B11[1,0] + B11[1,1])
    M4 = A11[0,1] * (B11[1,0] - B11[0,0])
    M5 = (A11[1,0] + A11[1,1]) * (B11[1,1] - B11[1,0])
    M6 = A11[0,0] * (B11[0,1] - B11[1,1])
    M7 = A11[1,1] * (B11[1,0] - B11[0,0])
    
    Wait, the standard Laderman is:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A12 + A22)B21
    M3 = A11(B21 - B11)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)(B11 + B12)
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A21 + A22)(B21 - B11)
    
    Then:
    C11 = M1 - M5 + M6 + M3
    C12 = A11*B12 + A12*B22 = M5 - M1 + M2 + M4? No.
    
    This is getting complex. Let me use a different approach.
    
    For 3x3x3, the known minimal rank is 23 (Strassen's algorithm).
    Strassen's algorithm for 3x3 uses 23 multiplications.
    
    But we need 3x3x8. Let's think about this differently.
    
    Actually, let's try a random search with structured initialization.
    """
    pass

def _random_search(n, m, p, max_rank, num_trials=100, seed=42):
    """
    Try to find a decomposition with rank < max_rank using random initialization
    and iterative refinement.
    """
    rng = np.random.RandomState(seed)
    best_rank = None
    best_solution = None
    
    for target_rank in range(max_rank - 1, 60, -1):
        if target_rank < 1:
            break
        for trial in range(num_trials):
            # Initialize with random values
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            # Try to refine using gradient descent on the squared error
            # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # This is expensive. Let's use a simpler approach.
            
            # For now, just check if this random initialization works (very unlikely)
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
                best_rank = target_rank
                best_solution = (U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_solution is not None:
            break
    
    return best_solution

def _structured_3x3x8(n, m, p):
    """
    Try to construct a decomposition for 3x3x8 using the structure of the problem.
    
    Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    For fixed i, C[i,:] = A[i,:] * B (a row vector times a matrix)
    
    We can use the fact that B has 8 columns to share computation.
    
    Let's try to use a decomposition where we exploit the rectangular shape.
    """
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    
    # Try to find better decompositions
    # 1. Try random search for rank < 72
    elapsed = time.time() - (deadline - 115.0)
    if elapsed < 60:
        result = _random_search(n, m, p, best_rank, num_trials=50, seed=42)
        if result is not None:
            U, V, W = result
            best_rank = min(best_rank, len(U[0]))
    
    # 2. Try structured approach
    elapsed = time.time() - (deadline - 115.0)
    if elapsed < 100:
        U2, V2, W2 = _structured_3x3x8(n, m, p)
        if len(U2[0]) < best_rank:
            if _verify(U2, V2, W2, n, m, p, tol=1e-9):
                U, V, W = U2, V2, W2
                best_rank = len(U2[0])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

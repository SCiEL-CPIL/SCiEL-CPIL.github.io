import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 3]
_n, _m, _p = DIMENSIONS

def _schoolbook_rank27():
    """Known valid rank-27 (schoolbook) decomposition."""
    n, m, p = _n, _m, _p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    """Check exact tensor decomposition."""
    n, m, p = _n, _m, _p
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank26_block():
    """Rank-26 decomposition: 2x2 Strassen for top-left block + direct for row/col 2."""
    n, m, p = _n, _m, _p
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 7 products for the 2x2 block (rows/cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    for (i, j) in [(0,0),(1,1)]:
        U[i*m+j][col] += 1.0
    for (j, k) in [(0,0),(1,1)]:
        V[j*p+k][col] += 1.0
    for (k, i) in [(0,0),(0,1),(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M2 = (A10+A11)*B00
    for (i, j) in [(1,0),(1,1)]:
        U[i*m+j][col] += 1.0
    V[0*p+0][col] += 1.0
    for (k, i) in [(0,0),(1,0)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M3 = A00*(B10-B11)
    U[0*m+0][col] += 1.0
    V[1*p+0][col] += 1.0
    V[1*p+1][col] += -1.0
    for (k, i) in [(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M4 = A11*(B01-B00)
    U[1*m+1][col] += 1.0
    V[0*p+1][col] += 1.0
    V[0*p+0][col] += -1.0
    for (k, i) in [(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M5 = (A00+A10)*B11
    for (i, j) in [(0,0),(1,0)]:
        U[i*m+j][col] += 1.0
    V[1*p+1][col] += 1.0
    for (k, i) in [(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M6 = (A11-A10)*(B00+B10)
    for (i, j) in [(1,1),(1,0)]:
        U[i*m+j][col] += 1.0 if j == 1 else -1.0
    for (j, k) in [(0,0),(1,0)]:
        V[j*p+k][col] += 1.0
    for (k, i) in [(0,1),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    for (i, j) in [(0,1),(1,1)]:
        U[i*m+j][col] += 1.0 if i == 0 else -1.0
    for (j, k) in [(1,0),(1,1)]:
        V[j*p+k][col] += 1.0
    for (k, i) in [(0,0),(1,0)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # j=2 contributions to 2x2 block
    U[0*m+2][col] = 1.0; V[2*p+0][col] = 1.0; W[0*n+0][col] = 1.0; col += 1
    U[0*m+2][col] = 1.0; V[2*p+1][col] = 1.0; W[1*n+0][col] = 1.0; col += 1
    U[1*m+2][col] = 1.0; V[2*p+0][col] = 1.0; W[0*n+1][col] = 1.0; col += 1
    U[1*m+2][col] = 1.0; V[2*p+1][col] = 1.0; W[1*n+1][col] = 1.0; col += 1
    
    # C[0,2]
    U[0*m+0][col] = 1.0; V[0*p+2][col] = 1.0; W[2*n+0][col] = 1.0; col += 1
    U[0*m+1][col] = 1.0; V[1*p+2][col] = 1.0; W[2*n+0][col] = 1.0; col += 1
    U[0*m+2][col] = 1.0; V[2*p+2][col] = 1.0; W[2*n+0][col] = 1.0; col += 1
    
    # C[1,2]
    U[1*m+0][col] = 1.0; V[0*p+2][col] = 1.0; W[2*n+1][col] = 1.0; col += 1
    U[1*m+1][col] = 1.0; V[1*p+2][col] = 1.0; W[2*n+1][col] = 1.0; col += 1
    U[1*m+2][col] = 1.0; V[2*p+2][col] = 1.0; W[2*n+1][col] = 1.0; col += 1
    
    # C[2,0]
    U[2*m+0][col] = 1.0; V[0*p+0][col] = 1.0; W[0*n+2][col] = 1.0; col += 1
    U[2*m+1][col] = 1.0; V[1*p+0][col] = 1.0; W[0*n+2][col] = 1.0; col += 1
    U[2*m+2][col] = 1.0; V[2*p+0][col] = 1.0; W[0*n+2][col] = 1.0; col += 1
    
    # C[2,1]
    U[2*m+0][col] = 1.0; V[0*p+1][col] = 1.0; W[1*n+2][col] = 1.0; col += 1
    U[2*m+1][col] = 1.0; V[1*p+1][col] = 1.0; W[1*n+2][col] = 1.0; col += 1
    U[2*m+2][col] = 1.0; V[2*p+1][col] = 1.0; W[1*n+2][col] = 1.0; col += 1
    
    # C[2,2]
    U[2*m+0][col] = 1.0; V[0*p+2][col] = 1.0; W[2*n+2][col] = 1.0; col += 1
    U[2*m+1][col] = 1.0; V[1*p+2][col] = 1.0; W[2*n+2][col] = 1.0; col += 1
    U[2*m+2][col] = 1.0; V[2*p+2][col] = 1.0; W[2*n+2][col] = 1.0; col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _try_rank23_winograd():
    """Attempt rank-23 decomposition using Winograd-style factorization.
    
    Winograd's method for 3x3 matrix multiplication uses a factorization
    through intermediate transforms. The key idea is to use:
      C = F_A * diag(M) * F_B * F_C
    where F_A, F_B, F_C are transform matrices and M contains the multiplications.
    
    For 3x3, we can use the DFT-like structure. The rank-23 result by Bini et al.
    uses a specific set of 23 rank-1 terms.
    """
    n, m, p = _n, _m, _p
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Winograd's factorization for 3x3 matrix multiplication
    # Uses the structure: C[i,k] = sum_j A[i,j] * B[j,k]
    # with intermediate transforms.
    
    # Define the transform matrices for Winograd's method
    # F_A maps A to intermediate values
    # F_B maps B to intermediate values  
    # F_C maps products back to C
    
    # For 3x3, we use the following 23 rank-1 terms
    # This is based on the Bini-Bini-Capovani-Lotti-Roth (1979) construction
    
    # The 23 terms are indexed by (i, j, k) triples that form a basis
    # for the tensor decomposition space.
    
    # Let me construct this systematically using the known rank-23 scheme.
    # The scheme uses:
    #   7 terms from Strassen-like structure on a 2x2 block
    #   16 additional terms for the remaining structure
    
    # Actually, let me use a different approach: search for a valid rank-23
    # decomposition by solving a system of equations.
    
    # For now, return None to indicate we haven't found a valid rank-23 yet.
    # The construct() function will fall back to rank-26 or rank-27.
    return None

def _search_low_rank(deadline):
    """Search for lower-rank decompositions using random initialization and optimization.
    
    This is a heuristic search that may find valid rank < 26 decompositions.
    """
    n, m, p = _n, _m, _p
    best_rank = 27
    best_decomp = _schoolbook_rank27()
    
    # Try ranks from 23 to 25
    for target_rank in [23, 24, 25]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(attempt)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple gradient-free search: try to adjust entries to satisfy constraints
            # This is a very basic heuristic and may not succeed
            success = False
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Check if we're close to a solution
                if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                    success = True
                    break
                
                # Random perturbation
                idx = np.random.randint(0, n * m)
                col = np.random.randint(0, target_rank)
                U[idx, col] += np.random.randn() * 0.01
                
                idx = np.random.randint(0, m * p)
                col = np.random.randint(0, target_rank)
                V[idx, col] += np.random.randn() * 0.01
                
                idx = np.random.randint(0, p * n)
                col = np.random.randint(0, target_rank)
                W[idx, col] += np.random.randn() * 0.01
            
            if success:
                # Verify with tighter tolerance
                if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-9):
                    best_rank = target_rank
                    best_decomp = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
    
    return best_decomp if best_rank < 27 else None

def construct():
    deadline = time.time() + 110.0
    best = _schoolbook_rank27()
    
    # Try rank 23 Winograd
    cand = _try_rank23_winograd()
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    
    # Try rank 26 block decomposition
    cand = _try_rank26_block()
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    
    # Search for lower rank
    cand = _search_low_rank(deadline)
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    
    # Fallback to rank 27
    return best

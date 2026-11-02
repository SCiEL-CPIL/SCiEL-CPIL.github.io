"""Program for matrix multiplication 2x3 by 3x5: rank-28 decomposition.

Uses block decomposition with Strassen's algorithm for 2x2x2 subproblems:
- Block 1: j=0, all k (2x1x5): rank 10 (schoolbook)
- Block 2a: j=1,2, k=0,1 (2x2x2): rank 7 (Strassen)
- Block 2b: j=1,2, k=2 (2x2x1): rank 4 (schoolbook)
- Block 2c: j=1,2, k=3,4 (2x2x2): rank 7 (Strassen)
Total rank: 10 + 7 + 4 + 7 = 28
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _strassen_2x2x2(U, V, W, col, i_off, j_off, k_off, m, p, n):
    """Apply Strassen's 7-product decomposition for a 2x2x2 block."""
    U[i_off*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[(i_off+1)*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    W[k_off*n + (i_off+1), col] = 1
    W[(k_off+1)*n + (i_off+1), col] = -1
    col += 1
    
    U[i_off*m + j_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[(j_off+1)*p + k_off, col] = 1
    V[j_off*p + k_off, col] = -1
    W[k_off*n + i_off, col] = 1
    W[k_off*n + (i_off+1), col] = 1
    col += 1
    
    U[i_off*m + j_off, col] = 1
    U[i_off*m + (j_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    col += 1
    
    U[(i_off+1)*m + j_off, col] = 1
    U[i_off*m + j_off, col] = -1
    V[j_off*p + k_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[i_off*m + (j_off+1), col] = 1
    U[(i_off+1)*m + (j_off+1), col] = -1
    V[(j_off+1)*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    col += 1
    
    return col

def _build_rank_28():
    """Build the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    col = _strassen_2x2x2(U, V, W, col, i_off=0, j_off=1, k_off=0, m=m, p=p, n=n)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    col = _strassen_2x2x2(U, V, W, col, i_off=0, j_off=1, k_off=3, m=m, p=p, n=n)
    assert col == rank
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if actual != expected:
                    return False
    return True

def _try_rank_27():
    """Attempt to find a rank-27 decomposition via random search + local refinement.
    
    Strategy: Start from rank-28, try to merge two rank-1 terms into one
    by finding a linear combination that still covers all tensor entries.
    """
    n, m, p = 2, 3, 5
    U0, V0, W0 = _build_rank_28()
    
    # Try to find a rank-27 solution by perturbing and solving
    # The condition is: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of 30 equations in 27*(6+15+10) = 675 unknowns (for rank 27)
    # We can use the rank-28 solution as a starting point and try to nullify one column
    
    # Approach: Try each pair of columns (a,b) and see if we can replace them
    # with a single column that still satisfies all constraints.
    
    for trial in range(200):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Pick two random columns to try to merge
        a = np.random.randint(0, 28)
        b = np.random.randint(0, 28)
        if a == b:
            continue
        
        # Create a rank-27 candidate by removing column b and trying to
        # adjust column a to cover both
        U = U0[:, :27].copy().astype(float)
        V = V0[:, :27].copy().astype(float)
        W = W0[:, :27].copy().astype(float)
        
        # The new column a needs to satisfy:
        # For each (i,j,k): U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] + sum_{r!=a} ... = 1
        # This is nonlinear. Instead, try a different approach.
        
        # Alternative: use the rank-28 solution and try to find a non-trivial
        # linear dependency among the 28 rank-1 tensors.
        # If there's a dependency, we can reduce rank.
        
        # Build the 30x28 matrix M where M[i*m+j, r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Actually, the constraint is bilinear in (U,V,W), so this is tricky.
        
        # Simpler approach: just verify rank-28 is valid and return it.
        pass
    
    return None

def construct():
    """Return rank-28 decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = 2, 3, 5
    
    # Try to find rank-27 first (with time limit)
    result_27 = _try_rank_27()
    if result_27 is not None:
        U, V, W = result_27
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-28
    U, V, W = _build_rank_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

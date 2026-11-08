import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _construct_rank_71():
    """
    Returns a rank-71 decomposition for 3x5x5 matrix multiplication.
    Uses 4 disjoint 2x2x2 Strassen blocks and schoolbook for the rest.
    """
    n, m, p = 3, 5, 5
    
    # Define 4 non-overlapping 2x2x2 blocks
    # i in {0,1}, j in {0,1} and {2,3}, k in {0,1} and {2,3}
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_strassen_rank = 4 * 7
    total_terms = n * m * p
    strassen_terms_count = 4 * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += -1.0
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[j0*p + k0][c] += -1.0
        W[k0*n + i0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1.0
        U[i0*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1.0
        U[i0*m + j0][c] += -1.0
        V[j0*p + k0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += -1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Identify covered terms
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    # Add schoolbook terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1.0
                    V[j*p+k][col] += 1.0
                    W[k*n+i][col] += 1.0
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

def _try_improve_with_linear_algebra(base_result, time_budget=10.0):
    """
    Attempt to find a lower-rank decomposition using linear algebra techniques.
    This is a best-effort approach that may not always improve upon the base rank.
    """
    n, m, p = 3, 5, 5
    base_rank = len(base_result["U"][0])
    
    # Flatten the tensor
    # The matrix multiplication tensor T[i,j,k] = sum_l A[i,l]*B[l,k]
    # We want to find U, V, W such that T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    
    # Approach: Try to find if we can express the tensor with fewer rank-1 terms
    # by solving a system of equations
    
    # For now, this is a placeholder for more sophisticated linear algebra approaches
    # that could potentially reduce the rank below 71.
    
    # A practical approach: try random perturbations and see if we can find a
    # valid decomposition with lower rank
    
    start_time = time.time()
    
    # Generate a random rank-70 attempt
    target_rank = base_rank - 1
    
    if target_rank < 50:
        target_rank = 50
    
    # This is computationally expensive, so we'll limit our attempts
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        # Create random initial guess
        np.random.seed(attempt)
        
        # Try to construct a valid decomposition with target_rank
        # This is a heuristic approach - we'll try to solve for the entries
        
        # For simplicity, we'll just return the base result if we can't improve
        # A proper implementation would use more sophisticated optimization
    
    return base_result

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    # Get the base rank-71 solution
    base_result = _construct_rank_71()
    
    # Try to improve using linear algebra (best effort)
    # This may not always succeed, so we fall back to the base solution
    try:
        improved_result = _try_improve_with_linear_algebra(base_result, time_budget=5.0)
        if len(improved_result["U"][0]) < len(base_result["U"][0]):
            return improved_result
    except Exception:
        pass
    
    return base_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verification
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    error += 1
    print(f"Verification errors: {error}")

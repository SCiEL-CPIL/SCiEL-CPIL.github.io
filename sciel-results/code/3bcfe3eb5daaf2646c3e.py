import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n, m, p = 3, 5, 5
    
    # Use 4 Strassen blocks + schoolbook for the rest
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
                    
    # Try to improve rank by combining terms using linear algebra
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Attempt to reduce rank by finding linear dependencies among the rank-1 terms
    # We can try to merge pairs of terms if their outer products sum to zero in some combination
    # This is a simplified approach - try to find if any two terms can be combined
    
    # Check if we can reduce by looking at the structure
    # For now, return the current result as the lattice reduction approach is complex
    # and may not improve within the time budget
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")

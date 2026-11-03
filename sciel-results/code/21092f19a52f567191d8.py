"""Tensor rank decomposition for 3x5x5 matrix multiplication."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid rank decomposition for 3x5x5 matrix multiplication.
    
    Strategy: Use 4 disjoint 2x2x2 Strassen blocks (rank 7 each) + schoolbook for remainder.
    This gives rank 4*7 + (75 - 4*8) = 28 + 43 = 71.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 75
    
    def build_decomposition(blocks2):
        """Build a decomposition using the given 2x2x2 Strassen blocks."""
        rank2 = 7
        
        # Count total rank
        total_rank = len(blocks2) * rank2
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks2:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        
        remainder_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
        total_rank += remainder_count
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        col = 0
        
        def add_strassen2(i0, j0, k0, col_offset):
            # Standard 2x2x2 Strassen with rank 7
            c = col_offset + 0
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 1
            U[(i0+1)*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += -1
            
            c = col_offset + 2
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 3
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            
            c = col_offset + 4
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1
            W[(k0+1)*n + i0][c] += 1
            
            c = col_offset + 5
            U[(i0+1)*m + j0][c] += 1
            U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 6
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
        
        for i0, j0, k0 in blocks2:
            add_strassen2(i0, j0, k0, col)
            col += rank2
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify the decomposition is correct using random test."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_reconstructed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_reconstructed[i, k] += np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
        
        error = np.max(np.abs(C_reconstructed - C_expected))
        return error < 1e-10, error
    
    # Known good configuration: 4 disjoint 2x2x2 Strassen blocks
    # Blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    # These cover i=0,1; j=0,1 and j=2,3; k=0,1 and k=2,3
    # Remaining: i=2 (all j,k) + i=0,1 with j=4 or k=4
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # Verify this configuration
    U, V, W, rank = build_decomposition(blocks)
    valid, error = verify_decomposition(U, V, W, n, m, p)
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U = [[0]*total_terms for _ in range(n*m)]
    V = [[0]*total_terms for _ in range(m*p)]
    W = [[0]*total_terms for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    
    # Verify correctness
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    C_reconstructed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_reconstructed[i, k] += np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
    
    error = np.max(np.abs(C_reconstructed - C_expected))
    print(f"Max reconstruction error: {error}")
    assert error < 1e-10, "Decomposition is incorrect!"
    print("Decomposition verified successfully.")

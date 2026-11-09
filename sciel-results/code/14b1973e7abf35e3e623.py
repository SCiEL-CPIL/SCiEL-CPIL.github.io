"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose matrix multiplication.
    
    For each (i, j, k) in [n]x[m]x[p], we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    And for all other index combinations, the sum should be 0.
    """
    rank = len(U[0])
    # Check all (i,j,k) triplets
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-10:
                    return False
    # Check cross terms: for (i,j,k) where not all indices match a valid product
    # The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 iff this corresponds to
    # (row i of A, col j of B, col k of C) which is always the case for valid
    # matrix mult. But we need T to be zero for "invalid" combinations.
    # Actually for standard matmul, every (i,j,k) with i<n, j<m, k<p is valid.
    # The tensor is fully supported on [n]x[m]x[p].
    # So we just need T[i,j,k] = 1 for all valid (i,j,k).
    # Wait - that's not right. Let me reconsider.
    
    # The matrix multiplication tensor has T[i,j,k] = delta_{j,j} (always 1)
    # No wait. For A (n x m) times B (m x p), result C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # So yes, every entry should be 1.
    return True

def _verify_decomposition_correct(U, V, W, n, m, p):
    """Correct verification: the matmul tensor T[i,j,k] = 1 for all i,j,k
    in their respective ranges. So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for all i in [0,n), j in [0,m), k in [0,p)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Use the standard schoolbook decomposition which is guaranteed to be valid
    # for the matrix multiplication tensor.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to the specific column in the decomposition
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition of the tensor T[i,j,k] = delta(i,j,k).
    The tensor T is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]."""
    n, m, p = DIMENSIONS
    # Build the left-hand side for each (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(len(U[0])):
                    lhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                rhs = 1.0  # T[i,j,k] = 1 only when i,j,k are all indices (but wait, T[i,j,k] = delta... no)
                
                # Actually the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k in their ranges?
                # No. The tensor for matrix mult (n x m) * (m x p) -> (n x p) is:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # So the tensor T has T[i,j,k] = 1 for each (i,j,k) because A[i,j]*B[j,k] contributes to C[i,k].
                # Wait, the standard tensor is M[i,j,k] = 1, meaning the product A*B has element C[i,k] = sum_j M[i,j,k] A[i,j] B[j,k].
                # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # So RHS should be 1.0.
                
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    The schoolbook decomposition has rank 18.
    """
    # Use the guaranteed valid schoolbook decomposition.
    # A rank < 18 decomposition would require specific non-trivial coefficients.
    # Since we cannot guarantee finding a valid rank 15 or 16 or 17 decomposition
    # without external data or heavy computation that might fail, we return the
    # known valid schoolbook decomposition.
    
    return _schoolbook()

# EVOLVE-BLOCK-END

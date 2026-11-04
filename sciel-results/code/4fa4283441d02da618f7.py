"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-minimal (or best found) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The decomposition is defined by three matrices U, V, W of shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Such that the matrix multiplication tensor T[i, j, k] = delta_{i,k} (for the identity part? No, it's the multiplication tensor)
    The multiplication tensor T for (n x m) * (m x p) -> (n x p) is defined such that:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In tensor form, T_{i, j, k} is 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    Actually, the standard tensor decomposition of matrix multiplication is:
    T = sum_c u_c \otimes v_c \otimes w_c
    where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n} (or similar indexing).
    
    The schoolbook decomposition uses rank R = n*m*p.
    Each term corresponds to a single multiplication A[i, j] * B[j, k] contributing to C[i, k].
    
    Indexing convention:
    U has rows indexed by (i, j) flattened to i*m + j.
    V has rows indexed by (j, k) flattened to j*p + k.
    W has rows indexed by (k, i) flattened to k*n + i. Note: The prompt says W[k*n+i, r].
    
    For the c-th term (corresponding to i, j, k):
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    All other entries in column c are 0.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the non-zero entries for this rank-1 term
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time just in case (though this loop is very fast)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

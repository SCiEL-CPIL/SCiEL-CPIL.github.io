"""Matrix multiplication 3x4x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]


# EVOLVE-BLOCK-START
def construct():
    """Return a rank-66 exact tensor decomposition for 3x4x6 matrix multiplication.
    
    Strategy:
      - Part 1: i in {0,1}, use 6 Strassen 2x2x2 blocks (B paired as (0,1),(2,3); C paired as (0,1),(2,3),(4,5))
        Each Strassen block has rank 7, so 6*7 = 42.
      - Part 2: i=2, schoolbook for all j,k: 4*6 = 24 terms.
      Total rank: 42 + 24 = 66.
    """
    deadline = time.time() + 100  # safety margin
    
    n, m, p = 3, 4, 6
    rank = 66
    
    U = np.zeros((n * m, rank))  # (12, 66)
    V = np.zeros((m * p, rank))  # (24, 66)
    W = np.zeros((p * n, rank))  # (18, 66)
    
    col = 0
    
    b_pairs = [(0, 1), (2, 3)]
    c_pairs = [(0, 1), (2, 3), (4, 5)]
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1, col):
        nonlocal col
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]) -> C[k0,i0] += M1, C[k1,i1] += M1
        U[i0*m+j0, col] += 1
        U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0] -> C[k1,i0] += M2
        U[i1*m+j0, col] += 1
        U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k1*n+i0, col] += 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1]) -> C[k0,i1] += M3
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1
        V[j1*p+k1, col] -= 1
        W[k0*n+i1, col] += 1
        col += 1
        
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0]) -> C[k1,i0] += M4
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1
        V[j0*p+k0, col] -= 1
        W[k1*n+i0, col] += 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1] -> C[k0,i1] += M5, C[k1,i1] += M5
        U[i0*m+j0, col] += 1
        U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1]) -> C[k1,i0] += M6
        U[i1*m+j0, col] += 1
        U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1
        V[j0*p+k1, col] += 1
        W[k1*n+i0, col] += 1
        col += 1
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1]) -> C[k1,i1] += M7, C[k0,i0] -= M7
        U[i0*m+j1, col] += 1
        U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1
        V[j1*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        W[k0*n+i0, col] -= 1
        col += 1
        
        return col
    
    for (j0, j1) in b_pairs:
        for (k0, k1) in c_pairs:
            col = add_strassen_block(0, 1, j0, j1, k0, k1, col)
            if time.time() > deadline:
                break
    
    # Part 2: i=2, schoolbook
    for j in range(m):
        for k in range(p):
            U[2*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+2, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Quick verification
    _verify_decomposition(result)
    
    return result


def _verify_decomposition(result, n=3, m=4, p=6):
    """Verify that the decomposition is correct."""
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += u[:, :, None] * v[:, :, None] * w.T[:, :, None]
    
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-10):
        diff = np.abs(T - expected)
        max_diff = np.max(diff)
        raise ValueError(f"Decomposition error: max diff = {max_diff}")
    
    return True
# EVOLVE-BLOCK-END

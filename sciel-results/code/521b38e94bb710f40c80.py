"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Kronecker product M_{2,3,3} ⊗ M_{1,1,2} giving rank ≤ 24.
    M_{2,3,3} is decomposed as M_{2,2,3} (rank 6) + M_{2,1,3} (rank 6) = rank 12.
    M_{1,1,2} is rank 2. Total rank ≤ 24.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def rank_18_decomposition():
        """
        Direct rank-18 decomposition for M_{2,3,6}.
        Based on the fact that M_{2,3,6} can be decomposed into 3 disjoint blocks of M_{2,3,2}
        (splitting the output columns k into 3 groups of 2), and each M_{2,3,2} has rank 6.
        Each M_{2,3,2} block is further decomposed into 3 M_{2,1,2} sub-blocks (splitting j),
        each of which has rank 2.
        Total rank = 3 (k-blocks) * 3 (j-blocks) * 2 (rank of M_{2,1,2}) = 18.
        """
        rank = 18
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        col = 0
        for j in range(3):
            for q in range(3): # q is the k-block index (0, 1, 2)
                # Term 1: corresponds to i=0 in M_{2,1,2}
                # U[3*0 + j, col] = 1
                U[j, col] = 1
                # V[6*j + 2*q + 0, col] = 1
                # V[6*j + 2*q + 1, col] = 1
                V[6*j + 2*q, col] = 1
                V[6*j + 2*q + 1, col] = 1
                # W[4*q + 0, col] = 1
                # W[4*q + 2, col] = 1
                W[4*q, col] = 1
                W[4*q + 2, col] = 1
                col += 1
                
                # Term 2: corresponds to i=1 in M_{2,1,2}
                # U[3*1 + j, col] = 1
                U[3 + j, col] = 1
                # V[6*j + 2*q + 0, col] = 1
                # V[6*j + 2*q + 1, col] = 1
                V[6*j + 2*q, col] = 1
                V[6*j + 2*q + 1, col] = 1
                # W[4*q + 1, col] = 1
                # W[4*q + 3, col] = 1
                W[4*q + 1, col] = 1
                W[4*q + 3, col] = 1
                col += 1
                
        return U, V, W

    # Try the new rank-18 decomposition
    best_U, best_V, best_W = rank_18_decomposition()
    best_rank = 18
    
    # Verify the new decomposition
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if rank 18 construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

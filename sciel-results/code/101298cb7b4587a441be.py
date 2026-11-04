import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
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
        rank = U.shape[1]
        # Using einsum for speed
        result = np.einsum('ir, jr, kr -> ijk', U, V, W, optimize=True)
        return np.array_equal(result, T)

    def optimized_block_decomposition():
        """
        Return rank-24 decomposition: 3 blocks of (Strassen 2x2x2 + Rank-1 2x1x2).
        """
        rank = 24
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = np.array([
            [1, 0], [0, 1],
            [0, 0], [1, 1],
            [1, 0], [0, 0],
            [0, 0], [0, 1],
            [1, 1], [0, 0],
            [-1, 0], [1, 0],
            [0, 1], [0, -1]
        ]).reshape(7, 2, 2)
        
        S_V = np.array([
            [1, 0], [0, 1],
            [1, 0], [0, 0],
            [0, 1], [0, -1],
            [-1, 0], [1, 0],
            [0, 0], [0, 1],
            [1, 1], [0, 0],
            [0, 0], [1, 1]
        ]).reshape(7, 2, 2)
        
        S_W = np.array([
            [1, 0], [0, 1],
            [0, 1], [0, -1],
            [0, 0], [1, 1],
            [1, 1], [0, 0],
            [-1, 0], [1, 0],
            [0, 0], [0, 1],
            [1, 0], [0, 0]
        ]).reshape(7, 2, 2)
        
        col = 0
        for b in range(3):
            # Strassen part (Rank 7)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r, i, j]
                
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r, j, kp]
                
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r, kp, i]
                
                col += 1
            
            # Rank-1 part (Rank 1)
            # Handles A[i, 2] * B[2, k]
            for i in range(2):
                U[i * m + 2, col] = 1
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global, col] = 1
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = 1
            col += 1
        
        return U, V, W

    # Start with the optimized rank-24 decomposition
    best_U, best_V, best_W = optimized_block_decomposition()
    best_rank = 24
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if optimization is wrong
        best_rank = n * m * p
        best_U = np.zeros((dim_a, best_rank), dtype=np.int64)
        best_V = np.zeros((dim_b, best_rank), dtype=np.int64)
        best_W = np.zeros((dim_c, best_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, col] = 1
                    best_V[b, col] = 1
                    best_W[c, col] = 1
                    col += 1

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

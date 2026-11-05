import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140

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
        # Use einsum for efficient verification
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(result, T)

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        This splits the 3x6 matrix B into three 3x2 blocks.
        Each block multiplication is 2x3 by 3x2.
        The rank of 2x3 by 3x2 is 11.
        Total rank = 3 * 11 = 33.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors (7 multiplications)
        # U, V, W are 2x2 matrices for each factor
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors (4 multiplications)
        # U is 2x1, V is 1x2, W is 2x2
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Block b handles columns 2b, 2b+1 of B (and corresponding C)
            # A is 2x3. We split A into A[:, 0:2] (2x2) and A[:, 2:3] (2x1)
            # B[:, 2b:2b+2] is 3x2. We split B into B[0:2, :] (2x2) and B[2:3, :] (1x2)
            
            # Part 1: Strassen for A[:, 0:2] * B[0:2, 2b:2b+2]
            for r in range(7):
                # U (2x2) maps to A[:, 0:2] -> rows i*m + j for i in 0,1 and j in 0,1
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                # V (2x2) maps to B[0:2, 2b:2b+2] -> rows j*p + k for j in 0,1 and k in 2b, 2b+1
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                # W (2x2) maps to C[:, 2b:2b+2] -> rows k*n + i for k in 2b, 2b+1 and i in 0,1
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Part 2: Standard for A[:, 2:3] * B[2:3, 2b:2b+2]
            for r in range(4):
                # U (2x1) maps to A[:, 2] -> rows i*m + 2
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                # V (1x2) maps to B[2, 2b:2b+2] -> row 2*p + k
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                # W (2x2) maps to C[:, 2b:2b+2] -> rows k*n + i
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    # Generate the rank-33 decomposition
    U, V, W = strassen_block_decomposition()
    
    # Verify
    if verify_decomposition(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
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
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

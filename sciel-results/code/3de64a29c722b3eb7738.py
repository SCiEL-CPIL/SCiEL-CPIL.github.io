import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0

    def check_decomposition(U, V, W, rank):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
        return np.all(result == 1)

    # Known optimal rank-23 decomposition for 3x3x3 matrix multiplication
    # Indices: U has shape (3,3), V has shape (3,3), W has shape (3,3)
    # U[i,j,c], V[j,k,c], W[k,i,c] with i,j,k in {0,1,2}, c in {0..22}
    U33 = [
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    V33 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    ]
    W33 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    # Wait, that's just the schoolbook for 3x3x3 with rank 27. Let me use a proper rank-23 decomposition.
    # Actually, let me use a known Strassen-like or specific rank-23 decomposition.
    # Let me try a different approach: use a known decomposition from literature.
    
    # Actually, let me reconsider. The known optimal rank for 3x3x3 is 23.
    # Let me use a specific decomposition. Here's one from the literature:
    # I'll use a decomposition that's been verified.
    
    # For now, let me use a simpler approach: block diagonal decomposition.
    # Split 5 = 2 + 3 for both m and p dimensions.
    # Then we have 4 blocks:
    # Block (0,0): 3x2x2, rank 3
    # Block (0,1): 3x2x3, rank 3  
    # Block (1,0): 3x3x2, rank 3
    # Block (1,1): 3x3x3, rank 23
    # Total: 3 + 3 + 3 + 23 = 32
    
    # But wait, that's not quite right. The blocks aren't independent because
    # the n-index is shared. Let me think more carefully.
    
    # Actually, for the tensor T_{ijk} = 1 where i in {0,1,2}, j in {0..4}, k in {0..4},
    # we can decompose j = j1 + 5*j2 and k = k1 + 5*k2 where j1,k1 in {0,1} and j2,k2 in {0,1}.
    # Then T_{i, j1+5*j2, k1+5*k2} = 1.
    # This factors as a product of tensors for each (j2,k2) block.
    
    # Block (j2=0,k2=0): j in {0,1}, k in {0,1} -> 3x2x2 tensor, rank 3
    # Block (j2=0,k2=1): j in {0,1}, k in {2,3,4} -> 3x2x3 tensor, rank 3
    # Block (j2=1,k2=0): j in {2,3,4}, k in {0,1} -> 3x3x2 tensor, rank 3
    # Block (j2=1,k2=1): j in {2,3,4}, k in {2,3,4} -> 3x3x3 tensor, rank 23
    
    # For 3x2x2: U_i = delta_{i,i}, V_j = 1, W_k = 1 for 3 terms
    # For 3x2x3: U_i = delta_{i,i}, V_j = 1, W_k = 1 for 3 terms
    # For 3x3x2: U_i = delta_{i,i}, V_j = 1, W_k = 1 for 3 terms
    # For 3x3x3: need rank-23 decomposition
    
    # Let me construct the full decomposition:
    rank = 32
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    # Block (j2=0,k2=0): j in {0,1}, k in {0,1}
    for i in range(n):
        U[i * m + 0, col] = 1  # j=0
        U[i * m + 1, col] = 1  # j=1
        for k in [0, 1]:
            V[0 * p + k, col] = 1  # j=0
            V[1 * p + k, col] = 1  # j=1
            W[k * n + i, col] = 1
        col += 1
    
    # Block (j2=0,k2=1): j in {0,1}, k in {2,3,4}
    for i in range(n):
        U[i * m + 0, col] = 1
        U[i * m + 1, col] = 1
        for k in [2, 3, 4]:
            V[0 * p + k, col] = 1
            V[1 * p + k, col] = 1
            W[k * n + i, col] = 1
        col += 1
    
    # Block (j2=1,k2=0): j in {2,3,4}, k in {0,1}
    for i in range(n):
        for j in [2, 3, 4]:
            U[i * m + j, col] = 1
        for k in [0, 1]:
            for j in [2, 3, 4]:
                V[j * p + k, col] = 1
            W[k * n + i, col] = 1
        col += 1
    
    # Block (j2=1,k2=1): j in {2,3,4}, k in {2,3,4}
    # Need 23 terms for 3x3x3
    # Use a known rank-23 decomposition
    # Let me use a specific one. Here's a rank-23 decomposition for 3x3x3:
    
    # Actually, let me just use the schoolbook for this block (rank 27) to be safe,
    # then the total rank is 3+3+3+27 = 36.
    for i in range(n):
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    if check_decomposition(U, V, W, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

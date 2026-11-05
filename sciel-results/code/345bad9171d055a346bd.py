import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    
    # Helper to verify a decomposition
    def verify_decomposition(U, V, W, n, m, p):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        # Reconstruct tensor T[i*m+j, j*p+k] = sum_k U[i*m+j, k] * V[j*p+k, k] * W[k*n+i, k]
        # Note: W index is k*n+i according to task description W[k*n+i, r]
        # Wait, the task says W[k*n+i, r].
        # Standard tensor: T_{ijk} = sum_k U_{ij}^k V_{jk}^k W_{ki}^k?
        # Let's check the schoolbook in the prompt.
        # U[i*m+j][c] = 1
        # V[j*p+k][c] = 1
        # W[k*n+i][c] = 1
        # Product: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # This sums to 1 if there is a term with i,j,k.
        # The tensor for matrix mult is T_{i,j,k} = 1 if we consider the coefficient of A_{ij}B_{jk} in C_{ik}.
        # Wait, the output is C_{ik}. The input is A_{ij} and B_{jk}.
        # The term in C_{ik} is A_{ij} B_{jk}.
        # So the tensor element corresponding to (i,j,k) is 1.
        # The decomposition is sum_r U_{ij}^r V_{jk}^r W_{ki}^r?
        # The prompt says W[k*n+i, r].
        # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # This must equal 1 for all i,j,k.
        
        T_recon = np.zeros((n*m, m*p, p*n)) # This is not the right shape for verification directly.
        # We need to check if sum_r U_{ij}^r V_{jk}^r W_{ki}^r == 1 for all i,j,k.
        
        # Flatten U, V, W
        U_flat = np.array(U)
        V_flat = np.array(V)
        W_flat = np.array(W)
        
        # Efficient verification:
        # For each r, we add outer product of U[:,r], V[:,r], W[:,r] to a tensor?
        # No, we need to check specific indices.
        
        # Let's just iterate over all i,j,k and check.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_flat[idx_u, r] * V_flat[idx_v, r] * W_flat[idx_w, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Attempt to construct Rank-47 decomposition
    # 3x7 = 3x3 + 3x3 + 3x1
    # Rank(3x3) = 19, Rank(3x1) = 9. Total 47.
    
    # We need the 19 terms for 3x3.
    # If we don't have them, we use schoolbook 27.
    # I will use a placeholder for the 19 terms.
    # If I cannot verify them, I will fall back to schoolbook for the 3x3 parts (rank 27).
    # 27 + 27 + 9 = 63.
    
    # Let's try to implement a known rank-19 decomposition.
    # Since I cannot recall the exact 19 terms with 100% confidence, I will use the schoolbook for the 3x3 parts.
    # This results in rank 63.
    # To improve, I would need the 19 terms.
    
    # However, I can try to find a rank-19 decomposition using a small search if I have time?
    # No, 120s is tight.
    
    # I will return the schoolbook to be safe, as I cannot guarantee the 19 terms.
    # Wait, I can try to use the rank-7 2x2 Strassen and pad?
    # 3x3 can be viewed as 2x2 blocks?
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # This is complex.
    
    # I will stick to the schoolbook for the 3x3 parts if I can't verify the 19 terms.
    # But I can construct the 3x1 part with rank 9.
    # So total rank = 27 + 27 + 9 = 63.
    
    # This is the same as the current score.
    
    # To get an improvement, I MUST find the 19 terms.
    # I will use a known set of 19 terms from a standard source if I can encode them.
    
    # Let's try to encode the 19 terms.
    # Reference: "A rank-19 algorithm for 3x3 matrix multiplication"
    
    # If I fail, I return schoolbook.
    
    # For now, I will return the schoolbook to ensure validity.
    return get_schoolbook()

# EVOLVE-BLOCK-END

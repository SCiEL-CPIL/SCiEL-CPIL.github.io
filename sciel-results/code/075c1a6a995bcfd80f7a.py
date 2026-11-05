import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2 Strassen factors (Rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # 2. Construct 6x6 factors using 3x3 Schoolbook grid of 2x2 Strassen blocks
    rank = 27 * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for s in range(7):
                    for dr in range(2):
                        for dc in range(2):
                            u_val = U2[dr * 2 + dc, s]
                            if u_val != 0:
                                u_row = (i * 2 + dr) * m + (j * 2 + dc)
                                U[u_row, col] = u_val
                                
                            v_val = V2[dr * 2 + dc, s]
                            if v_val != 0:
                                v_row = (j * 2 + dr) * p + (k * 2 + dc)
                                V[v_row, col] = v_val
                                
                            w_val = W2[dr * 2 + dc, s]
                            if w_val != 0:
                                w_row = (k * 2 + dr) * n + (i * 2 + dc)
                                W[w_row, col] = w_val
                    col += 1
                    
    # Verification
    # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == delta_{j1, j2} is not the right check for specific i,k.
    # The check is: for all i, k, sum_j A_{i,j} B_{j,k} == C_{i,k}
    # Using decomp: C'_{i,k} = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A_{i,j}) (sum_j V[j*p+k, r] B_{j,k})
    # This requires sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta_{j1, j2}
    
    # Efficient check using tensor contraction
    # T = sum_r U[:,r] V[:,r]^T W[:,r]^T ? No.
    # We can check a few random instances.
    np.random.seed(42)
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 6)
    C_expected = A @ B
    
    C_decomp = np.zeros((6, 6))
    for r in range(rank):
        # u_vec is 36x1, v_vec is 36x1
        # u_vec[j] corresponds to A[i, j]? No.
        # U is 36xR. Row index i*m+j.
        # So for a fixed i, the vector u_i is U[i*m : i*m+6, r]
        # This vector multiplies A[i, :]?
        # Let's check the formula:
        # C'_{i,k} = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        
        u_i = U[i*m : i*m+m, r] # This is wrong, i is not fixed in the loop
        # We need to sum over i and k.
        
    # Better check:
    # Construct the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i, j, k] to be 1 if j is the middle index?
    # No, the MM tensor T_MM has T_MM[i, j, k] = 1 for all i,j,k?
    # Let's re-verify the tensor definition.
    # <T, A \otimes B> = vec(C)
    # vec(C)_{i*n+k} = sum_j A_{i,j} B_{j,k}
    # So T[i, j, k] should be 1.
    
    # Let's verify T[i, j, k] == 1 for all i,j,k.
    valid = True
    for i in range(6):
        for j in range(6):
            for k in range(6):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(val, 1.0):
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook 216
        rank_sb = 216
        U_sb = np.zeros((36, rank_sb))
        V_sb = np.zeros((36, rank_sb))
        W_sb = np.zeros((36, rank_sb))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_sb[i*6+j, col] = 1
                    V_sb[j*6+k, col] = 1
                    W_sb[k*6+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

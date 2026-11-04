"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Rank-42 decomposition
    # 4 blocks of (2x2 by 2x2 rank 7 + 2x1 by 1x2 rank 2) = 4 * 9 = 36
    # 1 block of (2x3 by 3x1 rank 6) = 6
    # Total 42
    
    rank_total = 42
    U = np.zeros((dim_u, rank_total))
    V = np.zeros((dim_v, rank_total))
    W = np.zeros((dim_w, rank_total))
    
    # Rank-7 decomposition for 2x2 by 2x2
    # Indices for sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # U_idx: i*2 + j (local) -> global i*m + j
    # V_idx: j*2 + k (local) -> global j*p + (k_offset + k)
    # W_idx: k*2 + i (local) -> global (k_offset + k)*n + i
    
    # The 7 terms for 2x2x2 (Strassen)
    # U, V, W are 4x7 matrices in local coords
    U_7 = np.zeros((4, 7))
    V_7 = np.zeros((4, 7))
    W_7 = np.zeros((4, 7))
    
    # d=0
    U_7[0, 0] = 1; U_7[3, 0] = 1
    V_7[0, 0] = 1; V_7[3, 0] = 1
    W_7[0, 0] = 1; W_7[3, 0] = 1
    
    # d=1
    U_7[1, 1] = 1; U_7[3, 1] = 1
    V_7[0, 1] = 1
    W_7[0, 1] = 1; W_7[1, 1] = -1
    
    # d=2
    U_7[0, 2] = 1
    V_7[1, 2] = 1
    W_7[0, 2] = 1; W_7[1, 2] = -1
    
    # d=3
    U_7[3, 3] = 1
    V_7[1, 3] = 1; V_7[0, 3] = -1
    W_7[0, 3] = 1
    
    # d=4
    U_7[0, 4] = 1; U_7[1, 4] = 1
    V_7[3, 4] = 1
    W_7[0, 4] = 1; W_7[1, 4] = -1
    
    # d=5
    U_7[2, 5] = 1; U_7[0, 5] = -1
    V_7[0, 5] = 1; V_7[1, 5] = 1
    W_7[0, 5] = 1
    
    # d=6
    U_7[1, 6] = 1; U_7[3, 6] = -1
    V_7[2, 6] = 1; V_7[3, 6] = 1
    W_7[0, 6] = 1; W_7[1, 6] = -1
    
    # Verify 2x2x2
    # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
    # a=i*2+j, b=j*2+k, c=k*2+i
    valid_7 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i*2 + j
                b = j*2 + k
                c = k*2 + i
                val = np.dot(U_7[a,:], V_7[b,:] * W_7[c,:])
                if abs(val - 1) > 1e-10:
                    valid_7 = False
    # Check zeros
    for a in range(4):
        for b in range(4):
            for c in range(4):
                i = a // 2
                j = a % 2
                j2 = b // 2
                k = b % 2
                k2 = c // 2
                i2 = c % 2
                if not (j == j2 and k == k2 and i == i2):
                    val = np.dot(U_7[a,:], V_7[b,:] * W_7[c,:])
                    if abs(val) > 1e-10:
                        valid_7 = False
    
    if not valid_7:
        return best_result

    # Construct full decomposition
    col = 0
    # 4 blocks for k pairs (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k_offset = block_idx * 2
        
        # Add 7 terms for 2x2x2 part (j in 0,1)
        for d in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx, col] = U_7[i*2+j, d]
                    
                    v_idx = j * p + (k_offset + 0) # k is handled in V_7 index
                    # V_7 index is j*2 + k_local
                    # We need to map V_7[local_v_idx, d] to V[global_v_idx, col]
                    # local_v_idx = j*2 + k_local
                    # global_v_idx = j*p + (k_offset + k_local)
                    for k_local in range(2):
                        local_v_idx = j * 2 + k_local
                        global_v_idx = j * p + (k_offset + k_local)
                        V[global_v_idx, col] = V_7[local_v_idx, d]
                        
                    for k_local in range(2):
                        local_w_idx = k_local * 2 + i
                        global_w_idx = (k_offset + k_local) * n + i
                        W[global_w_idx, col] = W_7[local_w_idx, d]
                        
            col += 1
            
        # Add 2 terms for j=2 part (2x1 by 1x2)
        # C_i,k += A_i,2 B_2,k
        # Term 1: k_local=0
        for i in range(2):
            U[i*m + 2, col] = 1
        V[2*p + (k_offset + 0), col] = 1
        for i in range(2):
            W[(k_offset + 0)*n + i, col] = 1
        col += 1
        
        # Term 2: k_local=1
        for i in range(2):
            U[i*m + 2, col] = 1
        V[2*p + (k_offset + 1), col] = 1
        for i in range(2):
            W[(k_offset + 1)*n + i, col] = 1
        col += 1

    # Last block for k=8
    # C_i,8 = A_i,0 B_0,8 + A_i,1 B_1,8 + A_i,2 B_2,8
    # 6 terms (schoolbook for 2x3 by 3x1)
    for i in range(2):
        for j in range(3):
            U[i*m + j, col] = 1
            V[j*p + 8, col] = 1
            W[8*n + i, col] = 1
            col += 1

    # Verify full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                val = np.dot(U[a,:], V[b,:] * W[c,:])
                if abs(val - 1) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        # Check zeros
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.dot(U[a,:], V[b,:] * W[c,:])
                        if abs(val) > 1e-10:
                            valid = False
                            break
                if not valid: break
            if not valid: break
            
        if valid:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank_total

    return best_result
# EVOLVE-BLOCK-END

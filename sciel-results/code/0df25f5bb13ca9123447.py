import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # 1. Construct rank-27 schoolbook decomposition for 3x3
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*n3 + j*n3 + k # Wait, standard schoolbook for 3x3:
                    # U[i, j] corresponds to A[i, j]
                    # V[j, k] corresponds to B[j, k]
                    # W[k, i] corresponds to C[k, i] ? No, C[i, k] = sum_j A[i, j] B[j, k].
                    # Tensor T[i, j, k] = 1.
                    # U[i, j, r] V[j, k, r] W[k, i, r] ?
                    # The trace convention is W[k*n+i, r].
                    # So W index is k*n + i.
                    # For 3x3, W index is k*3 + i.
                    # Let's stick to the previous definition which was consistent with the check.
                    pass
        
        # Re-implementing correctly for 3x3
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Schoolbook: U[i, j, r] = 1 if r = i*3+j, V[j, k, r] = 1 if r = i*3+j, W[k, i, r] = 1 if r = i*3+j.
        # Wait, the index r must be the same.
        # Let r = i*3 + j.
        # Then U[i, j, i*3+j] = 1.
        # V[j, k, i*3+j] = 1.
        # W[k, i, i*3+j] = 1.
        # Let's check: sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Only r = i*3+j is non-zero for U and V.
        # Then term is 1 * 1 * W[k, i, i*3+j].
        # W[k, i, i*3+j] is 1.
        # So sum is 1. Correct.
        
        for i in range(3):
            for j in range(3):
                r = i*3 + j
                U3[i*3 + j, r] = 1.0
                # V index is j*3 + k? No, V is (m*p, rank) = (9, 27).
                # V[j, k, r] = 1.
                # But V depends on k.
                # In schoolbook, the term is A[i, j] * B[j, k].
                # So for a fixed i, j, the sum over k is separate?
                # No, the tensor is T[i, j, k].
                # The rank decomposition is T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
                # If we set U[i, j, i*3+j] = 1, V[j, k, i*3+j] = 1, W[k, i, i*3+j] = 1.
                # Then for any i, j, k:
                # Sum is over r.
                # U[i, j, r] is non-zero only for r = i*3+j.
                # So sum reduces to U[i, j, i*3+j] * V[j, k, i*3+j] * W[k, i, i*3+j].
                # = 1 * 1 * 1 = 1.
                # This works.
                V3[j*3 + k, r] # Wait, V row index is j*3+k.
                # We need to set V3[j*3+k, i*3+j] = 1 for ALL k?
                # If we set V3[j*3+k, i*3+j] = 1 for all k, then:
                # Term is 1 * 1 * W[k, i, i*3+j].
                # W3[k*3+i, i*3+j] must be 1.
                # So we set:
                # U3[i*3+j, i*3+j] = 1
                # V3[j*3+k, i*3+j] = 1 for all k
                # W3[k*3+i, i*3+j] = 1 for all k
                
                U3[i*3 + j, i*3 + j] = 1.0
                for k in range(3):
                    V3[j*3 + k, i*3 + j] = 1.0
                    W3[k*3 + i, i*3 + j] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27
    
    # 2. Construct 6x6 decomposition using Strassen on 2x2 blocks of 3x3
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Strassen formulas:
    # P1 = (A0 + A3)(B0 + B3) -> C0 += P1, C3 += P1
    # P2 = (A2 + A3)B0       -> C2 += P2, C3 -= P2
    # P3 = A0(B1 - B3)       -> C1 += P3, C3 += P3
    # P4 = A3(B2 - B0)       -> C0 += P4, C2 += P4
    # P5 = (A0 + A1)B3       -> C1 += P5, C0 -= P5
    # P6 = (A2 - A0)(B0 + B1)-> C3 += P6
    # P7 = (A1 - A3)(B2 + B3)-> C0 += P7
    
    # Define the structure for each term
    # Each term is a list of (block_idx, coeff) for U, V, W
    terms = [
        { # P1
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]
        },
        { # P2
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)]
        },
        { # P3
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)]
        },
        { # P4
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)]
        },
        { # P5
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(1, 1), (0, -1)]
        },
        { # P6
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)]
        },
        { # P7
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)]
        }
    ]
    
    def add_block_contribution(U_g, V_g, W_g, U3, V3, W3, 
                               u_blocks, v_blocks, w_blocks, start_col):
        for r in range(r3):
            # Extract 3x3 vectors for this rank component
            u_vec = U3[:, r]
            v_vec = V3[:, r]
            w_vec = W3[:, r]
            
            # Update U6
            for bi, coeff in u_blocks:
                row_block = bi // 2
                col_block = bi % 2
                # Local index in 3x3: i, j
                # Global index in 6x6: (row_block*3 + i)*6 + (col_block*3 + j)
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_vec[i*3 + j]
                        if val != 0:
                            global_i = row_block * 3 + i
                            global_j = col_block * 3 + j
                            global_u_idx = global_i * 6 + global_j
                            U6[global_u_idx, start_col + r] += val

            # Update V6
            for bi, coeff in v_blocks:
                row_block = bi // 2
                col_block = bi % 2
                # Local index in 3x3: j, k
                # Global index in 6x6: (row_block*3 + j)*6 + (col_block*3 + k)
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_vec[j*3 + k]
                        if val != 0:
                            global_j = row_block * 3 + j
                            global_k = col_block * 3 + k
                            global_v_idx = global_j * 6 + global_k
                            V6[global_v_idx, start_col + r] += val

            # Update W6
            for bi, coeff in w_blocks:
                row_block = bi // 2
                col_block = bi % 2
                # Local index in 3x3: k, i (W index is k*n + i)
                # Global index in 6x6: (col_block*3 + k)*6 + (row_block*3 + i)
                # Note: W is for C. C block (row_block, col_block) corresponds to C[row_block*3.., col_block*3..]
                # C[i, k] in local? No, C is 3x3.
                # Standard: C[i, k] = sum_j A[i, j] B[j, k].
                # W index is k*3 + i.
                # So local w_vec index is k*3 + i.
                # Global C index: row = row_block*3 + i, col = col_block*3 + k.
                # Global W index = col*6 + row = (col_block*3 + k)*6 + (row_block*3 + i).
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_vec[k*3 + i]
                        if val != 0:
                            global_k = col_block * 3 + k
                            global_i = row_block * 3 + i
                            global_w_idx = global_k * 6 + global_i
                            W6[global_w_idx, start_col + r] += val

    for k, term in enumerate(terms):
        start_col = k * r3
        add_block_contribution(U6, V6, W6, U3, V3, W3, 
                               term['u'], term['v'], term['w'], start_col)
        
    # Verify
    if time.time() - start_time > 100:
        # Fallback to schoolbook if we are running out of time
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    try:
        # Reshape for einsum check
        # U is (36, 189). Reshape to (6, 6, 189)
        U3d = U6.reshape(6, 6, -1)
        V3d = V6.reshape(6, 6, -1)
        W3d = W6.reshape(6, 6, -1)
        
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U3d, V3d, W3d)
        target = np.ones((6, 6, 6))
        
        if np.allclose(reconstructed, target, atol=1e-9):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        else:
            # Fallback
            rank = 216
            U_f = np.zeros((36, rank))
            V_f = np.zeros((36, rank))
            W_f = np.zeros((36, rank))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i*6 + j
                        jk = j*6 + k
                        ki = k*6 + i
                        col = i*36 + j*6 + k
                        U_f[idx, col] = 1.0
                        V_f[jk, col] = 1.0
                        W_f[ki, col] = 1.0
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    except:
        rank = 216
        U_f = np.zeros((36, rank))
        V_f = np.zeros((36, rank))
        W_f = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U_f[idx, col] = 1.0
                    V_f[jk, col] = 1.0
                    W_f[ki, col] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END

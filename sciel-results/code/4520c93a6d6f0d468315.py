"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
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
    
    # Construct rank-17 decomposition for 2x3 by 3x3 block
    # Using Strassen (rank 7) for j,k in {0,1} and schoolbook for the rest.
    
    n_block, m_block, p_block = 2, 3, 3
    rank_block = 17
    dim_u_b = n_block * m_block # 6
    dim_v_b = m_block * p_block # 9
    dim_w_b = p_block * n_block # 6
    
    U_b = np.zeros((dim_u_b, rank_block))
    V_b = np.zeros((dim_v_b, rank_block))
    W_b = np.zeros((dim_w_b, rank_block))
    
    col = 0
    
    # 1. Strassen for j in {0,1}, k in {0,1}
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # M1
    U_b[0, col] = 1; U_b[3, col] = 1 # a00, a11
    V_b[0, col] = 1; V_b[1, col] = 1 # b00, b11
    W_b[0, col] = 1; W_b[1, col] = 1; W_b[3, col] = 1; W_b[4, col] = 1 # c00, c01, c10, c11? No.
    # W indices: k*n+i.
    # c00 (k=0, i=0) -> W[0]
    # c01 (k=1, i=0) -> W[2]
    # c10 (k=0, i=1) -> W[1]
    # c11 (k=1, i=1) -> W[3]
    # M1 contributes to c00, c01, c10, c11?
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Let's just fill W_b with the coefficients for each output.
    # It's easier to define the U, V, W vectors for each multiplication and then sum their contributions to the outputs.
    # But the format is U[a,d], V[b,d], W[c,d].
    # For a specific multiplication d, U[a,d] and V[b,d] are fixed.
    # W[c,d] determines which output gets this multiplication.
    # We need sum_d U[a,d] V[b,d] W[c,d] = 1 if (a,b,c) is a valid term.
    
    # This means for each d, we define a set of (a,b) pairs that are multiplied, and a set of c indices that receive the result.
    # Actually, W[c,d] is a scalar.
    # If we use Strassen, the "multiplications" are linear combinations.
    # This requires U and V to contain the coefficients of the linear combinations.
    
    # Let's restart the block construction with explicit coefficients.
    
    U_b = np.zeros((dim_u_b, rank_block))
    V_b = np.zeros((dim_v_b, rank_block))
    W_b = np.zeros((dim_w_b, rank_block))
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U_b[idx, col] += val
        for idx, val in v_vals.items():
            V_b[idx, col] += val
        for idx, val in w_vals.items():
            W_b[idx, col] += val
        col += 1
        
    # Indices:
    # U: i*m+j. i in {0,1}, j in {0,1,2}.
    # 0: a00, 1: a01, 2: a02
    # 3: a10, 4: a11, 5: a12
    # V: j*p+k. j in {0,1,2}, k in {0,1,2}.
    # 0: b00, 1: b01, 2: b02
    # 3: b10, 4: b11, 5: b12
    # 6: b20, 7: b21, 8: b22
    # W: k*n+i. k in {0,1,2}, i in {0,1}.
    # 0: c00, 1: c10
    # 2: c01, 3: c11
    # 4: c02, 5: c12
    
    # Strassen Terms
    # M1 = (a00+a11)(b00+b11)
    # U: {0:1, 3:1}
    # V: {0:1, 4:1}
    # W: c00(0)=1, c01(2)=1, c10(1)=1, c11(3)=1?
    # c00 = M1 + ...
    # c01 = ... + M5 (no M1)
    # Wait, c01 = M3 + M5. M1 is not in c01.
    # c10 = M2 + M4. M1 is not in c10.
    # c11 = M1 - M2 + M3 + M6.
    # So M1 contributes to c00 and c11.
    # W: {0:1, 3:1}
    add_term({0:1, 3:1}, {0:1, 4:1}, {0:1, 3:1})
    
    # M2 = (a10+a11)b00
    # U: {3:1, 4:1}
    # V: {0:1}
    # c10 = M2 + ...
    # c11 = ... - M2 ...
    # W: {1:1, 3:-1}
    add_term({3:1, 4:1}, {0:1}, {1:1, 3:-1})
    
    # M3 = a00(b01-b11)
    # U: {0:1}
    # V: {1:1, 4:-1}
    # c01 = ... + M3
    # c11 = ... + M3 ...
    # W: {2:1, 3:1}
    add_term({0:1}, {1:1, 4:-1}, {2:1, 3:1})
    
    # M4 = a11(b10-b00)
    # U: {4:1}
    # V: {3:1, 0:-1}
    # c00 = ... + M4 ...
    # c10 = ... + M4
    # W: {0:1, 1:1}
    add_term({4:1}, {3:1, 0:-1}, {0:1, 1:1})
    
    # M5 = (a00+a01)b11
    # U: {0:1, 1:1}
    # V: {4:1}
    # c00 = ... - M5 ...
    # c01 = ... + M5
    # W: {0:-1, 2:1}
    add_term({0:1, 1:1}, {4:1}, {0:-1, 2:1})
    
    # M6 = (a10-a00)(b00+b01)
    # U: {3:1, 0:-1}
    # V: {0:1, 1:1}
    # c11 = ... + M6
    # W: {3:1}
    add_term({3:1, 0:-1}, {0:1, 1:1}, {3:1})
    
    # M7 = (a01-a11)(b10+b11)
    # U: {1:1, 4:-1}
    # V: {3:1, 4:1}
    # c00 = ... + M7
    # W: {0:1}
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Schoolbook Terms for j in {0,1}, k=2
    # a00*b02 -> c02
    add_term({0:1}, {2:1}, {4:1})
    # a01*b12 -> c02
    add_term({1:1}, {5:1}, {4:1})
    # a10*b02 -> c12
    add_term({3:1}, {2:1}, {5:1})
    # a11*b12 -> c12
    add_term({4:1}, {5:1}, {5:1})
    
    # Schoolbook Terms for j=2, k in {0,1,2}
    # a02*b20 -> c00
    add_term({2:1}, {6:1}, {0:1})
    # a02*b21 -> c01
    add_term({2:1}, {7:1}, {2:1})
    # a02*b22 -> c02
    add_term({2:1}, {8:1}, {4:1})
    # a12*b20 -> c10
    add_term({5:1}, {6:1}, {1:1})
    # a12*b21 -> c11
    add_term({5:1}, {7:1}, {3:1})
    # a12*b22 -> c12
    add_term({5:1}, {8:1}, {5:1})
    
    # Verify block
    err = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                a = i*m_block+j
                b = j*p_block+k
                c = k*n_block+i
                val = np.dot(U_b[a,:], V_b[b,:] * W_b[c,:])
                err += (val - 1)**2
    for a in range(dim_u_b):
        for b in range(dim_v_b):
            for c in range(dim_w_b):
                i = a // m_block
                j = a % m_block
                j2 = b // p_block
                k = b % p_block
                k2 = c // n_block
                i2 = c % n_block
                if not (j == j2 and k == k2 and i == i2):
                    val = np.dot(U_b[a,:], V_b[b,:] * W_b[c,:])
                    err += val**2
    
    if err < 1e-10:
        # Construct full decomposition
        full_rank = rank_block * 3
        U_full = np.zeros((dim_u, full_rank))
        V_full = np.zeros((dim_v, full_rank))
        W_full = np.zeros((dim_w, full_rank))
        
        for block_idx in range(3):
            k_offset = block_idx * p_block
            for c in range(rank_block):
                full_c = block_idx * rank_block + c
                U_full[:, full_c] = U_b[:, c]
                
                for j in range(m_block):
                    for k in range(p_block):
                        v_block_idx = j * p_block + k
                        v_full_idx = j * p + (k_offset + k)
                        V_full[v_full_idx, full_c] = V_b[v_block_idx, c]
                        
                for k in range(p_block):
                    for i in range(n_block):
                        w_block_idx = k * n_block + i
                        w_full_idx = (k_offset + k) * n + i
                        W_full[w_full_idx, full_c] = W_b[w_block_idx, c]
        
        # Verify full
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.dot(U_full[a,:], V_full[b,:] * W_full[c,:])
                    if abs(val - 1) > 1e-10:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            best_rank = full_rank
    
    return best_result
# EVOLVE-BLOCK-END

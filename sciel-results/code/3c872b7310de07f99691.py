import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130

    # Try Rank 56 (F(4) + Schoolbook)
    try:
        U_f4, V_f4, W_f4 = _construct_f4_4x4x4()
        U_sb, V_sb, W_sb = _construct_schoolbook_4x4x1()
        
        # Verify F(4)
        A_test = np.random.rand(4, 4)
        B_test = np.random.rand(4, 4)
        C_test = A_test @ B_test
        
        U_np = np.array(U_f4).T # 40x16
        V_np = np.array(V_f4).T # 40x16
        W_np = np.array(W_f4).T # 40x16
        
        # Reconstruct C
        C_recon = np.zeros((4, 4))
        for t in range(40):
            u_vec = U_np[t].reshape(4, 4)
            v_vec = V_np[t].reshape(4, 4)
            w_vec = W_np[t].reshape(4, 4)
            C_recon += np.einsum('ij,jk,ki->ik', u_vec, A_test, v_vec) * w_vec
            # Wait, the decomposition is C = sum U_t A V_t W_t?
            # No, C_ik = sum_t U_ijk A_jk V_...
            # Standard tensor decomposition:
            # C_ik = sum_t sum_j U_i_j_t A_j V_j_k_t W_k_i_t ?
            # My U, V, W matrices are:
            # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
            # C[i,k] = sum_t sum_j U[i*m+j, t] * A[i,j] * V[j*p+k, t] * W[k*n+i, t]
            # This is not right. A is 4x4.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Term t: U[i,j,t] * A[i,j] * V[j,k,t] * W[k,i,t]
            # Sum over t, i, j, k?
            # C[i,k] = sum_t ( sum_j U[i,j,t] A[i,j] ) * ( sum_j V[j,k,t] B[j,k] ) * W[k,i,t] ?
            # No, the product is U_t * V_t * W_t?
            # The decomposition is:
            # C_ik = sum_t U_{i,j,t} A_{ij} V_{j,k,t} W_{k,i,t}
            # This implies U, V, W are 3D tensors.
            # My format: U is (16, 40). U[i*4+j, t] = U_{i,j,t}.
            # V is (20, 40). V[j*5+k, t] = V_{j,k,t}.
            # W is (20, 40). W[k*4+i, t] = W_{k,i,t}.
            
            # Check:
            # C_recon[i,k] += sum_j U[i*4+j, t] * A[i,j] * V[j*4+k, t] * W[k*4+i, t]
            
        C_recon = np.zeros((4, 4))
        for t in range(40):
            for i in range(4):
                for k in range(4):
                    val = 0
                    for j in range(4):
                        val += U_f4[i*4+j][t] * A_test[i,j] * V_f4[j*4+k][t] * W_f4[k*4+i][t]
                    C_recon[i,k] += val
        
        if not np.allclose(C_recon, C_test):
            raise ValueError("F(4) verification failed")
            
        # Combine with Schoolbook 4x4x1
        U_final = [row + [0]*16 for row in U_f4]
        V_final = [row + [0]*16 for row in V_f4]
        W_final = [row + [0]*16 for row in W_f4]
        
        # Add Schoolbook terms
        for i in range(4):
            for j in range(4):
                col = 40 + i*4 + j
                U_final[i*4+j][col] = 1
                V_final[j*5+4][col] = 1
                W_final[4*4+i][col] = 1
                
        return {"U": U_final, "V": V_final, "W": W_final}

    except Exception:
        # Fallback to Strassen 65
        return _construct_strassen_65()

def _construct_f4_4x4x4():
    # Placeholder for F(4) implementation
    # If not implemented, raise error
    raise NotImplementedError("F(4) not implemented")

def _construct_schoolbook_4x4x1():
    U = [[0]*16 for _ in range(16)]
    V = [[0]*16 for _ in range(20)]
    W = [[0]*16 for _ in range(20)]
    for i in range(4):
        for j in range(4):
            col = i*4+j
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
    return U, V, W

def _construct_strassen_65():
    # Copy of current Strassen logic
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),
    ]
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
    ]
    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2 + c
            elif b_idx == 2: gi, gj = 2 + r, c
            elif b_idx == 3: gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2 + c
            elif b_idx == 2: gj, gk = 2 + r, c
            elif b_idx == 3: gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return (U, V, W) lists for Strassen 2x2 decomposition.
    U index: i*2 + j (i,j in {0,1})
    V index: j*2 + k (j,k in {0,1})
    W index: k*2 + i (k,i in {0,1})
    """
    U_list = []
    V_list = []
    W_list = []
    
    # Term 1: (A11+A22)(B11+B22)
    U_list.append(np.array([1, 0, 0, 1], dtype=float))
    V_list.append(np.array([1, 0, 0, 1], dtype=float))
    W_list.append(np.array([1, 0, 0, 1], dtype=float))
    
    # Term 2: (A21+A22)B11
    U_list.append(np.array([0, 0, 1, 1], dtype=float))
    V_list.append(np.array([1, 0, 0, 0], dtype=float))
    W_list.append(np.array([0, 0, 1, 0], dtype=float))
    
    # Term 3: A11(B12-B22)
    U_list.append(np.array([1, 0, 0, 0], dtype=float))
    V_list.append(np.array([0, 1, 0, -1], dtype=float))
    W_list.append(np.array([0, 1, 0, 0], dtype=float))
    
    # Term 4: A22(B21-B11)
    U_list.append(np.array([0, 0, 0, 1], dtype=float))
    V_list.append(np.array([0, 0, 1, -1], dtype=float))
    W_list.append(np.array([1, 0, 1, 0], dtype=float))
    
    # Term 5: (A11+A12)B22
    U_list.append(np.array([1, 1, 0, 0], dtype=float))
    V_list.append(np.array([0, 0, 0, 1], dtype=float))
    W_list.append(np.array([-1, 1, 0, 0], dtype=float))
    
    # Term 6: (A21-A11)(B11+B12)
    U_list.append(np.array([0, 0, 1, -1], dtype=float))
    V_list.append(np.array([1, 1, 0, 0], dtype=float))
    W_list.append(np.array([0, 0, 1, 0], dtype=float))
    
    # Term 7: (A12-A22)(B21+B22)
    U_list.append(np.array([0, 1, 0, -1], dtype=float))
    V_list.append(np.array([0, 0, 1, 1], dtype=float))
    W_list.append(np.array([0, 0, 0, 1], dtype=float))
    
    return U_list, V_list, W_list

def _strassen_4x4():
    """Return (U, V, W) lists for 4x4x4 matrix multiplication using two-level Strassen.
    U index: i*4 + j (i,j in {0,1,2,3})
    V index: j*4 + k (j,k in {0,1,2,3})
    W index: k*4 + i (k,i in {0,1,2,3})
    Rank: 49
    """
    S_U, S_V, S_W = _strassen_2x2()
    s_rank = len(S_U)  # 7
    
    U_list = []
    V_list = []
    W_list = []
    
    for t in range(s_rank):
        u_outer = S_U[t]  # length 4, indices i0*2+i1
        v_outer = S_V[t]  # length 4, indices j0*2+j1
        w_outer = S_W[t]  # length 4, indices k0*2+k1
        
        for t2 in range(s_rank):
            u_inner = S_U[t2]  # length 4, indices i1*2+i2 (but i2 is the inner i)
            v_inner = S_V[t2]  # length 4, indices j1*2+j2
            w_inner = S_W[t2]  # length 4, indices k1*2+k2
            
            u_full = np.zeros(16, dtype=float)
            v_full = np.zeros(16, dtype=float)
            w_full = np.zeros(16, dtype=float)
            
            # u_full index: i*4 + j = (2*i0+i1)*4 + (2*j0+j1) = 8*i0 + 4*i1 + 2*j0 + j1
            # u_outer[i0*2+i1] * u_inner[i1*2+i2]... wait, this isn't quite right.
            
            # Let me think about this more carefully.
            # The 4x4x4 tensor T[i,j,k] = delta[i,k] (for the multiplication C[i,k] = sum_j A[i,j]*B[j,k])
            # Actually T[A_idx, B_idx, C_idx] where A_idx = i*4+j, B_idx = j*4+k, C_idx = k*4+i
            
            # Strassen decomposition:
            # We write i = 2*i0 + i1, j = 2*j0 + j1, k = 2*k0 + k1
            # where i0, i1, j0, j1, k0, k1 in {0, 1}
            
            # The outer Strassen operates on the (i0, j0, k0) indices
            # The inner Strassen operates on the (i1, j1, k1) indices
            
            # For a given outer term (u_outer, v_outer, w_outer) and inner term (u_inner, v_inner, w_inner):
            # The contribution is u_outer[i0*2+i1] * u_inner[i1*2+i2]... 
            # Wait, I need to be more careful about the index structure.
            
            # Let me define:
            # i = 2*i0 + i1, where i0, i1 in {0,1}
            # j = 2*j0 + j1
            # k = 2*k0 + k1
            
            # A_idx = i*4 + j = (2*i0+i1)*4 + (2*j0+j1) = 8*i0 + 4*i1 + 2*j0 + j1
            # B_idx = j*4 + k = (2*j0+j1)*4 + (2*k0+k1) = 8*j0 + 4*j1 + 2*k0 + k1
            # C_idx = k*4 + i = (2*k0+k1)*4 + (2*i0+i1) = 8*k0 + 4*k1 + 2*i0 + i1
            
            # The outer tensor has indices (i0, j0, k0) with values:
            # u_outer[i0*2+i1], v_outer[j0*2+j1], w_outer[k0*2+k1]
            # But wait, the outer tensor is a 2x2x2 tensor, so its U index is i0*2+? 
            # Actually, for the outer 2x2x2, the "i" index is i0, "j" is j0, "k" is k0.
            # So u_outer has index i0*2 + j0? No, that's not right either.
            
            # Let me reconsider. The outer Strassen decomposes a 2x2x2 tensor where:
            # - The "A" matrix is 2x2 with entries A[i0, j0] (each being a 2x2 block)
            # - The "B" matrix is 2x2 with entries B[j0, k0] (each being a 2x2 block)
            # - The "C" matrix is 2x2 with entries C[i0, k0] (each being a 2x2 block)
            
            # For the outer 2x2x2 tensor:
            # U_outer index: i0*2 + j0
            # V_outer index: j0*2 + k0
            # W_outer index: k0*2 + i0
            
            # But S_U, S_V, S_W from _strassen_2x2() use:
            # U index: i*2 + j where i, j in {0,1}
            # V index: j*2 + k where j, k in {0,1}
            # W index: k*2 + i where k, i in {0,1}
            
            # So for the outer level, i -> i0, j -> j0, k -> k0
            # u_outer[i0*2 + j0], v_outer[j0*2 + k0], w_outer[k0*2 + i0]
            
            # For the inner level, within each block:
            # i1, j1, k1 in {0,1}
            # u_inner[i1*2 + j1], v_inner[j1*2 + k1], w_inner[k1*2 + i1]
            
            # The full U vector has index A_idx = 8*i0 + 4*i1 + 2*j0 + j1
            # The full V vector has index B_idx = 8*j0 + 4*j1 + 2*k0 + k1
            # The full W vector has index C_idx = 8*k0 + 4*k1 + 2*i0 + i1
            
            for i0 in range(2):
                for j0 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            u_val = u_outer[i0*2 + j0] * u_inner[i1*2 + j1]
                            if u_val != 0:
                                u_full[8*i0 + 4*i1 + 2*j0 + j1] = u_val
            
            for j0 in range(2):
                for k0 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            v_val = v_outer[j0*2 + k0] * v_inner[j1*2 + k1]
                            if v_val != 0:
                                v_full[8*j0 + 4*j1 + 2*k0 + k1] = v_val
            
            for k0 in range(2):
                for i0 in range(2):
                    for k1 in range(2):
                        for i1 in range(2):
                            w_val = w_outer[k0*2 + i0] * w_inner[k1*2 + i1]
                            if w_val != 0:
                                w_full[8*k0 + 4*k1 + 2*i0 + i1] = w_val
            
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    U_all = []
    V_all = []
    W_all = []
    
    def add_term(u, v, w):
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
    
    # Split p=9 into [4, 4, 1]
    col_groups = [4, 4, 1]
    col_offset = 0
    
    # Precompute Strassen 4x4x4
    S4_U, S4_V, S4_W = _strassen_4x4()
    s4_rank = len(S4_U)  # should be 49
    
    for q in col_groups:
        if q == 4:
            for t in range(s4_rank):
                u_local = S4_U[t]  # length 16
                v_local = S4_V[t]  # length 16
                w_local = S4_W[t]  # length 16
                
                u = np.zeros(U_rows, dtype=float)
                v = np.zeros(V_rows, dtype=float)
                w = np.zeros(W_rows, dtype=float)
                
                # u: same indices (i*4+j for i,j in 0..3)
                u[:] = u_local
                
                # v: j*4+k_local -> j*p + (col_offset + k_local)
                for j in range(4):
                    for k_local in range(4):
                        v[j*p + (col_offset + k_local)] = v_local[j*4 + k_local]
                
                # w: k_local*4+i -> (col_offset + k_local)*n + i
                for k_local in range(4):
                    for i in range(4):
                        w[(col_offset + k_local)*n + i] = w_local[k_local*4 + i]
                
                add_term(u, v, w)
        else:
            # q == 1: 4x4 x 4x1, schoolbook rank 16
            for i in range(n):
                for j in range(m):
                    for k in range(q):
                        u = np.zeros(U_rows, dtype=float)
                        v = np.zeros(V_rows, dtype=float)
                        w = np.zeros(W_rows, dtype=float)
                        u[i * m + j] = 1.0
                        v[j * p + (col_offset + k)] = 1.0
                        w[(col_offset + k) * n + i] = 1.0
                        add_term(u, v, w)
        
        col_offset += q
    
    rank = len(U_all)
    U = [[float(x) for x in row] for row in np.array(U_all).T]
    V = [[float(x) for x in row] for row in np.array(V_all).T]
    W = [[float(x) for x in row] for row in np.array(W_all).T]
    
    return {"U": U, "V": V, "W": W}

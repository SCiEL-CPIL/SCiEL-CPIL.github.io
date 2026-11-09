import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    # Strassen decomposition for 2x2 * 2x2
    # Returns list of (u_vec, v_vec, w_vec) for the 7 scalar multiplications
    # u_vec is length 4 (2x2 flat), v_vec is length 4, w_vec is length 4 (output 2x2 flat)
    def strassen_2x2():
        terms = []
        # Indices for X: 00:0, 01:1, 10:2, 11:3
        # Indices for Y: 00:0, 01:1, 10:2, 11:3
        # Indices for Z: 00:0, 01:1, 10:2, 11:3
        
        # M1 = (X00 + X11)(Y00 + Y11)
        u1 = np.array([1, 0, 0, 1])
        v1 = np.array([1, 0, 0, 1])
        # Z00 += M1, Z11 += M1
        w1 = np.array([1, 0, 0, 1])
        terms.append((u1, v1, w1))
        
        # M2 = (X10 + X11)Y00
        u2 = np.array([0, 0, 1, 1])
        v2 = np.array([1, 0, 0, 0])
        # Z10 += M2, Z11 -= M2
        w2 = np.array([0, 0, 1, -1])
        terms.append((u2, v2, w2))
        
        # M3 = X00(Y01 - Y11)
        u3 = np.array([1, 0, 0, 0])
        v3 = np.array([0, 1, 0, -1])
        # Z01 += M3, Z11 -= M3
        w3 = np.array([0, 1, 0, -1])
        terms.append((u3, v3, w3))
        
        # M4 = X11(Y10 - Y00)
        u4 = np.array([0, 0, 0, 1])
        v4 = np.array([-1, 0, 1, 0])
        # Z00 -= M4, Z01 += M4
        w4 = np.array([-1, 1, 0, 0])
        terms.append((u4, v4, w4))
        
        # M5 = (X00 + X01)Y11
        u5 = np.array([1, 1, 0, 0])
        v5 = np.array([0, 0, 0, 1])
        # Z00 += M5, Z11 += M5 (Wait, Z11 = M1 - M3 + M6 - M2. M5 is not in Z11?)
        # Check Strassen formula:
        # Z00 = M1 - M4 + M5 - M7
        # Z01 = M3 + M4
        # Z10 = M2 + M7
        # Z11 = M1 - M3 + M6 - M2
        # So M5 is only in Z00.
        w5 = np.array([1, 0, 0, 0])
        terms.append((u5, v5, w5))
        
        # M6 = (X10 - X00)(Y00 + Y01)
        u6 = np.array([-1, 0, 1, 0])
        v6 = np.array([1, 1, 0, 0])
        # Z11 += M6
        w6 = np.array([0, 0, 0, 1])
        terms.append((u6, v6, w6))
        
        # M7 = (X01 - X11)(Y10 + Y11)
        u7 = np.array([0, 1, 0, -1])
        v7 = np.array([0, 0, 1, 1])
        # Z00 -= M7, Z10 += M7
        w7 = np.array([-1, 0, 1, 0])
        terms.append((u7, v7, w7))
        
        return terms

    # We need to compute:
    # C_left = A1 * B11 + A2 * B21
    # C_right = A1 * B12 + A2 * B22
    # A1 is cols 0,1 of A. A2 is cols 2,3.
    # B11 is rows 0,1 cols 0,1 of B. B12 is rows 0,1 cols 2,3.
    # B21 is rows 2,3 cols 0,1 of B. B22 is rows 2,3 cols 2,3 of B.
    
    # Global A flat indices:
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3 (A1)
    # A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7 (A2)
    
    # Global B flat indices:
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3 (B11)
    # B[0,2]=4, B[0,3]=5, B[1,2]=6, B[1,3]=7 (B12)
    # B[2,0]=8, B[2,1]=9, B[3,0]=10, B[3,1]=11 (B21)
    # B[2,2]=12, B[2,3]=13, B[3,2]=14, B[3,3]=15 (B22)
    
    # Global C flat indices (k*n + i):
    # C is 4x2.
    # k=0 (row 0 of B, col 0 of C? No, C is 2x4? No, C is output of A(2x4)*B(4x4) -> 2x4)
    # Trace convention W[k*n + i, r]. k is row of C? i is col of C?
    # If C is 2x4, k is in 0..3, i is in 0..1.
    # C[k, i] = sum_j B[k, j] A[i, j]? No.
    # A is 2x4. B is 4x4.
    # (AB) is 2x4.
    # (AB)_{i, k} = sum_j A[i, j] B[j, k].
    # Trace convention: W[k*n + i, r].
    # This implies the output is indexed by k (0..3) and i (0..1).
    # So C_flat[k, i] = (AB)_{i, k}.
    # So C is effectively $C^T$ in standard row-major, or just indexed by (col, row).
    
    # C1 = A1 * B11. A1 is 2x2, B11 is 2x2. Result is 2x2.
    # This contributes to C[i, k] for i in 0..1, k in 0..1.
    # C1[i, k] = (A1 B11)_{i, k}.
    # In global C: C_flat[k*2 + i] += C1[i, k].
    
    # C2 = A1 * B12. Result 2x2.
    # Contributes to C[i, k] for i in 0..1, k in 2..3.
    # C2[i, k-2] = (A1 B12)_{i, k-2}.
    # Global: C_flat[(k-2+2)*2 + i] = C_flat[k*2 + i] += C2[i, k-2].
    
    # C3 = A2 * B21. Result 2x2.
    # Contributes to C[i, k] for i in 0..1, k in 0..1.
    # C3[i, k] = (A2 B21)_{i, k}.
    # Global: C_flat[k*2 + i] += C3[i, k].
    
    # C4 = A2 * B22. Result 2x2.
    # Contributes to C[i, k] for i in 0..1, k in 2..3.
    # C4[i, k-2] = (A2 B22)_{i, k-2}.
    # Global: C_flat[k*2 + i] += C4[i, k-2].

    strassen_terms = strassen_2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to expand local 2x2 term to global
    def add_term(local_u, local_v, local_w, a_offset, b_offset, c_row_start):
        # local_u is 4x1 (2x2 flat)
        # local_v is 4x1
        # local_w is 4x1
        # a_offset: starting index in A flat for the 2x2 block
        # b_offset: starting index in B flat for the 2x2 block
        # c_row_start: starting index in C flat for the 2x2 block?
        # C flat index is k*n + i.
        # For C1 (k=0,1; i=0,1), indices are 0,1,2,3.
        # For C2 (k=2,3; i=0,1), indices are 4,5,6,7.
        
        u_global = np.zeros(n*m)
        v_global = np.zeros(m*p)
        w_global = np.zeros(p*n)
        
        # Map local A indices (0..3) to global
        # Local A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global A for A1: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Offset 0
        # Global A for A2: 4:(0,2), 5:(0,3), 6:(1,2), 7:(1,3) -> Offset 4
        
        for j in range(4):
            u_global[a_offset + j] += local_u[j]
            
        # Map local B indices
        # Local B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global B for B11: 0,1,2,3 -> Offset 0
        # Global B for B12: 4,5,6,7 -> Offset 4
        # Global B for B21: 8,9,10,11 -> Offset 8
        # Global B for B22: 12,13,14,15 -> Offset 12
        
        for j in range(4):
            v_global[b_offset + j] += local_v[j]
            
        # Map local C indices
        # Local C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> C[i, k]
        # Global C index: k*n + i.
        # For C1 (k=0,1), c_row_start=0.
        # Local 0 (i=0, k=0) -> Global 0*2+0 = 0.
        # Local 1 (i=0, k=1) -> Global 1*2+0 = 2? No.
        # C_flat[k, i].
        # Local C1 is 2x2 matrix Z. Z[i, k].
        # We add Z[i, k] to Global C_flat[k, i].
        # Z[0,0] -> Global C[0,0] -> idx 0.
        # Z[0,1] -> Global C[1,0] -> idx 1*2+0 = 2.
        # Z[1,0] -> Global C[0,1] -> idx 0*2+1 = 1.
        # Z[1,1] -> Global C[1,1] -> idx 1*2+1 = 3.
        
        # Mapping from local flat index l to global flat index g
        # l=0 (i=0, k=0) -> g = 0*2+0 = 0
        # l=1 (i=0, k=1) -> g = 1*2+0 = 2
        # l=2 (i=1, k=0) -> g = 0*2+1 = 1
        # l=3 (i=1, k=1) -> g = 1*2+1 = 3
        
        # For C2 (k=2,3), c_row_start=4.
        # Z[0,0] (i=0, k=2) -> Global C[2,0] -> idx 2*2+0 = 4.
        # Z[0,1] (i=0, k=3) -> Global C[3,0] -> idx 3*2+0 = 6.
        # Z[1,0] (i=1, k=2) -> Global C[2,1] -> idx 2*2+1 = 5.
        # Z[1,1] (i=1, k=3) -> Global C[3,1] -> idx 3*2+1 = 7.
        
        # General mapping:
        # k_local = l % 2
        # i_local = l // 2
        # k_global = c_row_start // 2 + k_local  # c_row_start is 0 or 4. 0->0, 4->2
        # i_global = i_local
        # g = k_global * n + i_global
        
        k_base = c_row_start // 2
        for l in range(4):
            i_local = l // 2
            k_local = l % 2
            k_global = k_base + k_local
            i_global = i_local
            g = k_global * n + i_global
            w_global[g] += local_w[l]
            
        U_list.append(u_global)
        V_list.append(v_global)
        W_list.append(w_global)

    # 1. P11 = A1 * B11
    for u, v, w in strassen_terms:
        add_term(u, v, w, a_offset=0, b_offset=0, c_row_start=0)
        
    # 2. P12 = A1 * B12
    for u, v, w in strassen_terms:
        add_term(u, v, w, a_offset=0, b_offset=4, c_row_start=4)
        
    # 3. P21 = A2 * B21
    for u, v, w in strassen_terms:
        add_term(u, v, w, a_offset=4, b_offset=8, c_row_start=0)
        
    # 4. P22 = A2 * B22
    for u, v, w in strassen_terms:
        add_term(u, v, w, a_offset=4, b_offset=12, c_row_start=4)

    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We will build the decomposition block by block.
    # Global U: (n*m, Rank), V: (m*p, Rank), W: (p*n, Rank)
    
    # We define a function to get the decomposition for a sub-block
    # sub_n, sub_m, sub_p
    # It returns U_sub (sub_n*sub_m, r), V_sub (sub_m*sub_p, r), W_sub (sub_p*sub_n, r)
    
    def get_decomp(sn, sm, sp):
        # If it's a 2x2x2 block, use Strassen
        if sn == 2 and sm == 2 and sp == 2:
            # Rank 7
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # M1: (A00+A11)(B00+B11)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2: (A10+A11)B00
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = -1
            
            # M3: A00(B01-B11)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1; W[3][2] = 1
            
            # M4: A11(B10-B00)
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            
            # M5: (A00+A01)B11
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1
            
            # M6: (A10-A00)(B00+B01)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7: (A01-A11)(B10+B11)
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[3][6] = -1
            
            return U, V, W
        
        # If it's 2x1x2 or 1x1x2 or similar small cases, use simple decompositions
        # General case: Schoolbook
        # But we can be smarter for 2xkx2 and 1xkx2
        
        # For now, let's just use schoolbook for anything not 2x2x2
        # And handle 2x1x2, 1x2x2, 1x1x2 specifically if needed
        # Actually, the splitting logic will call this with specific sizes.
        
        # Schoolbook
        rank = sn * sm * sp
        U = [[0]*rank for _ in range(sn*sm)]
        V = [[0]*rank for _ in range(sm*sp)]
        W = [[0]*rank for _ in range(sp*sn)]
        
        col = 0
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    U[i*sm+j][col] = 1
                    V[j*sp+k][col] = 1
                    W[k*sn+i][col] = 1
                    col += 1
        return U, V, W

    # We need to handle the specific splits used in the analysis
    # 2x5x2 -> 2x2x2 + 2x2x2 + 2x1x2
    # 1x5x2 -> 1x2x2 + 1x2x2 + 1x1x2
    
    # Let's create a function to get decomp for 2x1x2 (Rank 1)
    def get_decomp_212():
        U = [[1], [1]] # A0, A1
        V = [[1]]      # B (only one col in middle? No, 1x2)
        # Wait, 2x1x2 means A is 2x1, B is 1x2.
        # U size 2x1, V size 1x2? No.
        # U: (2*1, 1) = 2x1.
        # V: (1*2, 1) = 2x1.
        # W: (2*2, 1) = 4x1.
        
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        # This is outer product.
        # Rank 1?
        # C = A B^T.
        # C_ik = A_i B_k.
        # Sum_r U_ir V_jr W_kr.
        # Here j is fixed to 0 (size 1).
        # So V has size 1*2 = 2? No, V is indexed by j, k.
        # V_{0, r} and V_{1, r}?
        # V size is m*p = 1*2 = 2.
        # So V is 2x1.
        # U is 2x1.
        # W is 4x1.
        
        # We need U_i V_j W_k = A_i B_k.
        # If we set V_{0,1} = 1, V_{1,1} = 1? No.
        # The term is A_i * B_k.
        # The tensor is T_{i, j, k} = 1 if j=0?
        # No, the multiplication is C_ik = sum_j A_ij B_jk.
        # Here j only takes value 0.
        # So C_ik = A_i0 B_0k.
        # We need sum_r U_ir V_{0,r} W_{kr} = A_i0 B_0k.
        # This is not a constant tensor.
        # The tensor T is defined by the structure.
        # T_{i, 0, k} = 1.
        # So we need sum_r U_ir V_{0,r} W_{kr} = 1 for all i,k.
        # This is the all-ones 2x2 matrix. Rank 1.
        # U = [1, 1]^T.
        # V = [1]^T (only one entry? No, V has 2 rows).
        # Wait, V is indexed by j*p+k.
        # j=0, k=0,1.
        # So V has entries V[0], V[1].
        # We need U_i * V_j * W_k?
        # The sum is over r.
        # Term r: U_{i,r} V_{j,r} W_{k,r}.
        # For fixed j=0, we need sum_r U_{i,r} V_{0,r} W_{k,r} = 1.
        # Let V_{0,r} = 1.
        # Then sum_r U_{i,r} W_{k,r} = 1.
        # This is rank 1 for 2x2.
        # U = [1, 1], W = [1, 1, 1, 1].
        # But V has 2 rows. V[1] is unused?
        # Yes, V[1] corresponds to j=0, k=1? No.
        # V index is j*p+k.
        # j=0, k=0 -> index 0.
        # j=0, k=1 -> index 1.
        # So V[0] and V[1] are both used?
        # No, the sum is over j.
        # C_ik = sum_j sum_r U_{i,r} V_{j,r} W_{k,r} A_{ij} B_{jk}.
        # For j=0, term is A_{i0} B_{0k}.
        # We need the coefficient of A_{i0} B_{0k} to be 1.
        # So sum_r U_{i,r} V_{0,r} W_{k,r} = 1.
        # V_{1,r} is multiplied by A_{i1} B_{1k}, which are 0 in this sub-problem.
        # So V_{1,r} can be 0.
        
        U = [[1], [1]]
        V = [[1], [0]]
        W = [[1], [1], [1], [1]]
        return U, V, W

    # 1x2x2 (Rank 2)
    def get_decomp_122():
        # A is 1x2, B is 2x2.
        # C is 1x2.
        # C_k = A0 B0k + A1 B1k.
        # Tensor T_{j, k} = 1?
        # j=0,1. k=0,1.
        # We need sum_r U_{j,r} V_{k,r} W_{?,r} = 1?
        # No, W is for output.
        # Output is 1x2. W size 2xR.
        # U size 2xR. V size 4xR.
        # C_k = sum_j sum_r U_{j,r} V_{j,r} W_{k,r} A_{0j} B_{jk}.
        # We need coeff of A_{0j} B_{jk} to be 1.
        # So sum_r U_{j,r} V_{j,r} W_{k,r} = 1? No.
        # The term is A_{0j} B_{jk}.
        # So for fixed j,k, sum_r U_{j,r} V_{j,r} W_{k,r} = 1.
        # This is a 2x2 matrix for each k?
        # Actually, this is just schoolbook for 1x2x2. Rank 2.
        U = [[1, 0], [0, 1]]
        V = [[1, 0], [0, 1], [0, 0], [0, 0]] # V[0]=1, V[1]=0 for r=0?
        # Let's just use schoolbook for 1x2x2.
        return get_decomp(1, 2, 2)

    # 1x1x2 (Rank 1)
    def get_decomp_112():
        return get_decomp(1, 1, 2)

    # Combine
    U_global = []
    V_global = []
    W_global = []
    
    def append_decomp(U_sub, V_sub, W_sub, i_offset, j_offset, k_offset):
        # U_sub is (sn*sm, r)
        # Map to global U (n*m, R)
        # i in [0, sn), j in [0, sm)
        # Global index: (i + i_offset)*m + (j + j_offset)
        
        # We need to insert rows into U_global.
        # Since we are appending columns, we just need to ensure U_global has enough rows.
        # But U_global rows are fixed.
        # We will build U_global as a list of lists, initialized to 0.
        
        pass

    # Better approach:
    # Initialize U, V, W with 0s of size (n*m, 0), etc.
    # Then for each block, generate U_sub, V_sub, W_sub.
    # And add them to the global matrices at the correct positions.
    
    U = [[0]*0 for _ in range(n*m)]
    V = [[0]*0 for _ in range(m*p)]
    W = [[0]*0 for _ in range(p*n)]
    
    def add_block(U_sub, V_sub, W_sub, i_off, j_off, k_off):
        r = len(U_sub[0])
        # Extend global matrices
        for row in U:
            row.extend([0]*r)
        for row in V:
            row.extend([0]*r)
        for row in W:
            row.extend([0]*r)
            
        # Fill
        for i in range(len(U_sub)//j_off): # No, len(U_sub) is sn*sm
            # We need sn, sm.
            pass

    # Let's refactor to pass sn, sm, sp to add_block
    def add_block(sn, sm, sp, U_sub, V_sub, W_sub, i_off, j_off, k_off):
        r = len(U_sub[0])
        for row in U:
            row.extend([0]*r)
        for row in V:
            row.extend([0]*r)
        for row in W:
            row.extend([0]*r)
            
        # U_sub indices: i in [0,sn), j in [0,sm)
        for i in range(sn):
            for j in range(sm):
                u_idx = i*sm + j
                g_u_idx = (i + i_off)*m + (j + j_off)
                for col in range(r):
                    U[g_u_idx][len(U[g_u_idx])-r+col] = U_sub[u_idx][col]
                    
        # V_sub indices: j in [0,sm), k in [0,sp)
        for j in range(sm):
            for k in range(sp):
                v_idx = j*sp + k
                g_v_idx = (j + j_off)*p + (k + k_off)
                for col in range(r):
                    V[g_v_idx][len(V[g_v_idx])-r+col] = V_sub[v_idx][col]
                    
        # W_sub indices: k in [0,sp), i in [0,sn)
        for k in range(sp):
            for i in range(sn):
                w_idx = k*sn + i
                g_w_idx = (k + k_off)*n + (i + i_off)
                for col in range(r):
                    W[g_w_idx][len(W[g_w_idx])-r+col] = W_sub[w_idx][col]

    # Block 1: i=0,1; j=0,1; k=0,1 (2x2x2)
    U1, V1, W1 = get_decomp(2, 2, 2)
    add_block(2, 2, 2, U1, V1, W1, 0, 0, 0)
    
    # Block 2: i=0,1; j=2,3; k=0,1 (2x2x2)
    U2, V2, W2 = get_decomp(2, 2, 2)
    add_block(2, 2, 2, U2, V2, W2, 0, 2, 0)
    
    # Block 3: i=0,1; j=4; k=0,1 (2x1x2)
    U3, V3, W3 = get_decomp_212()
    add_block(2, 1, 2, U3, V3, W3, 0, 4, 0)
    
    # Block 4: i=0,1; j=0,1; k=2,3 (2x2x2)
    U4, V4, W4 = get_decomp(2, 2, 2)
    add_block(2, 2, 2, U4, V4, W4, 0, 0, 2)
    
    # Block 5: i=0,1; j=2,3; k=2,3 (2x2x2)
    U5, V5, W5 = get_decomp(2, 2, 2)
    add_block(2, 2, 2, U5, V5, W5, 0, 2, 2)
    
    # Block 6: i=0,1; j=4; k=2,3 (2x1x2)
    U6, V6, W6 = get_decomp_212()
    add_block(2, 1, 2, U6, V6, W6, 0, 4, 2)
    
    # Block 7: i=2; j=0,1; k=0,1 (1x2x2)
    U7, V7, W7 = get_decomp(1, 2, 2)
    add_block(1, 2, 2, U7, V7, W7, 2, 0, 0)
    
    # Block 8: i=2; j=2,3; k=0,1 (1x2x2)
    U8, V8, W8 = get_decomp(1, 2, 2)
    add_block(1, 2, 2, U8, V8, W8, 2, 2, 0)
    
    # Block 9: i=2; j=4; k=0,1 (1x1x2)
    U9, V9, W9 = get_decomp(1, 1, 2)
    add_block(1, 1, 2, U9, V9, W9, 2, 4, 0)
    
    # Block 10: i=2; j=0,1; k=2,3 (1x2x2)
    U10, V10, W10 = get_decomp(1, 2, 2)
    add_block(1, 2, 2, U10, V10, W10, 2, 0, 2)
    
    # Block 11: i=2; j=2,3; k=2,3 (1x2x2)
    U11, V11, W11 = get_decomp(1, 2, 2)
    add_block(1, 2, 2, U11, V11, W11, 2, 2, 2)
    
    # Block 12: i=2; j=4; k=2,3 (1x1x2)
    U12, V12, W12 = get_decomp(1, 1, 2)
    add_block(1, 1, 2, U12, V12, W12, 2, 4, 2)
    
    return {"U": U, "V": V, "W": W}


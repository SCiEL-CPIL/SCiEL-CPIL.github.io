import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # 1. Construct the trivial rank-48 decomposition as a fallback
    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    
        return {"U": U, "V": V, "W": W}
    
    # 2. Attempt to construct a rank-42 decomposition using Strassen on 2x2 blocks
    # Strategy:
    # A is 2x4. Split into A_L (2x2) and A_R (2x2).
    # B is 4x6. Split into B_L (2x6) and B_R (2x6).
    # C = A_L B_L + A_R B_R.
    # 
    # We can split B_L (2x6) into three 2x2 blocks: B_L1, B_L2, B_L3.
    # We can split B_R (2x6) into three 2x2 blocks: B_R1, B_R2, B_R3.
    #
    # C = A_L [B_L1 B_L2 B_L3] + A_R [B_R1 B_R2 B_R3]
    #   = [A_L B_L1 + A_R B_R1] [A_L B_L2 + A_R B_R2] [A_L B_L3 + A_R B_R3]
    #
    # Let C_j = A_L B_Lj + A_R B_Rj for j=1,2,3.
    # C_j is a 2x2 matrix.
    #
    # The term A_L B_Lj is a 2x2 * 2x2 multiplication.
    # The term A_R B_Rj is a 2x2 * 2x2 multiplication.
    #
    # If we use Strassen for each 2x2 * 2x2, we need 7 multiplications per pair.
    # There are 3 such pairs (j=1,2,3).
    # Total Strassen multiplications = 3 * 7 = 21.
    #
    # However, we can potentially share some Strassen intermediate terms?
    # Strassen for A_L B_Lj:
    #   M1_j = (A_L00 + A_L11) * (B_Lj00 + B_Lj11)
    #   M2_j = (A_L10 + A_L11) * B_Lj00
    #   M3_j = A_L00 * (B_Lj10 - B_Lj11)
    #   M4_j = A_L11 * (B_Lj01 - B_Lj00)
    #   M5_j = (A_L00 + A_L01) * B_Lj11
    #   M6_j = (A_L10 - A_L00) * (B_Lj00 + B_Lj10)
    #   M7_j = (A_L01 - A_L11) * (B_Lj01 + B_Lj11)
    #
    # Strassen for A_R B_Rj:
    #   M1'_j = (A_R00 + A_R11) * (B_Rj00 + B_Rj11)
    #   ... etc.
    #
    # This approach yields 14 multiplications per column block j (7 for A_L, 7 for A_R).
    # Total 14 * 3 = 42 multiplications.
    #
    # We need to construct U, V, W such that:
    # U[i*m+j, r] corresponds to the coefficient of A[i,j] in the r-th product.
    # V[j*p+k, r] corresponds to the coefficient of B[j,k] in the r-th product.
    # W[k*n+i, r] corresponds to the coefficient of C[i,k] in the r-th product.
    #
    # Let's define the 42 products explicitly.
    # For each column block j in [0, 1, 2]:
    #   The 2x2 block of A is A[:, 2*j : 2*j+2].
    #   The 2x2 block of B is B[2*j : 2*j+2, :] ... wait.
    #   B is 4x6.
    #   B_L is B[0:2, 0:6]. B_R is B[2:4, 0:6].
    #   B_Lj is B[0:2, 2*j : 2*j+2].
    #   B_Rj is B[2:4, 2*j : 2*j+2].
    #
    # So for a fixed j (0, 1, 2):
    #   We are computing C_j = A_L B_Lj + A_R B_Rj.
    #   C_j contributes to C[:, 2*j : 2*j+2].
    #
    # Let's build the matrices U, V, W.
    # Rank = 42.
    # We will assign an index r to each of the 42 products.
    # r = j * 14 + s, where s is 0..6 for the Strassen steps of A_L B_Lj,
    # and s is 7..13 for the Strassen steps of A_R B_Rj?
    # Actually, we can just list them.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # For A_L B_Lj:
    # A_L indices in A: (0, 2j), (0, 2j+1), (1, 2j), (1, 2j+1)
    # B_Lj indices in B: (0, 2j), (0, 2j+1), (1, 2j), (1, 2j+1)
    # C_j indices in C: (0, 2j), (0, 2j+1), (1, 2j), (1, 2j+1)
    
    # For A_R B_Rj:
    # A_R indices in A: (0, 2j+2), (0, 2j+3), (1, 2j+2), (1, 2j+3)
    # B_Rj indices in B: (2, 2j), (2, 2j+1), (3, 2j), (3, 2j+1)
    # C_j indices in C: same as above.
    
    def add_strassen_block(A_i0, A_i1, B_j0, B_j1, C_i0, C_i1, C_k0, C_k1, r_base):
        """
        Adds 7 Strassen products for A[2x2] * B[2x2] to U, V, W.
        A is at rows A_i0, A_i1 and cols A_i0, A_i1? No, A is 2x2 block.
        Let's define indices carefully.
        A_block = A[A_i0:A_i0+2, A_i1:A_i1+2] ? No, A is 2x4.
        A_L is A[:, 0:2]. A_R is A[:, 2:4].
        So for A_L, rows are 0,1. Cols are 2j, 2j+1.
        For A_R, rows are 0,1. Cols are 2j+2, 2j+3.
        
        B_Lj is B[0:2, 2j:2j+2]. Rows 0,1. Cols 2j, 2j+1.
        B_Rj is B[2:4, 2j:2j+2]. Rows 2,3. Cols 2j, 2j+1.
        
        C_j is C[:, 2j:2j+2]. Rows 0,1. Cols 2j, 2j+1.
        
        Indices in U: row = i*m + j.
        Indices in V: row = j*p + k.
        Indices in W: row = k*n + i.
        """
        # A block indices
        a00 = A_i0 * m + A_i1
        a01 = A_i0 * m + (A_i1 + 1)
        a10 = (A_i0 + 1) * m + A_i1
        a11 = (A_i0 + 1) * m + (A_i1 + 1)
        
        # B block indices
        b00 = B_j0 * p + C_k0
        b01 = B_j0 * p + (C_k0 + 1)
        b10 = (B_j0 + 1) * p + C_k0
        b11 = (B_j0 + 1) * p + (C_k0 + 1)
        
        # C block indices
        c00 = C_k0 * n + C_i0
        c01 = C_k0 * n + (C_i0 + 1)
        c10 = (C_k0 + 1) * n + C_i0
        c11 = (C_k0 + 1) * n + (C_i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a00, r_base] += 1; U[a11, r_base] += 1
        V[b00, r_base] += 1; V[b11, r_base] += 1
        W[c00, r_base] += 1; W[c11, r_base] += 1
        
        # M2 = (c+d)e
        U[a10, r_base+1] += 1; U[a11, r_base+1] += 1
        V[b00, r_base+1] += 1
        W[c10, r_base+1] += 1
        
        # M3 = a(f-h)
        U[a00, r_base+2] += 1
        V[b01, r_base+2] += 1; V[b11, r_base+2] -= 1
        W[c01, r_base+2] += 1
        
        # M4 = d(g-e)
        U[a11, r_base+3] += 1
        V[b10, r_base+3] += 1; V[b00, r_base+3] -= 1
        W[c00, r_base+3] += 1; W[c10, r_base+3] += 1
        
        # M5 = (a+b)h
        U[a00, r_base+4] += 1; U[a01, r_base+4] += 1
        V[b11, r_base+4] += 1
        W[c00, r_base+4] -= 1; W[c01, r_base+4] += 1
        
        # M6 = (c-a)(e+g)
        U[a10, r_base+5] += 1; U[a00, r_base+5] -= 1
        V[b00, r_base+5] += 1; V[b10, r_base+5] += 1
        W[c11, r_base+5] += 1
        
        # M7 = (b-d)(f+h)
        U[a01, r_base+6] += 1; U[a11, r_base+6] -= 1
        V[b01, r_base+6] += 1; V[b11, r_base+6] += 1
        W[c00, r_base+6] += 1; W[c11, r_base+6] += 1
        
    # Loop over column blocks j = 0, 1, 2
    for j in range(3):
        # A_L part: A[:, 2j:2j+2] * B[0:2, 2j:2j+2]
        # A_i0 = 0, A_i1 = 2j
        # B_j0 = 0, C_k0 = 2j, C_i0 = 0
        r_base_L = j * 14
        add_strassen_block(0, 2*j, 0, 2*j, 0, 2*j, 0, 0, r_base_L)
        
        # A_R part: A[:, 2j+2:2j+4] * B[2:4, 2j:2j+2]
        # A_i0 = 0, A_i1 = 2j+2
        # B_j0 = 2, C_k0 = 2j, C_i0 = 0
        r_base_R = j * 14 + 7
        add_strassen_block(0, 2*j+2, 2, 2*j, 0, 2*j, 0, 0, r_base_R)
        
    # Verification
    # Construct A, B, C
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    # Reconstruct C from U, V, W
    C_rec = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        
        # u_vec is length n*m. Reshape to (n, m)
        u_mat = u_vec.reshape(n, m)
        # v_vec is length m*p. Reshape to (m, p)
        v_mat = v_vec.reshape(m, p)
        # w_vec is length p*n. Reshape to (n, p) using trace convention W[k*n+i, r]
        # w_vec[k*n + i] corresponds to C[i, k]
        # So w_mat[i, k] = w_vec[k*n + i]
        w_mat = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                w_mat[i, k] = w_vec[k*n + i]
        
        # The product term is (A * u_mat) * (v_mat * B) ? No.
        # The tensor decomposition is T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}
        # C_{ik} = sum_{ij, jk} A_{ij} B_{jk} T_{ijk}
        #       = sum_{ij, jk} A_{ij} B_{jk} sum_r U_{ij, r} V_{jk, r} W_{ki, r}
        #       = sum_r (sum_{ij} A_{ij} U_{ij, r}) * (sum_{jk} B_{jk} V_{jk, r}) * W_{ki, r}
        
        a_proj = np.sum(A * u_mat)
        b_proj = np.sum(B * v_mat)
        C_rec += a_proj * b_proj * w_mat
        
    if not np.allclose(C, C_rec, atol=1e-9):
        return trivial_decomposition()
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

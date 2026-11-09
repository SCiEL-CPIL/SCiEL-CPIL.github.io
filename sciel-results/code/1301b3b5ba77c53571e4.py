"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We aim for rank 88
    # Decomposition:
    # 1. 2x4x8 part (rank 56)
    # 2. 1x4x8 part (rank 32)
    
    # --- Part 1: 2x4x8 ---
    # Decomposed as (2x2x4) x (1x2x2)
    # (2x2x4) decomposed as (2x2x2) x (1x1x2)
    
    # Strassen 2x2x2 (Rank 7)
    # U_s (2*2, 7), V_s (2*2, 7), W_s (2*2, 7)
    # Indices: A: i,j. B: j,k. C: i,k.
    # W index: k*n + i.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1 # A11, A22
    V_s[0, 0] = 1; V_s[3, 0] = 1 # B11, B22
    W_s[0, 0] = 1; W_s[3, 0] = 1 # C11, C22 (Wait, C11 gets M1, C22 gets M1)
    # C11 index: 0*2+0=0. C22 index: 1*2+1=3.
    
    # M2 = (A21+A22)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1 # A21, A22
    V_s[0, 1] = 1 # B11
    W_s[2, 1] = 1 # C21 index: 1*2+0=2
    
    # M3 = A11(B12-B22)
    U_s[0, 2] = 1 # A11
    V_s[1, 2] = 1; V_s[3, 2] = -1 # B12, B22
    W_s[1, 2] = 1 # C12 index: 0*2+1=1
    
    # M4 = A22(B21-B11)
    U_s[3, 3] = 1 # A22
    V_s[2, 3] = 1; V_s[0, 3] = -1 # B21, B11
    W_s[0, 3] = 1; W_s[2, 3] = 1 # C11, C21
    
    # M5 = (A11+A12)B22
    U_s[0, 4] = 1; U_s[1, 4] = 1 # A11, A12
    V_s[3, 4] = 1 # B22
    W_s[0, 4] = -1; W_s[1, 4] = 1 # C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U_s[2, 5] = 1; U_s[0, 5] = -1 # A21, A11
    V_s[0, 5] = 1; V_s[1, 5] = 1 # B11, B12
    W_s[3, 5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U_s[1, 6] = 1; U_s[3, 6] = -1 # A12, A22
    V_s[2, 6] = 1; V_s[3, 6] = 1 # B21, B22
    W_s[0, 6] = 1; W_s[3, 6] = 1 # C11, C22
    
    # 1x1x2 (Rank 2)
    # A: 1x1. B: 1x2. C: 1x2.
    # U_1 (1, 2), V_1 (2, 2), W_1 (2, 2)
    U_1 = np.ones((1, 2))
    V_1 = np.eye(2) # B1, B2
    W_1 = np.eye(2) # C1, C2
    
    # Kronecker Product for 2x2x4
    # Rank 14
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((4, 14))
    W_224 = np.zeros((4, 14))
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            U_224[:, r] = U_s[:, r1] * U_1[:, r2].T
            V_224[:, r] = V_s[:, r1] * V_1[:, r2].T
            W_224[:, r] = W_s[:, r1] * W_1[:, r2].T
            
    # 1x2x2 (Rank 4)
    # A: 1x2. B: 2x2. C: 1x2.
    # U_122 (2, 4), V_122 (4, 4), W_122 (2, 4)
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((2, 4))
    
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # r=0: A1 B11
    U_122[0, 0] = 1
    V_122[0, 0] = 1 # B11
    W_122[0, 0] = 1 # C1
    
    # r=1: A2 B21
    U_122[1, 1] = 1
    V_122[2, 1] = 1 # B21
    W_122[0, 1] = 1 # C1
    
    # r=2: A1 B12
    U_122[0, 2] = 1
    V_122[1, 2] = 1 # B12
    W_122[1, 2] = 1 # C2
    
    # r=3: A2 B22
    U_122[1, 3] = 1
    V_122[3, 3] = 1 # B22
    W_122[1, 3] = 1 # C2
    
    # Kronecker Product for 2x4x8
    # Rank 56
    U_248 = np.zeros((4, 56)) # 2*2
    V_248 = np.zeros((8, 56)) # 2*4
    W_248 = np.zeros((16, 56)) # 4*2? No, p*n = 8*2 = 16.
    
    for r1 in range(14):
        for r2 in range(4):
            r = r1 * 4 + r2
            U_248[:, r] = U_224[:, r1] * U_122[:, r2].T
            V_248[:, r] = V_224[:, r1] * V_122[:, r2].T
            W_248[:, r] = W_224[:, r1] * W_122[:, r2].T
            
    # --- Part 2: 1x4x8 ---
    # Rank 32
    # Schoolbook
    U_148 = np.zeros((4, 32)) # 1*4
    V_148 = np.zeros((32, 32)) # 4*8
    W_148 = np.zeros((8, 32)) # 8*1
    
    for j in range(4):
        for k in range(8):
            r = j * 8 + k
            U_148[j, r] = 1
            V_148[j * 8 + k, r] = 1
            W_148[k, r] = 1
            
    # --- Combine ---
    # Total Rank 88
    U_total = np.zeros((12, 88)) # 3*4
    V_total = np.zeros((32, 88)) # 4*8
    W_total = np.zeros((24, 88)) # 8*3
    
    # Part 1: 2x4x8
    # Maps to i in 0,1
    # U_248 is 4x56. Global U is 12x88.
    # Global U index: i*m + j = i*4 + j.
    # U_248 index: i'*2 + j'? No, U_248 is for A (2x4).
    # A is 2x4. Flattened: i*4+j.
    # So U_248 row i*4+j maps to global U row i*4+j.
    
    for i in range(2):
        for j in range(4):
            local_u_idx = i * 4 + j
            U_total[local_u_idx, 0:56] = U_248[local_u_idx, :]
            
    # V_248 is 8x56. Global V is 32x88.
    # V index: j*p + k = j*8 + k.
    # V_248 index: j'*4 + k'?
    # In the Kronecker construction, V_248 was built from V_224 (4x14) and V_122 (4x4).
    # V_224 index: j1*2 + k1? No, V_224 is for B (2x4).
    # B is 2x4. Flattened: j*4+k.
    # V_248 is 8x56.
    # The mapping j = 2*j1 + j2, k = 4*k1 + 2*k2 + k3?
    # Wait, V_248 was constructed as V_224 (4x14) x V_122 (4x4).
    # V_224 row: j1*2 + k1? No, V_224 is for B (2x4).
    # B (2x4) flattened: j*4+k.
    # V_248 row: (j1*2+k1) * 4 + (j2*2+k2)?
    # This is getting confusing.
    
    # Let's just use the direct indices.
    # V_248 row index corresponds to B (4x8) flattened?
    # No, V_248 is for the 2x4x8 multiplication.
    # B is 4x8. Flattened: j*8+k.
    # V_248 has 8 rows? No, V_248 was 8x56.
    # B is 4x8. Size 32.
    # Wait, V_248 was constructed from V_224 (4x14) and V_122 (4x4).
    # V_224 is for B (2x4). Size 8.
    # V_122 is for B (2x2). Size 4.
    # Kronecker product of V (4x14) and V (4x4) gives (16x56)?
    # No, V_224 is 4x14. V_122 is 4x4.
    # The Kronecker product of matrices A (m x n) and B (p x q) is (mp x nq).
    # So V_248 should be (4*4 x 14*4) = 16x56.
    # But I defined V_248 as 8x56. That was a mistake.
    # B for 2x4x8 is 4x8. Size 32.
    # So V_248 should have 32 rows.
    
    # Let's re-verify the Kronecker dimensions.
    # 2x4x8 = (2x2x4) x (1x2x2).
    # B (4x8) = B1 (2x4) x B2 (2x2).
    # B1 size 8. B2 size 4.
    # B size 32.
    # V_224 (for B1) has 8 rows.
    # V_122 (for B2) has 4 rows.
    # V_248 should have 8*4 = 32 rows.
    
    # I will regenerate V_248 correctly.
    V_248 = np.zeros((32, 56))
    U_248 = np.zeros((8, 56)) # A is 2x4. Size 8.
    W_248 = np.zeros((16, 56)) # C is 2x8. Size 16.
    
    for r1 in range(14):
        for r2 in range(4):
            r = r1 * 4 + r2
            U_248[:, r] = U_224[:, r1] * U_122[:, r2].T
            V_248[:, r] = V_224[:, r1] * V_122[:, r2].T
            W_248[:, r] = W_224[:, r1] * W_122[:, r2].T
            
    # Now fill Total
    # U_248 (8x56) -> U_total (12x88)
    # A is 2x4. Flattened i*4+j.
    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            U_total[idx, 0:56] = U_248[idx, :]
            
    # V_248 (32x56) -> V_total (32x88)
    # B is 4x8. Flattened j*8+k.
    # V_248 index: (j1*2+k1)*4 + (j2*2+k2)?
    # B1 (2x4) index: j1*4+k1.
    # B2 (2x2) index: j2*2+k2.
    # B (4x8) index: j*8+k.
    # j = 2*j1 + j2.
    # k = 4*k1 + 2*k2 + k3?
    # This mapping is complex.
    # I will assume the Kronecker product ordering matches the flattening.
    # If B = B1 \otimes B2, then B_{j, k} = B1_{j1, k1} B2_{j2, k2}.
    # j = j1 * 2 + j2.
    # k = k1 * 2 + k2.
    # B1 is 2x4. B2 is 2x2.
    # B is 4x8.
    # j1 in 0,1. j2 in 0,1. j in 0..3.
    # k1 in 0..3. k2 in 0,1. k in 0..7.
    # Index in B: j*8+k.
    # Index in V_248: (j1*4+k1)*4 + (j2*2+k2)?
    # V_224 index: j1*4+k1.
    # V_122 index: j2*2+k2.
    # V_248 index: V_224_idx * 4 + V_122_idx.
    
    for v248_idx in range(32):
        # v248_idx = v224_idx * 4 + v122_idx
        v224_idx = v248_idx // 4
        v122_idx = v248_idx % 4
        
        # v224_idx = j1*4 + k1
        j1 = v224_idx // 4
        k1 = v224_idx % 4
        
        # v122_idx = j2*2 + k2
        j2 = v122_idx // 2
        k2 = v122_idx % 2
        
        # Global j, k
        j = 2 * j1 + j2
        k = 2 * k1 + k2
        
        global_v_idx = j * 8 + k
        V_total[global_v_idx, 0:56] = V_248[v248_idx, :]
        
    # W_248 (16x56) -> W_total (24x88)
    # C is 2x8. Flattened k*2+i?
    # W index: k*n + i.
    # C1 (2x4) index: k1*2+i1.
    # C2 (1x2) index: k2*1+i2?
    # C (2x8) index: k*2+i.
    # k = 2*k1 + k2.
    # i = i1.
    
    for w248_idx in range(16):
        # w248_idx = w224_idx * 4 + w122_idx
        w224_idx = w248_idx // 4
        w122_idx = w248_idx % 4
        
        # w224_idx = k1*2 + i1
        k1 = w224_idx // 2
        i1 = w224_idx % 2
        
        # w122_idx = k2*1 + i2?
        # 1x2x2 W index: k*1+i.
        # k2 in 0,1. i2=0.
        k2 = w122_idx // 1
        i2 = w122_idx % 1
        
        # Global k, i
        k = 2 * k1 + k2
        i = i1
        
        global_w_idx = k * 3 + i
        W_total[global_w_idx, 0:56] = W_248[w248_idx, :]
        
    # Part 2: 1x4x8
    # U_148 (4x32) -> U_total (12x88)
    # A is 1x4. Flattened i*4+j. i=2.
    # U_148 index: j.
    for j in range(4):
        global_u_idx = 2 * 4 + j
        U_total[global_u_idx, 56:88] = U_148[j, :]
        
    # V_148 (32x32) -> V_total (32x88)
    # B is 4x8. Flattened j*8+k.
    # V_148 index: j*8+k.
    for j in range(4):
        for k in range(8):
            global_v_idx = j * 8 + k
            V_total[global_v_idx, 56:88] = V_148[global_v_idx, :]
            
    # W_148 (8x32) -> W_total (24x88)
    # C is 1x8. Flattened k*1+i. i=2.
    # W_148 index: k.
    # Global W index: k*3+i.
    for k in range(8):
        global_w_idx = k * 3 + 2
        W_total[global_w_idx, 56:88] = W_148[k, :]
        
    # Return
    U_list = [[float(U_total[i, j]) for j in range(88)] for i in range(12)]
    V_list = [[float(V_total[i, j]) for j in range(88)] for i in range(32)]
    W_list = [[float(W_total[i, j]) for j in range(88)] for i in range(24)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

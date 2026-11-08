"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen 2x2x2 decomposition (Rank 7)
    # Indices: i (row A), j (col A / row B), k (col B)
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: (a00+a11)(b00+b11) -> c00 + c11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: (a10+a11)b00 -> c10 - c11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 3: a00(b01-b11) -> c01 + c11
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 4: a11(b10-b01) -> c10 + c00
    U2[3][3] = 1
    V2[2][3] = 1; V2[1][3] = -1
    W2[2][3] = 1; W2[0][3] = 1
    
    # Term 5: (a00+a01)b11 -> -c00 + c01
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6: (a10-a00)(b00+b01) -> -c10 - c01
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[2][5] = -1; W2[1][5] = -1
    
    # Term 7: (a01-a11)(b10+b11) -> -c00 + c11
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1; W2[3][6] = 1
    
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # 2. Construct 4x4x4 decomposition via Tensor Product (Rank 49)
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # where i = 2*i1 + i2, j = 2*j1 + j2
    
    rank_44 = 49
    U4 = np.zeros((16, rank_44))
    V4 = np.zeros((16, rank_44))
    W4 = np.zeros((16, rank_44))
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            idx_u = i * 4 + j
            
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    val = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                    U4[idx_u, r] = val
                    
                    val_v = V2[j1 * 2 + (i1*0 + 0), r1] # Wait, V indices depend on j and k
                    # V index for 2x2 is j*2+k.
                    # In 4x4, V index is j*4+k.
                    # j = 2*j1 + j2, k = 2*k1 + k2.
                    # V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                    # We need to fill V4 and W4 as well.
                    pass
    
    # Let's fill V4 and W4 in the same loop structure or separate.
    # Actually, let's just loop over all indices.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            idx_u = i * 4 + j
            idx_v = j * 4 + i # No, V index is j (row of B) and k (col of B).
            # Wait, V shape is (m*p, rank). m=4, p=4.
            # Index is j*4 + k.
            # So we need to loop over k too for V.
            
            # Let's restructure.
            # U4 depends on i, j.
            # V4 depends on j, k.
            # W4 depends on k, i.
            
            # Fill U4
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U4[idx_u, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            idx_v = j * 4 + k
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V4[idx_v, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            idx_w = k * 4 + i
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W4[idx_w, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    # 3. Construct 4x4x1 decomposition for the 5th column (Rank 16)
    # This handles A(4x4) * B_col(4x1)
    # U shape (16, 16), V shape (20, 16), W shape (20, 16)
    # We will append these to the 49-rank matrices.
    # Total Rank = 49 + 16 = 65.
    
    rank_total = 65
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total)) # m*p = 4*5 = 20
    W = np.zeros((20, rank_total)) # p*n = 5*4 = 20
    
    # Copy 4x4x4 part (first 49 columns)
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    # V and W need to be padded to 20 rows.
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    # Add 4x4x1 part (next 16 columns)
    # For each i, j in 0..3:
    # U[i*4+j, 49 + i*4+j] = 1
    # V[j*5 + 4, 49 + i*4+j] = 1  (k=4 is the 5th column)
    # W[4*4 + i, 49 + i*4+j] = 1  (k=4, i is col)
    
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * 4 + j, r] = 1
            V[j * 5 + 4, r] = 1
            W[4 * 4 + i, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

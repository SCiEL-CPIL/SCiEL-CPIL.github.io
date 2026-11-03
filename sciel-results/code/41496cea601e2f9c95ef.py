"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150 - 5  # 5s safety margin
    
    n, m, p = DIMENSIONS
    if not (n == 6 and m == 6 and p == 6):
        # Fallback to schoolbook if dimensions don't match 6x6x6
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U, "V": V, "W": W}
    
    # Decomposition: 6x6x6 via (2x2x2 Strassen) x (3x3x3 Schoolbook)
    # Rank = 7 * 27 = 189
    
    # 1. Define 2x2 Strassen Decomposition (Rank 7)
    # Indices: U(i1, j1), V(j1, k1), W(k1, i1)
    # Standard Strassen formula for C = A*B
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # U (A) terms:
    # M1: A11, A22 (coeff 1)
    # M2: A21, A22 (coeff 1)
    # M3: A11 (coeff 1)
    # M4: A22 (coeff 1)
    # M5: A11, A12 (coeff 1)
    # M6: A21, A11 (coeff 1, -1)
    # M7: A12, A22 (coeff 1, -1)
    
    # U2x2: shape (4, 7) where row = i1*m + j1 (m=2)
    U2x2 = np.zeros((4, 7), dtype=float)
    # M1
    U2x2[0, 0] = 1  # A11 (i=0,j=0)
    U2x2[3, 0] = 1  # A22 (i=1,j=1)
    # M2
    U2x2[2, 1] = 1  # A21 (i=1,j=0)
    U2x2[3, 1] = 1  # A22 (i=1,j=1)
    # M3
    U2x2[0, 2] = 1  # A11
    # M4
    U2x2[3, 3] = 1  # A22
    # M5
    U2x2[0, 4] = 1  # A11
    U2x2[1, 4] = 1  # A12 (i=0,j=1)
    # M6
    U2x2[2, 5] = 1  # A21
    U2x2[0, 5] = -1 # A11
    # M7
    U2x2[1, 6] = 1  # A12
    U2x2[3, 6] = -1 # A22
    
    # V (B) terms:
    # M1: B11, B22 (coeff 1)
    # M2: B11 (coeff 1)
    # M3: B21, B22 (coeff 1, -1)
    # M4: B12, B11 (coeff 1, -1)
    # M5: B22 (coeff 1)
    # M6: B11, B12 (coeff 1, 1)
    # M7: B21, B22 (coeff 1, 1)
    
    V2x2 = np.zeros((4, 7), dtype=float) # row = j1*p + k1 (p=2)
    # M1
    V2x2[0, 0] = 1  # B11 (j=0,k=0)
    V2x2[3, 0] = 1  # B22 (j=1,k=1)
    # M2
    V2x2[0, 1] = 1  # B11
    # M3
    V2x2[2, 2] = 1  # B21 (j=1,k=0)
    V2x2[3, 2] = -1 # B22
    # M4
    V2x2[1, 3] = 1  # B12 (j=0,k=1)
    V2x2[0, 3] = -1 # B11
    # M5
    V2x2[3, 4] = 1  # B22
    # M6
    V2x2[0, 5] = 1  # B11
    V2x2[1, 5] = 1  # B12
    # M7
    V2x2[2, 6] = 1  # B21
    V2x2[3, 6] = 1  # B22
    
    # W (C) terms:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    W2x2 = np.zeros((4, 7), dtype=float) # row = k1*n + i1 (n=2)
    # C11 (i=0,k=0) -> idx 0
    W2x2[0, 0] = 1  # M1
    W2x2[0, 3] = -1 # M4
    W2x2[0, 4] = 1  # M5
    W2x2[0, 6] = -1 # M7
    # C12 (i=0,k=1) -> idx 1 (k=1, i=0 -> 1*2+0=2? No. W[k*n+i]. n=2. k=1,i=0 -> 1*2+0=2. Wait.
    # Let's check indices.
    # C is 2x2. C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22.
    # W index is k*n + i.
    # C11: k=0, i=0 -> 0
    # C12: k=1, i=0 -> 2
    # C21: k=0, i=1 -> 1
    # C22: k=1, i=1 -> 3
    
    W2x2[2, 2] = 1  # C12 M3
    W2x2[2, 4] = 1  # C12 M5
    
    W2x2[1, 1] = 1  # C21 M2
    W2x2[1, 3] = 1  # C21 M4
    
    W2x2[3, 0] = 1  # C22 M1
    W2x2[3, 2] = -1 # C22 M3
    W2x2[3, 5] = 1  # C22 M6
    W2x2[3, 6] = -1 # C22 M7
    
    # 2. Define 3x3 Schoolbook Decomposition (Rank 27)
    # Rank = 27.
    # Terms indexed by (i, j, k) for i,j,k in 0..2.
    # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    
    U3x3 = np.zeros((9, 27), dtype=float)
    V3x3 = np.zeros((9, 27), dtype=float)
    W3x3 = np.zeros((9, 27), dtype=float)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i * 9 + j * 3 + k
                U3x3[i * 3 + j, idx] = 1
                V3x3[j * 3 + k, idx] = 1
                W3x3[k * 3 + i, idx] = 1
    
    # 3. Combine via Tensor Product
    # Total Rank R = 7 * 27 = 189
    R = 189
    
    U6x6 = np.zeros((36, R), dtype=float)
    V6x6 = np.zeros((36, R), dtype=float)
    W6x6 = np.zeros((36, R), dtype=float)
    
    for t2 in range(7):
        for t3 in range(27):
            t = t2 * 27 + t3
            
            # U6x6 row index: (i2 * 3 + i3) * 6 + (j2 * 3 + j3)
            # U2x2 row: i2 * 2 + j2
            # U3x3 row: i3 * 3 + j3
            
            # We need to iterate over all non-zero entries in U2x2[:, t2] and U3x3[:, t3]
            # Since they are sparse, we can just loop over all rows (36) and compute.
            # Or more efficiently:
            # U6x6[i6, t] = sum over valid splits of U2x2[i2, t2] * U3x3[i3, t3]
            # But U2x2 and U3x3 are single values per row for a fixed t.
            # Row i6 in 6x6 corresponds to block row i2 and inner row i3.
            # i2 = i6 // 6, i3 = i6 % 6.
            # However, the column index j is also split.
            # In U matrix, row is (i, j).
            # i6 = i2 * 3 + i3
            # j6 = j2 * 3 + j3
            # i2 = i6 // 3, i3 = i6 % 3
            # j2 = j6 // 3, j3 = j6 % 3
            
            # So for a fixed t2, t3:
            # U6x6[i6, t] = U2x2[i2, t2] * U3x3[i3, t3]
            # where i2 = i6 // 3, i3 = i6 % 3.
            # Wait, U2x2 row index is i2 * 2 + j2.
            # So we need to know j2 as well.
            
            # Let's just loop over all i6, j6 to construct U6x6? No, U6x6 is (36, 189).
            # 36 rows.
            for i6 in range(36):
                i2 = i6 // 3
                i3 = i6 % 3
                # We need to sum over j2, j3 such that j6 = j2*3+j3?
                # No, the row of U is a single index (i*m+j).
                # So for a specific row r6 in U6x6 (0..35), it corresponds to a specific i6 and j6.
                # r6 = i6 * 6 + j6.
                # i6 = r6 // 6, j6 = r6 % 6.
                # i2 = i6 // 3, i3 = i6 % 3.
                # j2 = j6 // 3, j3 = j6 % 3.
                
                # Let's rewrite the loop over r6
                pass
                
            # Let's do it properly by iterating r6
            for r6 in range(36):
                i6 = r6 // 6
                j6 = r6 % 6
                i2 = i6 // 3
                i3 = i6 % 3
                j2 = j6 // 3
                j3 = j6 % 3
                
                u2 = U2x2[i2 * 2 + j2, t2]
                u3 = U3x3[i3 * 3 + j3, t3]
                U6x6[r6, t] = u2 * u3
                
            # Similarly for V6x6
            for r6 in range(36):
                j6 = r6 // 6
                k6 = r6 % 6
                j2 = j6 // 3
                j3 = j6 % 3
                k2 = k6 // 3
                k3 = k6 % 3
                
                v2 = V2x2[j2 * 2 + k2, t2]
                v3 = V3x3[j3 * 3 + k3, t3]
                V6x6[r6, t] = v2 * v3
                
            # Similarly for W6x6
            for r6 in range(36):
                k6 = r6 // 6
                i6 = r6 % 6
                k2 = k6 // 3
                k3 = k6 % 3
                i2 = i6 // 3
                i3 = i6 % 3
                
                w2 = W2x2[k2 * 2 + i2, t2]
                w3 = W3x3[k3 * 3 + i3, t3]
                W6x6[r6, t] = w2 * w3
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
         # Return best effort? Or just the schoolbook if we haven't finished?
         # Actually, the loop is fast enough. 189 * 36 * 3 ~ 20k ops.
         pass
        
    return {"U": U6x6, "V": V6x6, "W": W6x6}
# EVOLVE-BLOCK-END

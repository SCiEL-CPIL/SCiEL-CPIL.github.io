import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    
    # 1. Strassen for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # A indices: 00:0, 01:1, 10:2, 11:3
    # B indices: 00:0, 01:1, 10:2, 11:3
    # C indices: 00:0, 01:1, 10:2, 11:3
    
    # M1
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[1, 0] = 1; W_S[2, 0] = 1; W_S[3, 0] = 1
    W_S[0, 0] -= 1 # C00 = M1 - M4 - M6 + M7 ?
    # Standard Strassen reconstruction:
    # C00 = M1 - M4 - M6 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # Let's fill W_S based on this.
    # W_S[k*2+i, c]
    
    # M1
    W_S[0, 0] = 1  # C00
    W_S[1, 0] = 1  # C01
    W_S[2, 0] = 0  # C10
    W_S[3, 0] = 1  # C11
    
    # M2
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[0, 1] = 0
    W_S[1, 1] = 0
    W_S[2, 1] = 1  # C10
    W_S[3, 1] = -1 # C11
    
    # M3
    U_S[0, 2] = 1
    V_S[2, 2] = 1; V_S[3, 2] = -1
    W_S[0, 2] = 0
    W_S[1, 2] = 0
    W_S[2, 2] = 0
    W_S[3, 2] = 1  # C11
    
    # M4
    U_S[3, 3] = 1
    V_S[1, 3] = 1; V_S[0, 3] = -1
    W_S[0, 3] = -1 # C00
    W_S[1, 3] = 0
    W_S[2, 3] = 1  # C10
    W_S[3, 3] = 0
    
    # M5
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = 0
    W_S[1, 4] = 1  # C01
    W_S[2, 4] = 0
    W_S[3, 4] = 1  # C11
    
    # M6
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[0, 5] = -1 # C00
    W_S[1, 5] = 0
    W_S[2, 5] = 0
    W_S[3, 5] = 0
    
    # M7
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[0, 6] = 1  # C00
    W_S[1, 6] = 1  # C01
    W_S[2, 6] = 0
    W_S[3, 6] = 0
    
    # 2. Decomposition for 2x2x3 (Rank 11)
    # U_223: 4x11, V_223: 6x11, W_223: 6x11
    # A: 2x2. B: 2x3. C: 2x3.
    # Indices:
    # U: i*2+j (0..3)
    # V: j*3+k (0..5)
    # W: k*2+i (0..5)
    
    # First 7 terms: Strassen for B cols 0,1.
    # B indices for cols 0,1:
    # B00:0, B01:1, B10:3, B11:4. (Since B is 2x3, j=0 has k=0,1,2; j=1 has k=3,4,5)
    # Strassen uses B00, B01, B10, B11.
    # Map B_222 indices to B_223 indices:
    # 00->0, 01->1, 10->3, 11->4.
    
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))
    W_223 = np.zeros((6, 11))
    
    # Copy U_S to U_223
    U_223[:, :7] = U_S
    
    # Map V_S to V_223
    # V_S index j*2+k.
    # j=0, k=0 -> V_223[0*3+0] = 0
    # j=0, k=1 -> V_223[0*3+1] = 1
    # j=1, k=0 -> V_223[1*3+0] = 3
    # j=1, k=1 -> V_223[1*3+1] = 4
    
    for c in range(7):
        for j in range(2):
            for k in range(2):
                if V_S[j*2+k, c] != 0:
                    V_223[j*3+k, c] = V_S[j*2+k, c]
                    
    # Map W_S to W_223
    # W_S index k*2+i.
    # k=0, i=0 -> W_223[0*2+0] = 0
    # k=0, i=1 -> W_223[0*2+1] = 1
    # k=1, i=0 -> W_223[1*2+0] = 2
    # k=1, i=1 -> W_223[1*2+1] = 3
    # Cols 0,1 of C correspond to k=0,1.
    # So W_223[0..3, :7] = W_S[:, :7]
    
    W_223[0:4, :7] = W_S
    
    # Last 4 terms: Schoolbook for B col 2.
    # B col 2 indices: j=0, k=2 -> 2; j=1, k=2 -> 5.
    # C col 2 indices: k=2, i=0 -> 4; k=2, i=1 -> 5.
    
    # C20 = A00 B02 + A01 B12
    # C21 = A10 B02 + A11 B12
    
    # Term 7: A00 B02
    U_223[0, 7] = 1
    V_223[2, 7] = 1 # B02
    W_223[4, 7] = 1 # C20
    
    # Term 8: A01 B12
    U_223[1, 8] = 1
    V_223[5, 8] = 1 # B12
    W_223[4, 8] = 1 # C20
    
    # Term 9: A10 B02
    U_223[2, 9] = 1
    V_223[2, 9] = 1 # B02
    W_223[5, 9] = 1 # C21
    
    # Term 10: A11 B12
    U_223[3, 10] = 1
    V_223[5, 10] = 1 # B12
    W_223[5, 10] = 1 # C21
    
    # 3. Tensor Product
    # Rank = 7 * 11 = 77
    rank = 77
    U_final = np.zeros((16, rank))
    V_final = np.zeros((24, rank))
    W_final = np.zeros((24, rank))
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            # U: (i1*2+i2) * 4 + (j1*2+j2) ?
            # A is 4x4. A[i,j].
            # i = i1*2 + i2. j = j1*2 + j2.
            # U_S index: i1*2 + j1.
            # U_223 index: i2*2 + j2.
            # So U_final index: (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2.
            # This is equivalent to:
            # idx = (U_S_idx * 4) + U_223_idx?
            # U_S_idx = i1*2 + j1.
            # U_223_idx = i2*2 + j2.
            # idx = (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2.
            # Wait. A[i,j] with i=i1*2+i2, j=j1*2+j2.
            # Standard row-major: i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2.
            # My formula: i1*8 + j1*4 + i2*2 + j2.
            # Mismatch.
            
            # Let's use explicit loops or correct mapping.
            # U_S[i1, j1] * U_223[i2, j2] -> U_final[i1*2+i2, j1*2+j2]
            
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U_S[i1*2+j1, c1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val2 = U_223[i2*2+j2, c2]
                            if val2 == 0: continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U_final[i*4+j, c] = val1 * val2
                            
            # V: B[j,k]. j=j1*2+j2, k=k1*3+k2?
            # B is 4x6.
            # j = j1*2 + j2 (j1,j2 in 0..1)
            # k = k1*3 + k2 (k1 in 0..1, k2 in 0..2)
            # V_S[j1, k1] * V_223[j2, k2] -> V_final[j, k]
            # V_S index: j1*2 + k1.
            # V_223 index: j2*3 + k2.
            
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V_S[j1*2+k1, c1]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(3):
                            val2 = V_223[j2*3+k2, c2]
                            if val2 == 0: continue
                            j = j1*2 + j2
                            k = k1*3 + k2
                            V_final[j*6+k, c] = val1 * val2
                            
            # W: C[k,i]. k=k1*3+k2, i=i1*2+i2.
            # W_S[k1, i1] * W_223[k2, i2] -> W_final[k, i]
            # W_S index: k1*2 + i1.
            # W_223 index: k2*2 + i2.
            
            for k1 in range(2):
                for i1 in range(2):
                    val1 = W_S[k1*2+i1, c1]
                    if val1 == 0: continue
                    for k2 in range(3):
                        for i2 in range(2):
                            val2 = W_223[k2*2+i2, c2]
                            if val2 == 0: continue
                            k = k1*3 + k2
                            i = i1*2 + i2
                            W_final[k*4+i, c] = val1 * val2
                            
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))

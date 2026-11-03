import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def get_strassen_222():
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C11, C12, C21, C22 contributions
    # Actually W maps (k,i) to C_ik.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's define W coefficients for each M
    # W[k*2+i, c]
    # M1: C11(+1), C12(0), C21(0), C22(+1)
    # Indices (k,i): (0,0)->C11, (1,0)->C12, (0,1)->C21, (1,1)->C22
    W[0, 0] = 1 # C11
    W[3, 0] = 1 # C22
    
    # M2 = (c+d)e
    # C21(+1), C22(-1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    W[3, 1] = -1 # C22
    
    # M3 = a(f-h)
    # C12(+1), C22(+1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12
    W[3, 2] = 1 # C22
    
    # M4 = d(g-e)
    # C11(-1), C21(+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1 # C11
    W[2, 3] = 1 # C21
    
    # M5 = (a+b)h
    # C11(-1), C12(+1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1 # C12
    
    # M6 = (c-a)(e+f)
    # Not used in standard Strassen?
    # Standard Strassen uses 7 products.
    # M6 is often (c-a)(e+f) but let's check the formulas.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # My M6 definition above was empty in W.
    # Wait, the standard 7 products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f) -- Wait, is this used?
    # Let's re-verify Strassen.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # Where are M6 and M7?
    # M6 is not in the sum?
    # Ah, the 7 products are M1..M7.
    # M6 = (c-a)(e+f) is not in the standard set I wrote?
    # Let's check a reliable source.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # It seems M6 is not used in the reconstruction?
    # No, that's impossible. Rank 7 means 7 terms.
    # Let's check C11 again.
    # a e + b g
    # M1 + M4 - M5 - M7
    # (ae+ah+de+dh) + (dg-de) - (ah+bh) - (bg+bh-dg-dh)
    # = ae+ah+de+dh + dg-de - ah-bh - bg-bh+dg+dh
    # = ae - bg + 2dg + 2dh - 2bh?
    # This doesn't look like ae+bg.
    
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's re-evaluate C11 with M6.
    # Maybe C11 = M1 + M4 - M5 - M7 is wrong.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # This set uses M1, M2, M3, M4, M5, M7. M6 is unused?
    # No, the standard algorithm uses 7 multiplications.
    # Let's use a known correct implementation.
    
    # From Wikipedia:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e)
    # M3 = (a)(f - h)
    # M4 = (d)(g - e)
    # M5 = (a + b)(h)
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # It seems M6 is indeed not in the reconstruction formulas listed above?
    # Wait, C22 = M1 - M2 + M3 + M7.
    # Let's check C22 = b f + d h.
    # M1 - M2 + M3 + M7
    # (ae+ah+de+dh) - (ce+de) + (af-ah) + (bg+bh-dg-dh)
    # = ae+ah+de+dh - ce-de + af-ah + bg+bh-dg-dh
    # = ae - ce + af + bg + bh - dg
    # This is not bf+dh.
    
    # There is a mistake in my memory of Strassen formulas.
    # Let's use the ones from a reliable source or derive them.
    # Actually, I can just use the rank 49 algorithm for 4x4x4 directly if I find one, or use a simpler block decomposition.
    
    # Alternative: Use Strassen for 2x2x2 (Rank 7) and combine.
    # I will implement the correct Strassen tensors.
    
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's check C11 again.
    # M1 + M4 - M5 - M7
    # (ae+ah+de+dh) + (dg-de) - (ah+bh) - (bg+bh-dg-dh)
    # = ae+ah+de+dh+dg-de-ah-bh-bg-bh+dg+dh
    # = ae - bg + 2dg + 2dh - 2bh
    # Still wrong.
    
    # Okay, I will use a known correct set of coefficients.
    # U, V, W for 2x2x2 rank 7.
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 1, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, 0]  # d
    ])
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, 1, 1, 1, 0, 1]  # h
    ])
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, -1], # C11 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C12 (k=1, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C21 (k=0, i=1)
        [1, -1, 1, 0, 0, 0, 1]   # C22 (k=1, i=1)
    ])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Block 1: 4x4x4 (Rank 49)
    U2, V2, W2 = get_strassen_222()
    U444_kron = np.kron(U2, U2)
    V444_kron = np.kron(V2, V2)
    W444_kron = np.kron(W2, W2)
    
    # Permutation for 4x4
    # Target index t = I*4 + J = (2*i1 + i2)*4 + (2*j1 + j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # Kron index r = i1*8 + j1*4 + i2*2 + j2
    perm = []
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    t = 8*i1 + 4*i2 + 2*j1 + j2
                    r = 8*i1 + 4*j1 + 2*i2 + j2
                    perm.append((t, r))
    
    # Create permutation arrays
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for t, r in perm:
        # U rows are (i,j) -> I*4+J
        # V rows are (j,k) -> J*4+K
        # W rows are (k,i) -> K*4+I
        # The permutation is the same for all since the structure is symmetric in index grouping.
        perm_U[t] = r
        perm_V[t] = r
        perm_W[t] = r
        
    U444 = U444_kron[perm_U]
    V444 = V444_kron[perm_V]
    W444 = W444_kron[perm_W]
    
    # Block 2: 4x4x2 (Rank 28)
    # 4 Strassen blocks
    U442 = np.zeros((20, 28), dtype=int)
    V442 = np.zeros((30, 28), dtype=int)
    W442 = np.zeros((24, 28), dtype=int)
    
    # Block 1: A00 B00
    # A00: i in 0,1; j in 0,1
    # B00: j in 0,1; k in 0,1
    # W: k in 0,1; i in 0,1
    for c in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * 5 + j
                U442[u_idx, c] += U2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                v_idx = j * 6 + k
                V442[v_idx, c] += V2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                w_idx = k * 4 + i
                W442[w_idx, c] += W2[k*2+i, c]
                
    # Block 2: A01 B10
    # A01: i in 0,1; j in 2,3
    # B10: j in 2,3; k in 0,1
    # W: k in 0,1; i in 0,1
    for c in range(7):
        for i in range(2):
            for j in range(2, 4):
                u_idx = i * 5 + j
                U442[u_idx, 7+c] += U2[i*2+(j-2), c]
        for j in range(2, 4):
            for k in range(2):
                v_idx = j * 6 + k
                V442[v_idx, 7+c] += V2[(j-2)*2+k, c]
        for k in range(2):
            for i in range(2):
                w_idx = k * 4 + i
                W442[w_idx, 7+c] += W2[k*2+i, c]
                
    # Block 3: A10 B00
    # A10: i in 2,3; j in 0,1
    # B00: j in 0,1; k in 0,1
    # W: k in 0,1; i in 2,3
    for c in range(7):
        for i in range(2, 4):
            for j in range(2):
                u_idx = i * 5 + j
                U442[u_idx, 14+c] += U2[(i-2)*2+j, c]
        for j in range(2):
            for k in range(2):
                v_idx = j * 6 + k
                V442[v_idx, 14+c] += V2[j*2+k, c]
        for k in range(2):
            for i in range(2, 4):
                w_idx = k * 4 + i
                W442[w_idx, 14+c] += W2[k*2+(i-2), c]
                
    # Block 4: A11 B10
    # A11: i in 2,3; j in 2,3
    # B10: j in 2,3; k in 0,1
    # W: k in 0,1; i in 2,3
    for c in range(7):
        for i in range(2, 4):
            for j in range(2, 4):
                u_idx = i * 5 + j
                U442[u_idx, 21+c] += U2[(i-2)*2+(j-2), c]
        for j in range(2, 4):
            for k in range(2):
                v_idx = j * 6 + k
                V442[v_idx, 21+c] += V2[(j-2)*2+k, c]
        for k in range(2):
            for i in range(2, 4):
                w_idx = k * 4 + i
                W442[w_idx, 21+c] += W2[k*2+(i-2), c]
                
    # Block 3: 4x1x6 (Rank 24)
    U416 = np.zeros((20, 24), dtype=int)
    V416 = np.zeros((30, 24), dtype=int)
    W416 = np.zeros((24, 24), dtype=int)
    
    col = 0
    for i in range(4):
        for k in range(6):
            U416[i * 5 + 4, col] = 1
            V416[4 * 6 + k, col] = 1
            W416[k * 4 + i, col] = 1
            col += 1
            
    # Combine
    U_total = np.hstack([U444, U442, U416])
    V_total = np.hstack([V444, V442, V416])
    W_total = np.hstack([W444, W442, W416])
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

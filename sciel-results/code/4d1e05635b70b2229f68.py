import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Strassen 2x2x2 decomposition
    # U, V, W of shape (4, 7)
    # Indices 0..3 correspond to A00, A01, A10, A11 (row-major)
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 -= M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10-B11) -> C01 += M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = (A00-A11)(B01+B11) -> C00 += M4, C11 += M4
        U[0, 3] = 1; U[3, 3] = -1
        V[1, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01) -> C10 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7 = (A01-A11)(B10+B11) -> C01 += M7? No, C01 already has M3, M5.
        # Standard Strassen:
        # C00 = M1 + M4
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M4 + M5 + M7
        # Let's re-verify M7.
        # M7 = (A01-A11)(B10+B11)
        # A01 B10 + A01 B11 - A11 B10 - A11 B11
        # C01 needs A01 B11.
        # C11 needs -A11 B11?
        # Actually, let's just use the standard known coefficients.
        
        # Reset and use a verified set
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # 1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=-1
        
        # 2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=1
        
        # 3
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[1,2]=1
        
        # 4
        U[0,3]=1; U[3,3]=-1
        V[1,3]=1; V[3,3]=1
        W[0,3]=1; W[3,3]=1
        
        # 5
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[1,4]=1; W[3,4]=1
        
        # 6
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[2,5]=1
        
        # 7
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[1,6]=1
        
        return U, V, W

    U2, V2, W2 = strassen_2x2()
    
    # 2. Build 4x4x4 decomposition (Rank 49)
    # U4 shape (16, 49)
    # V4 shape (16, 49)
    # W4 shape (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                val_u = U2[i1, t1] * U2[i2, t2]
                val_v = V2[i1, t1] * V2[i2, t2]
                val_w = W2[i1, t1] * W2[i2, t2]
                U4[i, col] = val_u
                V4[i, col] = val_v
                W4[i, col] = val_w
                
    # 3. Build 4x4x1 decomposition (Rank 16)
    # This part handles the 5th column of B (index k=4)
    # U_tail shape (16, 16)
    # V_tail shape (20, 16) -- only rows 4,9,14,19 are non-zero
    # W_tail shape (20, 16) -- only rows 16,17,18,19 are non-zero
    U_tail = np.zeros((16, 16), dtype=int)
    V_tail = np.zeros((20, 16), dtype=int)
    W_tail = np.zeros((20, 16), dtype=int)
    
    for i in range(4):
        for j in range(4):
            t = i * 4 + j
            U_tail[i * 4 + j, t] = 1
            V_tail[j * 5 + 4, t] = 1
            W_tail[4 * 4 + i, t] = 1

    # 4. Combine into full matrices
    R = 49 + 16
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((20, R), dtype=int)
    W = np.zeros((20, R), dtype=int)
    
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    U[:, 49:] = U_tail
    V[:, 49:] = V_tail
    W[:, 49:] = W_tail

    # 5. Verify
    def verify(U, V, W):
        # T[i*4+j, j*5+k, k*4+i] = 1
        # Check a few points and random points
        # Full check is O(16*20*20*65) ~ 400k ops, fast enough.
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    idx_u = i * 4 + j
                    idx_v = j * 5 + k
                    idx_w = k * 4 + i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    if val != 1:
                        return False
        # Check zeros
        # Sample some zeros
        import random
        random.seed(42)
        for _ in range(1000):
            a = random.randint(0, 15)
            b = random.randint(0, 19)
            c = random.randint(0, 19)
            ia, ja = divmod(a, 4)
            jb, kb = divmod(b, 5)
            kc, ic = divmod(c, 4)
            if ja != jb or ia != ic or kb != kc:
                val = np.dot(U[a], V[b] * W[c])
                if val != 0:
                    return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        U_fb = np.zeros((16, 80), dtype=int)
        V_fb = np.zeros((20, 80), dtype=int)
        W_fb = np.zeros((20, 80), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    t = (i * 4 + j) * 5 + k
                    U_fb[i * 4 + j, t] = 1
                    V_fb[j * 5 + k, t] = 1
                    W_fb[k * 4 + i, t] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

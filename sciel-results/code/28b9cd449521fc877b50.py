import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140.0

    # Schoolbook solution (rank 96) - Guaranteed correct
    rank_seed = n * m * p
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
        Local indices:
        A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    # Attempt Rank 88 Construction
    # Top 2 rows (i=0,1): 2x4x8
    #   Split j into [0,1] and [2,3]. Split k into 4 blocks of size 2.
    #   8 blocks of 2x2x2. Each uses 7 rank terms. Total 56.
    # Bottom 1 row (i=2): 1x4x8
    #   8 independent 1x4x1 blocks. Each uses 4 rank terms. Total 32.
    # Total 88.
    
    U_88 = np.zeros((n*m, 88))
    V_88 = np.zeros((m*p, 88))
    W_88 = np.zeros((p*n, 88))
    col = 0
    
    Us, Vs, Ws = strassen_2x2x2()
    
    # Embed Strassen blocks for i in {0, 1}
    for i_block in [0, 1]:
        for j_block in [0, 1]:
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                
                for s in range(7):
                    # U mapping: local (li, lj) -> global (i_block + li, j_start + lj)
                    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    for la in range(4):
                        if Us[la, s] != 0:
                            li = la // 2
                            lj = la % 2
                            gi = i_block + li
                            gj = j_start + lj
                            U_88[gi*m + gj, col] = Us[la, s]
                    
                    # V mapping: local (lj, lk) -> global (j_start + lj, k_start + lk)
                    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    for lb in range(4):
                        if Vs[lb, s] != 0:
                            lj = lb // 2
                            lk = lb % 2
                            gj = j_start + lj
                            gk = k_start + lk
                            V_88[gj*p + gk, col] = Vs[lb, s]
                    
                    # W mapping: local (li, lk) -> global (k_start + lk, i_block + li)
                    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    for lc in range(4):
                        if Ws[lc, s] != 0:
                            li = lc // 2
                            lk = lc % 2
                            gi = i_block + li
                            gk = k_start + lk
                            W_88[gk*n + gi, col] = Ws[lc, s]
                    
                    col += 1
    
    # Embed Schoolbook blocks for i = 2
    for k in range(8):
        for j in range(4):
            U_88[2*m + j, col] = 1
            V_88[j*p + k, col] = 1
            W_88[k*n + 2, col] = 1
            col += 1
    
    # Verification
    # Check T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # T should be 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    valid = True
    # Check a few critical points and random points
    # Point (1, 9, 3) which failed before: i=0, j=1, k=1
    # a=1, b=9, c=3
    val = np.sum(U_88[1, :] * V_88[9, :] * W_88[3, :])
    if not np.isclose(val, 1.0):
        valid = False
    
    # Check another point: i=2, j=0, k=0 -> a=8, b=0, c=2
    val = np.sum(U_88[8, :] * V_88[0, :] * W_88[2, :])
    if not np.isclose(val, 1.0):
        valid = False
    
    # Check a zero point: i=0, j=0, k=0 (a=0, b=0, c=0) is 1.
    # i=0, j=1, k=0 (a=1, b=8, c=0) should be 0.
    val = np.sum(U_88[1, :] * V_88[8, :] * W_88[0, :])
    if not np.isclose(val, 0.0):
        valid = False

    # Full verification if time permits
    if valid and time.time() - start_time < timeout - 10:
        # Iterate over all valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U_88[a, :] * V_88[b, :] * W_88[c, :])
                    if not np.isclose(val, 1.0):
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        # Iterate over some invalid triples (random sample)
        if valid:
            for _ in range(100):
                a = np.random.randint(0, n*m)
                b = np.random.randint(0, m*p)
                c = np.random.randint(0, p*n)
                # Check if this is a valid triple
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                val = np.sum(U_88[a, :] * V_88[b, :] * W_88[c, :])
                if is_valid:
                    if not np.isclose(val, 1.0):
                        valid = False
                        break
                else:
                    if not np.isclose(val, 0.0):
                        valid = False
                        break

    if valid:
        return {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
    else:
        # Fallback to Schoolbook
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}


import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32
    
    # Build Strassen 2x2x2 decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0: (A00+A11)(B00+B11) -> C00+C11
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # c=1: (A10+A11)B00 -> C10-C11
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # c=2: A00(B01-B11) -> C01+C11
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # c=3: A11(B10-B00) -> C00+C10
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    
    # c=4: (A00+A01)B11 -> -C00+C01
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # c=5: (A10-A00)(B00+B01) -> C11
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # c=6: (A01-A11)(B10+B11) -> C00
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # Block Strassen for 4x4x4
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    terms_444 = []
    
    for kp in range(7):
        for ms in range(7):
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(16)
            
            for coeff, a_block in A_specs[kp]:
                bi, bj = divmod(a_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = U2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            u[gi * 4 + gj] += coeff * val
            
            for coeff, b_block in B_specs[kp]:
                bi, bj = divmod(b_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = V2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            v[gi * 4 + gj] += coeff * val
            
            for coeff, c_block in C_specs[kp]:
                bi, bj = divmod(c_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = W2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            w[gj * 4 + gi] += coeff * val
            
            if np.any(u) and np.any(v) and np.any(w):
                terms_444.append((u, v, w))
    
    # Build 4x4x8 decomposition by splitting B into two 4x4 halves
    U_list = []
    V_list = []
    W_list = []
    
    for half in range(2):
        b_col_off = half * 4
        
        for u, v, w in terms_444:
            U_list.append(u.copy())
            
            v_shifted = np.zeros(32)
            for j in range(4):
                for k in range(4):
                    v_shifted[j * 8 + (k + b_col_off)] = v[j * 4 + k]
            V_list.append(v_shifted)
            
            w_shifted = np.zeros(32)
            for k in range(4):
                for i in range(4):
                    w_shifted[(k + b_col_off) * 4 + i] = w[k * 4 + i]
            W_list.append(w_shifted)
    
    U = np.array(U_list).T  # (16, 98)
    V = np.array(V_list).T  # (32, 98)
    W = np.array(W_list).T  # (32, 98)
    
    # Verify correctness
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    error += abs(val - 1)
    
    if error > 1e-8:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

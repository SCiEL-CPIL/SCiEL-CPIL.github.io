import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Define Strassen 7-term decomposition for 2x2x2
    # Local indices ii, jj, kk in {0, 1}
    # U[local_i, local_j], V[local_j, local_k], W[local_k, local_i]
    # The 7 terms (u, v, w) are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> Wait, this is not right.
    # Let's use the standard Strassen 7 multiplications mapped to tensor rank-1 terms.
    # Actually, for 2x2x2, the rank is 7.
    # The 7 rank-1 tensors are:
    # 1. U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]] ? No.
    
    # Let's use a known list of 7 tuples (u_vec, v_vec, w_vec) for 2x2x2.
    # Flattened 4-vectors for U (2x2), V (2x2), W (2x2).
    # Order: [00, 01, 10, 11]
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # These M's are used to form C:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express this as sum of 7 rank-1 terms.
    # Each term r corresponds to one M_r.
    # Term r: U_r, V_r, W_r.
    # U_r extracts the linear combo from A.
    # V_r extracts the linear combo from B.
    # W_r distributes the result to C.
    
    # For M1:
    # A_part = A00 + A11. U1 = [[1,0],[0,1]] (flattened [1,0,0,1])
    # B_part = B00 + B11. V1 = [[1,0],[0,1]] (flattened [1,0,0,1])
    # C_part: C00 += M1, C11 += M1. W1 = [[1,0],[0,1]] (flattened [1,0,0,1])
    
    # For M2:
    # A_part = A10 + A11. U2 = [[0,0],[1,1]] (flattened [0,0,1,1])
    # B_part = B00. V2 = [[1,0],[0,0]] (flattened [1,0,0,0])
    # C_part: C10 += M2, C11 -= M2. W2 = [[0,1],[-1,0]] (flattened [0,1,-1,0])? 
    # W index is k,i. C10 is k=1,i=0? No, C is n x p. C[i,k].
    # Prompt: W[k*n+i]. So W[1*n+0] is C[0,1]? No.
    # C is n x p. Indices i in [0,n), k in [0,p).
    # W[k*n+i]. If n=2, p=2.
    # C[0,0] -> W[0*2+0]=W[0]
    # C[0,1] -> W[1*2+0]=W[2]
    # C[1,0] -> W[0*2+1]=W[1]
    # C[1,1] -> W[1*2+1]=W[3]
    
    # Let's map C indices to W flattened index (k*n+i):
    # C[0,0] (i=0,k=0) -> W[0]
    # C[0,1] (i=0,k=1) -> W[2]
    # C[1,0] (i=1,k=0) -> W[1]
    # C[1,1] (i=1,k=1) -> W[3]
    
    # M1 contributes to C[0,0] (+1) and C[1,1] (+1).
    # W1[0]=1, W1[3]=1. Others 0.
    # W1 flattened: [1, 0, 0, 1]
    
    # M2 contributes to C[1,0] (+1) and C[1,1] (-1).
    # W2[1]=1, W2[3]=-1.
    # W2 flattened: [0, 1, 0, -1]
    
    # M3 contributes to C[0,1] (+1) and C[1,1] (+1).
    # W3[2]=1, W3[3]=1.
    # W3 flattened: [0, 0, 1, 1]
    
    # M4 contributes to C[0,0] (-1) and C[0,1] (+1).
    # W4[0]=-1, W4[2]=1.
    # W4 flattened: [-1, 0, 1, 0]
    
    # M5 contributes to C[0,0] (+1) and C[0,1] (0)?
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    #
    # M5: C[0,0] (+1).
    # W5[0]=1.
    # W5 flattened: [1, 0, 0, 0]
    
    # M6: C[1,1] (+1).
    # W6[3]=1.
    # W6 flattened: [0, 0, 0, 1]
    
    # M7: C[0,0] (-1), C[1,0] (+1).
    # W7[0]=-1, W7[1]=1.
    # W7 flattened: [-1, 1, 0, 0]
    
    # Now U and V for each M:
    # U flattened (i,j) order: [00, 01, 10, 11]
    # V flattened (j,k) order: [00, 01, 10, 11]
    
    # M1: A00+A11. U1=[1,0,0,1]. B00+B11. V1=[1,0,0,1].
    # M2: A10+A11. U2=[0,0,1,1]. B00. V2=[1,0,0,0].
    # M3: A00. U3=[1,0,0,0]. B10-B11. V3=[0,0,1,-1].
    # M4: A11. U4=[0,0,0,1]. B01-B00. V4=[-1,1,0,0].
    # M5: A00+A01. U5=[1,1,0,0]. B11. V5=[0,0,0,1].
    # M6: A10-A00. U6=[-1,0,1,0]. B00+B01. V6=[1,1,0,0].
    # M7: A01-A11. U7=[0,1,0,-1]. B10+B11. V7=[0,0,1,1].
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,0,1,1]),
        ([0,0,0,1], [-1,1,0,0], [-1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [1,0,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [-1,1,0,0])
    ]
    
    # Partition indices
    n_blocks = [[0,1], [2]]
    m_blocks = [[0,1], [2,3], [4]]
    p_blocks = [[0,1], [2,3], [4,5]]
    
    # Calculate total rank
    total_rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                if len(nb)==2 and len(mb)==2 and len(pb)==2:
                    total_rank += 7
                else:
                    total_rank += len(nb)*len(mb)*len(pb)
                    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                ni = len(nb)
                mj = len(mb)
                pk = len(pb)
                
                if ni==2 and mj==2 and pk==2:
                    # Strassen
                    for u_loc, v_loc, w_loc in strassen_terms:
                        # Map local to global
                        # u_loc is flat [00, 01, 10, 11]
                        # Global i = nb[ii], j = mb[jj]
                        for li in range(2):
                            for lj in range(2):
                                idx_u = li*2 + lj
                                val = u_loc[idx_u]
                                if val != 0:
                                    gi = nb[li]
                                    gj = mb[lj]
                                    U[gi*m + gj][col] = val
                                    
                        for lj in range(2):
                            for lk in range(2):
                                idx_v = lj*2 + lk
                                val = v_loc[idx_v]
                                if val != 0:
                                    gj = mb[lj]
                                    gk = pb[lk]
                                    V[gj*p + gk][col] = val
                                    
                        for lk in range(2):
                            for li in range(2):
                                idx_w = lk*2 + li
                                val = w_loc[idx_w]
                                if val != 0:
                                    gk = pb[lk]
                                    gi = nb[li]
                                    W[gk*n + gi][col] = val
                                    
                        col += 1
                else:
                    # Schoolbook for this block
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                gi = nb[ii]
                                gj = mb[jj]
                                gk = pb[kk]
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    if time.time() - start_time > 110:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

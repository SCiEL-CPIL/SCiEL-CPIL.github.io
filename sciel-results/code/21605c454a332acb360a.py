import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space

DIMENSIONS = [2, 2, 16]
n, m, p = DIMENSIONS
U_DIM = n * m  # 4
V_DIM = m * p  # 32
W_DIM = p * n  # 32

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    def time_check():
        return time.time() > deadline
    
    # Build the target tensor T[i, j, k] = 1 if (i,j,k) is a valid triple for 2x2 * 2x16 multiplication
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    # We need to find rank-r decomposition: T = sum_{c=0}^{r-1} u_c v_c w_c
    # where u_c is in R^4, v_c is in R^32, w_c is in R^32
    
    # The tensor entries: T[i,j,k] = 1 if and only if the triple (i,j,k) corresponds to a multiplication term
    # For matrix multiplication A(i,j) * B(j,k) -> C(i,k):
    # T[i, j, k] = 1 for all valid (i,j,k), else 0
    # Since all (i,j,k) triples with i in [0,n), j in [0,m), k in [0,p) are valid,
    # T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Wait, that's not right. The tensor for matrix multiplication has T[i,j,k] = 1 
    # when the term A[i,j]*B[j,k] contributes to C[i,k].
    # In the flattening: U index = i*m+j, V index = j*p+k, W index = k*n+i
    # The tensor T is a 3-tensor where T[um_idx, vm_idx, wm_idx] = 1 
    # if (i,j,k) are consistent with um_idx = i*m+j, vm_idx = j*p+k, wm_idx = k*n+i.
    # For each (i,j,k) triple, there is exactly one entry that is 1.
    
    # Actually, the tensor is sparse: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's build the tensor as a 3D array
    T = np.zeros((U_DIM, V_DIM, W_DIM))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Strategy: Use the structure. The multiplication of 2x2 by 2x16 can be seen as
    # 8 independent 2x2 * 2x2 multiplications (since 16 = 2*8), but with shared structure.
    # Actually, 2x2 * 2x16 = for each of the 8 column-pairs in the 2x16, we do a 2x2*2x2.
    # Block diagonal Strassen would give 8 * 7 = 56.
    
    # But we can do better by exploiting that all 8 blocks use the same 2x2 left matrix.
    # Let A = [[a,b],[c,d]], B = [B_0, B_1, ..., B_7] where B_t is 2x2.
    # C = A * B, so C[:, 2t:2t+2] = A * B_t.
    # Using Strassen on A*B_t for each t gives 7 products per t, total 56.
    # But the Strassen products for different t share A-linear forms, so we can combine.
    
    # The 7 Strassen products for 2x2*2x2:
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)(w+z)
    # M6 = (c-a)(x+y)
    # M7 = (b-d)(z+w)
    # where B_t = [[x,y],[z,w]]
    
    # For B_t = [[x_t, y_t],[z_t, w_t]], the 7 products are linear in both A and B_t.
    # Since A is shared across all 8 blocks, we can define:
    # For each Strassen product k=1..7, the A-linear form is fixed, and the B-linear form 
    # varies per block t.
    # So we have 7 A-forms and 8*7=56 B-forms, but the rank is determined by the 
    # number of independent rank-1 terms.
    
    # Actually, each rank-1 term is u_c v_c w_c. The 56 Strassen terms are:
    # For t=0..7, k=1..7: u_k (A-form) * v_{t,k} (B-form) * w_k (C-form)
    # But u_k is shared across t, so we can factor: the tensor is
    # sum_{k=1}^7 sum_{t=0}^7 u_k v_{t,k} w_k
    # This is a sum of 56 rank-1 terms, but we can potentially reduce further.
    
    # Let me try a different approach: use the fact that the problem is essentially
    # computing 8 independent 2x2*2x2 products with the same left factor.
    # The tensor rank of M_{2,2,2} is 7. For 8 copies with shared left, 
    # the rank is at most 7*8 = 56, but potentially less.
    
    # Let me try to find a rank-50 or lower decomposition using optimization.
    # Given time constraints, let me try a structured approach.
    
    # Alternative: Try rank = 48 by combining some Strassen terms.
    # Or try rank = 56 (block Strassen) and see if we can verify it.
    
    # Let me implement the block Strassen decomposition with rank 56.
    # For each block t (0..7), and each Strassen product k (0..6):
    # u_k depends only on A (indices i*m+j for i,j in 0..1)
    # v_{t,k} depends on B_t (indices j*p+k for j in 0..1, k in 2t..2t+1)
    # w_k depends on C (indices k*n+i for k in 2t..2t+1, i in 0..1)
    
    # Strassen 2x2*2x2 decomposition (rank 7):
    # Let A = [[a,b],[c,d]], B = [[x,y],[z,w]], C = [[p,q],[r,s]]
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)(w+z)
    # M6 = (c-a)(x+y)
    # M7 = (b-d)(z+w)
    # p = M1 - M5 + M7 - M4
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M3 - M6 + M2
    
    # So the 7 rank-1 terms have:
    # u_0 = [1,0,0,1], v_0 = [1,0,0,1], w_0 = [1,1,1,1]  # M1
    # u_1 = [1,1,0,0], v_1 = [0,0,1,1], w_1 = [-1,1,0,0]  # M5
    # u_2 = [0,1,1,0], v_2 = [1,0,0,1], w_2 = [1,0,0,1]   # M2
    # u_3 = [1,0,0,-1], v_3 = [0,1,1,0], w_3 = [0,1,-1,0] # M3
    # u_4 = [0,0,1,-1], v_4 = [0,1,0,-1], w_4 = [0,0,1,1] # M4
    # u_5 = [1,-1,0,0], v_5 = [1,1,0,0], w_5 = [0,0,-1,0] # M6
    # u_6 = [0,1,-1,0], v_6 = [0,0,1,1], w_6 = [0,0,0,1]  # M7
    
    # Wait, let me be more careful. The standard Strassen decomposition:
    # p = M1 - M5 + M7 - M4
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M3 - M6 + M2
    
    # So w_0 (for M1) = [1, 0, 0, 1]  (coefficients of M1 in p and s)
    # w_1 (for M5) = [-1, 1, 0, 0]    (coefficients of M5 in p and q)
    # w_2 (for M2) = [0, 0, 1, 1]     (coefficients of M2 in r and s)
    # w_3 (for M3) = [0, 1, 0, -1]    (coefficients of M3 in q and s)
    # w_4 (for M4) = [0, 0, 1, -1]    (coefficients of M4 in r and p) -- wait, M4 appears in r and p
    # p = M1 - M5 + M7 - M4, so M4 coefficient in p is -1
    # r = M2 + M4, so M4 coefficient in r is 1
    # w_4 = [0, 0, 1, -1]  (p=0, q=0, r=1, s=0? No, w is for C indices)
    
    # Let me redefine. C = [[p,q],[r,s]], so C indices are:
    # C[0,0]=p, C[0,1]=q, C[1,0]=r, C[1,1]=s
    # W index = k*n + i, where k is column of B/C, i is row of A/C
    # For block t, columns are 2t and 2t+1
    # C[0,2t]=p_t, C[0,2t+1]=q_t, C[1,2t]=r_t, C[1,2t+1]=s_t
    # W indices: 2t*2+0=4t, 2t*2+1=4t+1, (2t+1)*2+0=4t+2, (2t+1)*2+1=4t+3
    
    # So for block t, w_k has non-zero entries at positions 4t, 4t+1, 4t+2, 4t+3
    
    # Let me just hardcode the 56-term Strassen decomposition and verify.
    
    rank = 56
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    
    # Strassen u-vectors (for A, flattened as [a,b,c,d])
    u_vecs = [
        [1, 0, 0, 1],   # M1: a+d
        [1, 1, 0, 0],   # M5: a+b
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, -1],  # M3: a-b... wait, M3 = a(y-w), so u = [1,0,0,0]
        # Let me redo this carefully.
    ]
    
    # M1 = (a+d)(x+w): u=[1,0,0,1], v=[1,0,0,1]
    # M2 = (c+d)x: u=[0,0,1,1], v=[1,0,0,0]
    # M3 = a(y-w): u=[1,0,0,0], v=[0,1,0,-1]
    # M4 = d(z-x): u=[0,0,0,1], v=[0,0,1,-1]
    # M5 = (a+b)(w+z): u=[1,1,0,0], v=[0,0,1,1]
    # M6 = (c-a)(x+y): u=[-1,0,1,0], v=[1,1,0,0]
    # M7 = (b-d)(z+w): u=[0,1,0,-1], v=[0,0,1,1]
    
    u_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    v_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    # w vectors: coefficients in C = [[p,q],[r,s]]
    # p = M1 - M5 + M7 - M4
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M3 - M6 + M2
    # So:
    # M1: p=1, q=0, r=0, s=1 -> w=[1,0,0,1]
    # M2: p=0, q=0, r=1, s=1 -> w=[0,0,1,1]
    # M3: p=0, q=1, r=0, s=-1 -> w=[0,1,0,-1]
    # M4: p=-1, q=0, r=1, s=0 -> w=[-1,0,1,0]
    # M5: p=-1, q=1, r=0, s=0 -> w=[-1,1,0,0]
    # M6: p=0, q=0, r=0, s=-1 -> w=[0,0,0,-1]
    # M7: p=1, q=0, r=0, s=0 -> w=[1,0,0,0]
    w_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
    ]
    
    col = 0
    for t in range(8):
        for k in range(7):
            # u: A indices [i*m+j] = [i*2+j]
            for idx, val in enumerate(u_strassen[k]):
                U[idx, col] = val
            
            # v: B indices [j*p+k'] where k' is column in block t
            # B_t = [[x,y],[z,w]] with x=B[0,2t], y=B[0,2t+1], z=B[1,2t], w=B[1,2t+1]
            # v_strassen[k] = [vx, vy, vz, vw]
            vx, vy, vz, vw = v_strassen[k]
            V[0 * 16 + 2 * t, col] = vx       # j=0, k'=2t
            V[0 * 16 + 2 * t + 1, col] = vy   # j=0, k'=2t+1
            V[1 * 16 + 2 * t, col] = vz       # j=1, k'=2t
            V[1 * 16 + 2 * t + 1, col] = vw   # j=1, k'=2t+1
            
            # w: C indices [k''*n+i] = [k''*2+i]
            # C_t = [[p,q],[r,s]] with p=C[0,2t], q=C[0,2t+1], r=C[1,2t], s=C[1,2t+1]
            # w_strassen[k] = [wp, wq, wr, ws]
            wp, wq, wr, ws = w_strassen[k]
            W[2 * t * 2 + 0, col] = wp       # k''=2t, i=0
            W[2 * t * 2 + 1, col] = wq       # k''=2t, i=1
            W[(2 * t + 1) * 2 + 0, col] = wr # k''=2t+1, i=0
            W[(2 * t + 1) * 2 + 1, col] = ws # k''=2t+1, i=1
            
            col += 1
    
    # Verify the decomposition
    def verify_decomp(U, V, W, T, tol=1e-9):
        # Compute sum_c U[:,c] V[:,c] W[:,c] and compare with T
        # T is (4, 32, 32)
        # Reconstruction: T_recon[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        rank = U.shape[1]
        T_recon = np.zeros_like(T)
        for c in range(rank):
            T_recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        return np.allclose(T_recon, T, atol=tol)
    
    if verify_decomp(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}


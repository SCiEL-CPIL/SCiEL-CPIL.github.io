import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Build target tensor
    T1 = np.zeros((n*m, m*p * p*n))
    T2 = np.zeros((m*p, n*m * p*n))
    T3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T1[a, b * (p*n) + c] = 1.0
                T2[b, a * (p*n) + c] = 1.0
                T3[c, a * (m*p) + b] = 1.0
    
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T

    def khatri_rao(A, B):
        dim1, R = A.shape
        dim2, _ = B.shape
        prod = A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
        return prod.reshape(R, dim1 * dim2).T

    def check_exact(U, V, W):
        MWV = khatri_rao(W, V)
        if not np.allclose(U @ MWV.T, T1, atol=1e-9):
            return False
        MUW = khatri_rao(U, W)
        if not np.allclose(V @ MUW.T, T2, atol=1e-9):
            return False
        MVU = khatri_rao(V, U)
        if not np.allclose(W @ MVU.T, T3, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exact(U_i, V_i, W_i):
            return U_i, V_i, W_i
        return None

    def polish(U, V, W, max_iter=100):
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            MWV = khatri_rao(W, V)
            err = np.linalg.norm(U @ MWV.T - T1)
            if err < 1e-10:
                res = snap_and_verify(U, V, W)
                if res:
                    return res
        return None

    def solve_als(U, V, W, max_iter=50):
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                break
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            MWV = khatri_rao(W, V)
            err = np.linalg.norm(U @ MWV.T - T1)
            if err < 1e-7:
                res = polish(U, V, W)
                if res:
                    return res, True
            if err > prev_err * 1.5 and it > 10:
                break
            prev_err = err
        return (U, V, W), False

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # p1 = (a+d)(e+h)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # p2 = (c+d)e
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=1
        # p3 = a(f-h)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1
        # p4 = d(g-f)
        U[3,3]=1; V[2,3]=1; V[1,3]=-1; W[0,3]=1; W[3,3]=1
        # p5 = (a+b)h
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        # p6 = (c+d)(f-h)
        U[2,5]=1; U[3,5]=1; V[1,5]=1; V[3,5]=-1; W[0,5]=1
        # p7 = (a-b)(e+g)
        U[0,6]=1; U[1,6]=-1; V[0,6]=1; V[2,6]=1; W[3,6]=1
        return U, V, W

    def build_block_solution():
        """
        Build solution using block decomposition with Strassen.
        Split A (2x5) into A1(2x2), A2(2x2), A3(2x1)
        Split B (5x6) into B1(2x6), B2(2x6), B3(1x6)
        C = A1*B1 + A2*B2 + A3*B3
        """
        # A1*B1: 2x2 * 2x6, split B1 into 3 blocks of 2x2
        # Each 2x2*2x2 uses Strassen rank 7, total 21
        # A2*B2: same, 21
        # A3*B3: 2x1 * 1x6, rank 12
        # Total: 54
        
        S_U, S_V, S_W = strassen_2x2x2()
        
        U = np.zeros((10, 54))
        V = np.zeros((30, 54))
        W = np.zeros((12, 54))
        
        col = 0
        # A1*B1: rows 0-1 of A, cols 0-1 of B (indices 0-1 in j)
        # A1 is A[0:2, 0:2], B1 is B[0:2, :]
        # Split B1 into B1_0 (B[0:2, 0:2]), B1_1 (B[0:2, 2:4]), B1_2 (B[0:2, 4:6])
        for block in range(3):
            # Strassen for A1 * B1_block
            # A1 indices: i=0,1; j=0,1 -> a_idx = i*5+j = 0,1,5,6
            # B1_block indices: j=0,1; k=block*2,block*2+1 -> b_idx = j*6+k
            # C indices: i=0,1; k=block*2,block*2+1 -> c_idx = k*2+i
            
            for s in range(7):
                # U part
                for a_row in range(4):
                    if S_U[a_row, s] != 0:
                        i = a_row // 2
                        j = a_row % 2
                        U[i*5+j, col] += S_U[a_row, s]
                # V part
                for b_row in range(4):
                    if S_V[b_row, s] != 0:
                        j = b_row // 2
                        k = b_row % 2
                        V[j*6 + block*2 + k, col] += S_V[b_row, s]
                # W part
                for c_row in range(4):
                    if S_W[c_row, s] != 0:
                        k = c_row // 2
                        i = c_row % 2
                        W[(block*2+k)*2 + i, col] += S_W[c_row, s]
                col += 1
        
        # A2*B2: rows 0-1 of A, cols 2-3 of B
        for block in range(3):
            for s in range(7):
                for a_row in range(4):
                    if S_U[a_row, s] != 0:
                        i = a_row // 2
                        j = a_row % 2
                        U[i*5 + 2 + j, col] += S_U[a_row, s]
                for b_row in range(4):
                    if S_V[b_row, s] != 0:
                        j = b_row // 2
                        k = b_row % 2
                        V[(2+j)*6 + block*2 + k, col] += S_V[b_row, s]
                for c_row in range(4):
                    if S_W[c_row, s] != 0:
                        k = c_row // 2
                        i = c_row % 2
                        W[(block*2+k)*2 + i, col] += S_W[c_row, s]
                col += 1
        
        # A3*B3: A[0:2, 4] * B[4, 0:6]
        # C[i,k] = A[i,4] * B[4,k]
        for i in range(2):
            for k in range(6):
                U[i*5+4, col] = 1
                V[4*6+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
        
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
        if check_exact(U, V, W):
            return U, V, W
        return None

    # Try block solution first
    block_sol = build_block_solution()
    if block_sol:
        U, V, W = block_sol
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try ALS from low rank up
    min_rank = 40
    max_rank = 53
    
    for R in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        restarts = 20 if R < 45 else 10
        for r in range(restarts):
            if time.time() > deadline:
                break
            scale = np.sqrt(1.0 / R)
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            
            sol, success = solve_als(U, V, W)
            if success:
                return {"U": sol[0].tolist(), "V": sol[1].tolist(), "W": sol[2].tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

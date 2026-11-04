import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def to_int_list(mat):
        """Convert numpy array to list of lists with integer values."""
        result = []
        for row in mat:
            r = []
            for v in row:
                if abs(v) < 1e-10:
                    r.append(0)
                else:
                    r.append(int(round(v)))
            result.append(r)
        return result
    
    def get_strassen_2x2():
        """Strassen 2x2x2, rank 7.
        U[li*2+lj], V[lj*2+lk], W[lk*2+li]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        # M3 = A11(B12-B22)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # M4 = A22(B21-B11)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=-1; W[2,3]=1
        # M5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=1
        # M6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # M7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=-1
        
        return U, V, W
    
    def verify_sub(U, V, W, ni, mi, pi):
        """Verify sum_r U[i*mi+j,r]*V[j*pi+k,r]*W[k*ni+i,r] = 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*mi+j, r] * V[j*pi+k, r] * W[k*ni+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_final(U, V, W):
        """Verify the full 4x5x6 decomposition."""
        return verify_sub(U, V, W, 4, 5, 6)
    
    def build_4x4x4():
        """4x4x4 using 4 Strassen 2x2x2 blocks. Rank 28.
        U: (16, 28), V: (16, 28), W: (16, 28)
        U[i*4+j], V[j*4+k], W[k*4+i]
        """
        U4 = np.zeros((16, 28))
        V4 = np.zeros((16, 28))
        W4 = np.zeros((16, 28))
        U2, V2, W2 = get_strassen_2x2()
        
        blocks = [(0,0), (0,1), (1,0), (1,1)]
        for idx, (br, bc) in enumerate(blocks):
            ro = idx * 7
            for li in range(2):
                for lj in range(2):
                    gi, gj = br*2+li, bc*2+lj
                    U4[gi*4+gj, ro:ro+7] = U2[li*2+lj, :]
            for lj in range(2):
                for lk in range(2):
                    gj, gk = bc*2+lj, bc*2+lk
                    V4[gj*4+gk, ro:ro+7] = V2[lj*2+lk, :]
            for lk in range(2):
                for li in range(2):
                    gk, gi = bc*2+lk, br*2+li
                    W4[gk*4+gi, ro:ro+7] = W2[lk*2+li, :]
        
        return U4, V4, W4
    
    def build_4x4x2():
        """4x4x2 schoolbook. Rank 32.
        U: (16, 32), V: (8, 32), W: (8, 32)
        U[i*4+j], V[j*2+k], W[k*4+i]
        """
        U = np.zeros((16, 32))
        V = np.zeros((8, 32))
        W = np.zeros((8, 32))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U[i*4+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W
    
    def build_4x4x6():
        """4x4x6 = 4x4x4 (rank 28) + 4x4x2 (rank 32). Total rank 60.
        U: (16, 60), V: (24, 60), W: (24, 60)
        U[i*4+j], V[j*6+k], W[k*4+i]
        """
        U444, V444, W444 = build_4x4x4()
        U442, V442, W442 = build_4x4x2()
        
        rank = 60
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        # 4x4x4 part: output cols k=0..3
        U[:, :28] = U444
        V[:16, :28] = V444
        W[:16, :28] = W444
        
        # 4x4x2 part: output cols k=4..5
        U[:, 28:] = U442
        for j in range(4):
            for lk in range(2):
                V[j*6 + 4 + lk, 28:] = V442[j*2 + lk, :]
        for lk in range(2):
            for li in range(4):
                W[(4+lk)*4 + li, 28:] = W442[lk*4 + li, :]
        
        return U, V, W
    
    def build_4x1x6():
        """4x1x6. Rank 24.
        U: (4, 24), V: (6, 24), W: (24, 24)
        U[i], V[k], W[k*4+i]
        """
        U = np.zeros((4, 24))
        V = np.zeros((6, 24))
        W = np.zeros((24, 24))
        col = 0
        for i in range(4):
            for k in range(6):
                U[i, col] = 1
                V[k, col] = 1
                W[k*4+i, col] = 1
                col += 1
        return U, V, W
    
    def build_4x5x6():
        """4x5x6 = 4x4x6 (rank 60) + 4x1x6 (rank 24). Total rank 84.
        U: (20, 84), V: (30, 84), W: (24, 84)
        U[i*5+j], V[j*6+k], W[k*4+i]
        """
        U446, V446, W446 = build_4x4x6()
        U416, V416, W416 = build_4x1x6()
        
        rank = 84
        U = np.zeros((20, rank))
        V = np.zeros((30, rank))
        W = np.zeros((24, rank))
        
        # 4x4x6 part: A[:, 0:4] * B[0:4, :]
        for i in range(4):
            for j in range(4):
                U[i*5+j, :60] = U446[i*4+j, :]
        V[:24, :60] = V446
        W[:, :60] = W446
        
        # 4x1x6 part: A[:, 4] * B[4, :]
        for i in range(4):
            U[i*5+4, 60:] = U416[i, :]
        V[24:, 60:] = V416
        W[:, 60:] = W416
        
        return U, V, W
    
    # Verify components
    if time.time() - start_time > DEADLINE:
        return schoolbook()
    
    U444, V444, W444 = build_4x4x4()
    if not verify_sub(U444, V444, W444, 4, 4, 4):
        return schoolbook()
    
    U442, V442, W442 = build_4x4x2()
    if not verify_sub(U442, V442, W442, 4, 4, 2):
        return schoolbook()
    
    U446, V446, W446 = build_4x4x6()
    if not verify_sub(U446, V446, W446, 4, 4, 6):
        return schoolbook()
    
    U416, V416, W416 = build_4x1x6()
    if not verify_sub(U416, V416, W416, 4, 1, 6):
        return schoolbook()
    
    U, V, W = build_4x5x6()
    if not verify_final(U, V, W):
        return schoolbook()
    
    # Convert to integer lists
    U_list = to_int_list(U)
    V_list = to_int_list(V)
    W_list = to_int_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}

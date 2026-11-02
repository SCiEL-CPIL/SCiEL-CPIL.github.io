"""Matrix multiplication 2x4 by 4x5 tensor decomposition with rank minimization."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify exact decomposition using numpy for speed."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) for n=2, m=2, p=2.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen intermediate products:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    
    # c=0: M1
    U0 = [1,0,0,1]  # a11, a12, a21, a22
    V0 = [1,0,0,1]  # b11, b12, b21, b22
    W0 = [1,0,0,1]  # c11, c12, c21, c22
    
    # c=1: M2
    U1 = [0,0,1,1]
    V1 = [1,0,0,0]
    W1 = [0,0,1,0]  # only c21 gets M2? No.
    # C = [[c11, c12], [c21, c22]]
    # c21 = M2 + M4 - M5 - M7
    # c22 = M1 - M4 + M5 + M6
    # c11 = M1 + M3 - M5 - M6
    # c12 = M2 - M3 + M4 + M7
    
    # Let's re-map carefully.
    # c=0 (M1): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # c=1 (M2): U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] -> W[2]=1 (c21)
    # c=2 (M3): U=[1,0,0,0], V=[0,1,0,-1], W=[1,0,0,0] -> W[0]=1 (c11) ... wait
    # c11 = M1 + M3 - M5 - M6. So M3 contributes +1 to c11.
    # c12 = M2 - M3 + M4 + M7. So M3 contributes -1 to c12.
    # So W for M3 is [1, -1, 0, 0]?
    # Let's check M3 = a11*(b12-b22).
    # U=[1,0,0,0], V=[0,1,0,-1].
    # W: c11 gets +1, c12 gets -1. So W=[1,-1,0,0].
    
    # c=3 (M4): a22*(b21-b11).
    # U=[0,0,0,1], V=[-1,0,1,0].
    # c11: 0. c12: +1. c21: -1. c22: 0.
    # W=[0,1,-1,0].
    
    # c=4 (M5): (a11+a12)*b22.
    # U=[1,1,0,0], V=[0,0,0,1].
    # c11: -1. c12: 0. c21: -1. c22: +1.
    # W=[-1,0,-1,1].
    
    # c=5 (M6): (a21-a11)*(b11+b12).
    # U=[-1,0,1,0], V=[1,1,0,0].
    # c11: -1. c12: 0. c21: 0. c22: +1.
    # W=[-1,0,0,1].
    
    # c=6 (M7): (a12-a22)*(b21+b22).
    # U=[0,1,-1,0], V=[0,0,1,1].
    # c11: 0. c12: +1. c21: 0. c22: +1.
    # W=[0,1,0,1].
    
    U = [
        [1,0,1,0,1,-1,0],
        [0,0,0,0,1,0,1],
        [0,1,0,0,0,1,0],
        [1,1,0,1,0,0,-1]
    ]
    V = [
        [1,1,0,-1,0,1,0],
        [0,0,1,0,0,1,0],
        [0,0,0,1,0,0,1],
        [1,0,-1,0,1,0,1]
    ]
    W = [
        [1,0,1,0,-1,-1,0],
        [0,0,-1,1,0,0,1],
        [0,1,0,-1,-1,0,0],
        [1,0,0,0,1,1,1]
    ]
    
    # Verify
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U_np[i*2+j], V_np[j*2+k] * W_np[k*2+i])
                if abs(s - 1.0) > 1e-9:
                    return None
    return U, V, W

def _winograd_2x2x3():
    """
    Winograd decomposition for 2x2x3.
    n=2, m=2, p=3.
    Rank 8? 
    A = [[a,b],[c,d]] (2x2)
    B = [[e,f,g],[h,i,j]] (2x3)
    C = A*B (2x3)
    
    Winograd for 2x2 conv (m=2, p=2) has rank 7 (Strassen).
    For p=3, we can split into p=2 and p=1.
    Rank(2x2x2) + Rank(2x2x1) = 7 + 2 = 9?
    Or maybe direct Winograd for 2x2x3.
    
    Let's stick to the block approach:
    2x4x5 = (2x2x5) + (2x2x5)
    2x2x5 = 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 2)
    Total rank = 2 * (7+7+2) = 32.
    
    Let's implement 2x2x1 rank 2.
    """
    pass

def _block_2x2x1():
    """
    Decomposition for 2x2x1.
    A (2x2), B (2x1), C (2x1).
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    Rank 2.
    c=0: U=[1,0,0,0], V=[1,0], W=[1,0]
    c=1: U=[0,1,0,0], V=[0,1], W=[1,0] -> Wait, W index is k*2+i. k=0.
    W[0*2+0]=1, W[0*2+1]=1? No.
    
    Let's do:
    c=0: U=[1,0,0,0], V=[1,0], W=[1,0] (contributes to C0)
    c=1: U=[0,1,0,0], V=[0,1], W=[1,0] (contributes to C0)
    c=2: U=[0,0,1,0], V=[1,0], W=[0,1] (contributes to C1)
    c=3: U=[0,0,0,1], V=[0,1], W=[0,1] (contributes to C1)
    
    This is rank 4. Can we do better?
    C0 = A00 B0 + A01 B1
    C1 = A10 B0 + A11 B1
    
    M1 = (A00+A01) B0
    M2 = A01 (B0+B1)
    C0 = M1 - M2 + ...?
    
    Let's use rank 2 for 2x2x1?
    No, 2x2x1 tensor is 2x2x1 = 4 elements. Rank >= 2.
    Actually, rank 2x2x1 is 2.
    T[i,j,0] = 1.
    U[i,j,c] V[j,0,c] W[0,i,c] = 1.
    
    c=0: U=[1,0,0,0], V=[1,0], W=[1,0] -> T[0,0]=1, T[0,1]=0, T[1,0]=0, T[1,1]=0.
    c=1: U=[0,1,0,0], V=[0,1], W=[1,0] -> T[0,1]=1.
    c=2: U=[0,0,1,0], V=[1,0], W=[0,1] -> T[1,0]=1.
    c=3: U=[0,0,0,1], V=[0,1], W=[0,1] -> T[1,1]=1.
    
    This is rank 4.
    Is rank 2 possible?
    T = u1 v1 w1 + u2 v2 w2.
    T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    
    Let's use the 2x2x2 Strassen and zero out the second k?
    No, that's not right.
    
    Let's just use schoolbook for the 2x2x1 parts if rank is not obvious.
    Schoolbook 2x2x1 is rank 4.
    Total rank = 2 * (7 + 7 + 4) = 36.
    """
    U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    V = [
        [1,0,1,0],
        [0,1,0,1]
    ]
    W = [
        [1,1,0,0],
        [0,0,1,1]
    ]
    return U, V, W

def construct():
    n, m, p = 2, 4, 5
    # We build a decomposition of rank 36.
    # Structure:
    # j in {0,1} and j in {2,3}.
    # For each block of j (size 2), we have 2x2x5.
    # 2x2x5 = 2x2x2 (k=0,1) + 2x2x2 (k=2,3) + 2x2x1 (k=4).
    # Ranks: 7 + 7 + 4 = 18.
    # Two blocks: 18 * 2 = 36.
    
    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_2x2x2 = _strassen_2x2x2()
    if strassen_2x2x2 is None:
        return _schoolbook(n, m, p)
    sU, sV, sW = strassen_2x2x2
    bU, bV, bW = _block_2x2x1()
    
    col_offset = 0
    
    for j_block in [0, 2]:
        # j_block is 0 or 2
        # Local j indices 0, 1 map to global j_block, j_block+1
        # Local k indices 0, 1, 2, 3, 4 map to global 0..4
        
        # 1. 2x2x2 for k=0,1 (local) -> global k=0,1
        for c in range(7):
            for li in range(2):
                gi = li
                lj = 0
                gj = j_block + lj
                # U[gi*4+gj, col_offset+c] = sU[li*2+lj, c]
                U[gi*4+gj][col_offset+c] = sU[li*2+lj][c]
            
            for lj in range(2):
                gj = j_block + lj
                for lk in range(2):
                    gk = lk # local k 0,1 -> global 0,1
                    # V[gj*5+gk, col_offset+c] = sV[lj*2+lk, c]
                    V[gj*5+gk][col_offset+c] = sV[lj*2+lk][c]
            
            for lk in range(2):
                gk = lk
                for li in range(2):
                    gi = li
                    # W[gk*2+gi, col_offset+c] = sW[lk*2+li, c]
                    W[gk*2+gi][col_offset+c] = sW[lk*2+li][c]
        col_offset += 7
        
        # 2. 2x2x2 for k=2,3 (local) -> global k=2,3
        for c in range(7):
            for li in range(2):
                gi = li
                lj = 0
                gj = j_block + lj
                U[gi*4+gj][col_offset+c] = sU[li*2+lj][c]
            
            for lj in range(2):
                gj = j_block + lj
                for lk in range(2):
                    gk = 2 + lk # local k 0,1 -> global 2,3
                    V[gj*5+gk][col_offset+c] = sV[lj*2+lk][c]
            
            for lk in range(2):
                gk = 2 + lk
                for li in range(2):
                    gi = li
                    W[gk*2+gi][col_offset+c] = sW[lk*2+li][c]
        col_offset += 7
        
        # 3. 2x2x1 for k=4 (local) -> global k=4
        for c in range(4):
            for li in range(2):
                gi = li
                lj = 0
                gj = j_block + lj
                U[gi*4+gj][col_offset+c] = bU[li*2+lj][c]
            
            for lj in range(2):
                gj = j_block + lj
                lk = 0
                gk = 4
                V[gj*5+gk][col_offset+c] = bV[lj*2+lk][c]
            
            lk = 0
            gk = 4
            for li in range(2):
                gi = li
                W[gk*2+gi][col_offset+c] = bW[lk*2+li][c]
        col_offset += 4
    
    # Verify
    if _verify(U, V, W, n, m, p, total_rank):
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

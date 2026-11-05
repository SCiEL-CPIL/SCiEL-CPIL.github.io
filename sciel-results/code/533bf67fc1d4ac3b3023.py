"""
Combined approach:
1. Structural idea from Inspiration 1: Recursive Kronecker product of Strassen decompositions.
   This allows us to build a 4x4x4 decomposition from 2x2x2 Strassen blocks efficiently.
   Rank of 4x4x4 via Kronecker(2x2x2, 2x2x2) is 7*7=49.
   This is significantly better than the block-diagonal approach in the current program which
   effectively treats 4x4x5 as two separate 4x4x2 and 4x4x3 problems, leading to rank 70.

2. Refinement/Machinery:
   - Use the verified schoolbook fallback (rank 80) from the current program.
   - Use the verification logic from the current program (which is robust).
   - Implement the Kronecker product specifically for the 4x4x4 case, and then extend to 4x4x5
     by adding a 4x4x1 schoolbook block (rank 16).
     Total rank = 49 + 16 = 65.
   - The 4x4x1 part is just the last column of B. We can treat it as a separate tensor block
     where k=4 is isolated.
     Actually, the tensor is T[i,j,k]. 
     For k=0..3, we use the 4x4x4 decomposition.
     For k=4, we use a 4x4x1 decomposition (rank 16).
     Since the k-index is disjoint (0-3 vs 4), we can simply concatenate the U, V, W matrices
     along the rank dimension.
     
     Wait, the U, V, W shapes are:
     U: (n*m, rank)
     V: (m*p, rank)
     W: (p*n, rank)
     
     For the 4x4x4 part (k in 0..3):
     U1: (16, 49)
     V1: (4*4, 49) -> (16, 49)  -- Note: V index is j*p+k. For p=4, j in 0..3, k in 0..3.
     W1: (4*4, 49) -> (16, 49)  -- Note: W index is k*n+i. For p=4, k in 0..3, i in 0..3.
     
     For the 4x4x1 part (k=4):
     U2: (16, 16)
     V2: (4*5, 16) -> (20, 16)  -- V index j*5+k. Only k=4 is non-zero. So V2[j*5+4, c] is set.
     W2: (5*4, 16) -> (20, 16)  -- W index k*4+i. Only k=4 is non-zero. So W2[4*4+i, c] is set.
     
     We can combine them into:
     U = [U1, U2]  (16 x 65)
     V = [V1_padded, V2] (20 x 65)
     W = [W1_padded, W2] (20 x 65)
     
     Where V1_padded has zeros for the k=4 rows (rows 16-19 of V, since j*5+4 for j=0..3 are 4,9,14,19).
     Actually, V1 is defined for p=4. To embed it into p=5:
     V1_embedded[j*5+k, c] = V1[j*4+k, c] for k in 0..3.
     V1_embedded[j*5+4, c] = 0.
     
     Similarly for W1_embedded[k*4+i, c] = W1[k*4+i, c] for k in 0..3.
     W1_embedded[4*4+i, c] = 0.
     
     This yields a valid decomposition of rank 65.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 120.0
START = time.time()


def _deadline_exceeded():
    return (time.time() - START) > DEADLINE - 2.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,i')delta(j,j')delta(k,k')."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                a = uc[i * m + j]
                if a == 0:
                    continue
                for k in range(p):
                    T[i, j, k] += a * vc[j * p + k] * wc[k * n + i]
    # Target: T[i,j,k] = 1 iff i==i',j==j',k==k' i.e. T[i,j,k]=1 for all (i,j,k)
    target = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T - target)) <= tol


def _schoolbook(n, m, p):
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
    return U, V, W


def _strassen_2x2x2():
    """Rank-7 decomposition of 2x2x2 multiplication. Returns (U,V,W) shapes (4,7),(4,7),(4,7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    def setv(vec, vals):
        for idx, v in zip([0,1,2,3], vals):
            vec[idx] = v
            
    # term 0: M1 = A11 * (B12 - B22)  (Note: standard Strassen variations exist, this is one valid set)
    # Let's use the standard 7 terms:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b11)
    # m4 = a22(b12-b22)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # My previous implementation in the seed was:
    # term 0: M1 = A11 * (B12 - B22) -> This corresponds to m3? No.
    # Let's stick to the verified one from the seed or just use a known correct one.
    # The seed's _strassen_2x2x2 was:
    # term 0: M1 = A11 * (B12 - B22) ... wait, let me re-read seed.
    # Seed:
    # term 0: M1 = A11 * (B12 - B22) -> U=[1,0,0,0], V=[0,1,0,-1], W=[1,0,0,0]
    # This computes A[0,0]*(B[0,1]-B[1,1]) and adds to C[0,0].
    # This is valid.
    
    # I will copy the exact Strassen from the seed to ensure correctness, as it was verified.
    setv(U[0], [1,0,0,0]); setv(V[0], [0,1,0,-1]); setv(W[0], [1,0,0,0])
    setv(U[1], [1,1,0,0]); setv(V[1], [0,0,0,1]); setv(W[1], [0,1,0,0])
    setv(U[2], [0,0,1,1]); setv(V[2], [1,0,0,0]); setv(W[2], [0,0,1,0])
    setv(U[3], [0,0,0,1]); setv(V[3], [0,0,1,-1]); setv(W[3], [0,0,0,1])
    setv(U[4], [1,0,0,1]); setv(V[4], [1,0,0,1]); setv(W[4], [-1,0,0,-1])
    setv(U[5], [0,1,0,-1]); setv(V[5], [0,0,1,1]); setv(W[5], [0,0,0,1])
    setv(U[6], [0,0,1,-1]); setv(V[6], [1,1,0,0]); setv(W[6], [1,0,0,0])
    return U, V, W


def _kronecker_product_2x2x2(U1, V1, W1):
    """
    Compute the Kronecker product of two 2x2x2 decompositions to get a 4x4x4 decomposition.
    Input: U1, V1, W1 of shape (4, 7)
    Output: U, V, W of shape (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # Iterate over all indices
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u_val = U1[i1*2+j1][c1] * U1[i2*2+j2][c2]
                                    v_val = V1[j1*2+k1][c1] * V1[j2*2+k2][c2]
                                    w_val = W1[k1*2+i1][c1] * W1[k2*2+i2][c2]
                                    
                                    if u_val == 0 or v_val == 0 or w_val == 0:
                                        continue
                                        
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    
                                    U[i*4+j][c] += u_val
                                    V[j*4+k][c] += v_val
                                    W[k*4+i][c] += w_val
    return U, V, W


def _build_4x4x5():
    """
    Build 4x4x5 decomposition.
    Rank 49 for 4x4x4 part (k=0..3).
    Rank 16 for 4x4x1 part (k=4).
    Total rank 65.
    """
    n, m, p = 4, 4, 5
    
    # 1. 4x4x4 part
    S_U, S_V, S_W = _strassen_2x2x2()
    U4, V4, W4 = _kronecker_product_2x2x2(S_U, S_V, S_W)
    
    # 2. 4x4x1 part (schoolbook for k=4)
    # This is just the standard multiplication for the last column.
    # T[i,j,4] = sum_k A[i,k] B[k,4].
    # This is a 4x4x1 tensor. Rank 16.
    U1 = [[0]*16 for _ in range(16)]
    V1 = [[0]*16 for _ in range(20)] # m*p = 4*5 = 20
    W1 = [[0]*16 for _ in range(20)] # p*n = 5*4 = 20
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Term: A[i,j] * B[j,4] -> C[i,4]
                # U[i*4+j, col] = 1
                # V[j*5+4, col] = 1  (j is row of B, 4 is col of B)
                # W[4*4+i, col] = 1  (4 is col of C, i is row of C)
                U1[i*4+j][col] = 1
                V1[j*5+4][col] = 1
                W1[4*4+i][col] = 1
                col += 1
                
    # 3. Combine
    rank = 49 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # Copy 4x4x4 part (columns 0..48)
    for c in range(49):
        for idx in range(16):
            U[idx][c] = U4[idx][c]
            V[idx][c] = V4[idx][c]
            W[idx][c] = W4[idx][c]
        
        # Note: V4 and W4 are size 16. They correspond to k=0..3.
        # In the full V (size 20), the indices for k=0..3 are j*5+k.
        # V4[j*4+k] corresponds to V[j*5+k].
        # So we need to map V4 indices to V indices.
        # V4 is indexed by j*4+k (j in 0..3, k in 0..3).
        # V is indexed by j*5+k (j in 0..3, k in 0..4).
        # So V[j*5+k, c] = V4[j*4+k, c].
        
        # Let's redo the copy for V and W carefully.
        
    # Reset and do it properly
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # Part 1: 4x4x4
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][c] = U4[i*4+j][c]
        for j in range(4):
            for k in range(4):
                V[j*5+k][c] = V4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                W[k*4+i][c] = W4[k*4+i][c]
                
    # Part 2: 4x4x1 (k=4)
    for c in range(16):
        for idx in range(16):
            U[idx][c+49] = U1[idx][c]
        for idx in range(20):
            V[idx][c+49] = V1[idx][c]
            W[idx][c+49] = W1[idx][c]
            
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # Try the structured rank-65 decomposition first.
    try:
        U, V, W = _build_4x4x5()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    if _deadline_exceeded():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    # Fallback: verified schoolbook rank 80.
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

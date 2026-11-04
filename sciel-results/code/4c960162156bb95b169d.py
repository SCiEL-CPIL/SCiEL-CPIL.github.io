"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 120  # safety margin under the 150s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _to_halfint(Mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * Mat) / 2.0

def _strassen_2x2():
    """Strassen decomposition for 2x2x2. Returns U,V,W each (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A21+A22)B11: C21+=M2, C22-=M2
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3=A11(B21-B22): C12+=M3, C22+=M3
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4=A22(B12-B11): C11+=M4, C21+=M4
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5=(A11+A12)B22: C11-=M5, C12+=M5
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6=(A21-A11)(B11+B12): C22+=M6
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A12-A22)(B21+B22): C11+=M7
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _strassen_4x4():
    """Strassen decomposition for 4x4x4. Returns U,V,W each (16,49)."""
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Outer Strassen coefficients
    oU = np.zeros((2,2,7))
    oV = np.zeros((2,2,7))
    oW = np.zeros((2,2,7))
    # M1
    oU[0,0,0]=1; oU[1,1,0]=1; oV[0,0,0]=1; oV[1,1,0]=1; oW[0,0,0]=1; oW[1,1,0]=1
    # M2
    oU[1,0,1]=1; oU[1,1,1]=1; oV[0,0,1]=1; oW[0,1,1]=1; oW[1,1,1]=-1
    # M3
    oU[0,0,2]=1; oV[1,0,2]=1; oV[1,1,2]=-1; oW[1,0,2]=1; oW[1,1,2]=1
    # M4
    oU[1,1,3]=1; oV[0,1,3]=1; oV[0,0,3]=-1; oW[0,0,3]=1; oW[0,1,3]=1
    # M5
    oU[0,0,4]=1; oU[0,1,4]=1; oV[1,1,4]=1; oW[0,0,4]=-1; oW[1,0,4]=1
    # M6
    oU[1,0,5]=1; oU[0,0,5]=-1; oV[0,0,5]=1; oV[0,1,5]=1; oW[1,1,5]=1
    # M7
    oU[0,1,6]=1; oU[1,1,6]=-1; oV[1,0,6]=1; oV[1,1,6]=1; oW[0,0,6]=1
    
    for ro in range(7):
        for ri in range(7):
            r = ro*7 + ri
            for ii in range(2):
                for ij in range(2):
                    cu = U2[ii*2+ij, ri]
                    if cu != 0:
                        for oi in range(2):
                            for oj in range(2):
                                co = oU[oi,oj,ro]
                                if co != 0:
                                    U[(2*oi+ii)*4+(2*oj+ij), r] += co*cu
            for ij in range(2):
                for ik in range(2):
                    cv = V2[ij*2+ik, ri]
                    if cv != 0:
                        for oj in range(2):
                            for ok in range(2):
                                co = oV[oj,ok,ro]
                                if co != 0:
                                    V[(2*oj+ij)*4+(2*ok+ik), r] += co*cv
            for ik in range(2):
                for ii in range(2):
                    cw = W2[ik*2+ii, ri]
                    if cw != 0:
                        for ok in range(2):
                            for oi in range(2):
                                co = oW[ok,oi,ro]
                                if co != 0:
                                    W[(2*ok+ik)*4+(2*oi+ii), r] += co*cw
    return U, V, W

def _embed_strassen_3x4x5():
    """Embed 4x4x4 Strassen into 3x4x5 by padding and extracting.
    Pad A (3x4) to 4x4 (add zero row), B (4x5) to 4x4 (add zero col), C (3x5) to 4x4 (add zero row/col).
    Then extract rows 0-2 of U, cols 0-3 of V mapped to 5 cols, etc.
    Actually: U4[i*4+j,r] for i in 0..3,j in 0..3 -> U[i*4+j,r] for i in 0..2,j in 0..3
    V4[j*4+k,r] for j in 0..3,k in 0..3 -> V[j*5+k,r] for j in 0..3,k in 0..4 (k=4 is zero)
    W4[k*4+i,r] for k in 0..3,i in 0..3 -> W[k*3+i,r] for k in 0..4,i in 0..2 (k=4 is zero)
    
    But V4 only has k in 0..3, and we need k in 0..4. The 5th column of B is new.
    Similarly W4 only has k in 0..3, but we need k in 0..4.
    
    This approach doesn't directly work. Let me try a different padding:
    Pad A to 4x5 (add zero row), keep B as 4x5, pad C to 4x5 (add zero row).
    Then 4x5 * 4x5 -> 4x5? No, that's not right either.
    
    Let me just try to use the 4x4x4 Strassen and see which terms survive.
    """
    U4, V4, W4 = _strassen_4x4()
    n, m, p = N, M, P  # 3, 4, 5
    
    # From 4x4x4: U4[i*4+j,r], V4[j*4+k,r], W4[k*4+i,r] for i,j,k in 0..3
    # We need: U[i*4+j,r] for i in 0..2, j in 0..3 (12 rows)
    #           V[j*5+k,r] for j in 0..3, k in 0..4 (20 rows)
    #           W[k*3+i,r] for k in 0..4, i in 0..2 (15 rows)
    
    # The issue: V4 has j*4+k with k in 0..3, but we need j*5+k with k in 0..4
    # W4 has k*4+i with k in 0..3, but we need k*3+i with k in 0..4
    
    # This direct embedding doesn't work due to dimension mismatch.
    # Instead, let's try: pad B to 4x4 (drop 5th col), pad C to 4x4 (drop 5th row)
    # Then we get a valid 4x4x4 decomposition, and we just lose the 5th column/row.
    # But we need the full 3x4x5, so this won't be complete.
    
    # Alternative: use a different approach. Let's just do random search with better init.
    return None

def _try_rank(rank, seed, time_budget):
    """Try to find an exact half-integer decomposition of given rank."""
    np.random.seed(seed)
    n, m, p = N, M, P
    best_err = np.inf
    best = None
    
    # Initialize with schoolbook-like structure for lower ranks
    if rank >= n*m*p:
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
    else:
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
    
    steps = 300
    for it in range(steps):
        if time.time() > time_budget:
            break
        Ur = _to_halfint(U)
        Vr = _to_halfint(V)
        Wr = _to_halfint(W)
        U3r = Ur.reshape(n, m, rank)
        V3r = Vr.reshape(m, p, rank)
        W3r = Wr.reshape(p, n, rank)
        Tr = np.einsum('ijr,jkr,kir->ijk', U3r, V3r, W3r)
        err = np.max(np.abs(Tr - 1.0))
        if err < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
        if err < best_err:
            best_err = err
            best = (Ur.copy(), Vr.copy(), Wr.copy())
        
        choice = np.random.randint(3)
        if choice == 0:
            idx = (np.random.randint(n*m), np.random.randint(rank))
            U[idx] += np.random.randn() * 0.3
        elif choice == 1:
            idx = (np.random.randint(m*p), np.random.randint(rank))
            V[idx] += np.random.randn() * 0.3
        else:
            idx = (np.random.randint(p*n), np.random.randint(rank))
            W[idx] += np.random.randn() * 0.3
    
    if best is not None:
        Ur, Vr, Wr = best
        U3r = Ur.reshape(n, m, rank)
        V3r = Vr.reshape(m, p, rank)
        W3r = Wr.reshape(p, n, rank)
        Tr = np.einsum('ijr,jkr,kir->ijk', U3r, V3r, W3r)
        if np.max(np.abs(Tr - 1.0)) < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None

def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    for rank in range(20, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break
        for seed in range(40):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
    return best_result

# EVOLVE-BLOCK-END

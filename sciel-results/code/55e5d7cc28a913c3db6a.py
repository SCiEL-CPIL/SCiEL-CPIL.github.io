import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{jk}"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                expected = 1.0 if j == k else 0.0
                if abs(val - expected) > tol:
                    return False
    return True

def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2. Convention: U[i*2+j,c], V[j*2+k,c], W[k*2+i,c]"""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1=(A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3=A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4=A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5=(A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6=(A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=1
    # M7=(A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=-1; W[1,6]=1
    return U, V, W

def _strassen_4x4():
    """Rank-49 decomposition for 4x4 * 4x4 via Kronecker product of two 2x2 Strassen."""
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _strassen_2x2()
    n, m, p = 4, 4, 4
    r1, r2 = 7, 7
    rank = r1 * r2
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U1[i1*2+j1, c1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2, c2]
                            if u2 == 0: continue
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            U[i*4+j, c] = u1 * u2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1*2+k1, c1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2*2+k2, c2]
                            if v2 == 0: continue
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            V[j*4+k, c] = v1 * v2
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W1[k1*2+i1, c1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2[k2*2+i2, c2]
                            if w2 == 0: continue
                            k = 2*k1 + k2
                            i = 2*i1 + i2
                            W[k*4+i, c] = w1 * w2
    return U, V, W

def _build_rank98():
    """Build rank-98 decomposition for 4x4 * 4x8 (two 4x4x4 blocks)."""
    n, m, p = 4, 4, 8
    U4, V4, W4 = _strassen_4x4()
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    # Left half (k=0..3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[k*8+i, 0:49] = W4[k*4+i, :]
    # Right half (k=4..7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+4+k, 49:98] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*8+i, 49:98] = W4[k*4+i, :]
    return U, V, W

def _try_subspace_embedding(U, V, W, n, m, p, target_rank, deadline, rng):
    """
    Subspace embedding: project U to target_rank columns, then solve for V, W.
    Uses the linearity in V and W given fixed U.
    """
    if target_rank >= U.shape[1]:
        return None
    
    # Create random projection matrix
    P = rng.standard_normal((U.shape[1], target_rank))
    U_new = U @ P  # (16, target_rank)
    
    # Now solve for V_new and W_new using ALS
    V_new = rng.standard_normal((m*p, target_rank))
    W_new = rng.standard_normal((p*n, target_rank))
    
    for als_iter in range(50):
        if time.time() > deadline - 5:
            break
        
        # Fix V_new, solve for W_new
        W_new = np.zeros((p*n, target_rank))
        success = True
        for k in range(p):
            for i in range(n):
                # For each j: sum_c U_new[i*m+j,c] * V_new[j*p+k,c] * W_new[k*n+i,c] = delta_{jk}
                A_sys = np.zeros((m, target_rank))
                b_sys = np.zeros(m)
                for j in range(m):
                    for c in range(target_rank):
                        A_sys[j, c] = U_new[i*m+j, c] * V_new[j*p+k, c]
                    b_sys[j] = 1.0 if j == k else 0.0
                try:
                    x, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    W_new[k*n+i, :] = x
                except:
                    success = False
                    break
            if not success:
                break
        if not success:
            continue
        
        # Fix W_new, solve for V_new
        V_new = np.zeros((m*p, target_rank))
        success = True
        for j in range(m):
            for k in range(p):
                A_sys = np.zeros((n, target_rank))
                b_sys = np.zeros(n)
                for i in range(n):
                    for c in range(target_rank):
                        A_sys[i, c] = U_new[i*m+j, c] * W_new[k*n+i, c]
                    b_sys[i] = 1.0 if j == k else 0.0
                try:
                    x, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    V_new[j*p+k, :] = x
                except:
                    success = False
                    break
            if not success:
                break
        if not success:
            continue
        
        # Check residual
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c] for c in range(target_rank))
                    expected = 1.0 if j == k else 0.0
                    max_err = max(max_err, abs(val - expected))
        
        if max_err < 1e-10:
            # Check if close to half-integers
            U_r = np.round(U_new * 2) / 2
            V_r = np.round(V_new * 2) / 2
            W_r = np.round(W_new * 2) / 2
            max_err_r = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] for c in range(target_rank))
                        expected = 1.0 if j == k else 0.0
                        max_err_r = max(max_err_r, abs(val - expected))
            if max_err_r < 1e-10:
                return U_r, V_r, W_r
            else:
                # Return non-rounded if exact
                if max_err < 1e-12:
                    return U_new, V_new, W_new
    return None

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140
    n, m, p = 4, 4, 8
    
    try:
        # Get rank-98 baseline
        U98, V98, W98 = _build_rank98()
        
        if not _verify(U98, V98, W98, n, m, p):
            raise ValueError("Rank-98 verification failed")
        
        best_rank = 98
        best_result = (U98, V98, W98)
        
        # Try subspace embedding for various target ranks
        for target_rank in range(97, 60, -1):
            if time.time() > DEADLINE - 15:
                break
            
            # Multiple random projections
            for trial in range(3):
                if time.time() > DEADLINE - 10:
                    break
                
                rng = np.random.default_rng(target_rank * 1000 + trial)
                result = _try_subspace_embedding(U98, V98, W98, n, m, p, target_rank, DEADLINE, rng)
                if result is not None:
                    U_t, V_t, W_t = result
                    if _verify(U_t, V_t, W_t, n, m, p):
                        best_rank = target_rank
                        best_result = (U_t, V_t, W_t)
                        break  # Found valid solution for this rank, try lower
        
        if best_rank < 98:
            U_b, V_b, W_b = best_result
            return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
        
        # Return rank-98
        return {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}
        
    except Exception:
        pass
    
    # Fallback: schoolbook rank 128
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

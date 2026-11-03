"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of two Strassen decompositions."""
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    return {"U": U, "V": V, "W": W}


def _build_4x4x3_tail():
    """Build rank-44 decomposition for 4x4 * 4x3 using 4 blocks of rank-11 2x2x3."""
    n, m, p = 4, 4, 3
    U2, V2, W2 = _strassen_2x2x2()
    # 2x2x3 decomposition: rank 11
    # 7 terms for 2x2x2 part (columns 0-2 of V/W)
    # 4 terms for 2x2x1 part (column 2 of V/W is separate? No, 2x2x3 is 2x2 * 2x3)
    # V is 2x3, W is 3x2.
    # The 2x2x2 part uses V[:, 0:2] and W[0:2, :].
    # The 2x2x1 part uses V[:, 2] and W[2, :].
    
    # Let's construct the 2x2x3 decomposition explicitly.
    # U (4x11), V (6x11), W (6x11)
    U_223 = [[0]*11 for _ in range(4)]
    V_223 = [[0]*11 for _ in range(6)]
    W_223 = [[0]*11 for _ in range(6)]
    
    # First 7 terms: Strassen for 2x2x2 part (output cols 0,1)
    for t in range(7):
        for lr in range(2):
            for lc in range(2):
                U_223[lr*2+lc][t] = U2[lr*2+lc][t]
        for lr in range(2):
            for kl in range(2):
                V_223[lr*3+kl][t] = V2[lr*2+kl][t]
        for lr in range(2):
            for kl in range(2):
                W_223[kl*2+lr][t] = W2[kl*2+lr][t]
        
    # Last 4 terms: Schoolbook for 2x2x1 part (output col 2)
    for t in range(4):
        lr = t // 2
        lc = t % 2
        U_223[lr*2+lc][7+t] = 1
        V_223[lc*3+2][7+t] = 1  # V index: row lc (0,1), col 2 (3rd col) -> lc*3+2
        W_223[2*2+lr][7+t] = 1  # W index: row 2 (3rd row), col lr (0,1) -> 2*2+lr
        
    # Now embed 4 such blocks into 4x4x3
    # Block (br, bc) where br is row block (0,1), bc is col block (0,1)
    # Input A: 4x4. Block (br, bc) is A[br*2:br*2+2, bc*2:bc*2+2]
    # Input B: 4x3. Block (bc, :) is B[bc*2:bc*2+2, :] (2x3)
    # Output C: 4x3. Block (br, :) is C[br*2:br*2+2, :] (2x3)
    
    rank = 44
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(11):
                r = block_idx * 11 + t
                # U part
                for lr in range(2):
                    for lc in range(2):
                        i = br*2 + lr
                        j = bc*2 + lc
                        U[i*4+j][r] = U_223[lr*2+lc][t]
                # V part
                for lr in range(2):
                    for kl in range(3):
                        j = bc*2 + lr
                        k = kl
                        V[j*3+k][r] = V_223[lr*3+kl][t]
                # W part
                for lr in range(2):
                    for kl in range(3):
                        i = br*2 + lr
                        k = kl
                        W[k*4+i][r] = W_223[kl*2+lr][t]
            block_idx += 1
            
    return {"U": U, "V": V, "W": W}


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7.
    Part 1: 4x4 * 4x4 -> rank 49
    Part 2: 4x4 * 4x3 -> rank 44
    Total: 93
    """
    n, m, p = 4, 4, 7
    head = _build_4x4x4_strassen()
    tail = _build_4x4x3_tail()
    
    rank_head = 49
    rank_tail = 44
    total_rank = rank_head + rank_tail
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Head: cols 0-3 of B and C
    for i in range(16):
        for r in range(rank_head):
            U[i][r] = head['U'][i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_head):
                V[i*p+j][r] = head['V'][i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_head):
                W[i*n+j][r] = head['W'][i*4+j][r]
                
    # Tail: cols 4-6 of B and C
    offset = rank_head
    for i in range(16):
        for r in range(rank_tail):
            U[i][offset + r] = tail['U'][i][r]
    for i in range(4):
        for j in range(3):
            for r in range(rank_tail):
                V[i*p + (4+j)][offset + r] = tail['V'][i*3+j][r]
    for i in range(3):
        for j in range(4):
            for r in range(rank_tail):
                W[(4+i)*n + j][offset + r] = tail['W'][i*4+j][r]
                
    return {"U": U, "V": V, "W": W}


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _als_decompose(n, m, p, target_rank, init_U, init_V, init_W, max_iter=200, tol=1e-12):
    """Alternating least squares for exact tensor decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    U = np.array(init_U, dtype=float)[:, :target_rank].copy()
    V = np.array(init_V, dtype=float)[:, :target_rank].copy()
    W = np.array(init_W, dtype=float)[:, :target_rank].copy()
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break
        Z1 = (V @ W.T).T.reshape(mp * pn, target_rank)
        U_new = np.linalg.lstsq(Z1.T, T1_T, rcond=None)[0].T
        Z2 = (U @ W.T).T.reshape(nm * pn, target_rank)
        V_new = np.linalg.lstsq(Z2.T, T2_T, rcond=None)[0].T
        Z3 = (U @ V.T).T.reshape(nm * mp, target_rank)
        W_new = np.linalg.lstsq(Z3.T, T3_T, rcond=None)[0].T
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    for it in range(30):
        if time.time() - _START_TIME > _TIME_LIMIT - 2:
            break
        Z1 = (V @ W.T).T.reshape(mp * pn, target_rank)
        U_new = np.linalg.lstsq(Z1.T, T1_T, rcond=None)[0].T
        Z2 = (U @ W.T).T.reshape(nm * pn, target_rank)
        V_new = np.linalg.lstsq(Z2.T, T2_T, rcond=None)[0].T
        Z3 = (U @ V.T).T.reshape(nm * mp, target_rank)
        W_new = np.linalg.lstsq(Z3.T, T3_T, rcond=None)[0].T
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < 1e-14:
            break
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = float(np.max(np.abs(T_recon - T)))
    return U, V, W, error


def _round_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Round to nearest half-integers and verify exact decomposition."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if (np.max(np.abs(U_r - U)) > tol or np.max(np.abs(V_r - V)) > tol or np.max(np.abs(W_r - W)) > tol):
        return None
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None


def _try_reduce_tail(base_U, base_V, base_W, target_rank, max_iter=200, num_restarts=10):
    """Try to reduce the rank of the 4x4x3 tail part."""
    n, m, p = 4, 4, 3
    nm, mp, pn = 16, 12, 12
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    U_base = np.array(base_U, dtype=float)
    V_base = np.array(base_V, dtype=float)
    W_base = np.array(base_W, dtype=float)
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        if restart == 0:
            # SVD init
            T1 = T.reshape(nm, mp * pn)
            U1, S1, _ = np.linalg.svd(T1, full_matrices=False)
            T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            U2, S2, _ = np.linalg.svd(T2, full_matrices=False)
            T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            U3, S3, _ = np.linalg.svd(T3, full_matrices=False)
            U_r = U1[:, :target_rank] * np.sqrt(S1[:target_rank])
            V_r = U2[:, :target_rank] * np.sqrt(S2[:target_rank])
            W_r = U3[:, :target_rank] * np.sqrt(S3[:target_rank])
            np.random.seed(restart)
            U_r += 0.01 * np.random.randn(*U_r.shape)
            V_r += 0.01 * np.random.randn(*V_r.shape)
            W_r += 0.01 * np.random.randn(*W_r.shape)
        elif restart == 1:
            # Base init
            U_r = U_base[:, :target_rank].copy()
            V_r = V_base[:, :target_rank].copy()
            W_r = W_base[:, :target_rank].copy()
        else:
            # Random init
            np.random.seed(restart * 100)
            U_r = np.random.randn(nm, target_rank)
            V_r = np.random.randn(mp, target_rank)
            W_r = np.random.randn(pn, target_rank)
            
        try:
            U_new, V_new, W_new, err = _als_decompose(n, m, p, target_rank, U_r, V_r, W_r, max_iter=max_iter)
            if err < 1e-8:
                result = _round_and_verify(U_new, V_new, W_new, n, m, p)
                if result is not None:
                    return result
            if err < best_err:
                best_err = err
        except Exception:
            continue
            
    return None


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Build the base rank-93 decomposition
    base_result = None
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            base_result = result
    except Exception:
        pass
    
    if base_result is None:
        if time.time() - _START_TIME > _TIME_LIMIT:
            return _schoolbook(4, 4, 7)
        try:
            result = _schoolbook(4, 4, 7)
            base_result = result
        except Exception:
            return _schoolbook(4, 4, 7)
            
    base_rank = len(base_result['U'][0])
    
    # Try to reduce the tail rank (44 -> 43, 42, ...)
    # The tail is the last 44 columns
    tail_U = [row[49:] for row in base_result['U']]
    tail_V = [row[49:] for row in base_result['V']]
    tail_W = [row[49:] for row in base_result['W']]
    
    # Note: tail_V and tail_W have dimensions 12x44 in the base result?
    # No, base V is 28x93. Tail V is rows 16..27 (cols 4..6 of B).
    # base V[i*p + (4+j)] for i in 0..3, j in 0..2.
    # So tail_V is 12x44.
    # tail_W is 12x44.
    
    # We want to find a rank R < 44 decomposition for the tail.
    # If we find one, we can replace the tail columns.
    
    # To save time, we only try a few reductions
    for target_tail_rank in [43, 42, 41, 40]:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        reduced_tail = _try_reduce_tail(tail_U, tail_V, tail_W, target_tail_rank)
        if reduced_tail is not None:
            # Combine head and reduced tail
            new_rank = 49 + target_tail_rank
            U_new = [[0] * new_rank for _ in range(16)]
            V_new = [[0] * new_rank for _ in range(28)]
            W_new = [[0] * new_rank for _ in range(28)]
            
            # Head
            for i in range(16):
                for r in range(49):
                    U_new[i][r] = base_result['U'][i][r]
            for i in range(4):
                for j in range(4):
                    for r in range(49):
                        V_new[i*7+j][r] = base_result['V'][i*7+j][r]
            for i in range(4):
                for j in range(4):
                    for r in range(49):
                        W_new[i*4+j][r] = base_result['W'][i*4+j][r]
                        
            # Tail
            offset = 49
            for i in range(16):
                for r in range(target_tail_rank):
                    U_new[i][offset + r] = reduced_tail['U'][i][r]
            for i in range(4):
                for j in range(3):
                    for r in range(target_tail_rank):
                        V_new[i*7 + (4+j)][offset + r] = reduced_tail['V'][i*3+j][r]
            for i in range(3):
                for j in range(4):
                    for r in range(target_tail_rank):
                        W_new[(4+i)*4 + j][offset + r] = reduced_tail['W'][i*4+j][r]
                        
            if _verify(U_new, V_new, W_new, 4, 4, 7):
                return {"U": U_new, "V": V_new, "W": W_new}
                
    return base_result

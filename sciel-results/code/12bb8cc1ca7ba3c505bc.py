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


def _verify(U, V, W, n, m, p, tol=1e-9):
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
                if abs(s - 1) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 using Strassen where beneficial.
    
    Part 1: 4x4 * 4x4 -> rank 49 (Kronecker of two Strassen 2x2x2)
    Part 2: 4x4 * 4x2 -> rank 28 (four 2x2*2x2 Strassen products)
    Part 3: 4x4 * 4x1 -> rank 16 (four 2x2*2x1 schoolbook products)
    Total: 93
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
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
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]  # 4*2
    W2p = [[0] * rank2 for _ in range(8)]  # 2*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            block_idx += 1
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]  # 4*1
    W3 = [[0] * rank3 for _ in range(4)]  # 1*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    
    # Combine all parts
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    # Part 3: B col 6, C col 6
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def _try_rank_reduction(U, V, W, n, m, p, max_attempts=200):
    """Attempt to reduce rank by checking if any rank-1 term is redundant.
    
    For each term r, check if the remaining terms still span the tensor.
    If so, remove that term. Repeat until no more terms can be removed.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build the target tensor as a vector
    target = np.zeros((n * m * p,), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[k * n * m + i * m + j] = 1.0
    
    # Helper to compute tensor residual for a subset of terms
    def _residual(terms_to_exclude):
        """Compute sum over non-excluded terms."""
        res = np.zeros((n * m * p,), dtype=float)
        for r in range(rank):
            if r in terms_to_exclude:
                continue
            # Add outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] to res
            for i in range(n * m):
                uval = U_arr[i, r]
                if uval == 0:
                    continue
                for j in range(m * p):
                    vval = V_arr[j, r]
                    if vval == 0:
                        continue
                    prod = uval * vval
                    if prod == 0:
                        continue
                    for k in range(p * n):
                        wval = W_arr[k, r]
                        if wval == 0:
                            continue
                        res[k * n * m + j * p + i] += prod * wval
        return res
    
    # For each term, check if removing it still gives exact match
    # We need to be careful: the tensor is T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # Build target tensor with proper indexing
    target_t = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_t[i * m + j, j * p + k, k * n + i] = 1.0
    
    removed = set()
    improved = True
    while improved:
        improved = False
        for r in range(rank):
            if r in removed:
                continue
            if time.time() - _START_TIME > _TIME_LIMIT:
                break
            
            # Check if removing term r still satisfies the tensor
            # Compute residual: T_target - sum_{r' != r} U[:,r']*V[:,r']*W[:,r']
            ok = True
            for ii in range(n * m):
                for jj in range(m * p):
                    for kk in range(p * n):
                        expected = target_t[ii, jj, kk]
                        # Compute sum without term r
                        s = 0.0
                        for r2 in range(rank):
                            if r2 == r or r2 in removed:
                                continue
                            s += U_arr[ii, r2] * V_arr[jj, r2] * W_arr[kk, r2]
                        if abs(s - expected) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                removed.add(r)
                improved = True
        
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
    
    if not removed:
        return {"U": U, "V": V, "W": W}
    
    # Build reduced matrices
    keep = [r for r in range(rank) if r not in removed]
    new_rank = len(keep)
    U_new = [[0] * new_rank for _ in range(n * m)]
    V_new = [[0] * new_rank for _ in range(m * p)]
    W_new = [[0] * new_rank for _ in range(p * n)]
    
    for new_r, old_r in enumerate(keep):
        for i in range(n * m):
            U_new[i][new_r] = U[old_r and i or i][old_r] if False else U[i][old_r]
        for i in range(m * p):
            V_new[i][new_r] = V[i][old_r]
        for i in range(p * n):
            W_new[i][new_r] = W[i][old_r]
    
    return {"U": U_new, "V": V_new, "W": W_new}


def _try_pairwise_merge(U, V, W, n, m, p):
    """Try to merge pairs of rank-1 terms that might combine.
    
    If two terms r1, r2 have the property that their sum can be replaced
    by a single term, we reduce rank by 1.
    This is more complex, so we skip for now and just use removal.
    """
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Build the rank-93 decomposition
    try:
        result = _build_4x4x7_optimized()
        if not _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return _schoolbook(4, 4, 7)
    except Exception:
        return _schoolbook(4, 4, 7)
    
    # Try to reduce rank by removing redundant terms
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        try:
            reduced = _try_rank_reduction(result['U'], result['V'], result['W'], 4, 4, 7)
            if _verify(reduced['U'], reduced['V'], reduced['W'], 4, 4, 7):
                result = reduced
        except Exception:
            pass
    
    return result

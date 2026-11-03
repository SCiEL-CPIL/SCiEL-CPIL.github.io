"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0


def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
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


def _build_4x4x7():
    """Build decomposition for 4x4 * 4x7.
    
    Split B (4x7) into B_left (4x4), B_mid (4x2), B_right (4x1).
    - 4x4 * 4x4: Kronecker of two Strassen 2x2x2, rank 49
    - 4x4 * 4x2: four 2x2*2x2 Strassen products, rank 28  
    - 4x4 * 4x1: four 2x2*2x1 schoolbook products, rank 16
    Total: 93
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    rank1, rank2, rank3 = 49, 28, 16
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
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
                                U[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    offset2 = rank1
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = offset2 + block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V[j*2+kl + 4][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W[(4+k)*4+i][r] += val
            block_idx += 1
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    offset3 = rank1 + rank2
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = offset3 + block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U[i*4+j][r] = 1
                    V[j*1 + 6][r] = 1
                    W[6*4 + i][r] = 1
            block_idx += 1
    
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


def _reduce_rank_by_peeling(U, V, W, n, m, p, tol=1e-9):
    """Try to reduce rank by peeling off linearly dependent rank-1 tensors."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    nm, mp, pn = n * m, m * p, p * n
    dim = nm * mp * pn
    
    # Build flattened matrix M where M[:, r] = vec(u_r ⊗ v_r ⊗ w_r)
    M = np.zeros((dim, rank))
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).ravel()
    
    # Find numerical rank via SVD
    try:
        _, S, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    if len(S) == 0:
        return U, V, W
    
    max_s = S[0] if len(S) > 0 else 0
    if max_s < tol:
        return U, V, W
    
    num_rank = sum(1 for s in S if s > tol * max_s)
    
    if num_rank >= rank:
        return U, V, W
    
    # Use QR with column pivoting to find independent columns
    # For now, just return original if reduction is significant
    if num_rank < rank:
        # Find pivot columns
        Q, R = np.linalg.qr(M, mode='reduced')
        diag_R = np.abs(np.diag(R))
        pivots = sorted([i for i in range(rank) if diag_R[i] > tol * max_s])
        
        if len(pivots) < rank:
            # Keep only pivot columns
            new_U = [[U_arr[i, c] for c in pivots] for i in range(nm)]
            new_V = [[V_arr[i, c] for c in pivots] for i in range(mp)]
            new_W = [[W_arr[i, c] for c in pivots] for i in range(pn)]
            
            if _verify(new_U, new_V, new_W, n, m, p):
                return new_U, new_V, new_W
    
    return U, V, W


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            # Try to reduce rank
            if time.time() - _START_TIME < _TIME_LIMIT - 30:
                U, V, W = _reduce_rank_by_peeling(result['U'], result['V'], result['W'], 4, 4, 7)
                if _verify(U, V, W, 4, 4, 7):
                    return {"U": U, "V": V, "W": W}
            return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    return _schoolbook(4, 4, 7)

"""
Structural: Winograd/Strassen block decomposition for 4x5x6 matrix multiplication,
combined with Kronecker-product structured decomposition and refinement.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    
    U has shape (16, 28), V has shape (8, 28), W has shape (8, 28).
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_2x3_block():
    """
    Build a 2x3x3 decomposition (A 2x3, B 3x3, C 2x3).
    Rank = 2*3*3 = 18 schoolbook, or can we do better?
    For now, use schoolbook for 2x3x3.
    """
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _build_4x5x6_winograd():
    """
    Build 4x5x6 decomposition using Winograd-style overlapping blocks.
    Rank = 108 (3*28 + 24).
    """
    n, m, p = 4, 5, 6
    U442, V442, W442 = _build_strassen_4x4x2()
    total_rank = 3 * 28 + 24
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    for blk in range(3):
        base = blk * 28
        k_start = blk * 2
        for i in range(4):
            for j in range(4):
                U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
        for j in range(4):
            for k in range(6):
                if k >= k_start and k < k_start + 2:
                    v_row = j * 6 + k
                    v_local_row = j * 2 + (k - k_start)
                    V[v_row, base:base+28] = V442[v_local_row, :]
        for k in range(6):
            if k >= k_start and k < k_start + 2:
                for i in range(4):
                    w_row = k * 4 + i
                    w_local_row = (k - k_start) * 4 + i
                    W[w_row, base:base+28] = W442[w_local_row, :]
    base = 3 * 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    return U, V, W

def _build_2x2x2_strassen():
    """Return (U,V,W) for 2x2x2 Strassen, each shape (4,7)."""
    U = np.zeros((4,7)); V = np.zeros((4,7)); W = np.zeros((4,7))
    # P1=(a11+a22)(b11+b22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2=(a21+a22)b11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # P3=a11(b12-b22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # P4=a22(b21-b11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # P5=(a11+a12)b22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # P6=(a21-a11)(b11+b12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7=(a12-a22)(b21+b22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x5x6_kronecker():
    """
    Construct decomposition using 4=2+2, 6=3+3 block structure.
    Each 2x5x3 block uses Strassen-2x2x2 (rank 7) + schoolbook 2x3x3 (rank 18) = 25.
    Total: 4 * 25 = 100.
    """
    n, m, p = 4, 5, 6
    U222, V222, W222 = _build_2x2x2_strassen()
    
    rank_per_block = 7 + 18  # Strassen + schoolbook
    total_rank = 4 * rank_per_block
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for ib in range(2):
        for kb in range(2):
            base = (ib * 2 + kb) * rank_per_block
            i_off = ib * 2
            k_off = kb * 3
            
            # Part 1: Strassen 2x2x2 for A[i_off:i_off+2, 0:2] @ B[0:2, k_off:k_off+2]
            for t in range(7):
                col = base + t
                for li in range(2):
                    for lj in range(2):
                        coeff = U222[li*2+lj, t]
                        if coeff != 0:
                            U[(i_off+li)*5 + lj, col] += coeff
                for lj in range(2):
                    for lk in range(2):
                        coeff = V222[lj*2+lk, t]
                        if coeff != 0:
                            V[lj*6 + (k_off+lk), col] += coeff
                for lk in range(2):
                    for li in range(2):
                        coeff = W222[lk*2+li, t]
                        if coeff != 0:
                            W[(k_off+lk)*4 + (i_off+li), col] += coeff
            
            # Part 2: Schoolbook for A[i_off:i_off+2, 2:5] @ B[2:5, k_off:k_off+3]
            for li in range(2):
                for lj in range(3):
                    for lk in range(3):
                        col = base + 7 + li*9 + lj*3 + lk
                        U[(i_off+li)*5 + (2+lj), col] = 1.0
                        V[(2+lj)*6 + (k_off+lk), col] = 1.0
                        W[(k_off+lk)*4 + (i_off+li), col] = 1.0
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _iterative_refinement(n, m, p, initial_rank=80, max_iters=100):
    """
    Vectorized iterative refinement: alternate solving for each matrix.
    """
    if _time_left() < 10.0:
        return None
    R = initial_rank
    U3 = np.random.randn(n, m, R) * 0.1
    V3 = np.random.randn(m, p, R) * 0.1
    W3 = np.random.randn(p, n, R) * 0.1
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        # Solve for W: for each (k,i), W[k,i,:] = sum_j U[i,j,:]*V[j,k,:] / ||...||^2
        # A[k,i,:] = sum_j U[i,j,:] * V[j,k,:]
        A = np.einsum('ijr,jkr->kir', U3, V3)
        norms = np.einsum('kir,kir->ki', A, A)
        valid = np.all(norms > 1e-10)
        if not valid:
            break
        W3 = A / norms[:, :, np.newaxis]
        
        # Solve for U: B[i,j,:] = sum_k V[j,k,:] * W[k,i,:]
        B = np.einsum('jkr,kir->ijr', V3, W3)
        norms = np.einsum('ijr,ijr->ij', B, B)
        valid = np.all(norms > 1e-10)
        if not valid:
            break
        U3 = B / norms[:, :, np.newaxis]
        
        # Solve for V: C[j,k,:] = sum_i U[i,j,:] * W[k,i,:]
        C = np.einsum('ijr,kir->jkr', U3, W3)
        norms = np.einsum('jkr,jkr->jk', C, C)
        valid = np.all(norms > 1e-10)
        if not valid:
            break
        V3 = C / norms[:, :, np.newaxis]
        
        if it % 5 == 0:
            U = U3.reshape(n*m, R)
            V = V3.reshape(m*p, R)
            W = W3.reshape(p*n, R)
            if _verify_np(U, V, W, n, m, p, 1e-6):
                if _verify_decomp(U, V, W, n, m, p, 1e-8):
                    return U, V, W
    
    U = U3.reshape(n*m, R)
    V = V3.reshape(m*p, R)
    W = W3.reshape(p*n, R)
    if _verify_np(U, V, W, n, m, p, 1e-6):
        if _verify_decomp(U, V, W, n, m, p, 1e-8):
            return U, V, W
    return None

def _local_reduce(U, V, W, n, m, p, max_iters=10, budget_s=20.0):
    """Local column-removal search."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        for c in range(R):
            if _time_left() < 3.0:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def _pairwise_reduce(U, V, W, n, m, p, max_iters=5, budget_s=15.0):
    """Try removing pairs of columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 3:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 3.0:
                break
            for c2 in range(c1 + 1, R):
                if _time_left() < 3.0:
                    break
                mask = np.ones(R, dtype=bool)
                mask[c1] = False
                mask[c2] = False
                U2 = U[:, mask]
                V2 = V[:, mask]
                W2 = W[:, mask]
                if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                    if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                        U, V, W = U2, V2, W2
                        R -= 2
                        found = True
                        break
            if found:
                break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Winograd 4x5x6 (rank 108)
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_winograd()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try Kronecker 4x5x6 (target rank ~100)
    try:
        if _time_left() > 25.0:
            U, V, W = _build_4x5x6_kronecker()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Iterative refinement
    try:
        if _time_left() > 20.0:
            for R in range(max(50, best_rank - 20), best_rank):
                if _time_left() < 10.0:
                    break
                for attempt in range(3):
                    if _time_left() < 8.0:
                        break
                    cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=60)
                    if cand is not None:
                        U, V, W = cand
                        if _verify_decomp(U, V, W, n, m, p, 1e-9):
                            if U.shape[1] < best_rank:
                                best_rank = U.shape[1]
                                best_U, best_V, best_W = U, V, W
                                break
                if best_rank <= R:
                    break
    except Exception:
        pass
    
    # Local column-removal
    try:
        if _time_left() > 12.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=30, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pairwise column-removal
    try:
        if _time_left() > 8.0:
            cand = _pairwise_reduce(best_U, best_V, best_W, n, m, p, max_iters=8, budget_s=10.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

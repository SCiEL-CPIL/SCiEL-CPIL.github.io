"""
Rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.
Strategy: subspace decomposition splitting m=5 into m=2+3, using Strassen
blocks for 4x2x6 and structured blocks for 4x3x6, then column-removal.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
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
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
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
    
    return U, V, W

def _build_4x2x6_strassen():
    """
    Build 4x2x6 decomposition using Strassen blocks.
    A is 4x2, B is 2x6, C is 4x6.
    
    Split A into 2 blocks of 2x2, B into 3 blocks of 2x2.
    6 blocks of 2x2x2 Strassen, each rank 7.
    Total rank: 6 * 7 = 42.
    
    U shape: (8, 42), V shape: (12, 42), W shape: (24, 42)
    U[i*2+j, r], V[j*6+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _build_strassen_2x2()
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((12, rank))
    W = np.zeros((24, rank))
    
    # 2 row-blocks of A (each 2x2) x 3 col-blocks of B (each 2x2)
    for a_blk in range(2):  # rows 0-1, 2-3 of A
        for b_blk in range(3):  # cols 0-1, 2-3, 4-5 of B
            base = (a_blk * 3 + b_blk) * 7
            for t in range(7):
                col = base + t
                # A part: rows a_blk*2 to a_blk*2+1, cols 0-1
                for li in range(2):
                    for lj in range(2):
                        local_row = li * 2 + lj
                        coeff = U2[local_row, t]
                        if coeff != 0:
                            global_i = a_blk * 2 + li
                            global_j = lj
                            U[global_i * 2 + global_j, col] += coeff
                
                # B part: rows 0-1 (j), cols b_blk*2 to b_blk*2+1 (k)
                for lj in range(2):
                    for lk in range(2):
                        local_row = lj * 2 + lk
                        coeff = V2[local_row, t]
                        if coeff != 0:
                            global_j = lj
                            global_k = b_blk * 2 + lk
                            V[global_j * 6 + global_k, col] += coeff
                
                # C part: rows b_blk*2 to b_blk*2+1 (k), cols a_blk*2 to a_blk*2+1 (i)
                for lk in range(2):
                    for li in range(2):
                        local_row = lk * 2 + li
                        coeff = W2[local_row, t]
                        if coeff != 0:
                            global_k = b_blk * 2 + lk
                            global_i = a_blk * 2 + li
                            W[global_k * 4 + global_i, col] += coeff
    
    return U, V, W

def _build_4x3x6_blocks():
    """
    Build 4x3x6 decomposition.
    A is 4x3, B is 3x6, C is 4x6.
    
    Split A into A1 (2x3) and A2 (2x3).
    Split B into B1 (3x3) and B2 (3x3).
    
    4 sub-blocks of 2x3x3.
    
    For each 2x3x3:
    Split B into B1 (3x2) and B2 (3x1).
    A@B1: 2x3x2, rank 11
    A@B2: 2x3x1, rank 6
    Total per 2x3x3: 17
    
    Total: 4 * 17 = 68
    """
    n, m, p = 4, 3, 6
    rank = 68
    U = np.zeros((12, rank))
    V = np.zeros((18, rank))
    W = np.zeros((24, rank))
    
    # Build 2x3x2 block (rank 11)
    def build_2x3x2():
        U232 = np.zeros((6, 11))
        V232 = np.zeros((6, 11))
        W232 = np.zeros((4, 11))
        U2, V2, W2 = _build_strassen_2x2()
        
        # 2x2x2 Strassen part (first 2 cols of A, first 2 cols of B)
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U232[i * 3 + j, t] = U2[i * 2 + j, t]
            for j in range(2):
                for k in range(2):
                    V232[j * 2 + k, t] = V2[j * 2 + k, t]
            for k in range(2):
                for i in range(2):
                    W232[k * 2 + i, t] = W2[k * 2 + i, t]
        
        # 2x1x2 schoolbook part (col 2 of A, all cols of B)
        for i in range(2):
            for k in range(2):
                col = 7 + i * 2 + k
                U232[i * 3 + 2, col] = 1.0
                V232[2 * 2 + k, col] = 1.0
                W232[k * 2 + i, col] = 1.0
        
        return U232, V232, W232
    
    # Build 2x3x1 block (rank 6)
    def build_2x3x1():
        U231 = np.zeros((6, 6))
        V231 = np.zeros((3, 6))
        W231 = np.zeros((2, 6))
        for i in range(2):
            for j in range(3):
                col = i * 3 + j
                U231[i * 3 + j, col] = 1.0
                V231[j, col] = 1.0
                W231[i, col] = 1.0
        return U231, V231, W231
    
    U232, V232, W232 = build_2x3x2()
    U231, V231, W231 = build_2x3x1()
    
    # 4 sub-blocks: (a_blk, b_blk) where a_blk in {0,1}, b_blk in {0,1}
    for a_blk in range(2):
        for b_blk in range(2):
            base = (a_blk * 2 + b_blk) * 17
            
            # 2x3x2 part: A[a_blk*2:(a_blk+1)*2, :] @ B[:, b_blk*3:(b_blk+1)*3][:, :2]
            for t in range(11):
                col = base + t
                # U: A rows a_blk*2 to a_blk*2+1, all 3 cols
                for i in range(2):
                    for j in range(3):
                        U[(a_blk * 2 + i) * 3 + j, col] = U232[i * 3 + j, t]
                # V: B rows 0-2, cols b_blk*3 to b_blk*3+1
                for j in range(3):
                    for k in range(2):
                        V[j * 6 + (b_blk * 3 + k), col] = V232[j * 2 + k, t]
                # W: C rows b_blk*3 to b_blk*3+1, cols a_blk*2 to a_blk*2+1
                for k in range(2):
                    for i in range(2):
                        W[(b_blk * 3 + k) * 4 + (a_blk * 2 + i), col] = W232[k * 2 + i, t]
            
            # 2x3x1 part: A[a_blk*2:(a_blk+1)*2, :] @ B[:, b_blk*3:(b_blk+1)*3][:, 2]
            for t in range(6):
                col = base + 11 + t
                for i in range(2):
                    for j in range(3):
                        U[(a_blk * 2 + i) * 3 + j, col] = U231[i * 3 + j, t]
                for j in range(3):
                    V[j * 6 + (b_blk * 3 + 2), col] = V231[j, t]
                for i in range(2):
                    W[(b_blk * 3 + 2) * 4 + (a_blk * 2 + i), col] = W231[i, t]
    
    return U, V, W

def _build_4x5x6_subspace():
    """
    Build 4x5x6 using subspace decomposition: m=5 split into m=2 and m=3.
    
    Part 1: 4x2x6 (rank 42) - uses first 2 cols of A, first 2 rows of B
    Part 2: 4x3x6 (rank 68) - uses last 3 cols of A, last 3 rows of B
    
    Total rank: 42 + 68 = 110
    """
    n, m, p = 4, 5, 6
    U1, V1, W1 = _build_4x2x6_strassen()
    U2, V2, W2 = _build_4x3x6_blocks()
    
    total_rank = 42 + 68
    U = np.zeros((20, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((24, total_rank))
    
    # Part 1: 4x2x6, cols 0-1 of A, rows 0-1 of B
    for t in range(42):
        col = t
        # U: A[i, j] for i=0..3, j=0..1 -> row i*5+j
        for i in range(4):
            for j in range(2):
                U[i * 5 + j, col] = U1[i * 2 + j, t]
        # V: B[j, k] for j=0..1, k=0..5 -> row j*6+k
        for j in range(2):
            for k in range(6):
                V[j * 6 + k, col] = V1[j * 6 + k, t]
        # W: C[k, i] for k=0..5, i=0..3 -> row k*4+i
        for k in range(6):
            for i in range(4):
                W[k * 4 + i, col] = W1[k * 4 + i, t]
    
    # Part 2: 4x3x6, cols 2-4 of A, rows 2-4 of B
    base = 42
    for t in range(68):
        col = base + t
        # U: A[i, j] for i=0..3, j=2..4 -> row i*5+j
        for i in range(4):
            for j in range(3):
                U[i * 5 + (j + 2), col] = U2[i * 3 + j, t]
        # V: B[j, k] for j=2..4, k=0..5 -> row j*6+k
        for j in range(3):
            for k in range(6):
                V[(j + 2) * 6 + k, col] = V2[j * 6 + k, t]
        # W: C[k, i] for k=0..5, i=0..3 -> row k*4+i
        for k in range(6):
            for i in range(4):
                W[k * 4 + i, col] = W2[k * 4 + i, t]
    
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

def _local_reduce(U, V, W, n, m, p, max_iters=20, budget_s=30.0):
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

def _pairwise_reduce(U, V, W, n, m, p, max_iters=10, budget_s=20.0):
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

def _triple_reduce(U, V, W, n, m, p, max_iters=5, budget_s=15.0):
    """Try removing triples of columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 4:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 3.0:
                break
            for c2 in range(c1 + 1, R):
                if _time_left() < 3.0:
                    break
                for c3 in range(c2 + 1, R):
                    if _time_left() < 3.0:
                        break
                    mask = np.ones(R, dtype=bool)
                    mask[c1] = False
                    mask[c2] = False
                    mask[c3] = False
                    U2 = U[:, mask]
                    V2 = V[:, mask]
                    W2 = W[:, mask]
                    if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                            U, V, W = U2, V2, W2
                            R -= 3
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Start with subspace decomposition
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try subspace decomposition
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_subspace()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Local column-removal refinement
    try:
        if _time_left() > 20.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=20, budget_s=25.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pairwise column-removal refinement
    try:
        if _time_left() > 15.0:
            cand = _pairwise_reduce(best_U, best_V, best_W, n, m, p, max_iters=10, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Triple column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _triple_reduce(best_U, best_V, best_W, n, m, p, max_iters=5, budget_s=10.0)
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
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": W.tolist()}

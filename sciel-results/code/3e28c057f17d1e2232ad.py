"""Tensor decomposition search for 2x3x8 matrix multiplication."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE - time.time()


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == delta(i,i)delta(j,j)delta(k,k)."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Compute tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                target = 1.0 if (i == 0 or True) else 0.0
                # target: delta_ii * delta_jj * delta_kk
                target = (1.0 if i == i else 0.0) * (1.0 if j == j else 0.0) * (1.0 if k == k else 0.0)
                # Actually target is 1 only when this triple corresponds to identity mapping.
                # For matrix mult tensor: T[i,j,k] = 1 if (i,j,k) is a valid "diagonal" triple?
                # No. The tensor is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}?? 
                # Let me reconsider: T_{i,j,k} = 1 if the product C[i,k] uses A[i,j]*B[j,k].
                # So T[i,j,k] = 1 for ALL i,j,k! Because C[i,k] = sum_j A[i,j] B[j,k].
                # Wait no, the tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
                # That coefficient is 1 for every (i,j,k). So T[i,j,k] = 1 for all i,j,k.
                target = 1.0
                if abs(s - target) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok


def _verify2(U, V, W, n, m, p, tol=1e-9):
    """Proper verification: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all (i,j,k)."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank_of_matrix(M):
    """Numerical rank."""
    if len(M) == 0 or len(M[0]) == 0:
        return 0
    Ma = np.array(M, dtype=float)
    try:
        r = np.linalg.matrix_rank(Ma, tol=1e-8)
    except Exception:
        r = 0
    return int(r)


def _strassen_2x2x2():
    """Return (U,V,W) for 2x2x2 Strassen, rank 7."""
    # Standard Strassen with indices i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
    U = []
    V = []
    W = []
    # m1 = a00*b00, m2 = a11*b11, m3=(a00+a11)*(b00+b11), m4=(a10+a11)*b01,
    # m5=(a00+a10)*b11, m6=a01*(b10-b00), m7=(a11-a01)*(b01+b10)
    # C00 = m3+m4-m2-m7, C01=m1+m7, C10=m5+m2, C11=m1+m3-m4-m5
    # Build factor matrices.
    # For each rank factor c, we need U row (i*2+j), V row (j*2+k), W row (k*2+i).
    # This is complex; let me just use known rank-7 Strassen factors.
    # Actually, let me construct it properly.
    # The 7 multiplications correspond to bilinear forms.
    # I'll use the standard construction.
    factors = [
        # (U entries per (i,j), V entries per (j,k), W entries per (k,i))
        # m1 = A00*B00 -> C01 += m1, C11 += m1
        # m2 = A11*B11 -> C10 += m2, C00 -= m2
        # m3 = (A00+A11)*(B00+B11) -> C00 += m3, C11 += m3
        # m4 = (A10+A11)*B01 -> C00 += m4, C11 -= m4
        # m5 = (A00+A10)*B11 -> C10 += m5, C11 -= m5
        # m6 = A01*(B10-B00) -> C10 += m6  (wait, let me recheck)
        # Actually standard: C00=m3+m4-m2-m7, C01=m1+m7, C10=m5+m2, C11=m1+m3-m4-m5
        # Hmm, let me use a cleaner reference.
    ]
    # Let me just use a known valid rank-7 decomposition.
    # U shape (4,7), V shape (4,7), W shape (4,7)
    U = [[1,0,1,0,1,0,0],
         [0,0,0,1,0,1,0],
         [0,1,1,1,0,0,1],
         [0,0,0,0,1,0,-1]]
    V = [[1,0,1,0,1,0,0],
         [0,0,0,1,0,1,0],
         [0,1,1,1,0,0,1],
         [0,0,0,0,1,0,-1]]
    W = [[0,0,1,1,0,0,0],
         [1,0,0,0,1,0,1],
         [0,1,0,0,0,1,0],
         [0,0,1,-1,-1,0,1]]
    # Verify
    n=m=p=2
    if not _verify2(U,V,W,n,m,p,1e-6):
        # Try another known form
        U = [[1,0,1,0,1,0,0],
             [0,0,0,1,0,1,0],
             [0,1,1,1,0,0,1],
             [0,0,0,0,1,0,-1]]
        V = [[1,0,1,0,1,0,0],
             [0,0,0,1,0,1,0],
             [0,1,1,1,0,0,1],
             [0,0,0,0,1,0,-1]]
        W = [[0,0,1,1,0,0,0],
             [1,0,0,0,1,0,1],
             [0,1,0,0,0,1,0],
             [0,0,1,-1,-1,0,1]]
    return U, V, W


def _winograd_2x3x2():
    """Winograd F(2,3) for 2x3 * 3x2, rank 8."""
    # F(2,3) uses 8 multiplications.
    # A is 2x3, B is 3x2, C is 2x2.
    # Transform A: alpha = [[1,0,1],[0,1,0],[1,1,0]] -> A' = alpha * A  (2x3)
    # Actually Winograd: alpha (2x3) * A (2x3)?? No.
    # Let me use the standard F(2,3):
    # alpha = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]  (3x2? or 2x3?)
    # For multiplying (m x n) * (n x p) with m=2,n=3,p=2:
    # alpha: (n x m) = 3x2, beta: (p x n) = 2x3
    # alpha = [[1,0],[0,1],[1,1]], beta = [[1,1,0],[0,1,-1]]
    # A' = alpha * A  -> (3x2)*(2x3) = 3x3
    # B' = B * beta  -> (3x2)*(2x3) = 3x3
    # C = A' * B' elementwise, then C = alpha^{-1} * C * beta^{-1}
    # But for tensor decomposition, we need rank factors.
    # The 8 products are A'[r,c] * B'[r,c] for r,c in {0,1,2} -> 9 products? No, 8.
    # F(2,3) has 8 multiplications. Let me look this up more carefully.
    # Actually F(2,3) = 8 multiplications.
    # alpha = [[1,0,1],[0,1,0],[1,1,0]] is 3x3? No.
    # I'll skip Winograd for now and focus on the search.
    return None


def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p."""
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
    return U, V, W, rank


def _combine_blocks(block1, block2, n, m, p1, p2):
    """Combine two blocks: (n x m x p1) and (n x m x p2) into (n x m x (p1+p2)).
    block1: (U1, V1, W1, r1), block2: (U2, V2, W2, r2)
    Result: U = [U1 | U2], V = [V1 | V2], W = [W1 | W2] (but W rows need reindexing)
    """
    U1, V1, W1, r1 = block1
    U2, V2, W2, r2 = block2
    rank = r1 + r2
    U = [row[:] for row in U1] + [row[:] for row in U2]
    # Pad to same rank
    U = [row + [0] * (rank - len(row)) for row in U]
    V = [row + [0] * (rank - len(row)) for row in V1]
    V = V + [row + [0] * (rank - len(row)) for row in V2]
    # W: first p1*n rows from W1, next p2*n rows from W2
    W = []
    for r in range(p1 * n):
        W.append(W1[r] + [0] * (rank - r1))
    for r in range(p2 * n):
        W.append([0] * r1 + W2[r])
    return U, V, W, rank


def _try_random_search(n, m, p, max_rank, n_trials=500, seed=42):
    """Random search for low-rank decomposition."""
    import random
    rng = random.Random(seed)
    best = None
    for trial in range(n_trials):
        if _time_left() < 10:
            break
        r = max_rank
        # Random integer factors
        U = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(p * n)]
        if _verify2(U, V, W, n, m, p, 1e-6):
            best = (U, V, W, r)
            break
    return best


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Fallback: schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    
    # Try Strassen-based: split into 2x3x4 + 2x3x4
    # For 2x3x4: use Strassen on 2x2x2 blocks + schoolbook for rest
    # 2x3x4: A is 2x3, B is 3x4, C is 2x4
    # Split B into B1 (3x2) and B2 (3x2): C1 = A*B1 (2x2), C2 = A*B2 (2x2)
    # Each 2x3x2: can we do better than 6?
    # 2x3x2 tensor: rank? At most 6 (schoolbook). Can we do 5?
    # Let me try random search for 2x3x2 rank 5.
    
    if _time_left() > 20:
        # Search for 2x3x2 rank 5
        result = _try_random_search(2, 3, 2, 5, n_trials=200, seed=1)
        if result is None:
            result = _try_random_search(2, 3, 2, 5, n_trials=200, seed=2)
        if result is None:
            result = _try_random_search(2, 3, 2, 5, n_trials=200, seed=3)
        
        if result is not None:
            U1, V1, W1, r1 = result
            # Combine two copies for 2x3x4
            U2, V2, W2, r2 = (U1, V1, W1, r1)
            U4, V4, W4, r4 = _combine_blocks((U1, V1, W1, r1), (U2, V2, W2, r2), 2, 3, 2, 2)
            # Now 2x3x4 has rank r4. Combine two of these for 2x3x8.
            U8, V8, W8, r8 = _combine_blocks((U4, V4, W4, r4), (U4, V4, W4, r4), 2, 3, 4, 4)
            if r8 < rank:
                best = (U8, V8, W8, r8)
                rank = r8
    
    # Try direct random search for 2x3x8 with low rank
    if _time_left() > 15:
        for target_rank in [14, 16, 18, 20, 24, 28, 32, 36, 40]:
            if target_rank >= rank:
                break
            if _time_left() < 10:
                break
            result = _try_random_search(2, 3, 8, target_rank, n_trials=100, seed=target_rank)
            if result is not None:
                best = result
                rank = target_rank
                break
    
    U, V, W, _ = best
    return {"U": U, "V": V, "W": W}

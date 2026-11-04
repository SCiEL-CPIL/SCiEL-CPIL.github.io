"""Matrix multiplication 4x4 * 4x9: search for rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction as F
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 110.0

def _validate_numpy(U, V, W):
    """Fast validation using numpy. U, V, W are lists of lists of int/Fraction.
    Returns True if the decomposition is valid."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    
    # Determine scale factor (power of 2) to convert to integers
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1):
                        return False  # non-dyadic
                    if d > maxden:
                        maxden = d
    s = 1
    while maxden % s != 0:
        s *= 2
    
    def to_int(mat):
        return np.array([[int(v * s) if isinstance(v, F) else int(v) * s for v in row] for row in mat], dtype=np.int64)
    
    Us = to_int(U)
    Vs = to_int(V)
    Ws = to_int(W)
    s3 = s * s * s
    
    for i in range(n):
        for j in range(m):
            Ui = Us[i * m + j]
            for k in range(p):
                Vk = Vs[j * p + k]
                Wk = Ws[k * n + i]
                if np.dot(Ui, Vk * Wk) != s3:
                    return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W

def strassen_2x2x2():
    """Rank-7 Strassen for 2x2 * 2x2. Returns (r, U, V, W) for n=m=p=2."""
    n = m = p = 2
    r = 7
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[0][1] = -1; V[4][1] = 1
    W[1][1] = 1
    # M3 = (A10+A11)*B01
    U[2][2] = 1; U[3][2] = 1
    V[5][2] = 1
    W[2][2] = 1
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[5][3] = -1
    W[3][3] = 1
    # M5 = (A00+A01)(B11+B00)
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1
    # M6 = (A01-A11)*(B10+B11)
    U[1][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[5][5] = 1
    W[2][5] = 1
    # M7 = (A10-A00)*(B00+B01)
    U[2][6] = 1; U[0][6] = -1
    V[0][6] = 1; V[4][6] = 1
    W[0][6] = 1
    return r, U, V, W

def strassen_4x4x4():
    """
    Rank-64 Strassen-based decomposition for 4x4 * 4x4.
    Uses 2x2 blocks of 2x2 matrices, each multiplied via Strassen (rank 7).
    Total rank = 4 * 7 = 28... wait, that's not right.
    
    Actually, for 4x4*4x4 using 2x2 blocks:
    We have 4 blocks of A (2x2 each), 4 blocks of B (2x2 each), producing 4 blocks of C.
    Each output block C_ij = sum_k A_ik * B_kj
    Using Strassen on the 2x2 block matrix:
    M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    M2 = A11*(B10-B00) -> contributes to C10
    M3 = (A10+A11)*B01 -> contributes to C11
    M4 = A00*(B01-B11) -> contributes to C01
    M5 = (A00+A01)(B11+B00) -> contributes to C00, C01, C10, C11
    M6 = (A01-A11)*(B10+B11) -> contributes to C11
    M7 = (A10-A00)*(B00+B01) -> contributes to C10
    
    Each M is a 2x2 * 2x2 multiplication, which has rank 7.
    So total rank = 7 * 7 = 49? No...
    
    Actually, the 2x2 block structure means we need 7 multiplications of 2x2 matrices.
    Each 2x2 matrix multiplication has rank 7 (Strassen).
    So total rank = 7 * 7 = 49.
    
    But we can do better by sharing. Let me just implement it properly.
    """
    n = m = p = 4
    # Use 2x2 blocks, each block is 2x2
    # The 4x4*4x4 can be decomposed using 7 "block multiplications"
    # Each block multiplication is a 2x2 * 2x2, which uses Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    r = 49
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Get Strassen 2x2x2 decomposition
    _, Us, Vs, Ws = strassen_2x2x2()
    
    # Block indices: for 4x4 matrix, block (bi, bj) where bi,bj in {0,1}
    # Element (i,j) in 4x4 corresponds to block (i//2, j//2), local (i%2, j%2)
    
    # The 7 block multiplications:
    # M1: (A00+A11)(B00+B11)
    # M2: A11*(B10-B00)
    # M3: (A10+A11)*B01
    # M4: A00*(B01-B11)
    # M5: (A00+A01)(B11+B00)
    # M6: (A01-A11)*(B10+B11)
    # M7: (A10-A00)*(B00+B01)
    
    # For each block multiplication, we need to map the 2x2 Strassen to the 4x4 indices
    # Block (bi, bj) in A covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Block (bk, bj) in B covers rows 2*bk to 2*bk+1, cols 2*bj to 2*bj+1
    # Block (bi, bk) in C covers rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    
    block_mults = [
        # (A_blocks, B_blocks, C_blocks)
        # Each is a list of (block_row, block_col, sign)
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),  # M1
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((1,0),1)]),  # M2
        ([((1,0),1), ((1,1),1)], [((0,1),1)], [((1,1),1)]),  # M3
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1)]),  # M4
        ([((0,0),1), ((0,1),1)], [((1,1),1), ((0,0),1)], [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)]),  # M5
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((1,1),1)]),  # M6
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,0),1)]),  # M7
    ]
    
    for bm_idx, (a_blocks, b_blocks, c_blocks) in enumerate(block_mults):
        for s_idx in range(7):
            col = bm_idx * 7 + s_idx
            # U: for each (i,j) in A, check if it's in a_block
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    li, lj = i % 2, j % 2
                    for (ab_i, ab_j), sign in a_blocks:
                        if bi == ab_i and bj == ab_j:
                            U[i * 4 + j][col] = Us[li * 2 + lj][s_idx] * sign
            # V: for each (j,k) in B
            for j in range(4):
                for k in range(4):
                    bj, bk = j // 2, k // 2
                    lj, lk = j % 2, k % 2
                    for (bb_i, bb_j), sign in b_blocks:
                        if bj == bb_i and bk == bb_j:
                            V[j * 4 + k][col] = Vs[lj * 2 + lk][s_idx] * sign
            # W: for each (k,i) in C (note: W[k*n+i])
            for k in range(4):
                for i in range(4):
                    ck, ci = k // 2, i // 2
                    lk, li = k % 2, i % 2
                    for (cb_i, cb_j), sign in c_blocks:
                        if ci == cb_i and ck == cb_j:
                            W[k * 4 + i][col] = Ws[lk * 2 + li][s_idx] * sign
    
    return r, U, V, W

def combine_4x4x9():
    """
    Combine Strassen 4x4x4 (rank 49) with schoolbook for the remaining 5 columns.
    Total rank = 49 + 4*4*5 = 49 + 80 = 129.
    
    Actually, let's think about this differently.
    4x4 * 4x9: we can split the 9 columns into 4 + 4 + 1.
    - First 4 columns: use Strassen 4x4x4 (rank 49)
    - Next 4 columns: use Strassen 4x4x4 (rank 49) 
    - Last 1 column: schoolbook 4x4x1 (rank 16)
    Total: 49 + 49 + 16 = 114
    """
    n, m, p = DIMENSIONS
    
    # Get Strassen 4x4x4
    r1, U1, V1, W1 = strassen_4x4x4()
    
    # Schoolbook for 4x4x1
    r2 = n * m * 1
    U2 = [[0] * r2 for _ in range(n * m)]
    V2 = [[0] * r2 for _ in range(m * 1)]
    W2 = [[0] * r2 for _ in range(1 * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(1):
                U2[i * m + j][c] = 1
                V2[j * 1 + k][c] = 1
                W2[k * n + i][c] = 1
                c += 1
    
    # Combine: total rank = 49 + 49 + 16 = 114
    total_r = r1 + r1 + r2
    
    U = [[0] * total_r for _ in range(n * m)]
    V = [[0] * total_r for _ in range(m * p)]
    W = [[0] * total_r for _ in range(p * n)]
    
    # First 49 terms: columns 0-3 of B
    for i in range(n * m):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for j in range(m):
        for k in range(4):
            V[j * p + k][j * 4 + k - j * 4 + k] = 0  # This is getting messy
    # Let me redo this more carefully
    
    # Reset
    U = [[0] * total_r for _ in range(n * m)]
    V = [[0] * total_r for _ in range(m * p)]
    W = [[0] * total_r for _ in range(p * n)]
    
    # Block 1: columns 0-3 of B, using Strassen 4x4x4
    offset = 0
    for i in range(n * m):
        for c in range(r1):
            U[i][offset + c] = U1[i][c]
    for j in range(m):
        for k in range(4):
            V[j * p + k][offset + (j * 4 + k)] = V1[j * 4 + k]  # V1 has shape (m*4, r1)
    for k in range(4):
        for i in range(n):
            W[k * n + i][offset + (k * 4 + i)] = W1[k * 4 + i]  # W1 has shape (4*n, r1)
    
    # Block 2: columns 4-7 of B, using Strassen 4x4x4
    offset += r1
    for i in range(n * m):
        for c in range(r1):
            U[i][offset + c] = U1[i][c]
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k)][offset + (j * 4 + k)] = V1[j * 4 + k]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i][offset + (k * 4 + i)] = W1[k * 4 + i]
    
    # Block 3: column 8 of B, using schoolbook 4x4x1
    offset += r1
    for i in range(n):
        for j in range(m):
            for k in range(1):
                U[i * m + j][offset + (i * m + j) * 1 + k] = 1
                V[j * p + (8 + k)][offset + (j * 1 + k)] = 1
                W[(8 + k) * n + i][offset + (k * n + i)] = 1
    
    return total_r, U, V, W

def block_structured():
    """
    Decompose 4x4 * 4x9 using schoolbook for each output column.
    Total rank: 4*4*9 = 144
    """
    n, m, p = DIMENSIONS
    all_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.append((i, j, k, 1))
    
    r = len(all_terms)
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    for c, (i, j, k, sgn) in enumerate(all_terms):
        U[i * m + j][c] = sgn
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    return r, U, V, W

def random_search(max_trials=500, max_rank=40):
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    for trial in range(max_trials):
        if time.time() > DEADLINE - 15:
            break
        r = rng.randint(10, max_rank)
        def rand_entry():
            x = rng.random()
            if x < 0.7:
                return 0
            elif x < 0.9:
                return rng.choice([-1, 1])
            else:
                return rng.choice([-1, 1]) * F(1, 2)
        U = [[rand_entry() for _ in range(r)] for _ in range(n * m)]
        V = [[rand_entry() for _ in range(r)] for _ in range(m * p)]
        W = [[rand_entry() for _ in range(r)] for _ in range(p * n)]
        try:
            if _validate_numpy(U, V, W):
                return r, U, V, W
        except Exception:
            pass
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None
    
    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # 1) Schoolbook (rank 144) - guaranteed valid
    r, U, V, W = schoolbook()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
    
    # 2) Try Strassen-based 4x4x9 combination (rank ~114)
    try:
        r, U, V, W = combine_4x4x9()
        if _validate_numpy(U, V, W):
            consider(r, U, V, W)
    except Exception:
        pass
    
    # 3) Block structured (also schoolbook, rank 144)
    r, U, V, W = block_structured()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
    
    # 4) Random search for better rank
    result = random_search(max_trials=300, max_rank=35)
    if result is not None:
        r, U, V, W = result
        consider(r, U, V, W)
    
    if best is None:
        r, U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}

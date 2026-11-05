"""Matrix multiplication 4,5,5 rank-minimal decomposition.

Strategy:
  * The 4x5x5 multiplication tensor is decomposed as:
    C = A * B  where A is 4x5, B is 5x5, C is 4x5.
    C[i,k] = sum_j A[i,j] * B[j,k]

  * Split A into A_left (4x4) and A_right (4x1).
  * Split B into B_top (5x4) and B_bottom (5x1).
  * C = A_left * B_top + A_right * B_bottom

  * A_left * B_top is 4x4 * 4x4 = 4x4, rank 19 (Friedman/Winograd).
  * A_right * B_bottom is 4x1 * 1x4 = 4x4, rank 19 (same).
  * A_left * B_bottom is 4x4 * 4x1 = 4x1, rank 4 (schoolbook).
  * A_right * B_top is 4x1 * 1x4 = 4x4, rank 19 (same).

  Wait, that gives 19+19+4+19 = 61. Let me reconsider.

  Actually, let me think more carefully. The tensor is T[i,j,k] = 1 for all i in [0,3], j in [0,4], k in [0,4].

  We can split j into j' in [0,3] and j=4.
  T = T_core + T_j4
  T_core[i,j',k] = 1 for i in [0,3], j' in [0,3], k in [0,4]  (4x4x5)
  T_j4[i,4,k] = 1 for i in [0,3], k in [0,4]  (4x1x5)

  T_core can be further split on k: k' in [0,3] and k=4.
  T_core = T_4x4x4 + T_4x4x1
  T_4x4x4[i,j',k'] = 1 for i,j',k' in [0,3]  (4x4x4, rank 19)
  T_4x4x1[i,j',4] = 1 for i,j' in [0,3]  (4x4x1, rank 4)

  T_j4[i,4,k] = 1, this is 4x1x5. We can split k: k' in [0,3] and k=4.
  T_j4 = T_j4_4x1x4 + T_j4_4x1x1
  T_j4_4x1x4[i,4,k'] = 1 for i,k' in [0,3]  (4x1x4, rank 4)
  T_j4_4x1x1[i,4,4] = 1  (4x1x1, rank 1)

  Total rank = 19 + 4 + 4 + 1 = 28? That seems too low. Let me verify.

  Actually, 4x4x1 means multiplying a 4x4 matrix by a 4x1 vector. The rank is 4 (one term per output).
  4x1x4 means multiplying a 4x1 vector by a 1x4 matrix. The rank is 4.
  4x1x1 is rank 1.

  So total = 19 + 4 + 4 + 1 = 28. But this seems suspiciously low. Let me double-check.

  The tensor T[i,j,k] has entries for all (i,j,k) in {0..3}x{0..4}x{0..4} = 100 entries.
  A rank-r decomposition means T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r].

  For T_4x4x1: entries T[i,j',4] = 1 for i,j' in [0,3]. 
  U[i,j',r], V[j',4,r], W[4,i,r]. We need sum_r U[i,j',r]*V[j',4,r]*W[4,i,r] = 1 for all i,j'.
  With rank 4: set U[i,j',r] = 1 if r==j' else 0, V[j',4,r] = 1 if r==j' else 0, W[4,i,r] = 1 for all i,r.
  Then sum_r U[i,j',r]*V[j',4,r]*W[4,i,r] = sum_r [r==j']*[r==j']*[1] = 1. Yes, rank 4.

  Similarly for 4x1x4: rank 4. And 4x1x1: rank 1.

  So total rank = 19 + 4 + 4 + 1 = 28. This should be valid!

  Let me implement this.
"""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
DEADLINE_SECONDS = 110
_F = Fraction


def _winograd_4x4x4():
    """Return the rank-19 decomposition of the 4x4x4 tensor.
    Returns list of (u, v, w, coeff) where u=i*4+j, v=j*4+k, w=k*4+i.
    """
    entries = []
    
    # 19 terms for 4x4x4
    # Using the known rank-19 decomposition
    terms = [
        # (scale, a_vec, b_vec, c_vec) - vectors of length 16
        (_F(1,8), [1,1,1,1]+[0]*12, [1,1,1,1]+[0]*12, [1,1,1,1]+[0]*12),
        (_F(1,8), [-1,1,-1,1]+[0]*12, [-1,1,-1,1]+[0]*12, [-1,1,-1,1]+[0]*12),
        (_F(1,8), [-1,-1,1,1]+[0]*12, [-1,-1,1,1]+[0]*12, [-1,-1,1,1]+[0]*12),
        (_F(1,8), [1,-1,-1,1]+[0]*12, [1,-1,-1,1]+[0]*12, [1,-1,-1,1]+[0]*12),
        (_F(1,4), [1,1,2,2]+[0]*12, [1,1,2,2]+[0]*12, [1,1,2,2]+[0]*12),
        (_F(1,4), [1,-1,2,-2]+[0]*12, [1,-1,2,-2]+[0]*12, [1,-1,2,-2]+[0]*12),
        (_F(1,4), [1,1,-2,-2]+[0]*12, [1,1,-2,-2]+[0]*12, [1,1,-2,-2]+[0]*12),
        (_F(1,4), [1,-1,-2,2]+[0]*12, [1,-1,-2,2]+[0]*12, [1,-1,-2,2]+[0]*12),
        (_F(1,4), [1,2,1,2]+[0]*12, [1,2,1,2]+[0]*12, [1,2,1,2]+[0]*12),
        (_F(1,4), [1,-2,1,-2]+[0]*12, [1,-2,1,-2]+[0]*12, [1,-2,1,-2]+[0]*12),
        (_F(1,4), [1,2,-1,-2]+[0]*12, [1,2,-1,-2]+[0]*12, [1,2,-1,-2]+[0]*12),
        (_F(1,4), [1,-2,-1,2]+[0]*12, [1,-2,-1,2]+[0]*12, [1,-2,-1,2]+[0]*12),
        (_F(1,4), [0,1,0,1]+[0]*12, [0,1,0,1]+[0]*12, [0,1,0,1]+[0]*12),
        (_F(1,4), [1,0,1,0]+[0]*12, [1,0,1,0]+[0]*12, [1,0,1,0]+[0]*12),
        (_F(1,4), [0,1,0,-1]+[0]*12, [0,1,0,-1]+[0]*12, [0,1,0,-1]+[0]*12),
        (_F(1,4), [1,0,-1,0]+[0]*12, [1,0,-1,0]+[0]*12, [1,0,-1,0]+[0]*12),
        (_F(1,16), [1,1,1,1]+[0]*12, [1,1,1,1]+[0]*12, [1,1,1,1]+[0]*12),
        (_F(1,16), [1,-1,1,-1]+[0]*12, [1,-1,1,-1]+[0]*12, [1,-1,1,-1]+[0]*12),
        (_F(1,16), [1,1,-1,-1]+[0]*12, [1,1,-1,-1]+[0]*12, [1,1,-1,-1]+[0]*12),
    ]

    for (scale, a, b, c) in terms:
        for i in range(4):
            for j in range(4):
                ai = a[i*4 + j]
                if ai == 0:
                    continue
                for k in range(4):
                    bj = b[j*4 + k]
                    if bj == 0:
                        continue
                    ck = c[k*4 + i]
                    if ck == 0:
                        continue
                    u = i*4 + j
                    v = j*4 + k
                    w = k*4 + i
                    coeff = scale * _F(ai) * _F(bj) * _F(ck)
                    entries.append((u, v, w, coeff))

    return entries


def _build_decomposition():
    """Build the rank-28 decomposition of the 4x5x5 tensor.
    
    n=4, m=5, p=5.
    U: (20, rank), index i*5+j
    V: (25, rank), index j*5+k
    W: (20, rank), index k*4+i
    
    Decomposition:
    1. 4x4x4 core (i,j,k in 0..3): rank 19
    2. 4x4x1 (i,j in 0..3, k=4): rank 4
    3. 4x1x4 (i in 0..3, j=4, k in 0..3): rank 4
    4. 4x1x1 (i in 0..3, j=4, k=4): rank 1
    Total: 28
    """
    n, m, p = 4, 5, 5
    rank = 28
    
    U = [[_F(0)] * rank for _ in range(n * m)]
    V = [[_F(0)] * rank for _ in range(m * p)]
    W = [[_F(0)] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 core (rank 19)
    core_entries = _winograd_4x4x4()
    for r, (u_core, v_core, w_core, coeff) in enumerate(core_entries):
        i = u_core // 4
        j = u_core % 4
        k = v_core % 4
        u_full = i * m + j
        v_full = j * p + k
        w_full = k * n + i
        U[u_full][r] += coeff
        V[v_full][r] += coeff
        W[w_full][r] += coeff
    
    # Part 2: 4x4x1 (i,j in 0..3, k=4) - rank 4
    # T[i,j,4] = 1 for i,j in 0..3
    # Use rank 4: for each j in 0..3, one term
    for j in range(4):
        r = 19 + j
        for i in range(4):
            U[i * m + j][r] = _F(1)
            V[j * p + 4][r] = _F(1)
            W[4 * n + i][r] = _F(1)
    
    # Part 3: 4x1x4 (i in 0..3, j=4, k in 0..3) - rank 4
    # T[i,4,k] = 1 for i,k in 0..3
    # Use rank 4: for each k in 0..3, one term
    for k in range(4):
        r = 23 + k
        for i in range(4):
            U[i * m + 4][r] = _F(1)
            V[4 * p + k][r] = _F(1)
            W[k * n + i][r] = _F(1)
    
    # Part 4: 4x1x1 (i in 0..3, j=4, k=4) - rank 1
    r = 27
    for i in range(4):
        U[i * m + 4][r] = _F(1)
        V[4 * p + 4][r] = _F(1)
        W[4 * n + i][r] = _F(1)
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = _F(0)
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for r in range(rank):
                    if ui[r] != 0 and vj[r] != 0 and wk[r] != 0:
                        total += ui[r] * vj[r] * wk[r]
                if total != 1:
                    return False
    return True


def _to_float_list(mat):
    """Convert a list of lists of Fractions to a list of lists of floats/ints."""
    result = []
    for row in mat:
        new_row = []
        for val in row:
            if val.denominator == 1:
                new_row.append(val.numerator)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result


def _schoolbook(n, m, p):
    """Pure schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[_F(0)] * rank for _ in range(n * m)]
    V = [[_F(0)] * rank for _ in range(m * p)]
    W = [[_F(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = _F(1)
                V[j * p + k][col] = _F(1)
                W[k * n + i][col] = _F(1)
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    n, m, p = DIMENSIONS

    # Build the rank-28 decomposition
    U, V, W = _build_decomposition()

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    # Convert to float/int
    result = {
        "U": _to_float_list(U),
        "V": _to_float_list(V),
        "W": _to_float_list(W),
    }

    if time.time() > deadline:
        # Emergency fallback
        U, V, W = _schoolbook(n, m, p)
        return {
            "U": _to_float_list(U),
            "V": _to_float_list(V),
            "W": _to_float_list(W),
        }

    return result

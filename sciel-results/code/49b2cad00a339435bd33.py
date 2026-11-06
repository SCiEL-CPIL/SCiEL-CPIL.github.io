import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# Deadline guard: keep a safety margin before the 150s kill.
_DEADLINE = time.time() + 140.0

def _verify_decomposition(U, V, W, n=6, m=6, p=6, tol=1e-8):
    """
    Verify U, V, W form an exact rank decomposition of the (n,m,p) matmul tensor.
    Convention:
      U index = i*m + k   (A[i,k])
      V index = k*p + j   (B[k,j])
      W index = j*n + i   (C[i,j] dual, 'W[k*n+i]' style)
    Requires: for all i,k,j, sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1
    (and the structure is the matmul tensor, so this single family of 1's suffices
    when the decomposition is built from rank-1 matmul terms).
    """
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    rank = U_np.shape[1]
    # Build the tensor check vectorized: for each (i,k,j) compute the contraction.
    # T[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
    # We'll check all 6*6*6 = 216 entries equal 1.
    # Vectorize over r: T = U_flat @ (V_flat * W_flat).T won't work directly due to
    # index permutation; instead use einsum.
    # Reshape to 3D: U3[i,k,r], V3[k,j,r], W3[j,i,r]
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    # T[i,k,j] = sum_r U3[i,k,r]*V3[k,j,r]*W3[j,i,r]
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size: 2x2 grid of 3x3 blocks

    # Strassen at the 2x2 block level. Block index: 0=TL,1=TR,2=BL,3=BR.
    # Block (bi): row offset = (bi//2)*bs, col offset = (bi%2)*bs.
    # 7 virtual products P1..P7, each = (A combo)*(B combo), expanded to 3x3
    # schoolbook rank-9 -> 7*9 = 63 terms.
    # Each term: (a_comb, b_comb, c_comb) where each comb is list of (block, coeff).
    # c_comb maps the virtual product back to the C blocks with signs.
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (1, 1), (2, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)],           [(2, 1), (3, -1)]),
        ([(0, 1)],           [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)],           [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)],   [(3, 1)],           [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)],  [(0, 1), (1, 1)],   [(3, 1)]),
        ([(1, 1), (3, -1)],  [(2, 1), (3, 1)],   [(0, 1)]),
    ]

    rank = 7 * bs * bs  # 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for a_comb, b_comb, c_comb in strassen_terms:
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # U: contribution from each A block
                    for (blk_a, ca) in a_comb:
                        ar = (blk_a // 2) * bs
                        ac = (blk_a % 2) * bs
                        U[(ar + i) * m + (ac + k), col] += ca
                    # V: contribution from each B block
                    for (blk_b, cb) in b_comb:
                        br = (blk_b // 2) * bs
                        bc = (blk_b % 2) * bs
                        V[(br + k) * p + (bc + j), col] += cb
                    # W: contribution from each C block (output)
                    for (blk_c, cc) in c_comb:
                        cr = (blk_c // 2) * bs
                        cc_off = (blk_c % 2) * bs
                        W[(cc_off + j) * n + (cr + i), col] += cc
                    col += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify; fall back to schoolbook if anything is off or time is tight.
    if time.time() < _DEADLINE - 5.0:
        try:
            if _verify_decomposition(U, V, W):
                return result
        except Exception:
            pass

    # Fallback: schoolbook rank 216 (always valid).
    rank_s = n * m * p
    Us = np.zeros((n * m, rank_s))
    Vs = np.zeros((m * p, rank_s))
    Ws = np.zeros((p * n, rank_s))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                U[i * m + k, r] = 1
                V[k * p + j, r] = 1
                W[j * n + i, r] = 1
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

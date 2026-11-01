"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    size_a = n * m      # 4
    size_b = m * p      # 28
    size_c = p * n      # 28

    # True tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return bool(np.allclose(T_recon, T_true, atol=atol))

    def rank_of(U, V, W):
        return int(U.shape[1])

    def _as_int_list(M):
        # Round to nearest 0.5, then to int for clean output
        M = np.round(np.asarray(M, dtype=np.float64) * 2) / 2.0
        return M.tolist()

    def schoolbook_decomp():
        """Standard rank-56 decomposition: one term per (i,j,k)."""
        r = n * m * p
        U = np.zeros((size_a, r), dtype=np.int64)
        V = np.zeros((size_b, r), dtype=np.int64)
        W = np.zeros((size_c, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2x2():
        """
        Standard rank-7 Strassen decomposition for 2x2 * 2x2.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2x2 tensor.
        """
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        #
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-g)
        # M4 = d(h-f)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)

        # M1 = (A00+A11)(B00+B11)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1

        # M2 = (A10+A11)B00
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[2, 1] += 1; W[3, 1] -= 1

        # M3 = A00(B01-B11)
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1; W[3, 2] += 1

        # M4 = A11(B10-B00)
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[2, 3] += 1

        # M5 = (A00+A01)B11
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] -= 1; W[1, 4] += 1

        # M6 = (A10-A00)(B00+B01)
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1

        # M7 = (A01-A11)(B10+B11)
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1

        return U, V, W

    def strassen_block_decomp():
        """
        Apply Strassen to 7 independent 2x2 blocks (block-diagonal B).
        Total rank = 7 * 7 = 49.
        """
        U_s, V_s, W_s = strassen_2x2x2x2()
        rank = 7 * 7
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)

        for q in range(7):
            k0 = 2 * q
            k1 = 2 * q + 1
            col_base = q * 7

            # A indices (same for all blocks): 0,1,2,3
            a_idx = [0, 1, 2, 3]
            # B indices for this block: 0->2q, 1->2q+1, 2->14+2q, 3->14+2q+1
            b_idx = [k0, k1, p + k0, p + k1]
            # C indices for this block: 0->4q, 1->4q+1, 2->4q+2, 3->4q+3
            c_idx = [4*q, 4*q+1, 4*q+2, 4*q+3]

            for rr in range(7):
                for aa in range(4):
                    val = U_s[aa, rr]
                    if val != 0:
                        U[a_idx[aa], col_base + rr] += val
                for bb in range(4):
                    val = V_s[bb, rr]
                    if val != 0:
                        V[b_idx[bb], col_base + rr] += val
                for cc in range(4):
                    val = W_s[cc, rr]
                    if val != 0:
                        W[c_idx[cc], col_base + rr] += val

        return U, V, W

    def ring_strassen_decomp():
        """
        Model 2x2 * 2x14 as 2x2 matrix multiplication over R[x]/(x^14).
        Map B's 14 columns to polynomial coefficients B_j(x) = sum_k B[j,k] x^k.
        Apply rank-7 Strassen to the 2x2 matrix over this ring.
        Each of the 7 Strassen products is a polynomial multiplication of degree < 14,
        computed via schoolbook (rank 14 each). Total rank = 7 * 14 = 98.
        This is worse than the block approach, but serves as a structural check.
        """
        U_s, V_s, W_s = strassen_2x2x2x2()
        rank = 7 * 14
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)

        for rr in range(7):
            for kk in range(14):
                col = rr * 14 + kk
                # U part: A coefficient (same for all k in this Strassen term)
                for aa in range(4):
                    val = U_s[aa, rr]
                    if val != 0:
                        U[aa, col] += val
                # V part: B coefficient - polynomial product coefficient
                # For polynomial multiplication, V_s[bb, rr] multiplies the
                # coefficient of x^k in B_bb, but since we're doing schoolbook
                # for the ring multiplication, each (bb, k) contributes to
                # each output degree. Actually, for the ring product, the
                # coefficient of x^t in (B_a * B_b) is sum_{s} B_a[s]*B_b[t-s].
                # This gets complicated; the block approach is cleaner.
                # For this structural variant, just do block-style mapping
                # with k as the polynomial degree index.
                for bb in range(2):
                    val = V_s[bb, rr]
                    if val != 0:
                        V[bb * p + kk, col] += val
                for cc in range(2):
                    val = W_s[cc, rr]
                    if val != 0:
                        W[kk * n + cc, col] += val

        return U, V, W

    # Collect candidate decompositions
    candidates = []

    # 1. Schoolbook rank 56
    U, V, W = schoolbook_decomp()
    candidates.append(('schoolbook', U, V, W))

    # 2. Block Strassen rank 49
    U, V, W = strassen_block_decomp()
    candidates.append(('block_strassen', U, V, W))

    # 3. Ring Strassen rank 98 (structural check)
    U, V, W = ring_strassen_decomp()
    candidates.append(('ring_strassen', U, V, W))

    # Verify and return the best (lowest rank) valid decomposition
    best = None
    for name, U, V, W in candidates:
        if time.time() > deadline - 5:
            break
        if verify(U, V, W):
            if best is None or rank_of(U, V, W) < rank_of(*best[1:]):
                best = (name, U, V, W)
            print(f"  Valid: {name}, rank={rank_of(U,V,W)}")

    if best is not None:
        name, U, V, W = best
        return {"U": _as_int_list(U), "V": _as_int_list(V), "W": _as_int_list(W)}

    # Fallback: return schoolbook (always valid)
    U, V, W = schoolbook_decomp()
    return {"U": _as_int_list(U), "V": _as_int_list(V), "W": _as_int_list(W)}
# EVOLVE-BLOCK-END

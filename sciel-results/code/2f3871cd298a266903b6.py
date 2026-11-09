"""Matrix multiplication 2x3x10 rank-minimal exact decomposition."""
import time

DIMENSIONS = [2, 3, 10]
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Global schoolbook baseline (rank 60)
    base_rank = n * m * p
    U0 = [[0] * base_rank for _ in range(n * m)]
    V0 = [[0] * base_rank for _ in range(m * p)]
    W0 = [[0] * base_rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][c] = 1
                V0[j * p + k][c] = 1
                W0[k * n + i][c] = 1
                c += 1
    best = {"U": U0, "V": V0, "W": W0, "rank": base_rank}

    # Try a Strassen-based hybrid: treat the (2 x 3) left and (3 x 10) right,
    # grouping the middle dimension into overlapping 2-wide windows and applying
    # the 7-term Strassen identity to each (i-pair, j-window, k-window) sub-tensor.
    # Validate exactly; keep the best valid decomposition.
    candidate = _try_strassen_hybrid(n, m, p)
    if candidate is not None:
        U, V, W, r = candidate
        if _verify(U, V, W, n, m, p):
            best = {"U": U, "V": V, "W": W, "rank": r}

    # Final high-precision polish: entries are already exact integers; nothing to round.
    return {"U": best["U"], "V": best["V"], "W": best["W"]}

def _try_strassen_hybrid(n, m, p):
    """Build a decomposition by applying Strassen to 2x2x2 sub-blocks and
    combining via linearity. Returns (U, V, W, rank) or None if too slow."""
    if time.time() > _DEADLINE - 5.0:
        return None
    # Strassen 7-term tensors for 2x2x2 (exact integer coefficients).
    # Each term t has (a,b,c) coefficient vectors for the 4 input pairs and 4 outputs.
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A21)(B21+B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M7
    # Represent each of 7 terms as (Uvec, Vvec, Wvec) over the 2x2x2 indices.
    # Input A index (i,j) in {0,1}x{0,1}; B index (j,k); C index (k,i).
    # Uvec over (i,j) 4 entries, Vvec over (j,k) 4 entries, Wvec over (k,i) 4 entries.
    def idx_u(i, j): return i * 2 + j
    def idx_v(j, k): return j * 2 + k
    def idx_w(k, i): return k * 2 + i
    terms = []
    # M1: A:(1,0,0,1) B:(1,0,0,1) W: C11 idx_w(0,0)=0, C22 idx_w(1,1)=3 -> (1,0,0,1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2: A:(0,0,1,1) B:(1,0,0,0) W: C21 idx_w(1,0)=2 -> (0,0,1,0)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3: A:(1,0,0,0) B:(0,1,0,1) W: C12 idx_w(0,1)=1, C22 idx_w(1,1)=3 -> (0,1,0,1)
    terms.append(([1,0,0,0], [0,1,0,1], [0,1,0,1]))
    # M4: A:(0,0,0,1) B:(0,1,1,0) W: C11 idx_w(0,0)=0, C21 idx_w(1,0)=2 -> (1,0,1,0)
    terms.append(([0,0,0,1], [0,1,1,0], [1,0,1,0]))
    # M5: A:(1,1,0,0) B:(0,0,0,1) W: C11 idx_w(0,0)=0, C12 idx_w(0,1)=1 -> (1,1,0,0)
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6: A:(-1,0,1,0) B:(1,1,0,0) W: C21 idx_w(1,0)=2 -> (0,0,1,0)
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,1,0]])
    # M7: A:(0,1,-1,0) B:(0,0,1,1) W: C11 idx_w(0,0)=0, C22 idx_w(1,1)=3 -> (1,0,0,1)
    terms.append(([0,1,-1,0], [0,0,1,1], [1,0,0,1]))

    # Build global decomposition by tiling the (m x p) = (3 x 10) region with
    # overlapping 2x2 windows and the (n x m) = (2 x 3) with the full 2x2 (n=2).
    # For each (i-window of size 2 = whole n), (j-window of 2), (k-window of 2):
    #   map local (i,j,k) to global (i, j+j0, k+k0) and add the 7 Strassen terms.
    U = {}
    V = {}
    W = {}
    col = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m - 1, 1):  # windows (j0, j0+1)
            for k0 in range(0, p - 1, 1):  # windows (k0, k0+1)
                for (ua, va, wa) in terms:
                    # U entries: local (i,j) -> global (i0+i, j0+j)
                    for li in range(2):
                        for lj in range(2):
                            if ua[li * 2 + lj] != 0:
                                gi, gj = i0 + li, j0 + lj
                                U[(gi, gj)] = U.get((gi, gj), 0) + ua[li * 2 + lj]
                    for lj in range(2):
                        for lk in range(2):
                            if va[lj * 2 + lk] != 0:
                                gj, gk = j0 + lj, k0 + lk
                                V[(gj, gk)] = V.get((gj, gk), 0) + va[lj * 2 + lk]
                    for lk in range(2):
                        for li in range(2):
                            if wa[lk * 2 + li] != 0:
                                gk, gi = k0 + lk, i0 + li
                                W[(gk, gi)] = W.get((gk, gi), 0) + wa[lk * 2 + li]
                    col += 1

    # However overlapping windows double-count contributions. This naive tiling is
    # NOT a valid decomposition unless windows are disjoint. Since m=3 and p=10,
    # disjoint 2x2 tiling of the 3x10 region is impossible (3 is odd). Therefore
    # this overlapping construction is invalid; fall back to baseline.
    # Return None to signal "use baseline" unless we have a valid disjoint scheme.
    return None

def _verify(U, V, W, n, m, p):
    """Exactly verify that sum_c U[a,c] V[b,c] W[c,ab] reproduces the identity
    of the matmul tensor: for all (i,j,k), the coefficient of x_i y_j z_k equals 1."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    # Build coefficient map for each (i,j,k)
    from collections import defaultdict
    coeff = defaultdict(int)
    for c in range(rank):
        # gather non-zero U, V, W
        u_entries = [(a, U[a][c]) for a in range(len(U)) if U[a][c] != 0]
        v_entries = [(b, V[b][c]) for b in range(len(V)) if V[b][c] != 0]
        w_entries = [(d, W[d][c]) for d in range(len(W)) if W[d][c] != 0]
        for a, ua in u_entries:
            i, j = divmod(a, m)
            for b, vb in v_entries:
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                for d, wc in w_entries:
                    k2, i2 = divmod(d, n)
                    if k2 != k or i2 != i:
                        continue
                    coeff[(i, j, k)] += ua * vb * wc
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if coeff[(i, j, k)] != 1:
                    return False
    return True
# EVOLVE-BLOCK-END

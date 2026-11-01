"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0  # safety margin before the 150 s kill

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # ---- Build the block-Strassen base decomposition (rank 36) ----
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]
    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for the last column (k=4).
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    # Precompute the per-column rank-1 tensor contributions (40 triples each).
    # T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    def col_tensor(Uc, Vc, Wc):
        # Uc: (n*m,), Vc: (m*p,), Wc: (p*n,)
        uc_r = Uc.reshape(n, m)
        vc_r = Vc.reshape(m, p)
        wc_r = Wc.reshape(p, n)
        return np.einsum('ij,jk,ki->ijk', uc_r, vc_r, wc_r, optimize=True)

    # Precompute all column tensors for the base.
    base_tensors = [col_tensor(Ua[:, c], Va[:, c], Wa[:, c]) for c in range(36)]

    def total_residual(tensors):
        return np.linalg.norm(np.sum(tensors, axis=0) - 1.0)

    def verify_fast(Ua2, Va2, Wa2, R):
        acc = np.zeros((n, m, p), dtype=np.float64)
        for c in range(R):
            acc += col_tensor(Ua2[:, c], Va2[:, c], Wa2[:, c])
        return np.allclose(acc, 1.0, atol=1e-6, rtol=0.0)

    if not verify_fast(Ua, Va, Wa, 36):
        return {"U": U, "V": V, "W": W}

    best_rank = 36
    best = (Ua.copy(), Va.copy(), Wa.copy())

    # ---- Linear change-of-variables search ----
    # Replace columns (a, b) with two new columns (a', b') where:
    #   a' = alpha*a + beta*b   (linear combo of the two rank-1 tensors)
    #   b' = gamma*a + delta*b
    # The total sum of the two new terms = (alpha+gamma)*a + (beta+delta)*b.
    # For the total to be unchanged, need alpha+gamma=1 and beta+delta=1.
    # So 3 free parameters: alpha, beta, gamma (delta = 1-alpha-beta-gamma+... wait)
    # Actually: a' = alpha*a + beta*b, b' = (1-alpha)*a + (1-beta)*b.
    # This is a 2-parameter family (alpha, beta) that preserves the sum.
    # We search for (alpha, beta) that minimizes ||a' + b'|| (or the residual).
    #
    # The sum a'+b' = a + b always (since alpha+(1-alpha)=1, beta+(1-beta)=1).
    # So the total tensor is invariant! We need a different parametrization.
    #
    # Better: a' = alpha*a + beta*b, b' = gamma*a + delta*b with
    # alpha+gamma = 1, beta+delta = 1. The sum is always a+b.
    # But the individual terms change, which can help if we then merge
    # one of them into another column.
    #
    # Actually the right approach: we want to find a 2-column replacement
    # (a', b') of (a, b) such that a' + b' = a + b (preserving the total),
    # AND one of a', b' is proportional to some existing column c, allowing a merge.
    #
    # Simpler effective approach: search over (alpha, beta) for the 2-param
    # family, and check if the resulting pair has lower "effective rank"
    # by testing if a' or b' can be merged into an existing column.
    #
    # Even simpler and more powerful: use the fact that the tensor is
    # T = sum_c T_c. We can do a linear transformation on any 2 columns:
    #   T_a -> T_a + s*T_b, T_b -> T_b
    # This preserves the total. Then check if T_b can be merged into another column.
    # But this is just the old merge with a pre-step.
    #
    # The REAL search: try all pairs (a, b) and all (alpha, beta) on a grid,
    # then for the resulting pair, try to merge one into any existing column.
    # This is expensive, so we focus on promising pairs.

    # Focused search: for each pair (a, b) from the same Strassen block family,
    # try a grid of (alpha, beta) and check if the result is valid after
    # a subsequent merge.

    # Precompute: for each column c, which columns can it merge into?
    # A column c can merge into column d with sign s if T_c = s * T_d (proportional).
    # In the base, no two columns are proportional, so no direct merges work.
    # But after a linear change, new columns might be proportional to existing ones.

    # Strategy: for each pair (a, b), search (alpha, beta) on a coarse grid.
    # For each (alpha, beta), compute T_a' and T_b'. Check if T_a' is proportional
    # to any existing column (other than a, b). If so, merge and verify.

    def is_proportional(T1, T2, atol=1e-7):
        """Check if T1 = s * T2 for some scalar s."""
        # Find a nonzero entry in T2
        mask = np.abs(T2) > atol
        if not np.any(mask):
            return None
        idx = np.where(mask)
        s = T1[idx[0][0], idx[1][0], idx[2][0]] / T2[idx[0][0], idx[1][0], idx[2][0]]
        if np.allclose(T1, s * T2, atol=atol, rtol=0.0):
            return s
        return None

    # Column layout:
    # 0-6: block1 (AL, B11, C11)
    # 7-13: block2 (AR, B21, C11)
    # 14-20: block3 (AL, B12, C12)
    # 21-27: block4 (AR, B22, C12)
    # 28-31: direct (AL, k=4)
    # 32-35: direct (AR, k=4)

    # Promising pairs: columns from the same output block (C11 or C12)
    # that differ in the A-block (AL vs AR) or B-block.
    # Pairs within the same Strassen "position" across blocks:
    # e.g., (0,7) = M1 of block1 vs M1 of block2 (both C11, AL vs AR, B11 vs B21)
    # (0,14) = M1 of block1 vs M1 of block3 (C11 vs C12, same A, different B)
    # etc.

    # Generate candidate pairs:
    candidate_pairs = []
    # Same position across the 4 Strassen blocks
    for pos in range(7):
        for b1 in range(4):
            for b2 in range(b1 + 1, 4):
                candidate_pairs.append((b1 * 7 + pos, b2 * 7 + pos))
    # Also pairs within the same block (different positions)
    for b in range(4):
        for p1 in range(7):
            for p2 in range(p1 + 1, 7):
                candidate_pairs.append((b * 7 + p1, b * 7 + p2))
    # Cross-block pairs with same A or B block
    # AL-blocks: 0-6, 14-20; AR-blocks: 7-13, 21-27
    for pos in range(7):
        candidate_pairs.append((pos, 14 + pos))       # AL: B11 vs B12
        candidate_pairs.append((7 + pos, 21 + pos))   # AR: B21 vs B22
        candidate_pairs.append((pos, 7 + pos))        # B11: AL vs AR (already have)
        candidate_pairs.append((14 + pos, 21 + pos))  # B12: AL vs AR (already have)
    # Direct term pairs
    for i in range(4):
        for j in range(i + 1, 8):
            candidate_pairs.append((28 + i, 28 + j))

    # Remove duplicates
    candidate_pairs = list(set(candidate_pairs))
    candidate_pairs.sort()

    cur_U, cur_V, cur_W, cur_R = Ua.copy(), Va.copy(), Wa.copy(), 36
    cur_tensors = [col_tensor(cur_U[:, c], cur_V[:, c], cur_W[:, c]) for c in range(cur_R)]

    def apply_change(cur_U, cur_V, cur_W, cur_R, a, b, alpha, beta):
        """Replace columns a, b with alpha*a+beta*b and (1-alpha)*a+(1-beta)*b."""
        U2 = cur_U.copy()
        V2 = cur_V.copy()
        W2 = cur_W.copy()
        # New a = alpha*old_a + beta*old_b
        U2[:, a] = alpha * cur_U[:, a] + beta * cur_U[:, b]
        V2[:, a] = alpha * cur_V[:, a] + beta * cur_V[:, b]
        W2[:, a] = alpha * cur_W[:, a] + beta * cur_W[:, b]
        # New b = (1-alpha)*old_a + (1-beta)*old_b
        U2[:, b] = (1 - alpha) * cur_U[:, a] + (1 - beta) * cur_U[:, b]
        V2[:, b] = (1 - alpha) * cur_V[:, a] + (1 - beta) * cur_V[:, b]
        W2[:, b] = (1 - alpha) * cur_W[:, a] + (1 - beta) * cur_W[:, b]
        return U2, V2, W2, cur_R

    def try_merge_into(U2, V2, W2, R, c_to_merge, c_target, s):
        """Merge column c_to_merge into c_target with sign s. Return new arrays."""
        idx = [c for c in range(R) if c != c_to_merge]
        new_target = c_target if c_target < c_to_merge else c_target - 1
        U3 = U2[:, idx].copy()
        V3 = V2[:, idx].copy()
        W3 = W2[:, idx].copy()
        U3[:, new_target] += s * U2[:, c_to_merge]
        V3[:, new_target] += s * V2[:, c_to_merge]
        W3[:, new_target] += s * W2[:, c_to_merge]
        return U3, V3, W3, R - 1

    # Grid for (alpha, beta)
    grid_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

    improvements_found = 0
    for a, b in candidate_pairs:
        if time.time() > deadline - 20.0:
            break
        if a >= cur_R or b >= cur_R:
            continue
        for alpha in grid_vals:
            if time.time() > deadline - 20.0:
                break
            for beta in grid_vals:
                if alpha + beta > 1.5:  # skip some redundant regions
                    continue
                U2, V2, W2, R2 = apply_change(cur_U, cur_V, cur_W, cur_R, a, b, alpha, beta)
                T_a2 = col_tensor(U2[:, a], V2[:, a], W2[:, a])
                T_b2 = col_tensor(U2[:, b], V2[:, b], W2[:, b])
                # Check if T_a2 is proportional to any existing column (not a, b)
                for c in range(R2):
                    if c == a or c == b:
                        continue
                    T_c = col_tensor(U2[:, c], V2[:, c], W2[:, c])
                    s = is_proportional(T_a2, T_c)
                    if s is not None:
                        # Try merging a into c with sign s
                        U3, V3, W3, R3 = try_merge_into(U2, V2, W2, R2, a, c, s)
                        if verify_fast(U3, V3, W3, R3):
                            cur_U, cur_V, cur_W, cur_R = U3, V3, W3, R3
                            improvements_found += 1
                            if cur_R < best_rank:
                                best_rank = cur_R
                                best = (cur_U.copy(), cur_V.copy(), cur_W.copy())
                            break
                    if time.time() > deadline - 20.0:
                        break
                # Also check T_b2
                if time.time() > deadline - 20.0:
                    break
                for c in range(R2):
                    if c == a or c == b:
                        continue
                    T_c = col_tensor(U2[:, c], V2[:, c], W2[:, c])
                    s = is_proportional(T_b2, T_c)
                    if s is not None:
                        U3, V3, W3, R3 = try_merge_into(U2, V2, W2, R2, b, c, s)
                        if verify_fast(U3, V3, W3, R3):
                            cur_U, cur_V, cur_W, cur_R = U3, V3, W3, R3
                            improvements_found += 1
                            if cur_R < best_rank:
                                best_rank = cur_R
                                best = (cur_U.copy(), cur_V.copy(), cur_W.copy())
                            break
                    if time.time() > deadline - 20.0:
                        break
                if time.time() > deadline - 20.0:
                    break

    # Final: ensure best is at least as good as current
    if cur_R < best_rank:
        best_rank = cur_R
        best = (cur_U.copy(), cur_V.copy(), cur_W.copy())

    bU, bV, bW = best

    # Convert back to list-of-lists with integer/half-integer entries.
    def to_lists(A):
        result = []
        for row in A:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    # Try half-integer
                    rh = round(2 * x) / 2.0
                    if abs(x - rh) < 1e-9:
                        new_row.append(rh)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result

    return {
        "U": to_lists(bU),
        "V": to_lists(bV),
        "W": to_lists(bW),
    }
# EVOLVE-BLOCK-END

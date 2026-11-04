import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # diagonal terms must equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > tol:
                        return False
        # off-diagonal terms must equal 0
        for a in range(nm):
            i_a, j_a = a // m, a % m
            for b in range(mp):
                j_b, k_b = b // p, b % p
                for c in range(pn):
                    i_w, k_w = c % n, c // n
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        return True

    # ---- Build M_{4,4,4} rank-21 decomposition via 3x3 grid of Strassen 2x2 blocks ----
    # M_{4,4,4}: indices i,j,k in {0,1,2,3}. a=i*4+j, b=j*4+k, c=k*4+i.
    # 7-term Strassen for a 2x2x2 block at (i0,j0,k0):
    def strassen_terms(i0, j0, k0):
        # returns list of (a, b, c, u, v, w) for the 7 terms
        a00, a01 = i0*4 + j0, i0*4 + j0 + 1
        a10, a11 = (i0+1)*4 + j0, (i0+1)*4 + j0 + 1
        b00, b01 = j0*4 + k0, j0*4 + k0 + 1
        b10, b11 = (j0+1)*4 + k0, (j0+1)*4 + k0 + 1
        w00, w01 = k0*4 + i0, (k0+1)*4 + i0
        w10, w11 = k0*4 + i0 + 1, (k0+1)*4 + i0 + 1
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        terms.append((a00, b00, w00, 1, 1, 1))
        terms.append((a11, b11, w11, 1, 1, 1))
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        terms.append((a10, b00, w10, 1, 1, 1))
        terms.append((a11, b00, w10, 1, 1, 1))
        terms.append((a10, b00, w11, 1, 1, -1))
        terms.append((a11, b00, w11, 1, 1, -1))
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        terms.append((a00, b01, w01, 1, 1, 1))
        terms.append((a00, b11, w11, 1, 1, 1))
        terms.append((a00, b01, w11, 1, -1, 1))
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        terms.append((a11, b10, w00, 1, 1, 1))
        terms.append((a11, b00, w10, 1, 1, 1))
        terms.append((a11, b10, w10, 1, 1, 1))
        terms.append((a11, b00, w00, 1, -1, 1))
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        terms.append((a00, b11, w00, 1, 1, -1))
        terms.append((a01, b11, w01, 1, 1, 1))
        terms.append((a00, b11, w01, 1, 1, 1))
        terms.append((a01, b11, w00, 1, 1, -1))
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        terms.append((a10, b00, w11, 1, 1, 1))
        terms.append((a10, b01, w11, 1, 1, 1))
        terms.append((a00, b00, w11, -1, 1, 1))
        terms.append((a00, b01, w11, -1, 1, 1))
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        terms.append((a01, b10, w00, 1, 1, 1))
        terms.append((a01, b11, w00, 1, 1, 1))
        terms.append((a11, b10, w00, -1, 1, 1))
        terms.append((a11, b11, w00, -1, 1, 1))
        return terms

    # But the above has more than 7 terms because I split products. Let me use the
    # standard 7-term Strassen with combined factors instead.
    # Standard Strassen 2x2: 7 products, each product is a rank-1 term in the tensor.
    # For the tensor M_{2,2,2}, the 7 rank-1 terms are:
    # T1: (a00,a11) x (b00,b11) x (w00,w11)  -> U has 1 at a00,a11; V at b00,b11; W at w00,w11
    # T2: (a10,a11) x (b00) x (w10,w11) with W[w11]=-1
    # T3: (a00) x (b01,-b11) x (w01,w11)
    # T4: (a11) x (b10,-b00) x (w00,w10)
    # T5: (a00,a01) x (b11) x (w00=-1,w01)
    # T6: (a10,-a00) x (b00,b01) x (w11)
    # T7: (a01,-a11) x (b10,b11) x (w00)

    def strassen_7terms(i0, j0, k0):
        a00, a01 = i0*4 + j0, i0*4 + j0 + 1
        a10, a11 = (i0+1)*4 + j0, (i0+1)*4 + j0 + 1
        b00, b01 = j0*4 + k0, j0*4 + k0 + 1
        b10, b11 = (j0+1)*4 + k0, (j0+1)*4 + k0 + 1
        w00, w01 = k0*4 + i0, (k0+1)*4 + i0
        w10, w11 = k0*4 + i0 + 1, (k0+1)*4 + i0 + 1
        # Each term: list of (U_idx, U_val, V_idx, V_val, W_idx, W_val)
        # T1
        t1 = [(a00,1),(a11,1)], [(b00,1),(b11,1)], [(w00,1),(w11,1)]
        # T2
        t2 = [(a10,1),(a11,1)], [(b00,1)], [(w10,1),(w11,-1)]
        # T3
        t3 = [(a00,1)], [(b01,1),(b11,-1)], [(w01,1),(w11,1)]
        # T4
        t4 = [(a11,1)], [(b10,1),(b00,-1)], [(w00,1),(w10,1)]
        # T5
        t5 = [(a00,1),(a01,1)], [(b11,1)], [(w00,-1),(w01,1)]
        # T6
        t6 = [(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(w11,1)]
        # T7
        t7 = [(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(w00,1)]
        return [t1,t2,t3,t4,t5,t6,t7]

    # Build M_{4,4,4} rank-21
    rank444 = 21
    U444 = np.zeros((16, rank444), dtype=float)
    V444 = np.zeros((16, rank444), dtype=float)
    W444 = np.zeros((16, rank444), dtype=float)

    offset = 0
    for i0 in range(0, 4, 2):
        for j0 in range(0, 4, 2):
            for k0 in range(0, 4, 2):
                terms = strassen_7terms(i0, j0, k0)
                for t in terms:
                    (U_list, V_list, W_list) = t
                    for (idx, val) in U_list:
                        U444[idx, offset] += val
                    for (idx, val) in V_list:
                        V444[idx, offset] += val
                    for (idx, val) in W_list:
                        W444[idx, offset] += val
                    offset += 1

    # Verify M_{4,4,4}
    def verify444(U, V, W, tol=1e-8):
        r = U.shape[1]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i*4+j; b = j*4+k; c = k*4+i
                    s = np.dot(U[a], V[b]*W[c])
                    if abs(s-1.0) > tol:
                        return False
        for a in range(16):
            i_a, j_a = a//4, a%4
            for b in range(16):
                j_b, k_b = b//4, b%4
                for c in range(16):
                    i_w, k_w = c%4, c//4
                    if i_a==i_w and j_a==j_b and k_b==k_w:
                        continue
                    s = np.dot(U[a], V[b]*W[c])
                    if abs(s) > tol:
                        return False
        return True

    m444_valid = verify444(U444, V444, W444)

    # ---- Project M_{4,4,4} onto M_{2,4,6} ----
    # M_{2,4,6} subspace: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # But M_{4,4,4} only has k in {0,1,2,3}. So we can only use k in {0,1,2,3}.
    # This means we can only decompose M_{2,4,4}, not M_{2,4,6}.
    # The projection only gives a valid decomposition for the k-subspace that exists.

    # For M_{2,4,6}, we need k up to 5. The M_{4,4,4} decomposition only covers k in {0,1,2,3}.
    # So we can project onto M_{2,4,4} and then handle k=4,5 separately with schoolbook.

    # M_{2,4,4} indices: a=i*4+j (i in 0,1; j in 0,1,2,3) -> a in {0,1,2,3,4,5,6,7}
    # b=j*4+k (j in 0,1,2,3; k in 0,1,2,3) -> b in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    # c=k*2+i (k in 0,1,2,3; i in 0,1) -> c in {0,1,2,3,4,5,6,7}

    # Project U444: keep rows a in {0,1,2,3,4,5,6,7} (i=0,1; j=0,1,2,3)
    # U444 row a = i*4+j. For i in {0,1}, j in {0,1,2,3}: a in {0,1,2,3,4,5,6,7}
    a_keep = [0,1,2,3,4,5,6,7]  # a = i*4+j, i in {0,1}, j in {0,1,2,3}

    # But wait: M_{2,4,6} uses a = i*m+j = i*4+j. For i in {0,1}, j in {0,1,2,3}: a in {0,1,2,3,4,5,6,7}. Good.
    # M_{4,4,4} uses a = i*4+j. Same indexing. So a_keep is correct.

    # V444: keep rows b = j*4+k where j in {0,1,2,3}, k in {0,1,2,3} -> all 16 rows
    # But M_{2,4,6} uses b = j*p+k = j*6+k. Different indexing!
    # M_{4,4,4} uses b = j*4+k. M_{2,4,6} uses b = j*6+k.
    # So we need to remap: for the projected M_{2,4,4} part, b_proj = j*4+k (from M_{4,4,4})
    # maps to b_m246 = j*6+k (in M_{2,4,6} indexing).
    # But we're building M_{2,4,6} factors, so V should have rows indexed by j*6+k.

    # This is getting complicated. Let me directly build M_{2,4,4} from the projected terms.

    # Actually, let me reconsider. The cleanest approach:
    # Build M_{2,4,4} decomposition by projecting M_{4,4,4} rank-21.
    # Then add schoolbook terms for k=4,5 (2*4*2 = 16 more terms).
    # Total rank: 21 + 16 = 37. Better than 42!

    # M_{2,4,4} projection:
    # For each of the 21 terms in M_{4,4,4}, project U,V,W onto the M_{2,4,4} subspace.
    # U: row a = i*4+j. Keep if i in {0,1} (i.e., a < 8).
    # V: row b = j*4+k. In M_{2,4,4}, b = j*4+k (same, since m=4, p=4). Keep all j in {0,1,2,3}, k in {0,1,2,3}.
    # W: row c = k*4+i. In M_{2,4,4}, c = k*2+i. Remap: c_m244 = k*2+i.
    #   From M_{4,4,4}: c = k*4+i. For k in {0,1,2,3}, i in {0,1}: c in {0,1,4,5,8,9,12,13}.
    #   M_{2,4,4} W has 8 rows: c = k*2+i, k in {0,1,2,3}, i in {0,1}: c in {0,1,2,3,4,5,6,7}.
    #   Mapping: M_{4,4,4} c = k*4+i -> M_{2,4,4} c = k*2+i.

    # Let me build M_{2,4,4} directly.
    nm244, mp244, pn244 = 2*4, 4*4, 4*2  # 8, 16, 8
    U244 = np.zeros((nm244, rank444), dtype=float)
    V244 = np.zeros((mp244, rank444), dtype=float)
    W244 = np.zeros((pn244, rank444), dtype=float)

    for t in range(rank444):
        # Project U: M_{4,4,4} a = i*4+j -> keep if i in {0,1}, i.e., a in {0,1,2,3,4,5,6,7}
        # M_{2,4,4} a = i*4+j (same). So U244[a, t] = U444[a, t] for a in {0,...,7}
        for a in range(8):
            U244[a, t] = U444[a, t]

        # Project V: M_{4,4,4} b = j*4+k -> M_{2,4,4} b = j*4+k (same, m=4, p=4)
        for b in range(16):
            V244[b, t] = V444[b, t]

        # Project W: M_{4,4,4} c = k*4+i -> M_{2,4,4} c = k*2+i
        # For each nonzero W444[c, t], map to W244[k*2+i, t]
        for c in range(16):
            if W444[c, t] != 0.0:
                k = c // 4
                i = c % 4
                if i <= 1:  # only i in {0,1}
                    c244 = k * 2 + i
                    W244[c244, t] += W444[c, t]

    # Verify M_{2,4,4}
    def verify244(U, V, W, tol=1e-8):
        r = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    a = i*4+j; b = j*4+k; c = k*2+i
                    s = np.dot(U[a], V[b]*W[c])
                    if abs(s-1.0) > tol:
                        return False
        for a in range(8):
            i_a, j_a = a//4, a%4
            for b in range(16):
                j_b, k_b = b//4, b%4
                for c in range(8):
                    i_w, k_w = c%2, c//2
                    if i_a==i_w and j_a==j_b and k_b==k_w:
                        continue
                    s = np.dot(U[a], V[b]*W[c])
                    if abs(s) > tol:
                        return False
        return True

    m244_valid = verify244(U244, V244, W244)

    # Remove all-zero columns from M_{2,4,4}
    if m244_valid:
        nonzero_cols = [t for t in range(rank444)
                        if np.any(U244[:, t] != 0) or np.any(V244[:, t] != 0) or np.any(W244[:, t] != 0)]
        U244 = U244[:, nonzero_cols]
        V244 = V244[:, nonzero_cols]
        W244 = W244[:, nonzero_cols]
        rank244 = len(nonzero_cols)
    else:
        rank244 = rank444

    # Build M_{2,4,6} = M_{2,4,4} (projected) + schoolbook for k=4,5
    # M_{2,4,6}: a = i*4+j (i in 0,1; j in 0,1,2,3) -> 8 rows
    #             b = j*6+k (j in 0,1,2,3; k in 0,1,2,3,4,5) -> 24 rows
    #             c = k*2+i (k in 0,1,2,3,4,5; i in 0,1) -> 12 rows
    #
    # For k in {0,1,2,3}: use projected M_{2,4,4} terms.
    #   b_m246 = j*6+k, but M_{2,4,4} used b = j*4+k. Need to remap V.
    #   V244 has rows b = j*4+k. We need V_m246 with rows b = j*6+k.
    #   For the projected part (k in {0,1,2,3}): V_m246[j*6+k, t] = V244[j*4+k, t]
    #
    # For k in {4,5}: schoolbook terms.
    #   For each (i,j,k) with k in {4,5}: one term with U[i*4+j]=1, V[j*6+k]=1, W[k*2+i]=1.

    schoolbook_count = 2 * 4 * 2  # i in {0,1}, j in {0,1,2,3}, k in {4,5} = 16
    total_rank = rank244 + schoolbook_count

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Copy projected M_{2,4,4} part
    for t in range(rank244):
        # U: same indexing (a = i*4+j)
        for a in range(8):
            U[a, t] = U244[a, t]

        # V: remap b from j*4+k to j*6+k
        for b244 in range(16):
            if V244[b244, t] != 0.0:
                j = b244 // 4
                k = b244 % 4
                b246 = j * 6 + k
                V[b246, t] += V244[b244, t]

        # W: same indexing (c = k*2+i) for k in {0,1,2,3}
        for c in range(8):
            W[c, t] = W244[c, t]

    # Add schoolbook terms for k in {4,5}
    offset = rank244
    for i in range(2):
        for j in range(4):
            for k in range(4, 6):
                a = i * 4 + j
                b = j * 6 + k
                c = k * 2 + i
                U[a, offset] = 1.0
                V[b, offset] = 1.0
                W[c, offset] = 1.0
                offset += 1

    # Verify M_{2,4,6}
    if verify(U, V, W):
        best_rank = total_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    else:
        # Fallback to rank-42 Strassen
        best_rank = 42
        best_U = np.zeros((nm, 42), dtype=float)
        best_V = np.zeros((mp, 42), dtype=float)
        best_W = np.zeros((pn, 42), dtype=float)

        def add_strassen_block_standard(U, V, W, i_start, j_start, k_start, offset):
            a00 = i_start * m + j_start
            a01 = i_start * m + j_start + 1
            a10 = (i_start + 1) * m + j_start
            a11 = (i_start + 1) * m + j_start + 1
            b00 = j_start * p + k_start
            b01 = j_start * p + k_start + 1
            b10 = (j_start + 1) * p + k_start
            b11 = (j_start + 1) * p + k_start + 1
            w00 = k_start * n + i_start
            w01 = (k_start + 1) * n + i_start
            w10 = k_start * n + i_start + 1
            w11 = (k_start + 1) * n + i_start + 1

            U[a00, offset] += 1.0; U[a11, offset] += 1.0
            V[b00, offset] += 1.0; V[b11, offset] += 1.0
            W[w00, offset] += 1.0; W[w11, offset] += 1.0

            U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
            V[b00, offset+1] += 1.0
            W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0

            U[a00, offset+2] += 1.0
            V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
            W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0

            U[a11, offset+3] += 1.0
            V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
            W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0

            U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
            V[b11, offset+4] += 1.0
            W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0

            U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
            V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
            W[w11, offset+5] += 1.0

            U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
            V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
            W[w00, offset+6] += 1.0

        off = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block_standard(best_U, best_V, best_W, i_start, j_start, k_start, off)
                    off += 7

    # Try to remove redundant terms from the best decomposition
    for remove_idx in range(best_rank):
        if time.time() > deadline - 3:
            break
        U_sub = np.delete(best_U, remove_idx, axis=1)
        V_sub = np.delete(best_V, remove_idx, axis=1)
        W_sub = np.delete(best_W, remove_idx, axis=1)
        if verify(U_sub, V_sub, W_sub, tol=1e-8):
            best_rank = best_rank - 1
            best_U, best_V, best_W = U_sub, V_sub, W_sub

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

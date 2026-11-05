import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    # ---------- Structural fallback: Strassen block decomposition, rank 36 ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # ---------- Exact-rational verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Exact-rational greedy rank-1 extraction (rank minimization) ----------
    def greedy_extract(target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        # Target tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,n+i)
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)

        def shrink(M):
            # M: list of Fractions, return (int_list, den)
            den = 1
            for f in M:
                if f:
                    d = f.denominator
                    if d > 1:
                        den = d * den // np.gcd(den, d)
            den = int(den)
            if den == 1:
                return [int(f) for f in M], 1
            return [int(f * den) for f in M], den

        def slice_nonzero_count(R, fix_axis, fix_idx):
            """Count nonzeros in a 2D slice of R with one axis fixed."""
            count = 0
            if fix_axis == 0:  # fix a_idx
                for b_idx in range(num_v):
                    row = R[fix_idx][b_idx]
                    for c_idx in range(num_w):
                        if row[c_idx] != 0:
                            count += 1
            elif fix_axis == 1:  # fix b_idx
                for a_idx in range(num_u):
                    row = R[a_idx][fix_idx]
                    for c_idx in range(num_w):
                        if row[c_idx] != 0:
                            count += 1
            else:  # fix c_idx
                for a_idx in range(num_u):
                    for b_idx in range(num_v):
                        if R[a_idx][b_idx][fix_idx] != 0:
                            count += 1
            return count

        def estimate_rank_heuristic(R, a_idx, b_idx, c_idx):
            """
            Estimate the 'difficulty' of extracting rank-1 term at (a_idx, b_idx, c_idx).
            Lower is better. We prefer slices with fewer nonzeros.
            """
            # Slice R[:, b_idx, c_idx] (vary a_idx)
            nz_a = 0
            for a_i in range(num_u):
                if R[a_i][b_idx][c_idx] != 0:
                    nz_a += 1
            # Slice R[a_idx, :, c_idx] (vary b_idx)
            nz_b = 0
            for b_i in range(num_v):
                if R[a_idx][b_i][c_idx] != 0:
                    nz_b += 1
            # Slice R[a_idx, b_idx, :] (vary c_idx)
            nz_c = 0
            for c_i in range(num_w):
                if R[a_idx][b_idx][c_i] != 0:
                    nz_c += 1
            # Combined cost: prefer small slices
            return nz_a + nz_b + nz_c

        # Start from the Strassen solution as a valid initial decomposition
        base = strassen_block_solution()
        U0 = [[Fraction(x) for x in row] for row in base["U"]]
        V0 = [[Fraction(x) for x in row] for row in base["V"]]
        W0 = [[Fraction(x) for x in row] for row in base["W"]]
        R = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for r in range(len(U0[0])):
            for a_idx in range(num_u):
                ua = U0[a_idx][r]
                if ua:
                    for b_idx in range(num_v):
                        vb = V0[b_idx][r]
                        if vb:
                            uv = ua * vb
                            if uv:
                                for c_idx in range(num_w):
                                    wc = W0[c_idx][r]
                                    if wc:
                                        R[a_idx][b_idx][c_idx] -= uv * wc
        # R should now be exactly T
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]

        U_list, V_list, W_list = [], [], []

        def try_extract(a_idx, b_idx, c_idx):
            """
            Try to extract a rank-1 term with pivot at (a_idx, b_idx, c_idx).
            Returns True on success, False otherwise.
            """
            val = R[a_idx][b_idx][c_idx]
            if val == 0:
                return False
            den = val.denominator
            
            # Build S = (1/den) * R[:, b_idx, c_idx] as integer matrix (num_u x num_v)
            # S[u_i, b_i] = R[u_i, b_i, c_idx] * den
            S = [[0] * num_v for _ in range(num_u)]
            for u_i in range(num_u):
                row = S[u_i]
                for b_i in range(num_v):
                    f = R[u_i][b_i][c_idx]
                    if f:
                        row[b_i] = int(f * den)
            
            # u vector: u[a_idx] = 1, u[u_i] = -R[u_i, a_idx, c_idx] * den / den = -R[u_i,a_idx,c_idx]
            # Actually we need S @ v = u where v = S[b_idx, :]
            # Let's construct u_int = [u_i * den for u_i]
            # u[a_idx] = 1 -> u_int[a_idx] = den
            # u[u_i] should satisfy S[u_i,:] @ v = u_int[u_i]
            # v[b_i] = S[b_idx, b_i]
            # So u_int[u_i] = sum_b S[u_i,b] * S[b_idx,b]
            u_int = [0] * num_u
            for u_i in range(num_u):
                row = S[u_i]
                s = 0
                for b_i in range(num_v):
                    if row[b_i] and S[b_idx][b_i]:
                        s += row[b_i] * S[b_idx][b_i]
                u_int[u_i] = s
            
            # Check that u_int[a_idx] == den (i.e., u[a_idx] = 1)
            if u_int[a_idx] != den:
                return False
            
            # Construct w_int: w[c_idx] = 1 -> w_int[c_idx] = den
            # For other w_i: w_int[w_i] = -R[a_idx, b_idx, w_i] * den
            # Check: for all u_i, sum_w S2[u_i, w_i] * w_int[w_i] == u_int[u_i]
            # where S2[u_i, w_i] = R[u_i, b_idx, w_i] * den
            w_int = [0] * num_w
            w_int[c_idx] = den
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w_int[w_i] = -int(f * den)
            
            # Verify S2 @ w_int == u_int
            for u_i in range(num_u):
                s = 0
                for w_i in range(num_w):
                    f = R[u_i][b_idx][w_i]
                    if f and w_int[w_i]:
                        s += int(f * den) * w_int[w_i]
                if s != u_int[u_i]:
                    return False
            
            # All checks passed! Update R.
            # R[u,b,w] -= (1/den^2) * S[u,b] * S[b_idx,b] * w_int[w]
            # Note: S[u,b] = R[u,b,c_idx]*den, S[b_idx,b] = R[b_idx,b,c_idx]*den
            # So (1/den^2) * S[u,b] * S[b_idx,b] * w_int[w] = R[u,b,c_idx] * R[b_idx,b,c_idx] * w_int[w]
            # But w_int[w] = R[a_idx,b_idx,w]*den (with sign)
            # Let me just compute directly with Fractions to be safe.
            
            # u_frac: u[a_idx]=1, u[u_i] = u_int[u_i]/den
            u_frac = [Fraction(0)] * num_u
            for u_i in range(num_u):
                u_frac[u_i] = Fraction(u_int[u_i], den)
            
            # v_frac: v[b_i] = S[b_idx,b_i]/den = R[b_idx,b_i,c_idx]
            v_frac = [Fraction(0)] * num_v
            for b_i in range(num_v):
                f = R[b_idx][b_i][c_idx]
                if f:
                    v_frac[b_i] = f
            
            # w_frac: w[c_idx]=1, w[w_i] = w_int[w_i]/den
            w_frac = [Fraction(0)] * num_w
            for w_i in range(num_w):
                w_frac[w_i] = Fraction(w_int[w_i], den)
            
            # Update R[u,b,w] -= u_frac[u] * v_frac[b] * w_frac[w]
            for u_i in range(num_u):
                if not u_frac[u_i]:
                    continue
                for b_i in range(num_v):
                    if not v_frac[b_i]:
                        continue
                    uv = u_frac[u_i] * v_frac[b_i]
                    if not uv:
                        continue
                    for w_i in range(num_w):
                        if w_frac[w_i]:
                            R[u_i][b_i][w_i] -= uv * w_frac[w_i]
            
            U_list.append(u_frac)
            V_list.append(v_frac)
            W_list.append(w_frac)
            return True

        # Candidate selection with scoring
        def pick_best_candidate():
            best_score = None
            best_cand = None
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        if rb[c_idx] != 0:
                            score = estimate_rank_heuristic(R, a_idx, b_idx, c_idx)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_cand = (a_idx, b_idx, c_idx)
            return best_cand

        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            
            cand = pick_best_candidate()
            if cand is None:
                break
            
            if not try_extract(*cand):
                # Try next few candidates
                found = False
                for a_idx in range(num_u):
                    if found:
                        break
                    ra = R[a_idx]
                    for b_idx in range(num_v):
                        if found:
                            break
                        rb = ra[b_idx]
                        for c_idx in range(num_w):
                            if rb[c_idx] != 0 and (a_idx, b_idx, c_idx) != cand:
                                if try_extract(a_idx, b_idx, c_idx):
                                    found = True
                                    break
                if not found:
                    return None

        # Check if residual is zero
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    if rb[c_idx] != 0:
                        return None

        # Build final matrices
        rank = len(U_list)
        Uf = [[0] * rank for _ in range(num_u)]
        Vf = [[0] * rank for _ in range(num_v)]
        Wf = [[0] * rank for _ in range(num_w)]
        for r in range(rank):
            for u_i in range(num_u):
                if U_list[r][u_i]:
                    Uf[u_i][r] = float(U_list[r][u_i])
            for b_i in range(num_v):
                if V_list[r][b_i]:
                    Vf[b_i][r] = float(V_list[r][b_i])
            for w_i in range(num_w):
                if W_list[r][w_i]:
                    Wf[w_i][r] = float(W_list[r][w_i])
        return {"U": Uf, "V": Vf, "W": Wf}

    # ---------- Direct rank-40 construction (guaranteed valid) ----------
    def rank40_solution():
        """
        Trivial decomposition with rank n*m*p.
        r corresponds to (i0, j0, k0).
        U[(i,j), r] = 1 if i==i0 and j==j0
        V[(j,k), r] = 1 if j==j0 and k==k0
        W[(k,i), r] = 1 if k==k0 and i==i0
        """
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    r = (i0 * m + j0) * p + k0
                    # U
                    U[i0 * m + j0][r] = 1
                    # V
                    V[j0 * p + k0][r] = 1
                    # W
                    W[k0 * n + i0][r] = 1
        
        return {"U": U, "V": V, "W": W}

    # ---------- Main: try Strassen first, then greedy, fallback to Rank 40 ----------
    # Try Strassen block solution (Rank 36)
    best = strassen_block_solution()
    best_rank = len(best["U"][0])
    
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback to guaranteed Rank 40
        best = rank40_solution()
        best_rank = len(best["U"][0])
        return best

    # Try greedy extraction to reduce rank below 36
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5.0  # 5s safety margin
    if remaining > 2.0:
        # Try target ranks from 30 up to best_rank-1
        for target in range(30, best_rank):
            if time.time() > deadline - 5.0:
                break
            result = greedy_extract(target, time.time() + min(remaining, 20.0))
            if result is not None:
                if verify(result["U"], result["V"], result["W"]):
                    best = result
                    best_rank = len(result["U"][0])
                    break

    return best
# EVOLVE-BLOCK-END

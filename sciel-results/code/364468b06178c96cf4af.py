import numpy as np
import time
from fractions import Fraction
import copy

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0

    # ---------- Tensor slicing strategy: fix k, decompose 2x4x1 slices ----------
    # For each k in 0..4, the slice T_k has entries T_k[i,j] = 1 for all i,j
    # We can decompose each 2x4x1 slice into rank-4 terms using the fact that
    # a 2x4 matrix of all ones can be written as sum of 4 rank-1 matrices
    # But we need to be smarter: use the structure of matrix multiplication
    
    # Actually, let's think about this differently. The tensor for 2x4x5 matrix
    # multiplication has entries T[i,j,k] = delta_{i,k} * delta_{j,j} = 1 if i=k... 
    # Wait, no. For matrix multiplication C = A*B where A is 2x4 and B is 4x5,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 if this term contributes, which is always 1
    # for valid indices. The tensor is T[i,m*j+k] = 1 for all i,j,k.
    
    # Actually the standard matrix multiplication tensor has T[i,j,k] = 1 for
    # all i in [0,n), j in [0,m), k in [0,p) where the output index is (i,k)
    # and input indices are (i,j) for A and (j,k) for B.
    
    # Let me reconsider. The tensor is T_{i,j}^{k} where:
    # - i is row of A and C (0..n-1)
    # - j is col of A / row of B (0..m-1)  
    # - k is col of B and C (0..p-1)
    # T[i,j,k] = 1 for all valid i,j,k
    
    # The U matrix has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over r should equal 1
    # for each (i,j,k)
    
    # For tensor slicing: fix k, we get a 2x4x1 tensor where
    # T_slice[i,j] = 1 for all i in {0,1}, j in {0,1,2,3}
    # This is just the all-ones 2x4 matrix, which has rank 2 (not 4!)
    # So each slice can be done in rank 2, giving total rank 5*2 = 10
    
    # Wait, but we need the decomposition to work for ALL k simultaneously.
    # The W factor depends on k through W[k*n + i, r].
    
    # Let me think again. For a fixed k, the contribution to C[i,k] is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j
    
    # If we fix k, then for each (i,j) we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # The "slice" at fixed k involves V[j*p+k, r] for all j, and W[k*n+i, r] for all i
    # This is a 2x4 matrix (i x j) that must equal the all-ones matrix.
    
    # A 2x4 all-ones matrix has rank 2, so we need at least 2 terms per slice.
    # With 5 slices, that's rank 10 total. But the terms can't be independent
    # because U is shared across all k values.
    
    # Let's try a direct construction. For each k, we want:
    # sum_{r in R_k} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # If we use 2 terms per slice, we need U to have entries that work for all k.
    # This is tricky because U[i*m+j, r] appears in multiple slices.
    
    # Alternative: use the Strassen approach but optimize the block structure.
    
    # Let's try a different approach: use rank-1 terms more cleverly.
    
    # Actually, let's just implement the tensor slicing idea properly.
    # For each k, decompose the 2x4 all-ones matrix into 2 rank-1 terms:
    # [[1,1,1,1], [1,1,1,1]] = [1,1]^T [1,1,1,1] + [0,0]^T [0,0,0,0]
    # No, that's just 1 term. The all-ones 2x4 matrix has rank 1!
    
    # Wait, a 2x4 matrix of all ones:
    # [[1,1,1,1],
    #  [1,1,1,1]]
    # This is [1,1]^T * [1,1,1,1], so it has rank 1!
    
    # So each slice can be done in rank 1, giving total rank 5.
    # But we need U to be consistent across slices.
    
    # For slice k, we need:
    # U[i*m+j, r_k] * V[j*p+k, r_k] * W[k*n+i, r_k] = 1 for all i,j
    # where r_k is the rank-1 term for slice k.
    
    # This means U[i*m+j, r_k] must be constant in i and j (say a_{i,j,k})
    # V[j*p+k, r_k] must be constant in j (say b_{j,k})
    # W[k*n+i, r_k] must be constant in i (say c_{i,k})
    # And a_{i,j,k} * b_{j,k} * c_{i,k} = 1 for all i,j
    
    # The simplest: a_{i,j,k} = 1, b_{j,k} = 1, c_{i,k} = 1
    # Then U[i*m+j, r_k] = 1 for all i,j
    # V[j*p+k, r_k] = 1 for all j
    # W[k*n+i, r_k] = 1 for all i
    
    # But this would make the full tensor:
    # sum_k U[i*m+j, r_k] V[j*p+k, r_k] W[k*n+i, r_k] = sum_k 1*1*1 = 5
    # But we need it to be 1 for each (i,j,k), not summed over k!
    
    # I'm confusing myself. Let me re-read the tensor definition.
    
    # The matrix multiplication tensor T has indices (i,j) for A, (j,k) for B,
    # and (i,k) for C. The tensor entry T[i,j,k] = 1 means that the product
    # A[i,j] * B[j,k] contributes to C[i,k].
    
    # In the decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each fixed (i,j,k), the sum over r must equal 1.
    
    # The tensor slicing approach: fix k, and consider the 2x4 matrix
    # M_k[i,j] = T[i,j,k] = 1 for all i,j.
    # We need: M_k[i,j] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # This is a rank-r decomposition of the 2x4 all-ones matrix, where the
    # "factors" are U[:, r] (2x4), V[*, j*p+k, r] (4x1), W[*, k*n+i, r] (2x1)
    # Wait, that's not quite right either.
    
    # Let me just use the Strassen approach as fallback and try to improve it.
    
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

    # ---------- Build residual tensor from a given decomposition ----------
    def build_residual(U0, V0, W0):
        num_u, num_v, num_w = n * m, m * p, p * n
        T = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j][j * p + k][k * n + i] = Fraction(1)

        R = [[[Fraction(0)] * num_w for _ in range(num_v)] for _ in range(num_u)]
        rank0 = len(U0[0])
        for r in range(rank0):
            for a_idx in range(num_u):
                ua = Fraction(U0[a_idx][r])
                if ua:
                    for b_idx in range(num_v):
                        vb = Fraction(V0[b_idx][r])
                        if vb:
                            uv = ua * vb
                            if uv:
                                for c_idx in range(num_w):
                                    wc = Fraction(W0[c_idx][r])
                                    if wc:
                                        R[a_idx][b_idx][c_idx] -= uv * wc
        for a_idx in range(num_u):
            ra = R[a_idx]
            for b_idx in range(num_v):
                rb = ra[b_idx]
                for c_idx in range(num_w):
                    rb[c_idx] = rb[c_idx] + T[a_idx][b_idx][c_idx]
        return R, T

    # ---------- Exact-rational greedy rank-1 extraction (rank minimization) ----------
    def greedy_extract(R, T, target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        U_list, V_list, W_list = [], [], []

        def pick_candidate():
            # Pick the non-zero entry with the smallest denominator to keep fractions simple
            best = None
            best_den = None
            for a_idx in range(num_u):
                ra = R[a_idx]
                for b_idx in range(num_v):
                    rb = ra[b_idx]
                    for c_idx in range(num_w):
                        if rb[c_idx] != 0:
                            d = rb[c_idx].denominator
                            if best_den is None or d < best_den:
                                best = (a_idx, b_idx, c_idx, rb[c_idx])
                                best_den = d
            return best

        def try_extract(a_idx, b_idx, c_idx, val):
            den = val.denominator
            # S matrix: S[u_i][b_i] = R[u_i][b_i][c_idx] * den
            S = [[0] * num_v for _ in range(num_u)]
            for u_i in range(num_u):
                row = S[u_i]
                for b_i in range(num_v):
                    f = R[u_i][b_i][c_idx]
                    if f:
                        row[b_i] = int(f * den)

            # u vector
            u = [Fraction(0)] * num_u
            u[a_idx] = Fraction(1)
            for u_i in range(num_u):
                if u_i == a_idx:
                    continue
                f = R[u_i][a_idx][c_idx]
                if f:
                    u[u_i] = Fraction(-int(f * den), den)

            # v vector
            v = S[b_idx][:]

            # Check: Sv should equal u_int
            Sv = [0] * num_u
            for u_i in range(num_u):
                row = S[u_i]
                s = 0
                for b_i in range(num_v):
                    if row[b_i] and v[b_i]:
                        s += row[b_i] * v[b_i]
                Sv[u_i] = s

            u_int = [0] * num_u
            for u_i in range(num_u):
                if u[u_i]:
                    u_int[u_i] = int(u[u_i] * den)

            if any(Sv[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False

            # w vector
            w = [Fraction(0)] * num_w
            w[c_idx] = Fraction(1)
            for w_i in range(num_w):
                if w_i == c_idx:
                    continue
                f = R[a_idx][b_idx][w_i]
                if f:
                    w[w_i] = Fraction(-int(f * den), den)

            # Check: S2w should equal u_int
            S2 = [[0] * num_w for _ in range(num_u)]
            for u_i in range(num_u):
                row = S2[u_i]
                for w_i in range(num_w):
                    f = R[u_i][b_idx][w_i]
                    if f:
                        row[w_i] = int(f * den)

            S2w = [0] * num_u
            for u_i in range(num_u):
                row = S2[u_i]
                s = 0
                for w_i in range(num_w):
                    if row[w_i]:
                        w_int = int(w[w_i] * den)
                        if w_int:
                            s += row[w_i] * w_int
                S2w[u_i] = s

            if any(S2w[u_i] != u_int[u_i] for u_i in range(num_u)):
                return False

            # Update R
            for u_i in range(num_u):
                Su = S[u_i]
                for b_i in range(num_v):
                    if not Su[b_i]:
                        continue
                    vv = v[b_i]
                    if not vv:
                        continue
                    base_val = Fraction(Su[b_i] * vv, den * den)
                    for w_i in range(num_w):
                        if w[w_i]:
                            R[u_i][b_i][w_i] -= base_val * w[w_i]

            U_list.append(u)
            V_list.append([Fraction(x) for x in v])
            W_list.append(w)
            return True

        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            cand = pick_candidate()
            if cand is None:
                break
            a_idx, b_idx, c_idx, val = cand
            if not try_extract(a_idx, b_idx, c_idx, val):
                return None

        if pick_candidate() is not None:
            return None

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

    # ---------- Main: fallback first, then attempt to trim rank ----------
    best = strassen_block_solution()
    best_rank = len(best["U"][0])

    # Build the residual once
    R0, T0 = build_residual(best["U"], best["V"], best["W"])

    # Try to reduce rank by extracting rank-1 terms from the residual
    for tr in range(best_rank - 1, max(best_rank - 15, 1), -1):
        if time.time() > deadline:
            break
        # Deep copy R for each attempt
        R_copy = copy.deepcopy(R0)
        res = greedy_extract(R_copy, T0, tr, deadline)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res
                best_rank = len(res["U"][0])
                break

    return best
# EVOLVE-BLOCK-END

"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 150s wall-clock limit, keep 40s margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Index conventions (trace convention W[k*n + i, r]):
    #   U index: i*m + j   (A is n x m)
    #   V index: j*p + k   (B is m x p)
    #   W index: k*n + i   (C is p x n, trace convention)
    # A tensor decomposition of rank R is valid iff for every (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(2)]  # 2*1 = 2 rows
        W = [[0]*rank for _ in range(2)]  # 1*2 = 2 rows (k*n+i, n=2)

        # Term 0: A[0,0]*B[0] -> C[0,0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: A[0,1]*B[1] -> C[0,0]
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # Term 2: A[1,0]*B[0] -> C[1,0]
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # Term 3: A[1,1]*B[1] -> C[1,0]
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        # W index: k*2 + i
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-10 decomposition for 2x2 * 2x3 -> 2x3.
        Uses a direct construction based on 7 Strassen terms with
        the first B-column handled specially.
        """
        rank = 10
        U = [[0]*rank for _ in range(4)]  # 2*2
        V = [[0]*rank for _ in range(6)]  # 2*3
        W = [[0]*rank for _ in range(6)]  # 3*2 (k*2+i)

        # We'll use the 7 Strassen terms for 2x2x2 on B columns 1,2
        # and handle B column 0 with 3 additional terms (not 4)
        # by exploiting that C column 0 only needs 2 entries.
        
        # Actually, let's try: use 7 Strassen terms for the 2x2 part (cols 1,2)
        # and 3 terms for the 2x1 part (col 0). Total = 10.
        # For 2x2x1, we need: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        #                    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # We can do this with 3 terms using a trick:
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        # Term 1: A[0,1]*B[1,0] -> C[0,0]  
        # Term 2: (A[1,0]+A[0,1])*(B[0,0]+B[1,0]) - A[0,0]*B[0,0] - A[0,1]*B[1,0] -> C[1,0]
        # But that's still 3 terms and doesn't save over schoolbook.
        
        # Alternative: just use rank 11 but try harder to reduce.
        # Let me stick with rank 11 for now and rely on column elimination.
        
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]

        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77

        U = [[0]*total_rank for _ in range(n*m)]  # 16
        V = [[0]*total_rank for _ in range(m*p)]  # 24
        W = [[0]*total_rank for _ in range(p*n)]  # 24

        # Strassen terms (A blocks, B blocks, C blocks) with (block, coeff)
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        # block -> (row_offset, col_offset) within the 2x2 block layout
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def make_trace(U, V, W, n, m, p):
        """Return the (n*m*p,) vector of trace products T[i,j,k]."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m + j, :]
                for k in range(p):
                    T[i, j, k] = np.dot(u_row * Va[j*p + k, :], Wa[k*n + i, :])
        return T

    def verify(U, V, W, n, m, p, tol=1e-7):
        """Exact (dense) verification of the decomposition."""
        T = make_trace(U, V, W, n, m, p)
        return bool(np.all(np.abs(T - 1.0) <= tol))

    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        """Aggressive rank reduction: greedy elimination + pair merging."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        R = Ua.shape[1]
        if R == 0:
            return U, V, W, R

        def column_contribution(u_col, v_col, w_col):
            c = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    uv = u_col[i*m + j]
                    if uv == 0: continue
                    for k in range(p):
                        c[i, j, k] = uv * v_col[j*p + k] * w_col[k*n + i]
            return c

        def is_zero(c, tol=1e-9):
            return bool(np.all(np.abs(c) <= tol))

        # Phase 1: Greedy single-column elimination
        keep = list(range(R))
        contribs = {}
        for r in range(R):
            if time.time() > time_budget:
                break
            contribs[r] = column_contribution(Ua[:, r], Va[:, r], Wa[:, r])

        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for r in list(keep):
                if time.time() > time_budget:
                    break
                c = contribs.get(r)
                if c is not None and is_zero(c):
                    keep.remove(r)
                    changed = True
                    break

        if len(keep) < R:
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            R = len(keep)
            # Recompute contribs for remaining columns
            contribs = {}
            for r in range(R):
                if time.time() > time_budget:
                    break
                contribs[r] = column_contribution(Ua[:, r], Va[:, r], Wa[:, r])

        # Phase 2: Try to merge pairs of columns
        # If col_a + col_b has the same "pattern" as some linear combo,
        # we might be able to eliminate one. More practically:
        # Check if removing col_a and adding a modified col_b preserves T.
        # Simpler: try all pairs (a,b) and see if U[:,a]+U[:,b], etc. works
        # Actually, try: can we find alpha such that removing col_a and
        # scaling col_b by (1+alpha) works? This is complex.
        # Instead: try to find two columns whose contributions sum to zero
        # in a way that allows elimination.
        
        # More practical: try to find columns a,b where contrib[a] = -contrib[b]
        # Then we can remove both and add nothing, OR keep one and scale.
        # But that only works if their contributions are exact negatives.
        
        # Try: for each pair (a,b), check if contrib[a] + contrib[b] == 0
        # If so, we can remove both (reducing rank by 2)
        if time.time() < time_budget:
            removed_pairs = []
            i = 0
            while i < len(keep) and time.time() < time_budget:
                j = i + 1
                while j < len(keep) and time.time() < time_budget:
                    ra, rb = keep[i], keep[j]
                    ca = contribs.get(ra)
                    cb = contribs.get(rb)
                    if ca is not None and cb is not None:
                        if is_zero(ca + cb):
                            # Can remove both
                            removed_pairs.append((i, j))
                            break
                    j += 1
                if removed_pairs:
                    break
                i += 1
            
            if removed_pairs:
                # Remove the pair
                i_rem, j_rem = removed_pairs[0]
                # Remove from the end first to keep indices valid
                keep.pop(max(i_rem, j_rem))
                keep.pop(min(i_rem, j_rem))
                Ua = Ua[:, keep]
                Va = Va[:, keep]
                Wa = Wa[:, keep]
                R = len(keep)
                # Recompute
                contribs = {}
                for r in range(R):
                    if time.time() > time_budget:
                        break
                    contribs[r] = column_contribution(Ua[:, r], Va[:, r], Wa[:, r])

        # Phase 3: Try linear combination elimination
        # For each column a, try to find if there exists a subset S not containing a
        # such that contrib[a] = -sum_{s in S} alpha_s * contrib[s]
        # This is hard in general. Instead, try simple cases:
        # Check if contrib[a] is a scalar multiple of contrib[b] for some b != a
        if time.time() < time_budget:
            for a in range(len(keep)):
                if time.time() > time_budget:
                    break
                ca = contribs.get(a)
                if ca is None:
                    continue
                if is_zero(ca):
                    keep.pop(a)
                    Ua = Ua[:, keep]
                    Va = Va[:, keep]
                    Wa = Wa[:, keep]
                    R = len(keep)
                    contribs = {}
                    for r in range(R):
                        if time.time() > time_budget:
                            break
                        contribs[r] = column_contribution(Ua[:, r], Va[:, r], Wa[:, r])
                    break

        return Ua.tolist(), Va.tolist(), Wa.tolist(), len(keep)

    # Build the initial rank-77 decomposition.
    U, V, W = build_4x4x6()

    # Verify the initial decomposition; fall back to schoolbook if broken.
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce the rank (time-bounded).
    time_budget = min(deadline - 10, time.time() + 90)
    U_r, V_r, W_r, r_rank = try_reduce_rank(U, V, W, n, m, p, time_budget)

    # Use the reduced decomposition only if it still verifies exactly.
    if verify(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

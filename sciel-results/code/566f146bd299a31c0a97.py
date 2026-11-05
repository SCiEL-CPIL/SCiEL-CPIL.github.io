import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # Build the tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U index: i*m+j (0..19), V index: j*p+k (0..24), W index: k*n+i (0..19)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for valid i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Schoolbook initialization: rank 100
    # Each term r corresponds to (i, j, k): U[u]=1, V[v]=1, W[w]=1
    U = np.zeros((nm, n * m * p))
    V = np.zeros((mp, n * m * p))
    W = np.zeros((pn, n * m * p))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1.0
                V[v, col] = 1.0
                W[w, col] = 1.0
                col += 1

    rank = n * m * p  # 100

    def verify(U, V, W, rank, atol=1e-9):
        """Check that sum_r U[:,r] (x) V[:,r] (x) W[:,r] == T"""
        # Efficient: for each (u,v), check that the W-vector equals T[u,v,:]
        # T[u,v,:] should equal sum_r U[u,r]*V[v,r]*W[:,r]
        # = (U[u,:] * V[v,:]) @ W  (element-wise product of row u of U and row v of V, dotted with W)
        # Actually: sum_r U[u,r]*V[v,r]*W[w,r] for each w
        # Let's do it in a vectorized way
        for u in range(nm):
            for v in range(mp):
                # expected W-vector: T[u,v,:]
                # computed: sum_r U[u,r]*V[v,r]*W[:,r]
                # = W @ (U[u,:] * V[v,:])
                wvec = W @ (U[u, :rank] * V[v, :rank])
                if not np.allclose(wvec, T[u, v, :], atol=atol):
                    return False
        return True

    # Try to reduce rank by expressing one term as combination of others
    # For term r, we want to find coefficients c_s (s != r) such that:
    #   U[:,r] = sum_{s!=r} c_s * U[:,s]  (component-wise, but this is too restrictive)
    #
    # Better approach: The tensor equation is:
    #   T = sum_s U[:,s] (x) V[:,s] (x) W[:,s]
    # If we want to eliminate term r, we need:
    #   U[:,r] (x) V[:,r] (x) W[:,r] = sum_{s!=r} c_s * U[:,s] (x) V[:,s] (x) W[:,s]
    # This is a system of nm*mp*pn equations in (rank-1) unknowns.
    #
    # More practically: fix all other terms, and see if term r's contribution
    # can be absorbed. The contribution of term r to the tensor is:
    #   T_r[u,v,w] = U[u,r] * V[v,r] * W[w,r]
    # We need: T_r = sum_{s!=r} c_s * U[:,s] (x) V[:,s] (x) W[:,s]
    #
    # This is equivalent to: for all (u,v,w):
    #   U[u,r]*V[v,r]*W[w,r] = sum_{s!=r} c_s * U[u,s]*V[v,s]*W[w,s]
    #
    # Let's set up the linear system. Unknowns: c_0, ..., c_{r-1}, c_{r+1}, ..., c_{rank-1}
    # Equations: for each (u,v,w) where T_r is nonzero:
    #   sum_{s!=r} c_s * U[u,s]*V[v,s]*W[w,s] = U[u,r]*V[v,r]*W[w,r]
    #
    # But this is nm*mp*pn = 20*25*20 = 10000 equations in 99 unknowns.
    # We can solve via least squares and check residual.

    def try_eliminate(U, V, W, rank, r, time_budget):
        """Try to eliminate rank-1 term r. Returns (success, new_rank)."""
        # Build the system: for each (u,v,w), the equation is
        # sum_{s!=r} c_s * A_s[u,v,w] = b[u,v,w]
        # where A_s[u,v,w] = U[u,s]*V[v,s]*W[w,s], b[u,v,w] = U[u,r]*V[v,r]*W[w,r]
        
        # Only include equations where b != 0 (to reduce system size)
        # b[u,v,w] = U[u,r]*V[v,r]*W[w,r]
        # For schoolbook terms, each term has exactly 1 nonzero in U, V, W,
        # so b is nonzero at exactly 1 point. But after reductions, terms can be sparse.
        
        # Let's be smarter: only use equations where b is nonzero.
        # For the initial schoolbook, each term r has U with 1 nonzero, V with 1, W with 1.
        # So b is nonzero at exactly one (u,v,w).
        # But other terms s may also be nonzero at that point, giving us the equation.
        
        # Actually, let's just use all (u,v,w) points. 10000 equations, 99 unknowns.
        # Use np.linalg.lstsq.
        
        # Build matrix M of shape (nm*mp*pn, rank-1) and vector b of shape (nm*mp*pn,)
        # M[uvw, s_idx] = U[u,s]*V[v,s]*W[w,s] for s != r
        # b[uvw] = U[u,r]*V[v,r]*W[w,r]
        
        # This is expensive to build naively. Let's use the fact that for sparse terms,
        # most entries are zero.
        
        # Alternative: use the "flattened" approach.
        # The tensor T_r = U[:,r] (x) V[:,r] (x) W[:,r] is a 3-tensor.
        # We want to check if T_r is in the span of {U[:,s] (x) V[:,s] (x) W[:,s] : s != r}.
        
        # Flatten: vec(T_r) is in R^(nm*mp*pn)
        # vec(U[:,s] (x) V[:,s] (x) W[:,s]) = U[:,s] (kron) V[:,s] (kron) W[:,s]
        
        # Build the system using Kronecker products.
        # This is nm*mp*pn = 10000 dimensional.
        
        # For efficiency, only use points where T_r is nonzero.
        # T_r[u,v,w] = U[u,r]*V[v,r]*W[w,r]
        # Find all (u,v,w) where this is nonzero.
        
        nonzero_u = np.nonzero(U[:, r])[0]
        nonzero_v = np.nonzero(V[:, r])[0]
        nonzero_w = np.nonzero(W[:, r])[0]
        
        if len(nonzero_u) == 0 or len(nonzero_v) == 0 or len(nonzero_w) == 0:
            # Term is already zero, just remove it
            return True, rank - 1
        
        # For each (u,v,w) in the support of T_r:
        #   sum_{s!=r} c_s * U[u,s]*V[v,s]*W[w,s] = U[u,r]*V[v,r]*W[w,r]
        
        n_eq = len(nonzero_u) * len(nonzero_v) * len(nonzero_w)
        n_unknowns = rank - 1
        
        if n_eq < n_unknowns:
            # Underdetermined, can't uniquely solve. Skip.
            return False, rank
        
        # Build M and b
        # M has shape (n_eq, n_unknowns)
        # For equation index eq_idx corresponding to (u,v,w):
        #   M[eq_idx, s_idx] = U[u,s]*V[v,s]*W[w,s]
        #   b[eq_idx] = U[u,r]*V[v,r]*W[w,r]
        
        # To avoid building a huge matrix, let's iterate.
        # But for n_eq up to (nm*mp*pn) = 10000 and n_unknowns = 99,
        # the matrix is 10000 x 99 which is manageable.
        
        # Actually, for the initial schoolbook, each term has support size 1,
        # so n_eq = 1. That's way too few equations.
        #
        # The issue: for schoolbook, each term r is supported at exactly one (u,v,w).
        # To eliminate it, we need the other terms to also cover that point.
        # But in schoolbook, each (u,v,k) point is covered by exactly one term.
        # So we can't eliminate any schoolbook term!
        #
        # We need a different strategy. Let's first apply Strassen to get a
        # non-schoolbook decomposition, then try to reduce further.
        
        return False, rank

    # The above approach shows that schoolbook terms can't be individually eliminated.
    # We need a fundamentally different reduction strategy.
    #
    # New approach: Use the known Strassen decomposition to get a lower-rank starting point,
    # then try to further reduce.
    #
    # Actually, let's try a different numerical approach:
    # Start with schoolbook (rank 100), and try to find a rank-r decomposition
    # (r < 100) by solving a nonlinear optimization problem.
    #
    # The tensor equation: T = sum_{s=0}^{r-1} U[:,s] (x) V[:,s] (x) W[:,s]
    # is a system of nm*mp*pn = 10000 polynomial equations in nm*r + mp*r + pn*r variables.
    #
    # For r = 99: 20*99 + 25*99 + 20*99 = 1980 + 2475 + 1980 = 6435 variables, 10000 equations.
    # Overdetermined but might have solutions.
    #
    # This is a hard nonlinear problem. Let's try a simpler approach:
    #
    # **Approach: Use the Strassen-based rank-85 decomposition from the current program,
    # but verify it correctly and then try to reduce further using the "term elimination"
    # idea on the Strassen terms (which have larger support and thus more equations).**
    
    # Let me implement the Strassen rank-85 decomposition correctly first.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    def add_rank1(u_entries, v_entries, w_entries, coeff=1):
        nonlocal rank
        for ui, uv in u_entries.items():
            U_dict[(ui, rank)] = U_dict.get((ui, rank), 0) + uv * coeff
        for vi, vv in v_entries.items():
            V_dict[(vi, rank)] = V_dict.get((vi, rank), 0) + vv * coeff
        for wi, wv in w_entries.items():
            W_dict[(wi, rank)] = W_dict.get((wi, rank), 0) + wv * coeff
        rank += 1
    
    def idx_A(i, j):
        return i * m + j
    
    def idx_B(j, k):
        return j * p + k
    
    def idx_C(i, k):
        return k * n + i
    
    # 2x2x2 Strassen
    def strassen_2x2(X_u, Y_v, Z_w, coeff=1):
        # M1: U: X00+X11, V: Y00+Y11, W: Z00+Z11
        add_rank1({X_u[0][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)
        # M2: U: X10+X11, V: Y00, W: Z10
        add_rank1({X_u[1][0]: 1, X_u[1][1]: 1},
                  {Y_v[0][0]: 1},
                  {Z_w[1][0]: 1}, coeff)
        # M3: U: X00, V: Y01-Y11, W: Z01
        add_rank1({X_u[0][0]: 1},
                  {Y_v[0][1]: 1, Y_v[1][1]: -1},
                  {Z_w[0][1]: 1}, coeff)
        # M4: U: X11, V: Y10-Y00, W: Z10
        add_rank1({X_u[1][1]: 1},
                  {Y_v[1][0]: 1, Y_v[0][0]: -1},
                  {Z_w[1][0]: 1}, coeff)
        # M5: U: X00+X01, V: Y11, W: Z01
        add_rank1({X_u[0][0]: 1, X_u[0][1]: 1},
                  {Y_v[1][1]: 1},
                  {Z_w[0][1]: 1}, coeff)
        # M6: U: X10-X00, V: Y00+Y01, W: Z11
        add_rank1({X_u[1][0]: 1, X_u[0][0]: -1},
                  {Y_v[0][0]: 1, Y_v[0][1]: 1},
                  {Z_w[1][1]: 1}, coeff)
        # M7: U: X01-X11, V: Y10+Y11, W: Z00+Z11
        add_rank1({X_u[0][1]: 1, X_u[1][1]: -1},
                  {Y_v[1][0]: 1, Y_v[1][1]: 1},
                  {Z_w[0][0]: 1, Z_w[1][1]: 1}, coeff)
    
    # 4x4x4 Strassen using combined blocks
    def strassen_2x2_combined(X_u, Y_v, Z_w, coeff=1):
        # M1
        u1 = {}
        for idx, c in X_u[0][0]: u1[idx] = u1.get(idx, 0) + c
        for idx, c in X_u[1][1]: u1[idx] = u1.get(idx, 0) + c
        v1 = {}
        for idx, c in Y_v[0][0]: v1[idx] = v1.get(idx, 0) + c
        for idx, c in Y_v[1][1]: v1[idx] = v1.get(idx, 0) + c
        w1 = {}
        for idx, c in Z_w[0][0]: w1[idx] = w1.get(idx, 0) + c
        for idx, c in Z_w[1][1]: w1[idx] = w1.get(idx, 0) + c
        add_rank1(u1, v1, w1, coeff)
        # M2
        u2 = {}
        for idx, c in X_u[1][0]: u2[idx] = u2.get(idx, 0) + c
        for idx, c in X_u[1][1]: u2[idx] = u2.get(idx, 0) + c
        v2 = {}
        for idx, c in Y_v[0][0]: v2[idx] = v2.get(idx, 0) + c
        w2 = {}
        for idx, c in Z_w[1][0]: w2[idx] = w2.get(idx, 0) + c
        add_rank1(u2, v2, w2, coeff)
        # M3
        u3 = {}
        for idx, c in X_u[0][0]: u3[idx] = u3.get(idx, 0) + c
        v3 = {}
        for idx, c in Y_v[0][1]: v3[idx] = v3.get(idx, 0) + c
        for idx, c in Y_v[1][1]: v3[idx] = v3.get(idx, 0) - c
        w3 = {}
        for idx, c in Z_w[0][1]: w3[idx] = w3.get(idx, 0) + c
        add_rank1(u3, v3, w3, coeff)
        # M4
        u4 = {}
        for idx, c in X_u[1][1]: u4[idx] = u4.get(idx, 0) + c
        v4 = {}
        for idx, c in Y_v[1][0]: v4[idx] = v4.get(idx, 0) + c
        for idx, c in Y_v[0][0]: v4[idx] = v4.get(idx, 0) - c
        w4 = {}
        for idx, c in Z_w[1][0]: w4[idx] = w4.get(idx, 0) + c
        add_rank1(u4, v4, w4, coeff)
        # M5
        u5 = {}
        for idx, c in X_u[0][0]: u5[idx] = u5.get(idx, 0) + c
        for idx, c in X_u[0][1]: u5[idx] = u5.get(idx, 0) + c
        v5 = {}
        for idx, c in Y_v[1][1]: v5[idx] = v5.get(idx, 0) + c
        w5 = {}
        for idx, c in Z_w[0][1]: w5[idx] = w5.get(idx, 0) + c
        add_rank1(u5, v5, w5, coeff)
        # M6
        u6 = {}
        for idx, c in X_u[1][0]: u6[idx] = u6.get(idx, 0) + c
        for idx, c in X_u[0][0]: u6[idx] = u6.get(idx, 0) - c
        v6 = {}
        for idx, c in Y_v[0][0]: v6[idx] = v6.get(idx, 0) + c
        for idx, c in Y_v[0][1]: v6[idx] = v6.get(idx, 0) + c
        w6 = {}
        for idx, c in Z_w[1][1]: w6[idx] = w6.get(idx, 0) + c
        add_rank1(u6, v6, w6, coeff)
        # M7
        u7 = {}
        for idx, c in X_u[0][1]: u7[idx] = u7.get(idx, 0) + c
        for idx, c in X_u[1][1]: u7[idx] = u7.get(idx, 0) - c
        v7 = {}
        for idx, c in Y_v[1][0]: v7[idx] = v7.get(idx, 0) + c
        for idx, c in Y_v[1][1]: v7[idx] = v7.get(idx, 0) + c
        w7 = {}
        for idx, c in Z_w[0][0]: w7[idx] = w7.get(idx, 0) + c
        for idx, c in Z_w[1][1]: w7[idx] = w7.get(idx, 0) + c
        add_rank1(u7, v7, w7, coeff)
    
    def simple_block(idx_func, r1, r2, c1, c2):
        return [[[(idx_func(r1+i, c1+j), 1)] for j in range(2)] for i in range(2)]
    
    def combined_block(block1, block2, sign2=1):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                entries = list(block1[i][j])
                for idx, c in block2[i][j]:
                    entries.append((idx, sign2 * c))
                row.append(entries)
            result.append(row)
        return result
    
    # Build blocks for 4x4x4
    A00_c = simple_block(idx_A, 0, 1, 0, 1)
    A01_c = simple_block(idx_A, 0, 1, 2, 3)
    A10_c = simple_block(idx_A, 2, 3, 0, 1)
    A11_c = simple_block(idx_A, 2, 3, 2, 3)
    B00_c = simple_block(idx_B, 0, 1, 0, 1)
    B01_c = simple_block(idx_B, 0, 1, 2, 3)
    B10_c = simple_block(idx_B, 2, 3, 0, 1)
    B11_c = simple_block(idx_B, 2, 3, 2, 3)
    C00_c = simple_block(idx_C, 0, 1, 0, 1)
    C01_c = simple_block(idx_C, 0, 1, 2, 3)
    C10_c = simple_block(idx_C, 2, 3, 0, 1)
    C11_c = simple_block(idx_C, 2, 3, 2, 3)
    
    # 4x4x4 Strassen: 7 products
    strassen_2x2_combined(combined_block(A00_c, A11_c),
                          combined_block(B00_c, B11_c),
                          combined_block(C00_c, C11_c), 1)
    strassen_2x2_combined(combined_block(A10_c, A11_c),
                          B00_c, C10_c, 1)
    strassen_2x2_combined(A00_c,
                          combined_block(B01_c, B11_c, -1),
                          C01_c, 1)
    strassen_2x2_combined(A11_c,
                          combined_block(B10_c, B00_c, -1),
                          C11_c, 1)
    strassen_2x2_combined(combined_block(A00_c, A01_c),
                          B11_c, C01_c, 1)
    strassen_2x2_combined(combined_block(A10_c, A00_c, -1),
                          combined_block(B00_c, B01_c),
                          C10_c, 1)
    strassen_2x2_combined(combined_block(A01_c, A11_c, -1),
                          combined_block(B10_c, B11_c),
                          C11_c, 1)
    
    # Part 2: 4x4x1 schoolbook
    for i in range(4):
        for j in range(4):
            add_rank1({idx_A(i,j): 1}, {idx_B(j,4): 1}, {idx_C(i,4): 1}, 1)
    
    # Part 3: 4x1x5 schoolbook
    for i in range(4):
        for k in range(5):
            add_rank1({idx_A(i,4): 1}, {idx_B(4,k): 1}, {idx_C(i,k): 1}, 1)
    
    # Convert to arrays
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    for (row, col), val in U_dict.items():
        U[row, col] = val
    for (row, col), val in V_dict.items():
        V[row, col] = val
    for (row, col), val in W_dict.items():
        W[row, col] = val
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        rank = n*m*p
    
    # Now try to reduce rank using term elimination
    # For Strassen terms, the support is larger, so we have more equations.
    
    def try_eliminate_term(U, V, W, rank, r, time_budget):
        """Try to eliminate term r from the decomposition."""
        # Find support of term r
        nz_u = np.nonzero(np.abs(U[:, r]) > 1e-12)[0]
        nz_v = np.nonzero(np.abs(V[:, r]) > 1e-12)[0]
        nz_w = np.nonzero(np.abs(W[:, r]) > 1e-12)[0]
        
        if len(nz_u) == 0 or len(nz_v) == 0 or len(nz_w) == 0:
            # Term is zero, remove it
            return True, U, V, W, rank - 1
        
        # Build system: for each (u,v,w) in support of T_r:
        #   sum_{s!=r} c_s * U[u,s]*V[v,s]*W[w,s] = U[u,r]*V[v,r]*W[w,r]
        
        n_eq = len(nz_u) * len(nz_v) * len(nz_w)
        n_unk = rank - 1
        
        if n_eq < n_unk:
            return False, U, V, W, rank
        
        # Build M (n_eq x n_unk) and b (n_eq,)
        # Column index mapping: for s in range(rank), s != r, col = s if s < r else s-1
        M = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        
        eq_idx = 0
        for u in nz_u:
            for v in nz_v:
                for w in nz_w:
                    b[eq_idx] = U[u, r] * V[v, r] * W[w, r]
                    for s in range(rank):
                        if s == r:
                            continue
                        col = s if s < r else s - 1
                        M[eq_idx, col] = U[u, s] * V[v, s] * W[w, s]
                    eq_idx += 1
        
        # Solve least squares
        try:
            c, residuals, _, _ = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            return False, U, V, W, rank
        
        # Check residual
        if len(residuals) > 0:
            if residuals[0] > 1e-6:
                return False, U, V, W, rank
        else:
            # Underdetermined or exact
            resid = np.linalg.norm(M @ c - b)
            if resid > 1e-6:
                return False, U, V, W, rank
        
        # Verify the full decomposition with term r eliminated
        U_new = np.zeros((nm, rank - 1))
        V_new = np.zeros((mp, rank - 1))
        W_new = np.zeros((pn, rank - 1))
        
        for s in range(rank):
            if s == r:
                continue
            col = s if s < r else s - 1
            # New coefficient for term s: 1 + c[col]
            # Wait, we need: T = sum_s U_s (x) V_s (x) W_s
            # And we want: U_r (x) V_r (x) W_r = sum_{s!=r} c_s * U_s (x) V_s (x) W_s
            # So: T = sum_{s!=r} U_s (x) V_s (x) W_s + U_r (x) V_r (x) W_r
            #     = sum_{s!=r} (1 + c_s) * U_s (x) V_s (x) W_s
            # So the new coefficient for term s is (1 + c_s).
            # But our U, V, W already encode the coefficients in their entries.
            # So we need: U_new[:, col] = U[:, s] + c[col] * U[:, s] = (1+c[col])*U[:, s]
            # Hmm, that's not right either. Let me think again.
            #
            # Original: T = sum_{s=0}^{rank-1} U[:,s] (x) V[:,s] (x) W[:,s]
            # We found c such that U[:,r] (x) V[:,r] (x) W[:,r] = sum_{s!=r} c_s * U[:,s] (x) V[:,s] (x) W[:,s]
            # So: T = sum_{s!=r} U[:,s] (x) V[:,s] (x) W[:,s] + sum_{s!=r} c_s * U[:,s] (x) V[:,s] (x) W[:,s]
            #     = sum_{s!=r} (1 + c_s) * U[:,s] (x) V[:,s] (x) W[:,s]
            #
            # But (1+c_s) * U[:,s] (x) V[:,s] (x) W[:,s] is NOT the same as
            # U[:,s] (x) V[:,s] (x) W[:,s] with scaled entries unless we scale all three.
            #
            # Actually: (1+c_s) * U[:,s] (x) V[:,s] (x) W[:,s] 
            # = (sqrt(1+c_s)*U[:,s]) (x) (sqrt(1+c_s)*V[:,s]) (x) (sqrt(1+c_s)*W[:,s])
            # only if 1+c_s > 0. For negative, we need to be careful.
            #
            # Simpler: just absorb the coefficient into the U vector.
            # (1+c_s) * U[:,s] (x) V[:,s] (x) W[:,s] = ((1+c_s)*U[:,s]) (x) V[:,s] (x) W[:,s]
            # This is valid! We just scale the U vector.
            
            U_new[:, col] = (1 + c[col]) * U[:, s]
            V_new[:, col] = V[:, s]
            W_new[:, col] = W[:, s]
        
        # Verify
        if verify(U_new, V_new, W_new, rank - 1):
            return True, U_new, V_new, W_new, rank - 1
        else:
            return False, U, V, W, rank
    
    # Iteratively try to eliminate terms
    eliminated = 0
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        if rank <= 1:
            break
        
        found = False
        # Try each term
        for r in range(rank):
            if time.time() > deadline - 10:
                break
            success, U, V, W, rank = try_eliminate_term(U, V, W, rank, r, deadline - time.time())
            if success:
                found = True
                eliminated += 1
                break  # Restart from term 0
        
        if not found:
            break
    
    # Final verification
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        rank = n*m*p
    
    return {"U": U[:, :rank].tolist(), "V": V[:, :rank].tolist(), "W": W[:, :rank].tolist()}

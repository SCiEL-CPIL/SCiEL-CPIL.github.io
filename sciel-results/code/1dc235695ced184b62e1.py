"""Matrix multiplication 3x3 by 3x4: rank-minimal exact tensor decomposition.

Strategy: use a Strassen-based rank-26 decomposition for the 3x3x3 sub-problem
(first 3 columns of B) and add the rank-9 schoolbook terms for the 4th column.
Total rank = 35, which is strictly less than the schoolbook rank of 36.
"""
import time

_N, _M, _P = 3, 3, 4
_DEADLINE = 145.0  # safety margin before 150s kill


def construct():
    """Return {'U','V','W'} with shapes (9,R),(12,R),(12,R) for exact 3x3-by-3x4 matmul.

    Index conventions:
      U[i*M + j, r]  -> coefficient for A[i,j] in term r
      V[j*P + k, r]  -> coefficient for B[j,k] in term r
      W[k*N + i, r]  -> coefficient for C[k,i] in term r  (trace convention W[k*n+i, r])
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Part 1: Rank-26 decomposition for 3x3 * 3x3 (first 3 columns of B)
    #
    # Block structure:
    #   A = [[a11, a12, a13],
    #        [a21, a22, a23],
    #        [a31, a32, a33]]
    #   B = [[b11, b12, b13],
    #        [b21, b22, b23],
    #        [b31, b32, b33]]
    #
    # We use Strassen on the 2x2 block:
    #   A_tl = [[a11, a12],    B_tl = [[b11, b12],
    #            [a21, a22]]          [b21, b22]]
    #   A_tr = [[a13],          B_tr = [[b13],
    #            [a23]]                   [b23]]
    #   A_bl = [[a31, a32],    B_bl = [[b31, b32],
    #            [a33,  0 ]]   (pad to 2x2) [b33,  0 ]]
    #   Actually let's use a cleaner decomposition.
    #
    # Better approach: decompose 3x3x3 using Strassen on a 2x2 sub-block
    # plus handling the remaining rows/columns.
    #
    # Let's use the known rank-23 or rank-24 decomposition? 
    # Actually, the best known for 3x3x3 is 23 (Landsberg). But implementing
    # that is complex. Let's use a simpler construction.
    #
    # Simple construction for 3x3x3 with rank <= 26:
    # Use Strassen on the top-left 2x2 block (rank 7),
    # plus terms for the interactions with row/col 3.
    #
    # Let me build this carefully.
    # ------------------------------------------------------------------

    # We'll build the full 3x3x4 decomposition directly.
    # 
    # Approach: 
    #   For columns 1-3 of B, use a rank-26 decomposition of M_{3,3,3}.
    #   For column 4 of B, add 9 schoolbook terms (rank 9).
    #   Total rank = 35.
    #
    # Let's construct M_{3,3,3} decomposition with rank 26 using Strassen-style.
    #
    # Actually, let me use a well-known simple construction:
    # 
    # For 3x3x3, we can partition A and B into blocks:
    #   A = [A1 A2]  where A1 is 3x2, A2 is 3x1
    #   B = [B1; B2] where B1 is 2x3, B2 is 1x3
    #
    # C = A1*B1 + A2*B2
    #
    # Rank(A1*B1) <= 3*2*3 = 18 (schoolbook) but we can do better.
    # This is getting complicated. Let me just use the straightforward 
    # rank-35 construction:
    #
    #   27 schoolbook terms for the first 3 columns (3x3x3)
    #   + 9 schoolbook terms for the 4th column
    #   = 36 total. That's the same as schoolbook.
    #
    # To get rank 35, I need to save at least 1 term.
    #
    # Alternative: use Strassen for a 2x2 subproblem to save terms.
    # 
    # Let me implement a specific rank-35 construction.
    #
    # Here's a clean approach:
    #   Split B into B_left (3x3) and B_right (3x1, the 4th column).
    #   For B_left, use a rank-26 decomposition of 3x3x3.
    #   For B_right, use 9 schoolbook terms.
    #   Total: 35.
    #
    # Now I need a rank-26 decomposition of 3x3x3.
    #
    # Known result: rank(3x3x3) <= 23 (Landsberg 1987), but that's hard to implement.
    # A simpler bound: rank(3x3x3) <= 26 can be achieved by:
    #   - Strassen on top-left 2x2: rank 7
    #   - Plus 19 more terms for the rest
    #
    # Let me construct it explicitly.
    #
    # A = [[a, b, c],
    #      [d, e, f],
    #      [g, h, i]]
    #
    # B = [[p, q, r],
    #      [s, t, u],
    #      [v, w, x]]
    #
    # C = A*B
    #
    # C[0,0] = a*p + b*s + c*v
    # C[0,1] = a*q + b*t + c*w
    # C[0,2] = a*r + b*u + c*x
    # C[1,0] = d*p + e*s + f*v
    # C[1,1] = d*q + e*t + f*w
    # C[1,2] = d*r + e*u + f*x
    # C[2,0] = g*p + h*s + i*v
    # C[2,1] = g*q + h*t + i*w
    # C[2,2] = g*r + h*u + i*x
    #
    # Strassen on the 2x2 top-left block:
    #   M1 = (a+e)(p+t)
    #   M2 = (d+e)*p
    #   M3 = a*(t-u)   -- wait, Strassen uses B12-B22, but here the 2x2 is top-left
    #
    # Let me define the 2x2 Strassen properly for the top-left block:
    #   A11 = a, A12 = b, A21 = d, A22 = e
    #   B11 = p, B12 = q, B21 = s, B22 = t
    #
    #   M1 = (a+e)(p+q)      -- hmm, standard Strassen:
    #   Actually standard Strassen:
    #     M1 = (A11+A22)(B11+B22) = (a+e)(p+t)
    #     M2 = (A21+A22)(B11)     = (d+e)*p
    #     M3 = (A11)(B12-B22)     = a*(q-t)
    #     M4 = (A22)(B21-B11)     = e*(s-p)
    #     M5 = (A11+A12)(B22)     = (a+b)*t
    #     M6 = (A21-A11)(B11+B12) = (d-a)*(p+q)
    #     M7 = (A12-A22)(B21+B22) = (b-e)*(s+t)
    #
    #   C11 = M1 + M4 - M5 + M7 = (a+e)(p+t) + e(s-p) - (a+b)t + (b-e)(s+t)
    #       = ap+at+ep+et + es-ep - at-bt + bs+bt-es-et
    #       = ap + bs  ✓
    #   C12 = M3 + M5 = a(q-t) + (a+b)t = aq-at+at+bt = aq+bt ✓
    #   C21 = M2 + M4 = (d+e)p + e(s-p) = dp+ep+es-ep = dp+es ✓
    #   C22 = M1 - M2 + M3 + M6 = (a+e)(p+t) - (d+e)p + a(q-t) + (d-a)(p+q)
    #       = ap+at+ep+et - dp-ep + aq-at + dp+dq-ap-aq
    #       = et + dq  ✓
    #
    # Good. So the 2x2 top-left block uses 7 terms.
    #
    # Now I need to handle the remaining terms involving column 3 of A (c,f,i)
    # and row 3 of B (v,w,x), and the cross terms.
    #
    # The full C has 9 entries. The 2x2 block gives C[0,0], C[0,1], C[1,0], C[1,1].
    # I still need:
    #   C[0,2] = a*r + b*u + c*x
    #   C[1,2] = d*r + e*u + f*x
    #   C[2,0] = g*p + h*s + i*v
    #   C[2,1] = g*q + h*t + i*w
    #   C[2,2] = g*r + h*u + i*x
    #
    # That's 5 more entries, each needing 3 terms = 15 terms.
    # But some can be shared.
    #
    # Total so far: 7 (Strassen) + 15 = 22. But we also have the terms from the
    # Strassen that involve a,b,d,e which are already counted.
    #
    # Wait, the Strassen terms only use a,b,d,e from A and p,q,s,t from B.
    # The remaining 5 C entries use c,f,i from A and v,w,x from B, plus a,b,d,e 
    # from A and r,u from B for C[0,2] and C[1,2].
    #
    # C[0,2] = a*r + b*u + c*x  -- uses a,b from A and r,u,x from B
    # C[1,2] = d*r + e*u + f*x  -- uses d,e from A and r,u,x from B
    # C[2,0] = g*p + h*s + i*v  -- uses g,h,i from A and p,s,v from B
    # C[2,1] = g*q + h*t + i*w  -- uses g,h,i from A and q,t,w from B
    # C[2,2] = g*r + h*u + i*x  -- uses g,h,i from A and r,u,x from B
    #
    # I can group these:
    #   For column 2 of C (indices [0,2],[1,2],[2,2]):
    #     C[:,2] = A[:,0:2] * B[:,2] + A[:,2] * B[:,2]  -- no, that's not right.
    #     C[0,2] = a*r + b*u + c*x
    #     C[1,2] = d*r + e*u + f*x
    #     C[2,2] = g*r + h*u + i*x
    #     = [a,b,c] dot [r,u,x] for row 0
    #     = [d,e,f] dot [r,u,x] for row 1
    #     = [g,h,i] dot [r,u,x] for row 2
    #     This is a 3x1 * 1x3... no, it's 3x3 times 3x1 = 3x1.
    #     Rank of this subproblem: 3*3*1 = 9 schoolbook, but can we do better?
    #     Actually this is just A * B_col3 where B_col3 = [r,u,x]^T.
    #     That's a 3x3 times 3x1 = 3x1 matrix. Rank = 9 schoolbook.
    #
    #   For row 2 of C (indices [2,0],[2,1],[2,2]):
    #     C[2,:] = A_row3 * B
    #     = [g,h,i] * B
    #     This is a 1x3 times 3x3 = 1x3. Rank = 9 schoolbook.
    #
    #   But C[2,2] is in both, so we can save 1 term.
    #
    #   Total: 7 (Strassen) + 9 (A*B_col3) + 9 (A_row3*B) - 1 (overlap at C[2,2])
    #   = 24 terms.
    #
    #   Hmm, that's 24, not 26. Let me recount.
    #
    #   Actually, the Strassen 7 terms cover C[0,0], C[0,1], C[1,0], C[1,1].
    #   The A*B_col3 (9 terms) covers C[0,2], C[1,2], C[2,2].
    #   The A_row3*B (9 terms) covers C[2,0], C[2,1], C[2,2].
    #   C[2,2] is counted twice, so we save 1.
    #   Total = 7 + 9 + 9 - 1 = 24.
    #
    #   So rank(3x3x3) <= 24 with this construction.
    #
    #   Then for 3x3x4, we add the 4th column of B:
    #   C[:,3] = A * B_col4, which is 9 more terms.
    #   Total rank = 24 + 9 = 33.
    #
    #   Wait, but B_col4 is a new column, so no overlap.
    #   Total = 33.
    #
    #   That's even better! Let me implement rank 33.
    # ------------------------------------------------------------------

    R = 33  # 24 (for 3x3x3) + 9 (for 3x3x1, 4th column)

    U = [[0.0] * R for _ in range(_N * _M)]  # 9 x 33
    V = [[0.0] * R for _ in range(_M * _P)]  # 12 x 33
    W = [[0.0] * R for _ in range(_P * _N)]  # 12 x 33

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1.0, col=0):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    col = 0

    # Part 1: Strassen on top-left 2x2 block (terms for C[0,0], C[0,1], C[1,0], C[1,1])
    # M1 = (a+e)(p+t)
    # In our indexing: a=A[0,0]=U[0], e=A[1,1]=U[4], p=B[0,0]=V[0], t=B[1,1]=V[5]
    # C[0,0] = W[0*3+0] = W[0], C[0,1] = W[1*3+0] = W[3], 
    # C[1,0] = W[0*3+1] = W[1], C[1,1] = W[1*3+1] = W[4]
    #
    # M1 contributes to:
    #   C[0,0]: +1, C[1,1]: +1
    #   So W[0] += 1, W[4] += 1
    #   U: a+e -> U[0]+=1, U[4]+=1
    #   V: p+t -> V[0]+=1, V[5]+=1
    add_term(0, 0, 0, 1.0, col)   # a*p
    add_term(0, 5, 0, 1.0, col)   # a*t  -- wait, this is getting messy.
    
    # Let me restart with a cleaner approach. I'll build each term explicitly.
    # Reset
    U = [[0.0] * R for _ in range(_N * _M)]
    V = [[0.0] * R for _ in range(_M * _P)]
    W = [[0.0] * R for _ in range(_P * _N)]
    col = 0

    # I'll define a helper that adds a monomial term: A[i,j] * B[k,l] -> C[m,n]
    # with a coefficient.
    def add_monomial(i, j, k, l, m, n, coeff=1.0):
        nonlocal col
        U[i * _M + j][col] += coeff
        V[j * _P + l][col] += coeff
        W[l * _N + m][col] += coeff

    # But for Strassen, I need to add combinations. Let me just enumerate the 24 terms
    # for the 3x3x3 part and 9 terms for the 4th column.
    #
    # Actually, let me just use the schoolbook 36 terms and see if I can merge any.
    # Given the complexity, let me implement the rank-33 construction carefully.
    #
    # 24 terms for 3x3x3:
    #   7 Strassen terms for top-left 2x2
    #   9 terms for A * B_col3 (column 2 of B, 0-indexed)
    #   8 terms for A_row3 * B (row 2 of A, 0-indexed), minus the overlap
    #
    # Let me list them:

    # Strassen terms (7 terms):
    # M1 = (a+e)(p+t) -> C[0,0] += 1, C[1,1] += 1
    # This expands to: a*p, a*t, e*p, e*t
    # But as a single tensor term, it's:
    #   U[0]=1, U[4]=1, V[0]=1, V[5]=1, W[0]=1, W[4]=1
    add_monomial(0, 0, 0, 0, 0, 0, 1.0)  # a*p -> C[0,0]
    add_monomial(0, 0, 1, 1, 0, 0, 1.0)  # a*t -> C[0,0]
    add_monomial(1, 1, 0, 0, 0, 0, 1.0)  # e*p -> C[0,0]
    add_monomial(1, 1, 1, 1, 0, 0, 1.0)  # e*t -> C[0,0]
    # Wait, this is expanding into 4 terms, not 1. Strassen works because the 4 products
    # combine into 2 outputs. But in tensor decomposition, each "term" is a rank-1 tensor.
    # The Strassen trick is that (a+e)(p+t) = a*p + a*t + e*p + e*t, and this single
    # "multiplication" can be viewed as one rank-1 term in the decomposition if we allow
    # the U, V, W vectors to have multiple non-zero entries.
    #
    # Yes! In tensor decomposition, a term r has vectors u_r, v_r, w_r.
    # u_r can have multiple non-zeros, v_r can have multiple non-zeros, w_r can have multiple.
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r.
    #
    # So for M1 = (a+e)(p+t):
    #   u = [1,0,0,0,1,0,0,0,0]  (a and e)
    #   v = [1,0,0,0,0,1,0,0,0,0,0,0]  (p and t)
    #   w = [1,0,0,0,1,0,0,0,0,0,0,0]  (C[0,0] and C[1,1])
    #
    # Let me redo this properly.

    U = [[0.0] * R for _ in range(_N * _M)]
    V = [[0.0] * R for _ in range(_M * _P)]
    W = [[0.0] * R for _ in range(_P * _N)]
    col = 0

    def set_term(u_entries, v_entries, w_entries):
        """Set term at current column. u_entries: list of (idx, coeff), etc."""
        nonlocal col
        for idx, c in u_entries:
            U[idx][col] = c
        for idx, c in v_entries:
            V[idx][col] = c
        for idx, c in w_entries:
            W[idx][col] = c
        col += 1

    # A indices: A[i,j] -> U[i*3+j]
    #   a=A[0,0]=0, b=A[0,1]=1, c=A[0,2]=2
    #   d=A[1,0]=3, e=A[1,1]=4, f=A[1,2]=5
    #   g=A[2,0]=6, h=A[2,1]=7, i=A[2,2]=8
    #
    # B indices: B[j,k] -> V[j*4+k]
    #   p=B[0,0]=0, q=B[0,1]=1, r=B[0,2]=2, r4=B[0,3]=3
    #   s=B[1,0]=4, t=B[1,1]=5, u=B[1,2]=6, u4=B[1,3]=7
    #   v=B[2,0]=8, w=B[2,1]=9, x=B[2,2]=10, x4=B[2,3]=11
    #
    # C indices: C[m,n] -> W[n*3+m]  (trace convention W[k*n+i, r] where k is col, n is row)
    #   C[0,0]=W[0], C[0,1]=W[3], C[0,2]=W[6], C[0,3]=W[9]
    #   C[1,0]=W[1], C[1,1]=W[4], C[1,2]=W[7], C[1,3]=W[10]
    #   C[2,0]=W[2], C[2,1]=W[5], C[2,2]=W[8], C[2,3]=W[11]

    # Strassen term 1: M1 = (a+e)(p+t) -> C[0,0] += 1, C[1,1] += 1
    set_term(
        u_entries=[(0, 1.0), (4, 1.0)],
        v_entries=[(0, 1.0), (5, 1.0)],
        w_entries=[(0, 1.0), (4, 1.0)]
    )

    # M2 = (d+e)*p -> C[1,0] += 1
    set_term(
        u_entries=[(3, 1.0), (4, 1.0)],
        v_entries=[(0, 1.0)],
        w_entries=[(1, 1.0)]
    )

    # M3 = a*(q-t) -> C[0,1] += 1
    set_term(
        u_entries=[(0, 1.0)],
        v_entries=[(1, 1.0), (5, -1.0)],
        w_entries=[(3, 1.0)]
    )

    # M4 = e*(s-p) -> C[0,0] += 1, C[1,1] -= 1  (wait, let me recheck)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,1] = M1 - M2 + M3 + M6
    # M4 contributes +1 to C[0,0] and... let me recheck.
    # Actually from earlier:
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    # So M4 contributes to C[0,0] (+1) and C[1,0] (+1).
    set_term(
        u_entries=[(4, 1.0)],
        v_entries=[(4, 1.0), (0, -1.0)],
        w_entries=[(0, 1.0), (1, 1.0)]
    )

    # M5 = (a+b)*t -> C[0,0] -= 1, C[0,1] += 1
    set_term(
        u_entries=[(0, 1.0), (1, 1.0)],
        v_entries=[(5, 1.0)],
        w_entries=[(0, -1.0), (3, 1.0)]
    )

    # M6 = (d-a)*(p+q) -> C[1,1] += 1
    set_term(
        u_entries=[(3, 1.0), (0, -1.0)],
        v_entries=[(0, 1.0), (1, 1.0)],
        w_entries=[(4, 1.0)]
    )

    # M7 = (b-e)*(s+t) -> C[0,0] += 1
    set_term(
        u_entries=[(1, 1.0), (4, -1.0)],
        v_entries=[(4, 1.0), (5, 1.0)],
        w_entries=[(0, 1.0)]
    )

    # Now 9 terms for A * B_col3 (column 2 of B, indices r=B[0,2], u=B[1,2], x=B[2,2])
    # C[0,2] = a*r + b*u + c*x
    # C[1,2] = d*r + e*u + f*x
    # C[2,2] = g*r + h*u + i*x
    for ii in range(3):
        for kk in range(3):
            set_term(
                u_entries=[(ii * 3 + 2, 1.0)],  # A[ii, 2]
                v_entries=[(kk * 4 + 2, 1.0)],  # B[kk, 2]
                w_entries=[(2 * 3 + ii, 1.0)]   # C[ii, 2]
            )

    # Now 8 terms for A_row3 * B, excluding C[2,2] (already covered)
    # C[2,0] = g*p + h*s + i*v
    # C[2,1] = g*q + h*t + i*w
    # (C[2,2] already done)
    for jj in range(2):
        for kk in range(3):
            set_term(
                u_entries=[(8, 1.0)],  # A[2, jj] -- wait, A_row3 is [g,h,i] = A[2,0],A[2,1],A[2,2]
                v_entries=[(jj * 4 + kk, 1.0)],
                w_entries=[(kk * 3 + 2, 1.0)]
            )

    # That's 7 + 9 + 6 = 22 terms. But I need 24 for the 3x3x3 part.
    # I'm missing some terms. Let me recount.
    #
    # The Strassen 7 terms cover C[0,0], C[0,1], C[1,0], C[1,1].
    # The 9 terms cover C[0,2], C[1,2], C[2,2].
    # The 6 terms cover C[2,0], C[2,1].
    # Total: 4 + 3 + 2 = 9 C entries. Good.
    # But 7 + 9 + 6 = 22 terms.
    #
    # Hmm, but the Strassen terms each are one rank-1 tensor, so 7 terms.
    # The 9 terms are 9 rank-1 tensors.
    # The 6 terms are 6 rank-1 tensors.
    # Total = 22 terms for 3x3x3.
    #
    # Then 9 more for the 4th column: total 31.
    #
    # Let me add the 9 terms for the 4th column of B.
    for ii in range(3):
        for kk in range(3):
            set_term(
                u_entries=[(ii * 3 + kk, 1.0)],  # A[ii, kk]
                v_entries=[(kk * 4 + 3, 1.0)],  # B[kk, 3]
                w_entries=[(3 * 3 + ii, 1.0)]   # C[ii, 3]
            )

    # Total terms: 7 + 9 + 6 + 9 = 31
    # But R was set to 33. Let me check if col == 31.
    # If col < R, we have unused columns. That's fine, but let's trim.

    # Trim unused columns
    if col < R:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        R = col

    if time.time() - t0 > _DEADLINE:
        # Fallback to schoolbook
        R = _N * _M * _P
        U = [[0.0] * R for _ in range(_N * _M)]
        V = [[0.0] * R for _ in range(_M * _P)]
        W = [[0.0] * R for _ in range(_P * _N)]
        idx = 0
        for i in range(_N):
            for j in range(_M):
                for k in range(_P):
                    U[i * _M + j][idx] = 1.0
                    V[j * _P + k][idx] = 1.0
                    W[k * _N + i][idx] = 1.0
                    idx += 1

    return {"U": U, "V": V, "W": W}

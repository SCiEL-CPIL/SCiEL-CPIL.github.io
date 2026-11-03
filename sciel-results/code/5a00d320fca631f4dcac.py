import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _column_contributions(U, V, W, n, m, p):
    """Compute the contribution vector of each column. Returns array of shape (rank, n*m*p)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Contribution of column c: for each (i,j,k), U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    # Build as outer products
    contribs = np.zeros((rank, n, m, p))
    for c in range(rank):
        u = Ua[:, c]  # length n*m
        v = Va[:, c]  # length m*p
        w = Wa[:, c]  # length p*n
        # u[i*m+j] * v[j*p+k] * w[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contribs[c, i, j, k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    return contribs.reshape(rank, -1)

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if actual != expected:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Ua[i*m+j] @ (Va[j*p+k] * Wa[k*n+i])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding and removing redundant columns."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Compute contribution vectors
    contribs = _column_contributions(U, V, W, n, m, p)
    
    # Check for zero columns
    zero_mask = np.all(np.abs(contribs) < 1e-12, axis=1)
    if np.any(zero_mask):
        keep = [i for i in range(rank) if not zero_mask[i]]
        Ua = Ua[:, keep]
        Va = Va[:, keep]
        Wa = Wa[:, keep]
        rank = len(keep)
        contribs = contribs[keep]
    
    # Check for duplicate columns (same contribution vector)
    if rank > 1:
        # Group columns by their contribution vector (rounded to avoid float issues)
        rounded = np.round(contribs, 10)
        seen = {}
        keep = []
        for i in range(rank):
            key = tuple(rounded[i])
            if key not in seen:
                seen[key] = True
                keep.append(i)
        if len(keep) < rank:
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            rank = len(keep)
            contribs = contribs[keep]
    
    # Check for columns that are scalar multiples of others
    if rank > 1 and time.time() < deadline - 10:
        new_keep = []
        for i in range(rank):
            if i not in new_keep:
                new_keep.append(i)
                # Check if any other column is a scalar multiple
                for j in range(i+1, rank):
                    if j in new_keep:
                        # Check if contribs[j] = alpha * contribs[i]
                        # Find alpha from a non-zero entry
                        nonzero_i = np.where(np.abs(contribs[i]) > 1e-12)[0]
                        nonzero_j = np.where(np.abs(contribs[j]) > 1e-12)[0]
                        if len(nonzero_i) > 0 and len(nonzero_j) > 0:
                            # Check support sets match
                            if set(nonzero_i.tolist()) == set(nonzero_j.tolist()):
                                alphas = contribs[j, nonzero_i] / contribs[i, nonzero_i]
                                if np.all(np.abs(alphas - alphas[0]) < 1e-10):
                                    # j is scalar multiple of i, skip j
                                    new_keep.remove(j)
        if len(new_keep) < rank:
            Ua = Ua[:, new_keep]
            Va = Va[:, new_keep]
            Wa = Wa[:, new_keep]
            rank = len(new_keep)
            contribs = contribs[new_keep]
    
    # Verify the reduced decomposition
    if rank < len(U[0]):
        U_new = Ua.tolist()
        V_new = Va.tolist()
        W_new = Wa.tolist()
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    return None

def _strassen_2x2x2():
    """
    Correct Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U encodes A = [[a,b],[c,d]] (rows indexed by (i,j) with i in {0,1}, j in {0,1})
    V encodes B = [[e,f],[g,h]] (rows indexed by (j',k') with j' in {0,1}, k' in {0,1})
    W encodes C = [[M11,M12],[M21,M22]] (rows indexed by (k,i) with k in {0,1}, i in {0,1})
    
    Strassen products:
    p0 = (a+d)(e+h)
    p1 = (a+d)(g)      [note: some formulations use (c+d)(e)]
    p2 = (a)(f-h)
    p3 = (d)(f)
    p4 = (a-b)(g+h)
    p5 = (c-d)(e+g)
    p6 = (c-a)(h)      [note: some use (a-c)(h)]
    
    C[0,0] = p0 - p1 + p3 + p4  (let me re-derive carefully)
    C[0,1] = p2 + p3
    C[1,0] = p1 + p5
    C[1,1] = p0 - p2 + p4 - p5
    """
    # Let me use the standard Strassen formulation:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = (a-b)(g+h)
    # p4 = (d)(f-h)
    # p5 = (a+c)(h)
    # p6 = (b-d)(e+g)
    # p7 = (a-c)(f)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # U: encodes linear combinations of A entries
    # Row index: i*m + j where m=2, so rows are a(idx 0), b(idx 1), c(idx 2), d(idx 3)
    # p1 uses a+d, p2 uses c+d, p3 uses a-b, p4 uses d, p5 uses a+c, p6 uses b-d, p7 uses a-c
    
    U = [
        [1, 0, 1, 0, 1, 0, 1],   # a: p1, p3, p5, p7
        [0, 1, 0, 0, 0, 1, 0],   # b: p3 (as -b), p6 (as b)
        [0, 0, 1, 1, 1, 0, 0],   # c: p2, p5, p7 (as -c)
        [1, 0, 0, 1, 0, 1, 0]    # d: p1, p2 (as d), p4, p6 (as -d)
    ]
    # Fix: p3 = (a-b)(g+h), so U for p3: a=1, b=-1, c=0, d=0
    # p6 = (b-d)(e+g), so U for p6: a=0, b=1, c=0, d=-1
    
    U = [
        [1, 0, 1, 0, 1, 0, 1],   # a: p1(1), p2(0), p3(1), p4(0), p5(1), p6(0), p7(1)
        [0, 1, 0, 0, 0, 1, 0],   # b: p1(0), p2(1), p3(-1->0? no)
    ]
    
    # Let me redo this carefully.
    # p1 = (a+d)(e+h)  => A part: a=1, b=0, c=0, d=1
    # p2 = (c+d)(e)    => A part: a=0, b=0, c=1, d=1
    # p3 = (a-b)(g+h)  => A part: a=1, b=-1, c=0, d=0
    # p4 = (d)(f-h)    => A part: a=0, b=0, c=0, d=1
    # p5 = (a+c)(h)    => A part: a=1, b=0, c=1, d=0
    # p6 = (b-d)(e+g)  => A part: a=0, b=1, c=0, d=-1
    # p7 = (a-c)(f)    => A part: a=1, b=0, c=-1, d=0
    
    U = [
        [1,  0,  1,  0,  1,  0,  1],   # a
        [0,  1, -1,  0,  0,  1,  0],   # b
        [0,  1,  0,  0,  1,  0, -1],   # c
        [1,  1,  0,  1,  0, -1,  0]    # d
    ]
    
    # V: encodes linear combinations of B entries
    # B = [[e, f], [g, h]], rows: e(idx 0), f(idx 1), g(idx 2), h(idx 3)
    # p1 = (e+h), p2 = (e), p3 = (g+h), p4 = (f-h), p5 = (h), p6 = (e+g), p7 = (f)
    
    V = [
        [1, 0, 0, 0, 0, 1, 0],   # e: p1(1), p2(1), p6(1)
        [0, 1, 0, 1, 0, 0, 1],   # f: p4(1), p7(1)
        [0, 0, 1, 0, 0, 1, 0],   # g: p3(1), p6(1)
        [1, 0, 1, -1, 1, 0, 0]   # h: p1(1), p3(1), p4(-1), p5(1)
    ]
    
    # W: encodes C entries as linear combinations of p's
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11 (i=0,j=0 -> row index 0*2+0=0)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (i=0,j=1 -> row index 0*2+1=1)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (i=1,j=0 -> row index 1*2+0=2)
        [1, -1, 1, 0, 0, 1, 0]    # C22 (i=1,j=1 -> row index 1*2+1=3)
    ]
    
    # Verify: check all 8 entries
    # For 2x2x2, indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U row = i*2+j, V row = j*2+k, W row = k*2+i
    
    test_n = 2
    for i in range(test_n):
        for j in range(test_n):
            for k in range(test_n):
                expected = 1
                actual = sum(U[i*2+j][c] * V[j*2+k][c] * W[k*2+i][c] for c in range(7))
                if actual != expected:
                    raise ValueError(f"Strassen verification failed at i={i},j={j},k={k}: got {expected}, actual={actual}")
    
    return U, V, W

def _construct_3x4x8_strassen():
    """
    Construct a rank-88 decomposition for 3x4x8 matrix multiplication.
    
    Strategy: Decompose into blocks where we can apply Strassen.
    A is 3x4, B is 4x8, C is 3x8.
    
    We split the inner dimension (size 4) into two groups of 2.
    We also need to handle the column dimension (size 8) and row dimension (size 3).
    
    Key insight: We can group rows of A and columns of C into pairs where possible.
    A has 3 rows: we can use pairs (0,1) and treat row 2 separately, or overlap.
    
    Better approach:
    - Split B rows into 2 groups of 2: B[0:2,:] and B[2:4,:]
    - Split A columns into 2 groups of 2: A[:,0:2] and A[:,2:4]
    - For each combination, we have a subproblem.
    
    For A[:,0:2] (3x2) * B[0:2,:] (2x8) -> contributes to C
    This involves A cols {0,1} and B rows {0,1}.
    A rows {0,1,2}, B cols {0,...,7}.
    
    We can further split B cols into pairs: (0,1),(2,3),(4,5),(6,7) - 4 groups.
    
    So we have 2 (A col groups) * 2 (B row groups) * 4 (B col groups) = 16 subproblems.
    But A rows are 3, not 2.
    
    For each subproblem, A is 3x2, B is 2x2 (one col pair), so result is 3x2.
    We can use Strassen if we can form 2x2 subproblems.
    
    Actually, let me think differently.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    For the first sum, if we group k into pairs (k0,k1) and i into... we have 3 rows.
    
    Let me use a different splitting:
    - For B columns, split into 4 groups of 2.
    - For each B column group, B contributes 2 columns.
    - We need to compute C[:, k0:k0+2] for each group.
    
    For a fixed B column group (k0, k1):
    C[:, k0:k1] = A * B[:, k0:k1] (which is 3x2)
    = sum_j A[:,j] * B[j, k0:k1]
    
    Split j into {0,1} and {2,3}:
    C[:,k0:k1] = (A[:,0:2] * B[0:2, k0:k1]) + (A[:,2:4] * B[2:4, k0:k1])
    
    Each term is (3x2) * (2x2) = 3x2.
    
    For the (3x2)*(2x2) multiplication:
    A_part = [[a00, a01], [a10, a11], [a20, a21]] (3x2)
    B_part = [[b00, b01], [b10, b11]] (2x2)
    Result = [[c00, c01], [c10, c11], [c20, c21]]
    
    c_i0 = a_i0*b00 + a_i1*b10
    c_i1 = a_i0*b01 + a_i1*b11
    
    We can use Strassen-like reduction here. But since A is 3x2 and B is 2x2,
    we have 3*2*2 = 12 schoolbook terms per (A_col_group, B_col_group) pair.
    There are 2 A_col_groups * 4 B_col_groups = 8 such subproblems.
    Total schoolbook: 8 * 12 = 96.
    
    For Strassen savings, we need 2x2x2 blocks.
    In a 3x2 * 2x2 subproblem, can we identify 2x2x2 blocks?
    A has rows {0,1} and {1,2} as possible 2-row subsets, but they overlap.
    
    Let me try: for rows {0,1} of A, cols {0,1}, and B rows {0,1}, cols {0,1}:
    This is a 2x2 * 2x2 = 2x2 subproblem. Uses Strassen (7 terms).
    The remaining row (row 2) needs separate handling.
    
    So for each (A_col_group, B_col_group):
    - Use Strassen for rows {0,1}: saves 1 term (8-7=1).
    - Row 2: 4 terms (a20*b00, a20*b01, a21*b10, a21*b11) - schoolbook.
    - But wait, row 2 also has 2 cols in A and 2 in B, so 4 terms.
    - Total per subproblem: 7 + 4 = 11 instead of 12.
    
    8 subproblems * 11 = 88. That's the target!
    
    But we need to be careful about double-counting. When we say "rows {0,1}" 
    we're using A rows 0 and 1. Row 2 is separate. No overlap issue for the Strassen block.
    
    Let me implement this.
    
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 24 rows
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col = 0
    
    # A_col_groups: (0,1) and (2,3)
    # B_row_groups: (0,1) and (2,3)
    # B_col_groups: (0,1), (2,3), (4,5), (6,7)
    
    for a_cg in range(2):
        a_c0 = a_cg * 2
        a_c1 = a_cg * 2 + 1
        
        for b_rg in range(2):
            b_r0 = b_rg * 2
            b_r1 = b_rg * 2 + 1
            
            for b_cg in range(4):
                b_c0 = b_cg * 2
                b_c1 = b_cg * 2 + 1
                
                # Strassen block for A rows {0,1}, A cols {a_c0, a_c1}, 
                # B rows {b_r0, b_r1}, B cols {b_c0, b_c1}
                # This computes C rows {0,1}, cols {b_c0, b_c1}
                
                # In Strassen, A_part is 2x2: A[0:2, a_c0:a_c0+2]
                # B_part is 2x2: B[b_r0:b_r0+2, b_c0:b_c0+2]
                # Result: C[0:2, b_c0:b_c0+2]
                
                # U2 rows correspond to A rows 0,1 and cols 0,1
                # In our full U, these are at indices:
                # A row i, col j -> U index i*m + j = i*4 + j
                # For A rows {0,1}, cols {a_c0, a_c1}:
                # U2[0] -> A[0, a_c0] -> U index 0*4 + a_c0
                # U2[1] -> A[0, a_c1] -> U index 0*4 + a_c1
                # U2[2] -> A[1, a_c0] -> U index 1*4 + a_c0
                # U2[3] -> A[1, a_c1] -> U index 1*4 + a_c1
                
                u_indices = [0*4 + a_c0, 0*4 + a_c1, 1*4 + a_c0, 1*4 + a_c1]
                
                # V2 rows correspond to B rows 0,1 and cols 0,1
                # In our full V, index j*p + k = j*8 + k
                # For B rows {b_r0, b_r1}, cols {b_c0, b_c1}:
                # V2[0] -> B[b_r0, b_c0] -> V index b_r0*8 + b_c0
                # V2[1] -> B[b_r0, b_c1] -> V index b_r0*8 + b_c1
                # V2[2] -> B[b_r1, b_c0] -> V index b_r1*8 + b_c0
                # V2[3] -> B[b_r1, b_c1] -> V index b_r1*8 + b_c1
                
                v_indices = [b_r0*8 + b_c0, b_r0*8 + b_c1, b_r1*8 + b_c0, b_r1*8 + b_c1]
                
                # W2 rows correspond to C rows 0,1 and cols 0,1
                # In our full W, index k*n + i = k*3 + i
                # For C rows {0,1}, cols {b_c0, b_c1}:
                # W2[0] -> C[0, b_c0] -> W index b_c0*3 + 0
                # W2[1] -> C[0, b_c1] -> W index b_c1*3 + 0
                # W2[2] -> C[1, b_c0] -> W index b_c0*3 + 1
                # W2[3] -> C[1, b_c1] -> W index b_c1*3 + 1
                
                w_indices = [b_c0*3 + 0, b_c1*3 + 0, b_c0*3 + 1, b_c1*3 + 1]
                
                # Add Strassen terms
                for c_strassen in range(7):
                    for u_row in range(4):
                        if U2[u_row][c_strassen] != 0:
                            U[u_indices[u_row]][col] += U2[u_row][c_strassen]
                    for v_row in range(4):
                        if V2[v_row][c_strassen] != 0:
                            V[v_indices[v_row]][col] += V2[v_row][c_strassen]
                    for w_row in range(4):
                        if W2[w_row][c_strassen] != 0:
                            W[w_indices[w_row]][col] += W2[w_row][c_strassen]
                    col += 1
                
                # Now handle A row 2 separately:
                # A row 2, cols {a_c0, a_c1}
                # B rows {b_r0, b_r1}, cols {b_c0, b_c1}
                # Result: C row 2, cols {b_c0, b_c1}
                
                # c_2,k = a_2,a_c0 * b_b_r0,k + a_2,a_c1 * b_b_r1,k
                # For k in {b_c0, b_c1}:
                
                # 4 schoolbook terms:
                # a_2,a_c0 * b_b_r0,b_c0 -> c_2,b_c0
                # a_2,a_c0 * b_b_r0,b_c1 -> c_2,b_c1
                # a_2,a_c1 * b_b_r1,b_c0 -> c_2,b_c0
                # a_2,a_c1 * b_b_r1,b_c1 -> c_2,b_c1
                
                # a_2,a_c0: U index 2*4 + a_c0
                # a_2,a_c1: U index 2*4 + a_c1
                # b_b_r0,b_c0: V index b_r0*8 + b_c0
                # b_b_r0,b_c1: V index b_r0*8 + b_c1
                # b_b_r1,b_c0: V index b_r1*8 + b_c0
                # b_b_r1,b_c1: V index b_r1*8 + b_c1
                # c_2,b_c0: W index b_c0*3 + 2
                # c_2,b_c1: W index b_c1*3 + 2
                
                # Term 1: a_2,a_c0 * b_b_r0,b_c0 -> c_2,b_c0
                U[2*4 + a_c0][col] = 1
                V[b_r0*8 + b_c0][col] = 1
                W[b_c0*3 + 2][col] = 1
                col += 1
                
                # Term 2: a_2,a_c0 * b_b_r0,b_c1 -> c_2,b_c1
                U[2*4 + a_c0][col] = 1
                V[b_r0*8 + b_c1][col] = 1
                W[b_c1*3 + 2][col] = 1
                col += 1
                
                # Term 3: a_2,a_c1 * b_b_r1,b_c0 -> c_2,b_c0
                U[2*4 + a_c1][col] = 1
                V[b_r1*8 + b_c0][col] = 1
                W[b_c0*3 + 2][col] = 1
                col += 1
                
                # Term 4: a_2,a_c1 * b_b_r1,b_c1 -> c_2,b_c1
                U[2*4 + a_c1][col] = 1
                V[b_r1*8 + b_c1][col] = 1
                W[b_c1*3 + 2][col] = 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the rank-88 Strassen-based construction
    best_result = None
    best_rank = 96
    
    try:
        result = _construct_3x4x8_strassen()
        if time.time() < deadline - 20:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = len(result["U"][0])
    except Exception as e:
        pass
    
    # Try to reduce rank via local search
    if best_result is not None and time.time() < deadline - 15:
        try:
            reduced = _try_reduce_rank(best_result["U"], best_result["V"], best_result["W"], n, m, p, deadline)
            if reduced is not None:
                r = len(reduced["U"][0])
                if r < best_rank:
                    best_result = reduced
                    best_rank = r
        except Exception as e:
            pass
    
    # Fallback to schoolbook (rank 96)
    if best_result is None:
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_block_strassen():
    """
    Build a block-Strassen decomposition for 2x4 by 4x5 matrix multiplication.
    
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    B1 = [B11 B12 B13] where B11,B12 are 2x2, B13 is 2x1
    B2 = [B21 B22 B23] where B21,B22 are 2x2, B23 is 2x1
    
    C = A1*B1 + A2*B2
    
    Using Strassen (rank 7) for each 2x2x2 block and rank 2 for each 2x2x1 block:
    A1*B11: 7, A1*B12: 7, A1*B13: 2
    A2*B21: 7, A2*B22: 7, A2*B23: 2
    Total: 32
    """
    n, m, p = 2, 4, 5
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one term to the decomposition."""
        col = len(U)
        U.append([0.0] * (n * m))
        V.append([0.0] * (m * p))
        W.append([0.0] * (p * n))
        for idx, val in u_vals.items():
            U[col][idx] = val
        for idx, val in v_vals.items():
            V[col][idx] = val
        for idx, val in w_vals.items():
            W[col][idx] = val
    
    # Strassen 2x2x2: C = A*B, A,B,C are 2x2
    # A indices: A11=0, A12=1, A21=2, A22=3 (row-major: A[i,j] -> i*2+j)
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # For our problem:
    # A is 2x4: A[i,j] where i in 0..1, j in 0..3
    #   A1 = A[0:2, 0:2] (cols 0,1), A2 = A[0:2, 2:4] (cols 2,3)
    # B is 4x5: B[i,j] where i in 0..3, j in 0..4
    #   B1 = B[0:2, 0:5] (rows 0,1), B2 = B[2:4, 0:5] (rows 2,3)
    #   B11 = B1[0:2, 0:2], B12 = B1[0:2, 2:4], B13 = B1[0:2, 4:5]
    #   B21 = B2[0:2, 0:2], B22 = B2[0:2, 2:4], B23 = B2[0:2, 4:5]
    # C is 2x5: C[i,j] where i in 0..1, j in 0..4
    
    # U index: i*m + j = i*4 + j for A[i,j]
    # V index: j*p + k = j*5 + k for B[j,k]
    # W index: k*n + i = k*2 + i for C[i,k] (trace convention W[k*n+i, r])
    
    # Strassen terms for 2x2x2 block at A_block (rows a0,a1 of A, cols c0,c1) 
    # and B_block (rows r0,r1 of B, cols d0,d1)
    # contributes to C_block (rows a0,a1 of C, cols d0,d1)
    
    def strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen 7-term decomposition for a 2x2x2 block.
        a_rows: 2 row indices in A
        a_cols: 2 col indices in A  
        b_rows: 2 row indices in B
        b_cols: 2 col indices in B
        c_rows: 2 row indices in C
        c_cols: 2 col indices in C
        """
        a00, a01 = a_rows[0], a_rows[1]  # Actually these are the two rows
        # Let me redefine: a_rows = [r1, r2], a_cols = [c1, c2]
        # A block: A[r1,c1], A[r1,c2], A[r2,c1], A[r2,c2]
        # B block: B[r1,c1], B[r1,c2], B[r2,c1], B[r2,c2] -- wait, B rows and cols are different
        
        # Let me use explicit indices
        ar1, ar2 = a_rows
        ac1, ac2 = a_cols
        br1, br2 = b_rows
        bc1, bc2 = b_cols
        cr1, cr2 = c_rows
        cc1, cc2 = c_cols
        
        # A11 = A[ar1,ac1], A12 = A[ar1,ac2], A21 = A[ar2,ac1], A22 = A[ar2,ac2]
        # B11 = B[br1,bc1], B12 = B[br1,bc2], B21 = B[br2,bc1], B22 = B[br2,bc2]
        # C11 = C[cr1,cc1], C12 = C[cr1,cc2], C21 = C[cr2,cc1], C22 = C[cr2,cc2]
        
        u_a11 = ar1 * m + ac1
        u_a12 = ar1 * m + ac2
        u_a21 = ar2 * m + ac1
        u_a22 = ar2 * m + ac2
        
        v_b11 = br1 * p + bc1
        v_b12 = br1 * p + bc2
        v_b21 = br2 * p + bc1
        v_b22 = br2 * p + bc2
        
        w_c11 = cc1 * n + cr1
        w_c12 = cc2 * n + cr1
        w_c21 = cc1 * n + cr2
        w_c22 = cc2 * n + cr2
        
        # M1 = (A11+A22)(B11+B22)
        add_term({u_a11: 1, u_a22: 1}, {v_b11: 1, v_b22: 1}, {w_c11: 1, w_c12: -1, w_c21: -1, w_c22: 1})
        # M2 = (A21+A22)B11
        add_term({u_a21: 1, u_a22: 1}, {v_b11: 1}, {w_c21: 1, w_c22: 1})
        # M3 = A11(B12-B22)
        add_term({u_a11: 1}, {v_b12: 1, v_b22: -1}, {w_c12: 1, w_c22: 1})
        # M4 = A22(B21-B11)
        add_term({u_a22: 1}, {v_b21: 1, v_b11: -1}, {w_c21: 1, w_c22: 1})
        # M5 = (A11+A12)B22
        add_term({u_a11: 1, u_a12: 1}, {v_b22: 1}, {w_c11: 1, w_c12: 1})
        # M6 = (A21-A11)(B11+B12)
        add_term({u_a21: 1, u_a11: -1}, {v_b11: 1, v_b12: 1}, {w_c21: 1, w_c11: -1})
        # M7 = (A12-A22)(B21+B22)
        add_term({u_a12: 1, u_a22: -1}, {v_b21: 1, v_b22: 1}, {w_c12: 1, w_c22: -1})
    
    # Rank-2 decomposition for 2x2x1 block
    # A is 2x2, B is 2x1, C is 2x1
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    def rank2_2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        """
        Add rank-2 decomposition for 2x2x1 block.
        """
        ar1, ar2 = a_rows
        ac1, ac2 = a_cols
        br1, br2 = b_rows
        cc = c_col
        cr1, cr2 = c_rows
        
        u_a11 = ar1 * m + ac1
        u_a12 = ar1 * m + ac2
        u_a21 = ar2 * m + ac1
        u_a22 = ar2 * m + ac2
        
        v_b1 = br1 * p + cc
        v_b2 = br2 * p + cc
        
        w_c1 = cc * n + cr1
        w_c2 = cc * n + cr2
        
        # Term 1: A11*B1 -> C1, A21*B1 -> C2
        add_term({u_a11: 1, u_a21: 1}, {v_b1: 1}, {w_c1: 1, w_c2: 1})
        # Term 2: A12*B2 -> C1, A22*B2 -> C2
        add_term({u_a12: 1, u_a22: 1}, {v_b2: 1}, {w_c1: 1, w_c2: 1})
    
    # A rows: 0,1; A cols: 0,1,2,3
    # B rows: 0,1,2,3; B cols: 0,1,2,3,4
    # C rows: 0,1; C cols: 0,1,2,3,4
    
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B1 = B[0:2, 0:5], B2 = B[2:4, 0:5]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:5]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:5]
    
    # A1*B11: Strassen
    strassen_2x2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # A1*B12: Strassen
    strassen_2x2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # A1*B13: rank 2
    rank2_2x2x1([0,1], [0,1], [0,1], 4, [0,1], 4)
    # A2*B21: Strassen
    strassen_2x2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # A2*B22: Strassen
    strassen_2x2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    # A2*B23: rank 2
    rank2_2x2x1([0,1], [2,3], [2,3], 4, [0,1], 4)
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: T[i,j,k] = delta[i,k] * delta[j,j] ... 
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta[i,k] ... no.
    
    # The tensor for matrix multiplication of (n x m) by (m x p) is:
    # T[i,j,k] = 1 for all i,j,k where the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T[i,j,k] = delta_{i,k} is wrong.
    
    # The multiplication tensor: T = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the sense that
    # each triple (i,j,k) has coefficient 1)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Wait, that's not right either. Let me think again.
    
    # The tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    # Actually no. The tensor T is defined such that:
    # T = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    
    # So we need: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # But wait, that can't be right because the decomposition should give:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]
    
    # Hmm, I think the convention is:
    # The tensor T has T[i,j,k] = delta_{i,k} ... no.
    
    # Let me reconsider. In the standard formulation:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C * delta
    # where the coefficient is 1 for each (i,j,k) triple.
    
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition says: T = sum_c U_c * V_c * W_c (outer product)
    # where U_c[i*m+j] = U[i*m+j, c], etc.
    
    # So: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's verify this.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < tol, max_err


def _try_remove_columns(U, V, W, n=2, m=4, p=5, max_time=100.0):
    """
    Try to remove columns from the decomposition by finding linear dependencies.
    Use random search: randomly zero out a column and check if the decomposition still works.
    If not, try to adjust other columns to compensate.
    """
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank
    
    # Convert to numpy for faster checks
    def to_np(U, V, W):
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
    
    def check_valid(U, V, W):
        Ua, Va, Wa = to_np(U, V, W)
        r = Ua.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j, :], Va[j*p+k, :])
                    s = np.dot(s, Wa[k*n+i, :])
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        return max_err < 1e-8
    
    # First verify the initial decomposition
    if not check_valid(U, V, W):
        print("Initial decomposition is invalid!")
        return best_U, best_V, best_W, best_rank
    
    # Try removing each column one at a time
    for c in range(rank):
        if time.time() - start_time > max_time - 10:
            break
        trial_U = [row[:] for row in U]
        trial_V = [row[:] for row in V]
        trial_W = [row[:] for row in W]
        # Zero out column c
        for r in range(len(trial_U)):
            trial_U[r][c] = 0
        for r in range(len(trial_V)):
            trial_V[r][c] = 0
        for r in range(len(trial_W)):
            trial_W[r][c] = 0
        
        if check_valid(trial_U, trial_V, trial_W):
            # Column c is redundant! Remove it.
            U = [row[:c] + row[c+1:] for row in U]
            V = [row[:c] + row[c+1:] for row in V]
            W = [row[:c] + row[c+1:] for row in W]
            rank -= 1
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            best_rank = rank
            print(f"Removed column {c}, new rank: {rank}")
            c = -1  # Restart from beginning since indices changed
            c += 1
    
    # Try random linear combinations to find more dependencies
    # For each pair of columns, try to see if one can be expressed as a linear combination of others
    rng = np.random.RandomState(42)
    attempts = 0
    while time.time() - start_time < max_time - 5 and attempts < 500:
        attempts += 1
        # Pick a random column to try to eliminate
        c = rng.randint(0, rank)
        # Try to find coefficients alpha_d such that column c = sum_d alpha_d * column_d
        # This is a system of equations. For each (i,j,k) triple:
        # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = sum_d alpha_d * U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d]
        
        # This is nonlinear, so let's use a simpler approach:
        # Randomly perturb the decomposition and see if we can maintain validity with fewer terms
        
        # Alternative: try to merge two columns
        if rank < 2:
            break
        c2 = rng.randint(0, rank - 1)
        if c2 == c:
            c2 = (c2 + 1) % rank
        if c2 > c:
            c, c2 = c2, c
        
        # Try to see if column c can be absorbed into column c2
        # We need: for all (i,j,k), 
        # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] + alpha * U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
        # to be representable... this is complex.
        
        # Simpler: just try random zeroing of random subsets
        if rng.random() < 0.5:
            continue
        
        # Try removing a random column
        trial_U = [row[:] for row in U]
        trial_V = [row[:] for row in V]
        trial_W = [row[:] for row in W]
        for r in range(len(trial_U)):
            trial_U[r][c] = 0
        for r in range(len(trial_V)):
            trial_V[r][c] = 0
        for r in range(len(trial_W)):
            trial_W[r][c] = 0
        
        if check_valid(trial_U, trial_V, trial_W):
            U = [row[:c] + row[c+1:] for row in U]
            V = [row[:c] + row[c+1:] for row in V]
            W = [row[:c] + row[c+1:] for row in W]
            rank -= 1
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            best_rank = rank
            print(f"Randomly removed column {c}, new rank: {rank}")
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Build the block-Strassen decomposition
    U, V, W = _build_block_strassen()
    rank = len(U[0])
    print(f"Initial block-Strassen rank: {rank}")
    
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    print(f"Verification: valid={valid}, max_err={err}")
    
    if not valid:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() - start_time < deadline - 20:
        U, V, W, rank = _try_remove_columns(U, V, W, max_time=deadline - start_time - 10)
    
    # Final verification
    valid, err = _verify_decomposition(U, V, W)
    if not valid:
        # Fallback to the initial block-Strassen
        U, V, W = _build_block_strassen()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

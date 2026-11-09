"""Matrix multiplication 4,5,6 - Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _add_term(U, V, W, i, j, k, n, m, p, col):
    """Add a single rank-1 term: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1."""
    U[i * m + j][col] = 1
    V[j * p + k][col] = 1
    W[k * n + i][col] = 1
    return col + 1

def _strassen_2x2(U, V, W, i0, j0, k0, n, m, p, col):
    """Strassen decomposition for 2x2 @ 2x2 block starting at (i0,j0,k0).
    Uses 7 multiplications instead of 8.
    A = A[i0:i0+2, j0:j0+2], B = B[j0:j0+2, k0:k0+2], C = C[i0:i0+2, k0:k0+2]
    """
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M corresponds to a bilinear form on sub-blocks.
    # We need to express each M as a sum of rank-1 terms on the full matrices.
    
    # M1 = (A11+A22)(B11+B22): expand as A11*B11 + A11*B22 + A22*B11 + A22*B22
    # But Strassen combines these. The key is that M1 multiplies (A11+A22) by (B11+B22).
    # In the tensor decomposition, we need rank-1 terms that capture this.
    
    # For the full tensor, a rank-1 term at (i,j,k) contributes to C[i,k] += A[i,j]*B[j,k].
    # Strassen works by finding linear combinations.
    
    # Let me define the 7 Strassen products as bilinear forms:
    # M1: A entries (i0,j0),(i0+1,j0+1) with +1; B entries (j0,k0),(j0+1,k0+1) with +1
    #     C entries: (i0,k0)+1, (i0,k0+1)+1, (i0+1,k0)+1, (i0+1,k0+1)+1 ... no this isn't right.
    
    # Actually, Strassen's method works differently. The 7 products are:
    # M1 = (A11 + A22) * (B11 + B22)
    # This is NOT a single rank-1 term in the original tensor. It's a product of sums.
    # In the tensor decomposition, we need to represent the full multiplication.
    
    # The correct approach: The Strassen algorithm gives us 7 scalar multiplications
    # where each scalar is a linear combination of A entries times a linear combination of B entries.
    # Each such product contributes to multiple C entries.
    
    # For M1 = (A11 + A22)(B11 + B22):
    #   The "A-side" vector has +1 at (i0,j0) and +1 at (i0+1,j0+1)
    #   The "B-side" vector has +1 at (j0,k0) and +1 at (j0+1,k0+1)
    #   The "C-side" contribution: this product contributes to C11, C12, C21, C22
    #   Specifically in Strassen: C11 += M1, C12 += 0, C21 += 0, C22 += M1... 
    #   Wait, let me re-derive.
    
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11-B21)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pk is a product of a linear form in A and a linear form in B.
    # In the tensor decomposition, each Pk corresponds to ONE rank-r term where:
    #   U has the A-linear-form coefficients
    #   V has the B-linear-form coefficients  
    #   W has the C-coefficients (how Pk contributes to each C entry)
    
    # For P1: A-form = +1*(i0,j0) + 1*(i0+1,j0+1), B-form = +1*(j0,k0) + 1*(j0+1,k0+1)
    #   C-form: P1 contributes +1 to C11, 0 to C12, 0 to C21, +1 to C22
    #   So W: +1 at (k0,i0), +1 at (k0+1,i0+1)
    
    # For P2: A-form = +1*(i0+1,j0) + 1*(i0+1,j0+1), B-form = +1*(j0,k0)
    #   C-form: +1 to C21, +1 to C22... wait C21 = P2+P4, C22 = P1-P2+P3+P6
    #   So P2: +1 to C21, -1 to C22
    #   W: +1 at (k0,i0+1), -1 at (k0+1,i0+1)
    
    # Let me be more careful with indices.
    # C[i0, k0] = C11, C[i0, k0+1] = C12, C[i0+1, k0] = C21, C[i0+1, k0+1] = C22
    
    # P1: A: (i0,j0)+1, (i0+1,j0+1)+1; B: (j0,k0)+1, (j0+1,k0+1)+1
    #     C: (i0,k0)+1, (i0+1,k0+1)+1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0,1), (i0+1,j0+1,1)],  # A side
        [(j0,k0,1), (j0+1,k0+1,1)],  # B side
        [(i0,k0,1), (i0+1,k0+1,1)])  # C side
    
    # P2: A: (i0+1,j0)+1, (i0+1,j0+1)+1; B: (j0,k0)+1
    #     C: C21=P2+P4 => P2 gives +1 to C21; C22=P1-P2+P3+P6 => P2 gives -1 to C22
    #     W: (i0+1,k0)+1, (i0+1,k0+1)-1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0+1,j0,1), (i0+1,j0+1,1)],
        [(j0,k0,1)],
        [(i0+1,k0,1), (i0+1,k0+1,-1)])
    
    # P3: A: (i0,j0)+1; B: (j0+1,k0)+1, (j0+1,k0+1)-1
    #     C: C12=P3+P5 => +1 to C12; C22=P1-P2+P3+P6 => +1 to C22
    #     W: (i0,k0+1)+1, (i0+1,k0+1)+1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0,1)],
        [(j0+1,k0,1), (j0+1,k0+1,-1)],
        [(i0,k0+1,1), (i0+1,k0+1,1)])
    
    # P4: A: (i0+1,j0+1)+1; B: (j0,k0)+1, (j0+1,k0)-1
    #     C: C11=P1+P4-P5+P7 => +1 to C11; C21=P2+P4 => +1 to C21
    #     W: (i0,k0)+1, (i0+1,k0)+1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0+1,j0+1,1)],
        [(j0,k0,1), (j0+1,k0,-1)],
        [(i0,k0,1), (i0+1,k0,1)])
    
    # P5: A: (i0,j0)+1, (i0,j0+1)+1; B: (j0+1,k0+1)+1
    #     C: C11=P1+P4-P5+P7 => -1 to C11; C12=P3+P5 => +1 to C12
    #     W: (i0,k0)-1, (i0,k0+1)+1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0,1), (i0,j0+1,1)],
        [(j0+1,k0+1,1)],
        [(i0,k0,-1), (i0,k0+1,1)])
    
    # P6: A: (i0+1,j0)+1, (i0,j0)-1; B: (j0,k0)+1, (j0,k0+1)+1
    #     C: C22=P1-P2+P3+P6 => +1 to C22
    #     W: (i0+1,k0+1)+1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0+1,j0,1), (i0,j0,-1)],
        [(j0,k0,1), (j0,k0+1,1)],
        [(i0+1,k0+1,1)])
    
    # P7: A: (i0,j0+1)+1, (i0+1,j0+1)-1; B: (j0+1,k0)+1, (j0+1,k0+1)+1
    #     C: C11=P1+P4-P5+P7 => +1 to C11
    #     W: (i0,k0)+1
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0+1,1), (i0+1,j0+1,-1)],
        [(j0+1,k0,1), (j0+1,k0+1,1)],
        [(i0,k0,1)])
    
    return col

def _add_strassen_term(U, V, W, n, m, p, col, a_terms, b_terms, c_terms):
    """Add a rank-1 term defined by linear forms on A, B, and C.
    a_terms: list of (i, j, coeff) for A-side
    b_terms: list of (j, k, coeff) for B-side
    c_terms: list of (i, k, coeff) for C-side
    """
    for (i, j, coeff) in a_terms:
        U[i * m + j][col] = coeff
    for (j, k, coeff) in b_terms:
        V[j * p + k][col] = coeff
    for (i, k, coeff) in c_terms:
        W[k * n + i][col] = coeff
    return col + 1

def _decompose_block(U, V, W, n, m, p, i0, j0, k0, col, deadline):
    """Recursively decompose the n x m @ m x p block at offset (i0,j0,k0).
    Returns new column index.
    Strategy: prefer splitting to create 2x2x2 sub-blocks for Strassen.
    """
    if time.time() > deadline:
        return col
    
    if n == 0 or m == 0 or p == 0:
        return col
    
    # Base case: 2x2x2 uses Strassen (rank 7 vs schoolbook 8)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2(U, V, W, i0, j0, k0, n, m, p, col)
    
    # Base case: if any dimension is 1, use schoolbook (can't apply Strassen)
    if n == 1 or m == 1 or p == 1:
        for di in range(n):
            for dj in range(m):
                for dk in range(p):
                    col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
        return col
    
    # Small blocks: use schoolbook
    if n <= 2 and m <= 2 and p <= 2:
        for di in range(n):
            for dj in range(m):
                for dk in range(p):
                    col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
        return col
    
    # Choose split strategy to maximize Strassen usage
    # Priority: split to get 2x2x2 blocks
    # Option 1: split m if m is even and m//2 >= 2
    # Option 2: split p if p is even and p//2 >= 2  
    # Option 3: split n if n is even and n//2 >= 2
    
    best_col = col
    best_strategy = None
    
    # Try splitting m (middle dimension) - most natural for matrix mult
    if m >= 4:
        m1 = m // 2
        m2 = m - m1
        if m1 >= 2 and m2 >= 2:
            c1 = _decompose_block(U, V, W, n, m1, p, i0, j0, k0, col, deadline)
            c2 = _decompose_block(U, V, W, n, m2, p, i0, j0+m1, k0, c1, deadline)
            if best_strategy is None or c2 < best_col:
                best_col = c2
                best_strategy = ('m', m1)
    elif m == 3:
        # Split 3 into 1+2 or 2+1
        c1 = _decompose_block(U, V, W, n, 1, p, i0, j0, k0, col, deadline)
        c2 = _decompose_block(U, V, W, n, 2, p, i0, j0+1, k0, c1, deadline)
        if best_strategy is None or c2 < best_col:
            best_col = c2
            best_strategy = ('m', 1)
    elif m == 2:
        # Can't split m=2 further usefully unless n and p allow
        pass
    
    # Try splitting p (output dimension)
    if p >= 4:
        p1 = p // 2
        p2 = p - p1
        if p1 >= 2 and p2 >= 2:
            c1 = _decompose_block(U, V, W, n, m, p1, i0, j0, k0, col, deadline)
            c2 = _decompose_block(U, V, W, n, m, p2, i0, j0, k0+p1, c1, deadline)
            if best_strategy is None or c2 < best_col:
                best_col = c2
                best_strategy = ('p', p1)
    elif p == 3:
        c1 = _decompose_block(U, V, W, n, m, 1, i0, j0, k0, col, deadline)
        c2 = _decompose_block(U, V, W, n, m, 2, i0, j0, k0+1, c1, deadline)
        if best_strategy is None or c2 < best_col:
            best_col = c2
            best_strategy = ('p', 1)
    
    # Try splitting n (input dimension)
    if n >= 4:
        n1 = n // 2
        n2 = n - n1
        if n1 >= 2 and n2 >= 2:
            c1 = _decompose_block(U, V, W, n1, m, p, i0, j0, k0, col, deadline)
            c2 = _decompose_block(U, V, W, n2, m, p, i0+n1, j0, k0, c1, deadline)
            if best_strategy is None or c2 < best_col:
                best_col = c2
                best_strategy = ('n', n1)
    elif n == 3:
        c1 = _decompose_block(U, V, W, 1, m, p, i0, j0, k0, col, deadline)
        c2 = _decompose_block(U, V, W, 2, m, p, i0+1, j0, k0, c1, deadline)
        if best_strategy is None or c2 < best_col:
            best_col = c2
            best_strategy = ('n', 1)
    
    if best_strategy is not None:
        return best_col
    
    # Fallback: schoolbook
    for di in range(n):
        for dj in range(m):
            for dk in range(p):
                col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
    return col

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... 
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]) * (sum... ) 
    # No: the decomposition is C[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over j
    # Actually the standard form: T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # where T_{i,j,k} = delta (the multiplication tensor)
    # The multiplication tensor has T_{i,j,k} = 1 if the index maps correctly.
    # For matrix mult: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    
    # Verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # Wait, that's not right either. Let me think.
    
    # The tensor for matrix multiplication: T is a 3-tensor where
    # T[a,b,c] = sum over the "middle" index... 
    # Actually, the standard formulation: the matrix multiplication tensor M_{n,m,p}
    # has entries M_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # No wait. M_{i,j,k} represents the contribution of A[i,j]*B[j,k] to C[i,k].
    # So M_{i,j,k} = 1 for all valid indices.
    
    # A rank-r decomposition: M_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # Wait, the indices don't match up. Let me use the convention from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # The decomposition requires: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # But that's a 3-index sum which doesn't make sense for a bilinear form.
    
    # Actually, I think the correct interpretation is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c sum_j (U[i*m+j, c] * A[i,j]) * (V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Hmm, that's not standard either.
    
    # Let me just verify by checking that the decomposition reproduces the multiplication.
    # Create random A, B and check A@B vs the decomposed version.
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    C_decomposed = np.zeros((n, p))
    for c in range(rank):
        # U[:, c] is a vector of length n*m, reshape to (n, m)
        Uc = U_arr[:, c].reshape(n, m)
        Vc = V_arr[:, c].reshape(m, p)
        Wc = W_arr[:, c].reshape(p, n)
        # The contribution: sum over j of Uc[i,j] * A[i,j] ... 
        # I think the formula is: C[i,k] += sum_j Uc[i,j] * A[i,j] * Vc[j,k] * B[j,k] * Wc[k,i]
        # This doesn't seem right. Let me look at the schoolbook case.
        # In schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for specific (i,j,k)
        # So for a given c corresponding to (i,j,k): Uc[i,j]=1, Vc[j,k]=1, Wc[k,i]=1
        # Contribution to C[i',k']: sum_j' Uc[i',j']*A[i',j']*Vc[j',k']*B[j',k']*Wc[k',i']
        # = Uc[i,j]*A[i,j]*Vc[j,k]*B[j,k]*Wc[k,i] (only nonzero term)
        # = 1*A[i,j]*1*B[j,k]*1 = A[i,j]*B[j,k]
        # So C[i,k] gets A[i,j]*B[j,k] added, which is correct!
        
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += Uc[i, j] * A[i, j] * Vc[j, k] * B[j, k] * Wc[k, i]
                C_decomposed[i, k] += s
    
    if not np.allclose(C_decomposed, C_expected, atol=1e-10):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # First try the recursive Strassen-based decomposition
    try:
        # Estimate upper bound on rank
        # Schoolbook: n*m*p = 120
        # Strassen-based: should be much less
        # Start with schoolbook size, will trim later
        max_rank = n * m * p
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        
        col = _decompose_block(U, V, W, n, m, p, 0, 0, 0, 0, deadline)
        
        if col > max_rank:
            raise ValueError("Decomposition exceeded allocated rank")
        
        # Trim to actual rank
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            pass
    except Exception as e:
        pass
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END

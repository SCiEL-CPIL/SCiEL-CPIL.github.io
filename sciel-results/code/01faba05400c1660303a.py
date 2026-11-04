"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_tensors():
    """Return U, V, W (each 4x7) for Strassen's 2x2 matrix multiplication.
    Indexing: U row = i*m+j (i,j in {0,1}), V row = j*p+k, W row = k*n+i.
    The 7 products for Strassen:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B10-B11)
      M4 = A11*(B01-B11)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B10)
      M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U: 4 rows (i*2+j for i,j in 0,1), 7 columns
    # V: 4 rows (j*2+k for j,k in 0,1), 7 columns
    # W: 4 rows (k*2+i for k,i in 0,1), 7 columns
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: A00=row0, A01=row1, A10=row2, A11=row3
    # B indices: B00=row0, B01=row1, B10=row2, B11=row3
    # C indices: C00=row0, C01=row1, C10=row2, C11=row3 (W row = k*n+i, n=2)
    #   C00: k=0,i=0 -> row 0
    #   C01: k=0,i=1 -> row 1
    #   C10: k=1,i=0 -> row 2
    #   C11: k=1,i=1 -> row 3
    
    # M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)*B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # M3 = A00*(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A11*(B01-B11)
    U[3,3] = 1
    V[1,3] = 1; V[3,3] = -1
    W[0,3] = 1; W[3,3] = 1
    
    # M5 = (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A10-A00)*(B00+B10)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)*(B01+B11)
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[1,6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    # Compute the multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    # For each rank-1 component: U[:,r] * V[:,r] * W[:,r]
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Check a random sample
    rng = np.random.default_rng(42)
    for _ in range(200):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        val = 0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
        if abs(val - 1.0) > tol:
            return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
    """Vectorized verification."""
    rank = U.shape[1]
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute sum over rank: U[a,r]*V[b,r]*W[c,r]
    # Use einsum: result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    if np.max(np.abs(result - T)) > tol:
        return False
    return True

def _strassen_4x4_via_kron():
    """Build 4x4 Strassen decomposition using Kronecker products.
    
    Split 4x4 into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each block is 2x2
    B = [[B00, B01], [B10, B11]] where each block is 2x2
    C = [[C00, C01], [C10, C11]] where each block is 2x2
    
    Apply Strassen to the 2x2 block structure, where each "entry" is a 2x2 matrix.
    But Strassen's algorithm uses additions/multiplications of blocks.
    Each block multiplication is a 2x2*2x2 = 4x4 multiplication.
    
    Actually, the standard approach: if we have a rank-r decomposition for 2x2,
    we get a rank-r decomposition for 4x4 by "tensoring" with the 2x2 identity.
    Wait, that's not quite right either.
    
    Let me think more carefully. For 4x4 matrix multiplication:
    A (4x4) * B (4x4) = C (4x4)
    
    If we write A, B, C as 2x2 block matrices with 2x2 blocks:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = A*B
    
    Then C00 = A00*B00 + A01*B10, etc.
    
    Now, using Strassen's 7 multiplications for the block-level 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B11)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    Each of these M_i is a 2x2 * 2x2 = 4x4 matrix.
    So each M_i requires a 4x4 matrix multiplication... which is what we're trying to do!
    
    This is circular. The issue is that Strassen at the block level doesn't directly
    give us a rank decomposition for 4x4 unless we can handle the 2x2 block multiplications.
    
    However, if we use the schoolbook 2x2 decomposition (rank 8) for the inner 2x2
    multiplications, we'd get rank 7*8 = 56, which is worse than schoolbook 64... wait no,
    7*8=56 < 64. Hmm, but that's not how it works for tensor decompositions.
    
    Actually, for tensor decompositions, if we have a rank-r decomposition for n x n
    multiplication, we can get a rank-r^2 decomposition for (2n) x (2n) multiplication
    by the tensor product construction. But that gives rank 7^2 = 49 for 4x4.
    
    Wait, let me reconsider. The standard result: if μ(2) = 7, then μ(4) ≤ 49?
    No, that's not right either. The tensor product of two rank-r decompositions
    gives a rank-r^2 decomposition. But we can do better.
    
    Actually, the correct statement: if we have a rank-r algorithm for n×n matrix
    multiplication, then we have a rank-r^2 algorithm for (n×n)⊗(n×n) = (n^2)×(n^2)
    matrix multiplication. So μ(4) ≤ μ(2)^2 = 49.
    
    But actually, we can do even better by using the structure more carefully.
    
    Let me just implement the tensor product construction:
    - Take the 2x2 Strassen decomposition (rank 7)
    - Tensor it with itself to get 4x4 (rank 49)
    
    For the tensor product:
    If (U, V, W) is a rank-r decomposition for n×n, then
    (U⊗U, V⊗V, W⊗W) is a rank-r^2 decomposition for (n^2)×(n^2).
    
    The indexing works out with the Kronecker product.
    """
    U2, V2, W2 = _strassen_2x2_tensors()
    
    # Kronecker product: U4 = U2 ⊗ U2, etc.
    # Shape: (4*4, 7*7) = (16, 49)
    U4 = np.kron(U2, U2)  # (4,7) ⊗ (4,7) = (16, 49)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # But we need to permute the indices to match the standard 4x4 indexing.
    # U row index: i*m+j where i,j in {0,1,2,3}, m=4
    # With Kronecker: row index = i1*2*2 + i2*2 + j1*2 + j2? No...
    # Actually, np.kron(A, B) for A of shape (r1,c1) and B of shape (r2,c2)
    # gives shape (r1*r2, c1*c2).
    # Row index in kron: r1*2 + r2 (0-indexed), i.e., r1*r2_dim + r2
    # For U2 of shape (4,7): row i in U2 corresponds to i*2+j in the 2x2 A matrix
    #   i=0: A00 -> rows 0,1 (j=0,1)
    #   i=1: A01 -> rows 2,3 (j=0,1)
    #   i=2: A10 -> rows 4,5
    #   i=3: A11 -> rows 6,7
    # Wait, U2 has 4 rows. Let me re-examine.
    
    # In my _strassen_2x2_tensors(), U has 4 rows indexed 0..3:
    # Row 0: A00
    # Row 1: A01
    # Row 2: A10
    # Row 3: A11
    # This is i*2+j with i being the block row, j being the element within block.
    # So row = i*2 + j where i∈{0,1} is block index, j∈{0,1} is element index.
    
    # For 4x4, we want row = I*4 + J where I∈{0,1,2,3} is the row of the 4x4 matrix.
    # Block structure: row I = i*2 + i_local, col J = j*2 + j_local
    # where i,j ∈ {0,1} are block indices, i_local, j_local ∈ {0,1} are local indices.
    # So the 4x4 matrix has rows indexed 0..3, which decompose as (block_row, local_row).
    
    # The Kronecker product U2⊗U2 has rows indexed (row_a, row_b) where row_a, row_b ∈ {0,1,2,3}
    # The row index in the kron is row_a*4 + row_b.
    # row_a = i_a*2 + j_a (block row i_a, local row j_a)
    # row_b = i_b*2 + j_b (block row i_b, local row j_b)
    # Combined row index: row_a*4 + row_b = (i_a*2+j_a)*4 + (i_b*2+j_b) = i_a*8 + j_a*4 + i_b*2 + j_b
    
    # But we want the 4x4 row index to be: I = i_a*2 + i_b (block-level row), and within that,
    # the local row is j = j_a*2 + j_b? No, that's not standard.
    
    # Actually for a 4x4 matrix written as 2x2 blocks of 2x2 matrices:
    # A[I,J] where I = i*2 + a, J = j*2 + b, with i,j,a,b ∈ {0,1}
    # The block A[i,j] is a 2x2 matrix with entries A[i,j][a,b] = A[i*2+a, j*2+b]
    
    # So the 4x4 matrix element at (I, J) where I = i*2+a, J = j*2+b
    # The row index I goes from 0 to 3.
    
    # In the Kronecker product construction for tensor decompositions:
    # The "flattened" input index for A is I*4 + J (row-major for 4x4)
    # But wait, in our convention, U has n*m = 16 rows, indexed by i*m+j where i is row of A, j is col of A.
    # So U row index = I*4 + J where I = row of A (0..3), J = col of A (0..3).
    
    # In the Kronecker product of U2⊗U2:
    # U2 row index = i*2 + j where i ∈ {0,1} (block row), j ∈ {0,1} (local row within block)
    # Wait no. In the 2x2 case, U row index = i*m+j = i*2+j where i is row of 2x2 A, j is col of 2x2 A.
    # So U2 row 0 = (i=0,j=0) = A[0,0]
    # U2 row 1 = (i=0,j=1) = A[0,1]
    # U2 row 2 = (i=1,j=0) = A[1,0]
    # U2 row 3 = (i=1,j=1) = A[1,1]
    
    # For 4x4: U row index = I*4 + J where I is row (0..3), J is col (0..3) of the 4x4 A.
    # Decompose I = i*2 + a, J = j*2 + b where i,j,a,b ∈ {0,1}
    # So U row = (i*2+a)*4 + (j*2+b) = i*8 + a*4 + j*2 + b
    
    # In Kronecker U2⊗U2, row index = u2_row_a * 4 + u2_row_b
    # where u2_row_a = i_a*2 + j_a and u2_row_b = i_b*2 + j_b
    # (using different letters to avoid confusion)
    # So kron row = (i_a*2+j_a)*4 + (i_b*2+j_b) = i_a*8 + j_a*4 + i_b*2 + j_b
    
    # We want: i*8 + a*4 + j*2 + b
    # Comparing: i_a=i, j_a=a, i_b=j, j_b=b
    # So the kron row index (i_a*8 + j_a*4 + i_b*2 + j_b) equals the desired 4x4 row index
    # (i*8 + a*4 + j*2 + b) when i_a=i, j_a=a, i_b=j, j_b=b.
    # This is exactly the natural ordering! No permutation needed for U.
    
    # Similarly for V: V row index = j*p + k (col of A * p + row of B)
    # For 2x2: V row = j*2 + k
    # For 4x4: V row = J*4 + K where J is col of A (0..3), K is row of B (0..3)
    # J = j*2 + b, K = k*2 + c (using different letters)
    # V row = (j*2+b)*4 + (k*2+c) = j*8 + b*4 + k*2 + c
    # Kron V2⊗V2 row = (j_a*2+b_a)*4 + (j_b*2+b_b) = j_a*8 + b_a*4 + j_b*2 + b_b
    # Same structure, no permutation needed.
    
    # For W: W row index = k*n + i (row of B * n + row of C)
    # For 2x2: W row = k*2 + i
    # For 4x4: W row = K*4 + I where K is row of B (0..3), I is row of C (0..3)
    # K = k*2 + c, I = i*2 + a
    # W row = (k*2+c)*4 + (i*2+a) = k*8 + c*4 + i*2 + a
    # Kron W2⊗W2 row = (k_a*2+c_a)*4 + (i_a*2+a_a) = k_a*8 + c_a*4 + i_a*2 + a_a
    # Same structure, no permutation needed.
    
    # So the Kronecker product directly gives the correct indexing!
    # Let's verify with a small test.
    
    return U4, V4, W4

def _convert_to_lists(U, V, W):
    """Convert numpy arrays to lists of lists with integer/half-integer entries."""
    U_list = []
    for row in U:
        U_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                U_row.append(int(round(val)))
            elif abs(val*2 - round(val*2)) < 1e-9:
                U_row.append(round(val*2) / 2.0)
            else:
                U_row.append(float(val))
        U_list.append(U_row)
    
    V_list = []
    for row in V:
        V_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                V_row.append(int(round(val)))
            elif abs(val*2 - round(val*2)) < 1e-9:
                V_row.append(round(val*2) / 2.0)
            else:
                V_row.append(float(val))
        V_list.append(V_row)
    
    W_list = []
    for row in W:
        W_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                W_row.append(int(round(val)))
            elif abs(val*2 - round(val*2)) < 1e-9:
                W_row.append(round(val*2) / 2.0)
            else:
                W_row.append(float(val))
        W_list.append(W_row)
    
    return U_list, V_list, W_list

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 4x4x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a fallback
    rank_school = n * m * p  # 64
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = (rank_school, U_school, V_school, W_school)
    
    # Try Strassen 4x4 via Kronecker product (rank 49)
    try:
        U4, V4, W4 = _strassen_4x4_via_kron()
        if _verify_decomposition_fast(U4, V4, W4, n, m, p):
            rank_49 = U4.shape[1]
            best_result = (rank_49, U4, V4, W4)
    except Exception:
        pass
    
    # Try to find a lower rank decomposition using optimization
    # The minimum rank for 4x4 is known to be 49 (from Strassen's construction)
    # Actually, it's known that μ(4) ≤ 49, and it's conjectured to be less.
    # Let's try to verify if we can do better with a search.
    
    # For now, return the best we have
    rank, U, V, W = best_result
    U_list, V_list, W_list = _convert_to_lists(U, V, W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

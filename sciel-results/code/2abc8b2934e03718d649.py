"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_decomposition(n, m, p):
    """
    Build a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying an n x m matrix by an m x p matrix.
    
    Strategy:
    - If m is even, split the m columns/rows into pairs of 2.
    - For each pair of columns (j, j+1), use a Strassen-like 2x2x2 decomposition
      for the 2 (output rows) x 2 (input cols) x p (output cols) sub-problem.
    - Actually, we can view the multiplication as:
        C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
      For each j, we have a term A[i, j] * B[j, k].
      
    A better approach:
    - The tensor is T[i, j, k] = 1 for all valid i, j, k.
    - We can decompose this by grouping the j dimension.
    
    For 2x3x6:
    - n=2, m=3, p=6.
    - Schoolbook rank = 36.
    - Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
    
    Let's try a block decomposition.
    We can split the 3 columns of A into 1 + 2.
    Or 1 + 1 + 1.
    
    Actually, a known result for 2x2x2 is rank 7 (Strassen).
    For 2x3x6, we can think of it as:
    C = A * B where A is 2x3, B is 3x6.
    
    We can split B into 3 blocks of 2 columns each: B = [B1, B2, B3] where Bi is 3x2.
    Then C = A * B = A1 * B1 + A2 * B2 + A3 * B3? No, that's not right.
    
    Let's use the Kronecker product structure.
    The multiplication tensor for n x m by m x p is the Kronecker product of:
    - Identity tensor for n x 1 by 1 x p? No.
    
    Let's try a direct construction for 2x3x6.
    We can use the fact that 3 = 2 + 1.
    C = A[:, 0] * B[0, :] + A[:, 1:3] * B[1:3, :]
    
    The first term is an outer product of a 2-vector and a 6-vector, which has rank 1.
    The second term is a 2x2 by 2x6 multiplication.
    
    For 2x2 by 2x6:
    We can split the 6 columns into 3 pairs of 2.
    So it's 3 times a 2x2 by 2x2 multiplication.
    Each 2x2 by 2x2 can be done with rank 7 (Strassen).
    So total rank = 1 + 3 * 7 = 22.
    
    Let's verify this.
    A is 2x3, B is 3x6.
    A = [a0, a1, a2] where a0 is 2x1, a1, a2 are 2x1.
    B = [b0, b1, b2] where b0, b1, b2 are 1x6.
    
    C = a0 * b0 + a1 * b1 + a2 * b2.
    
    This is just a sum of 3 outer products, each of rank 1.
    So the total rank is 3? No, that's not right.
    The tensor T[i, j, k] = 1.
    If we use the decomposition C = sum_j a_j * b_j, then:
    T[i, j, k] = sum_c U[i, j, c] * V[c, k] ... no.
    
    The standard tensor decomposition for matrix multiplication is:
    T[i, j, k] = sum_c U[i, c] * V[c, j] * W[j, k] ... no.
    
    The correct form is:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the outer product a_j * b_j:
    This corresponds to a rank-1 term where:
    U[i*m+j, c] = A[i, j]
    V[j*p+k, c] = B[j, k]
    W[k*n+i, c] = 1
    
    But we want a decomposition with constant coefficients, not depending on A and B.
    
    Let's go back to the block decomposition idea.
    For 2x2x2, Strassen uses 7 multiplications.
    For 2x2x6, we can split the 6 into 3*2, so 3 * 7 = 21.
    For 2x3x6, we can split the 3 into 1+2.
    The 1x2x6 part is just 6 outer products, rank 6.
    The 2x2x6 part is 21.
    Total rank = 6 + 21 = 27.
    
    Wait, the 1x2x6 part:
    A is 1x2, B is 2x6.
    C = A * B is 1x6.
    C[k] = A[0] * B[0, k] + A[1] * B[1, k].
    This is a sum of 2 outer products, each of rank 1.
    So rank 2? No, the tensor is T[i, j, k] = 1 for i=0, j=0,1, k=0..5.
    This is a 1x2x6 tensor.
    The rank of this tensor is 2 (since it's a sum of 2 rank-1 tensors).
    
    So total rank = 2 + 21 = 23.
    
    Let's try to construct this.
    
    For the 1x2x6 part (i=0, j=0,1, k=0..5):
    We need T[0, 0, k] = 1 and T[0, 1, k] = 1 for all k.
    This can be done with 2 rank-1 terms:
    Term 1: U[0, 0] = 1, V[0, k] = 1, W[k, 0] = 1 for all k.
    Term 2: U[0, 1] = 1, V[1, k] = 1, W[k, 0] = 1 for all k.
    Wait, W is indexed by [k*n+i, c]. For n=2, i=0, W[k*2+0, c].
    
    Let's use a systematic approach.
    We'll build the decomposition by combining:
    1. A 1x2x6 decomposition (rank 2)
    2. A 2x2x6 decomposition (rank 21)
    
    For the 1x2x6 part:
    We need T[0, 0, k] = 1 and T[0, 1, k] = 1 for k=0..5.
    This is equivalent to two 1x1x6 tensors, each of rank 1.
    So we need 2 terms.
    
    For the 2x2x6 part:
    We need T[1, 0, k] = 1 and T[1, 1, k] = 1 for k=0..5.
    Wait, no. The 2x2x6 part corresponds to i=1, j=0,1, k=0..5? No.
    
    Let's re-index.
    A is 2x3, so i=0,1 and j=0,1,2.
    B is 3x6, so j=0,1,2 and k=0..5.
    C is 2x6, so i=0,1 and k=0..5.
    
    We split j into j=0 and j=1,2.
    For j=0: T[i, 0, k] = 1 for i=0,1 and k=0..5.
    This is a 2x1x6 tensor, which has rank 2.
    
    For j=1,2: T[i, j, k] = 1 for i=0,1, j=1,2, k=0..5.
    This is a 2x2x6 tensor, which has rank 21 (3 * 7).
    
    Total rank = 2 + 21 = 23.
    
    Let's implement this.
    """
    n, m, p = 2, 3, 6
    
    # Part 1: j=0, i=0,1, k=0..5 (rank 2)
    # Part 2: j=1,2, i=0,1, k=0..5 (rank 21)
    
    # We'll build the U, V, W matrices.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # Part 1: j=0
    # We need T[i, 0, k] = 1 for i=0,1, k=0..5.
    # This is a 2x1x6 tensor.
    # Rank 2 decomposition:
    # Term 1: U[i*3+0, c1] = 1, V[0*6+k, c1] = 1, W[k*2+i, c1] = 1
    # Term 2: U[i*3+0, c2] = 1, V[0*6+k, c2] = 1, W[k*2+i, c2] = 1
    # Wait, this doesn't work because W[k*2+i, c] would be 1 for both i=0 and i=1.
    
    # Let's use a different approach.
    # For the 2x1x6 tensor, we can use 2 terms:
    # Term 1: U[0*3+0, c1] = 1, V[0*6+k, c1] = 1, W[k*2+0, c1] = 1
    # Term 2: U[1*3+0, c2] = 1, V[0*6+k, c2] = 1, W[k*2+1, c2] = 1
    # This gives T[0, 0, k] = 1 and T[1, 0, k] = 1.
    
    # Part 2: j=1,2
    # We need T[i, j, k] = 1 for i=0,1, j=1,2, k=0..5.
    # This is a 2x2x6 tensor.
    # We can split k into 3 pairs: (0,1), (2,3), (4,5).
    # For each pair, we use a 2x2x2 Strassen decomposition (rank 7).
    
    # Let's implement the Strassen 2x2x2 decomposition.
    # For a 2x2x2 tensor T[i, j, k] = 1:
    # We use 7 terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B11 - B01)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # The output is:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map this to our indices.
    # For the 2x2x2 tensor with i=0,1, j=0,1, k=0,1:
    # A[i, j] corresponds to U[i*2+j, c]
    # B[j, k] corresponds to V[j*2+k, c]
    # C[i, k] corresponds to W[k*2+i, c]
    
    # We'll build the decomposition for the 2x2x6 tensor.
    # For each k-pair (2*t, 2*t+1), we use a 2x2x2 Strassen decomposition.
    
    # Let's create the U, V, W matrices.
    rank_part1 = 2
    rank_part2 = 21
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: j=0
    # Term 1: i=0
    U[0 * m + 0, 0] = 1
    for k in range(p):
        V[0 * p + k, 0] = 1
        W[k * n + 0, 0] = 1
    
    # Term 2: i=1
    U[1 * m + 0, 1] = 1
    for k in range(p):
        V[0 * p + k, 1] = 1
        W[k * n + 1, 1] = 1
    
    # Part 2: j=1,2
    # For each k-pair (2*t, 2*t+1), we use a 2x2x2 Strassen decomposition.
    # We'll use 7 terms per k-pair.
    
    # Strassen coefficients for 2x2x2:
    # M1: A00 + A11, B00 + B11
    # M2: A10 + A11, B00
    # M3: A00, B10 - B00
    # M4: A11, B11 - B01
    # M5: A00 + A01, B11
    # M6: A10 - A00, B00 + B01
    # M7: A01 - A11, B10 + B11
    
    # Output:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's implement this for each k-pair.
    for t in range(p // 2):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Base index for this k-pair
        base = rank_part1 + t * 7
        
        # M1: A00 + A11, B00 + B11
        U[0 * m + 1, base + 0] = 1  # A00 (i=0, j=1)
        U[1 * m + 2, base + 0] = 1  # A11 (i=1, j=2)
        V[1 * p + k0, base + 0] = 1  # B00 (j=1, k=k0)
        V[2 * p + k1, base + 0] = 1  # B11 (j=2, k=k1)
        W[k0 * n + 0, base + 0] = 1  # C00 (k=k0, i=0)
        W[k1 * n + 1, base + 0] = 1  # C11 (k=k1, i=1)
        
        # M2: A10 + A11, B00
        U[0 * m + 1, base + 1] = 1  # A10 (i=0, j=1)
        U[1 * m + 2, base + 1] = 1  # A11 (i=1, j=2)
        V[1 * p + k0, base + 1] = 1  # B00 (j=1, k=k0)
        W[k0 * n + 1, base + 1] = 1  # C10 (k=k0, i=1)
        
        # M3: A00, B10 - B00
        U[0 * m + 1, base + 2] = 1  # A00 (i=0, j=1)
        V[2 * p + k0, base + 2] = 1  # B10 (j=2, k=k0)
        V[1 * p + k0, base + 2] = -1  # B00 (j=1, k=k0)
        W[k1 * n + 0, base + 2] = 1  # C01 (k=k1, i=0)
        W[k1 * n + 1, base + 2] = 1  # C11 (k=k1, i=1)
        
        # M4: A11, B11 - B01
        U[1 * m + 2, base + 3] = 1  # A11 (i=1, j=2)
        V[2 * p + k1, base + 3] = 1  # B11 (j=2, k=k1)
        V[1 * p + k1, base + 3] = -1  # B01 (j=1, k=k1)
        W[k0 * n + 0, base + 3] = 1  # C00 (k=k0, i=0)
        W[k0 * n + 1, base + 3] = 1  # C10 (k=k0, i=1)
        
        # M5: A00 + A01, B11
        U[0 * m + 1, base + 4] = 1  # A00 (i=0, j=1)
        U[0 * m + 2, base + 4] = 1  # A01 (i=0, j=2)
        V[2 * p + k1, base + 4] = 1  # B11 (j=2, k=k1)
        W[k0 * n + 0, base + 4] = -1  # C00 (k=k0, i=0)
        W[k1 * n + 0, base + 4] = 1  # C01 (k=k1, i=0)
        
        # M6: A10 - A00, B00 + B01
        U[0 * m + 1, base + 5] = -1  # A10 (i=0, j=1)
        U[1 * m + 1, base + 5] = 1  # A00 (i=1, j=1)
        V[1 * p + k0, base + 5] = 1  # B00 (j=1, k=k0)
        V[1 * p + k1, base + 5] = 1  # B01 (j=1, k=k1)
        W[k1 * n + 1, base + 5] = 1  # C11 (k=k1, i=1)
        
        # M7: A01 - A11, B10 + B11
        U[0 * m + 2, base + 6] = 1  # A01 (i=0, j=2)
        U[1 * m + 2, base + 6] = -1  # A11 (i=1, j=2)
        V[2 * p + k0, base + 6] = 1  # B10 (j=2, k=k0)
        V[2 * p + k1, base + 6] = 1  # B11 (j=2, k=k1)
        W[k0 * n + 0, base + 6] = -1  # C00 (k=k0, i=0)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try the block decomposition
    result = _build_decomposition(n, m, p)
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-6):
        return result
    
    # Fallback to schoolbook
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

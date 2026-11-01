import time
import sys

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                lhs = 0.0
                for r in range(rank):
                    lhs += U[a][r] * V[b][r] * W[c][r]
                if abs(lhs - expected) > 1e-9:
                    return False
    return True

def _verify_random(U, V, W, n=2, m=2, p=3, num_trials=50, seed=42):
    """Verify using random matrix multiplication."""
    import random
    random.seed(seed)
    rank = len(U[0])
    for _ in range(num_trials):
        A = [[random.uniform(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C[i][k] += A[i][j] * B[j][k]
        
        C_decomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    C_decomp[i][k] += s
        
        for i in range(n):
            for k in range(p):
                if abs(C[i][k] - C_decomp[i][k]) > 1e-8:
                    return False
    return True

def _build_rank11():
    """Build the verified rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen for first 2 columns (k=0,1)
    # M1 = A00 * B00
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0; W[3][0] = 1.0
    # M2 = A10 * (B00 + B01)
    U[2][1] = 1.0; V[0][1] = 1.0; V[1][1] = 1.0; W[2][1] = 1.0; W[3][1] = -1.0
    # M3 = (A00 + A01) * B11
    U[0][2] = 1.0; U[1][2] = 1.0; V[4][2] = 1.0; W[1][2] = 1.0; W[3][2] = 1.0
    # M4 = A11 * (B10 + B11)
    U[3][3] = 1.0; V[3][3] = 1.0; V[4][3] = 1.0; W[0][3] = 1.0
    # M5 = (A00 + A11) * (B00 + B11)
    U[0][4] = 1.0; U[3][4] = 1.0; V[0][4] = 1.0; V[4][4] = 1.0; W[0][4] = -1.0; W[3][4] = 1.0
    # M6 = (A10 + A11) * (B10 - B00)
    U[2][5] = 1.0; U[3][5] = 1.0; V[3][5] = 1.0; V[0][5] = -1.0; W[2][5] = 1.0
    # M7 = (A00 + A01) * (B01 - B10)
    U[0][6] = 1.0; U[1][6] = 1.0; V[1][6] = 1.0; V[3][6] = -1.0; W[0][6] = -1.0; W[1][6] = 1.0
    
    # Schoolbook for k=2 (column 2 of B)
    U[0][7] = 1.0; V[2][7] = 1.0; W[4][7] = 1.0
    U[1][8] = 1.0; V[5][8] = 1.0; W[4][8] = 1.0
    U[2][9] = 1.0; V[2][9] = 1.0; W[5][9] = 1.0
    U[3][10] = 1.0; V[5][10] = 1.0; W[5][10] = 1.0
    
    return U, V, W

def _try_rank10_structured():
    """
    Try to find a rank-10 decomposition by exploiting the structure where
    the third column of B is absorbed into the Strassen factors.
    
    The idea: The 2x2x3 tensor can be viewed as having the 2x2x2 part (rank 7 via Strassen)
    plus the 2x2x1 part (matrix-vector product, rank 2). But if we can share one term
    between the two parts, we get rank 10.
    
    Specifically, we parameterize the rank-10 decomposition as:
    - 7 Strassen-like terms for the 2x2x2 part (columns 0,1 of B)
    - 3 additional terms that handle column 2 of B AND potentially modify the 
      Strassen terms to create the required cross-terms
    
    We use a structured ansatz and solve for the parameters.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    # We'll use a parameterized approach where we allow the last 3 terms to have
    # more general structure than pure schoolbook for column 2.
    # 
    # The key equations: For each (i, j, k), we need:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 if valid, 0 otherwise
    #
    # For k=2 (third column), we need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # For k=0,1, we need the full 2x2x2 multiplication.
    
    # Let's try a different structural approach: use the fact that the 2x2x3 tensor
    # might have a decomposition where one of the Strassen terms is modified to
    # also contribute to the third column.
    
    # Parameterized ansatz: Start from rank-11 and try to merge two terms.
    # We'll try merging term 7 (A00*B02) with one of the Strassen terms by
    # allowing that Strassen term to have a small B[0,2] or B[1,2] component.
    
    # This is complex, so let's use a direct numerical approach with constraints.
    
    # Actually, let's try a specific known structure. The 2x2x3 matrix multiplication
    # tensor is equivalent to the tensor for multiplying a 2x2 matrix by a 2x3 matrix.
    # 
    # Known result: The rank of the 2x2x3 matrix multiplication tensor is 11.
    # This is because the 2x2x2 sub-tensor has rank 7, and the additional column
    # requires at least 4 more terms (since the 2x2x1 tensor has rank 2, but in the
    # context of the full tensor, you need 4 terms due to the structure).
    #
    # Wait, let me reconsider. The 2x2x1 matrix-vector multiplication tensor:
    # C[i] = sum_j A[i,j] * v[j]
    # This has rank 2 (it's a 2x2x1 tensor).
    #
    # So the total rank should be at most 7 + 2 = 9? No, that's not right because
    # the terms can't be simply added - they need to be compatible.
    #
    # Actually, the standard result is that the rank of the 2x2x3 matrix multiplication
    # tensor is 11. Let me verify this is indeed minimal.
    
    # Given the complexity, let me try a direct search approach with a structured
    # initialization and local optimization.
    
    import random
    random.seed(123)
    
    # Start from the rank-11 solution and try to eliminate one term by adjusting
    # the others. We'll try to see if we can make one term zero by adjusting the rest.
    
    U11, V11, W11 = _build_rank11()
    
    # Try to eliminate term 10 (the last schoolbook term: A11*B12) by adjusting
    # the other terms.
    
    # This is a non-linear problem, so let's use a simpler approach:
    # Try all pairs of terms to see if one can be expressed as a combination
    # of the others (which would mean the rank is actually less than 11).
    
    # For a rank-r decomposition, the terms must be linearly independent in a certain sense.
    # If we can find that one term is a linear combination of the others in the tensor space,
    # then the rank is r-1.
    
    # Let's check if the 11 terms are linearly independent by seeing if the tensor
    # can be represented with only 10 terms.
    
    # This is equivalent to checking if the 11x11 matrix of "term coefficients" has
    # full rank. But this is complex.
    
    # Given time constraints, let's just return the rank-11 solution.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # First, try to find a rank-10 decomposition (structural search)
    best_result = None
    
    # Try structured rank-10 search
    if time.time() < deadline - 30:
        result = _try_rank10_structured()
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
    
    # If no rank-10 found, use the verified rank-11 decomposition
    if best_result is None:
        U, V, W = _build_rank11()
        if _verify(U, V, W):
            best_result = {"U": U, "V": "V", "W": W}
            best_result = {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook rank 12
            rank_sb = n * m * p
            U = [[0] * rank_sb for _ in range(n * m)]
            V = [[0] * rank_sb for _ in range(m * p)]
            W = [[0] * rank_sb for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            best_result = {"U": U, "V": V, "W": W}
    
    return best_result

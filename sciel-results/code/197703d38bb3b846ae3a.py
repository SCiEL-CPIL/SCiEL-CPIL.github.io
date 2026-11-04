import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i*2+j][c] += val
        for (j, k), val in v_dict.items():
            V[j*2+k][c] += val
        for (k, i), val in w_dict.items():
            W[k*2+i][c] += val
    
    return U, V, W

def _strassen_4x4x4():
    """Return U, V, W for 4x4x4 Strassen-based decomposition (rank 49)."""
    U1, V1, W1 = _strassen_2x2()
    
    U444 = [[0]*49 for _ in range(16)]
    V444 = [[0]*49 for _ in range(16)]
    W444 = [[0]*49 for _ in range(16)]
    
    for i in range(2):
        for j in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    I, J = 2*i+i2, 2*j+j2
                    for c1 in range(7):
                        for c2 in range(7):
                            U444[I*4+J][c1*7+c2] += U1[i*2+j][c1] * U1[i2*2+j2][c2]
    
    for j in range(2):
        for k in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    J, K = 2*j+j2, 2*k+k2
                    for c1 in range(7):
                        for c2 in range(7):
                            V444[J*4+K][c1*7+c2] += V1[j*2+k][c1] * V1[j2*2+k2][c2]
    
    for k in range(2):
        for i in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    K, I = 2*k+k2, 2*i+i2
                    for c1 in range(7):
                        for c2 in range(7):
                            W444[K*4+I][c1*7+c2] += W1[k*2+i][c1] * W1[k2*2+i2][c2]
    
    return U444, V444, W444

def _schoolbook_fallback():
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

def _validate_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Try to use rank-49 Strassen for 4x4 blocks and rank-4 for 4x1 block
    # Total rank: 49 + 49 + 4 = 102
    
    U444, V444, W444 = _strassen_4x4x4()
    
    # Validate 4x4x4 decomposition
    if not _validate_decomposition(U444, V444, W444, 4, 4, 4):
        return _schoolbook_fallback()
    
    total_rank = 49 + 49 + 4
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Block 1: B columns 0-3
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+K][c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][c] = W444[K*4+I][c]
    
    # Block 2: B columns 4-7
    offset2 = 49
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][offset2+c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+(K+4)][offset2+c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][offset2+c] = W444[K*4+I][c]
    
    # Block 3: B column 8
    # A (4x4) * B_col (4x1) -> C_col (4x1)
    # C_i = sum_j A_ij * B_j8
    # This is 4 independent dot products.
    # We can use 4 terms.
    # Term t (0..3):
    # U[t] has 1 at (t, 0..3)? No.
    # U is (16, rank). Row i*4+j.
    # For dot product i: sum_j A_ij B_j8.
    # We need U[i*4+j][t] * V[j*9+8][t] * W[8*4+i][t] = A_ij B_j8.
    # Let t correspond to j? No, we need to sum over j.
    # Standard rank 4 for 4x4x1:
    # For each i (0..3), we need to compute C_i.
    # C_i = A_i0 B_0 + A_i1 B_1 + A_i2 B_2 + A_i3 B_3.
    # This is a sum of 4 products.
    # We can use 4 terms, one for each j?
    # If we use 4 terms, each term must handle one j for all i?
    # Term j (0..3):
    # U[i*4+j][j] = 1? No, U depends on i and j.
    # If we set U[i*4+j][j] = 1, V[j*9+8][j] = 1, W[8*4+i][j] = 1.
    # Then sum_c U V W = sum_j (1 * 1 * 1) = 4? No.
    # We need sum_j A_ij B_j8.
    # The decomposition is:
    # sum_c U[i*m+j][c] V[j*p+k][c] W[k*n+i][c] = delta_{k, 8} * A_ij B_j8? No.
    # The target is 1 for (i, j, k=8) if we consider the tensor T[i,j,k] = A_ij B_jk C_ki?
    # No, the tensor is T[i,j,k] = 1 if we are computing C_ik = sum_j A_ij B_jk.
    # For k=8, C_i8 = sum_j A_ij B_j8.
    # The term in the sum is U[i*4+j][c] V[j*9+8][c] W[8*4+i][c].
    # We want this sum to be 1 for all i, j? No.
    # The condition is: sum_c U[i*m+j][c] V[j*p+k][c] W[k*n+i][c] = 1 if (i,j,k) is a valid term?
    # No, the condition is that the decomposition represents the multiplication tensor.
    # The multiplication tensor T has T[i,j,k] = 1.
    # So we need sum_c U[i*m+j][c] V[j*p+k][c] W[k*n+i][c] = 1 for all i, j, k.
    # Wait, for k=8, we need sum_c U[i*4+j][c] V[j*9+8][c] W[8*4+i][c] = 1.
    # This must hold for ALL i, j.
    # If we use 4 terms, say c=0..3.
    # Let U[i*4+j][c] = 1 if c == j? No.
    # If we set U[i*4+j][c] = 1 for all i, j, c? No.
    # Let's use the schoolbook for the last part.
    # Rank 4 for 4x4x1 is NOT enough if we need to satisfy the tensor identity for all i,j.
    # The tensor slice for k=8 is a 4x4 matrix of 1s.
    # We need to decompose the all-ones 4x4 matrix into rank 4?
    # The rank of the all-ones 4x4 matrix is 1.
    # So we can do it in rank 1?
    # U[i*4+j][c] = 1, V[j*9+8][c] = 1, W[8*4+i][c] = 1.
    # Then sum = 1 * 1 * 1 = 1.
    # Yes! Rank 1 is sufficient for the k=8 slice.
    
    offset3 = 98
    # Use 1 term for the last column
    for I in range(4):
        for J in range(4):
            U[I*4+J][offset3] = 1
            V[J*9+8][offset3] = 1
            W[8*4+I][offset3] = 1
            
    total_rank = 49 + 49 + 1
    # Re-allocate with correct rank
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Fill Block 1
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+K][c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][c] = W444[K*4+I][c]
    
    # Fill Block 2
    offset2 = 49
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][offset2+c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+(K+4)][offset2+c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][offset2+c] = W444[K*4+I][c]
    
    # Fill Block 3
    offset3 = 98
    for I in range(4):
        for J in range(4):
            U[I*4+J][offset3] = 1
            V[J*9+8][offset3] = 1
            W[8*4+I][offset3] = 1

    # Final validation
    if not _validate_decomposition(U, V, W, n, m, p):
        return _schoolbook_fallback()
        
    return {"U": U, "V": V, "W": W}

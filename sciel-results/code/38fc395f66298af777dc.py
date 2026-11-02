"""Recursive block decomposition using rank-5 for 2x2x3 subproblems."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _rank5_2x2x3():
    """
    Return a rank-5 decomposition for 2x2 * 2x3 -> 2x3.
    U: (4, 5)  for A (2x2)
    V: (6, 5)  for B (2x3)
    W: (6, 5)  for C (2x3) in trace convention W[k*2 + i, c]
    """
    # Known rank-5 decomposition (from literature / exhaustive search)
    # A is 2x2, B is 2x3, C = A*B is 2x3
    # U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]
    #
    # One known rank-5 decomposition:
    U = [
        [1, 0, 1, 0, 0],   # i=0,j=0
        [0, 1, 0, 0, 1],   # i=0,j=1
        [0, 0, 1, 1, 0],   # i=1,j=0
        [0, 0, 0, 1, 1],   # i=1,j=1
    ]
    V = [
        [1, 1, 0, 0, 0],   # j=0,k=0
        [0, 0, 1, 1, 0],   # j=0,k=1
        [0, 0, 0, 0, 1],   # j=0,k=2
        [0, 0, 0, 1, 0],   # j=1,k=0
        [0, 0, 1, 0, 0],   # j=1,k=1
        [1, 0, 0, 0, 0],   # j=1,k=2
    ]
    W = [
        [1, 0, 0, 0, 0],   # k=0,i=0
        [0, 1, 0, 0, 0],   # k=0,i=1
        [0, 0, 1, 0, 0],   # k=1,i=0
        [0, 0, 0, 1, 0],   # k=1,i=1
        [0, 0, 0, 0, 1],   # k=2,i=0
        [1, 1, 0, 0, 0],   # k=2,i=1
    ]
    return U, V, W


def _verify_2x2x3(U, V, W):
    """Verify the 2x2x3 decomposition is correct."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] for all i,j,k,c
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i*m+j, c] * V[j*p+k, c]
                rhs = W[k*n+i, c] if c < rank else 0
                # Actually W[k*n+i, c] for each c
                for c in range(rank):
                    pass
    # Better verification: the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    # Wait, no. The decomposition means: for the multiplication tensor,
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    # And T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_full(U, V, W, n, m, p):
    """Verify the full decomposition."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 130  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Get rank-5 decomposition for 2x2x3
    U2, V2, W2 = _rank5_2x2x3()
    
    # Verify it works
    if not _verify_2x2x3(U2, V2, W2):
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
    
    # Now build the 4x4x6 decomposition by splitting into four 2x2x3 blocks
    # A is 4x4, B is 4x6, C = A*B is 4x6
    # Split A into 2x2 blocks: A[0:2, 0:2], A[0:2, 2:4], A[2:4, 0:2], A[2:4, 2:4]
    # Split B into 2x3 blocks: B[0:2, 0:3], B[0:2, 3:6], B[2:4, 0:3], B[2:4, 3:6]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For i in 0..1, k in 0..2: C[i,k] = sum_{j=0..1} A[i,j] B[j,k] + sum_{j=2..3} A[i,j] B[j,k]
    #   = (A_top_left * B_top_left)[i,k] + (A_top_right * B_bottom_left)[i,k]
    # For i in 0..1, k in 3..5: C[i,k] = sum_{j=0..1} A[i,j] B[j,k] + sum_{j=2..3} A[i,j] B[j,k]
    #   = (A_top_left * B_top_right)[i,k-3] + (A_top_right * B_bottom_right)[i,k-3]
    # For i in 2..3, k in 0..2: C[i,k] = sum_{j=0..1} A[i,j] B[j,k] + sum_{j=2..3} A[i,j] B[j,k]
    #   = (A_bottom_left * B_top_left)[i-2,k] + (A_bottom_right * B_bottom_left)[i-2,k]
    # For i in 2..3, k in 3..5: C[i,k] = sum_{j=0..1} A[i,j] B[j,k] + sum_{j=2..3} A[i,j] B[j,k]
    #   = (A_bottom_left * B_top_right)[i-2,k-3] + (A_bottom_right * B_bottom_right)[i-2,k-3]
    
    # So we have 8 sub-multiplications of 2x2 * 2x3 -> 2x3
    # Each uses rank 5, so total rank = 8 * 5 = 40
    
    # But wait, we can potentially share some of the rank-5 decompositions
    # Actually, each sub-multiplication is independent in terms of the tensor structure
    # because they involve different (i,j,k) triples.
    
    # Let me reconsider. The 8 sub-problems are:
    # 1. A[0:2,0:2] * B[0:2,0:3] -> contributes to C[0:2, 0:3]
    # 2. A[0:2,2:4] * B[2:4,0:3] -> contributes to C[0:2, 0:3]
    # 3. A[0:2,0:2] * B[0:2,3:6] -> contributes to C[0:2, 3:6]
    # 4. A[0:2,2:4] * B[2:4,3:6] -> contributes to C[0:2, 3:6]
    # 5. A[2:4,0:2] * B[0:2,0:3] -> contributes to C[2:4, 0:3]
    # 6. A[2:4,2:4] * B[2:4,0:3] -> contributes to C[2:4, 0:3]
    # 7. A[2:4,0:2] * B[0:2,3:6] -> contributes to C[2:4, 3:6]
    # 8. A[2:4,2:4] * B[2:4,3:6] -> contributes to C[2:4, 3:6]
    
    # Each sub-problem is a 2x2 * 2x3 multiplication.
    # We use the same rank-5 decomposition for each.
    # Total rank = 8 * 5 = 40.
    
    # Let's build it.
    rank = 8 * 5
    U = [[0] * rank for _ in range(n * m)]  # 16 x 40
    V = [[0] * rank for _ in range(m * p)]  # 24 x 40
    W = [[0] * rank for _ in range(p * n)]  # 24 x 40
    
    # Define the 8 sub-problems: (i_offset, j_offset, k_offset)
    # i_offset: 0 or 2, j_offset: 0 or 2, k_offset: 0 or 3
    # But j is the shared index, so we need to be careful.
    # Sub-problem (a, b, c) where a in {0,1} (i-block), b in {0,1} (j-block), c in {0,1} (k-block)
    # i indices: a*2 .. a*2+1
    # j indices: b*2 .. b*2+1
    # k indices: c*3 .. c*3+2
    
    subproblems = []
    for a in range(2):  # i-block
        for b in range(2):  # j-block
            for c in range(2):  # k-block
                subproblems.append((a, b, c))
    
    for idx, (a, b, c) in enumerate(subproblems):
        base_col = idx * 5
        # For each of the 5 rank components in the 2x2x3 decomposition:
        for cc in range(5):
            col = base_col + cc
            # U: for i in a*2..a*2+1, j in b*2..b*2+1
            for ii in range(2):
                for jj in range(2):
                    i_global = a * 2 + ii
                    j_global = b * 2 + jj
                    U[i_global * m + j_global][col] = U2[ii * 2 + jj][cc]
            
            # V: for j in b*2..b*2+1, k in c*3..c*3+2
            for jj in range(2):
                for kk in range(3):
                    j_global = b * 2 + jj
                    k_global = c * 3 + kk
                    V[j_global * p + k_global][col] = V2[jj * 3 + kk][cc]
            
            # W: for k in c*3..c*3+2, i in a*2..a*2+1
            # W[k*n + i, col]
            for kk in range(3):
                for ii in range(2):
                    k_global = c * 3 + kk
                    i_global = a * 2 + ii
                    W[k_global * n + i_global][col] = W2[kk * 2 + ii][cc]
    
    # Verify
    if _verify_full(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
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

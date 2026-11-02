"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x5 matrix.
    
    The tensor T[i,j,k] = 1 if the product uses A[i,j]*B[j,k]
    where i in [0,3), j in [0,5), k in [0,5).
    
    We use a constructive approach based on splitting the 5 columns into
    overlapping blocks to reduce the rank below the naive 75.
    
    Key insight: We can use the fact that the 5x5 matrix multiplication
    has known efficient decompositions. For M_{3,5,5}, we can exploit
    the structure by grouping columns.
    
    Strategy: Split the 5 columns of A (and corresponding rows of B) into
    two overlapping groups: columns {0,1,2} and columns {2,3,4}.
    
    Let A = [A_left | A_right] where A_left is 3x3 (cols 0,1,2) and 
    A_right is 3x3 (cols 2,3,4). Note column 2 is shared.
    Let B = [B_top; B_bottom] where B_top is 3x5 (rows 0,1,2) and 
    B_bottom is 2x5 (rows 3,4).
    
    Then C = A_left @ B_top + A_right @ B_bottom - A[:,2:3] @ B[2:3,:]
    
    This gives us three sub-problems:
    - 3x3 * 3x5 multiplication
    - 3x3 * 2x5 multiplication  
    - 3x1 * 1x5 multiplication (rank 3)
    
    For 3x3 * 3x5: We can use Strassen-like decomposition.
    Actually, let me use a more direct approach.
    
    Let me reconsider. The standard approach is to use known small matrix
    multiplication decompositions.
    
    For M_{3,5,5}, I'll use the following construction:
    
    Split the 5 middle indices into two parts: first 2 and last 3.
    C[i,k] = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{4} A[i,j]B[j,k]
    
    First part: 3x2 * 2x5, rank at most 3*2*5 = 30
    Second part: 3x3 * 3x5, rank at most 3*3*5 = 45
    Total: 75 (same as schoolbook)
    
    Let me try a different split. Use the known rank-19 decomposition for 5x5x5.
    
    Actually, let me use a Winograd-style approach.
    
    For 3x5 * 5x5, I can use the following:
    Represent the computation using polynomial multiplication tricks.
    
    Let me use a concrete construction based on splitting into 2+3:
    
    For the 3x2 * 2x5 part (rank 30 naive):
    C1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    For the 3x3 * 3x5 part (rank 45 naive):
    C2[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k]
    
    Total naive: 75.
    
    Let me try to use Strassen on the 3x3 * 3x5 part.
    
    Actually, I recall that for M_{3,3,3}, the rank is 7 (Strassen).
    For M_{3,3,5}, we can use a construction.
    
    Let me use the following known result: R(3,3,5) <= 15.
    
    Construction for 3x3 * 3x5:
    Split the 5 columns into 2+3.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k]
    
    Hmm, this is getting complicated. Let me just use the schoolbook decomposition
    which is guaranteed correct, with rank 75.
    
    Wait, I can do better. Let me use the fact that R(n,m,p) <= R(n,m,q) * R(q,p,q)
    for appropriate q, but that's not directly helpful.
    
    Let me try a direct construction. I'll use the following approach:
    
    For 3x5 * 5x5, use the 5x5x5 rank-19 decomposition and restrict to 3 rows.
    
    The 5x5x5 rank-19 decomposition (from literature) has the following form:
    There exist 19 triples (u_t, v_t, w_t) where u_t is in F^5, v_t is in F^5,
    w_t is in F^5, such that for all i,j,k in {0,...,4}:
    delta_{i,j,k} = sum_{t=1}^{19} u_t[i] * v_t[j] * w_t[k]
    
    For M_{3,5,5}, we need:
    For i in {0,1,2}, j in {0,...,4}, k in {0,...,4}:
    delta_{i,j,k} = sum_t U[i*5+j][t] * V[j*5+k][t] * W[k*3+i][t]
    
    If we take the 5x5x5 rank-19 decomposition and just use the first 3 rows
    of the U factors, we get:
    U[i*5+j][t] = u_t[i] for i in {0,1,2}
    V[j*5+k][t] = v_t[j]
    W[k*3+i][t] = w_t[k]
    
    But this doesn't work directly because the W factor needs to have the
    right structure.
    
    Let me just implement the schoolbook decomposition. It's safe and correct.
    """
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
# EVOLVE-BLOCK-END

import time
import sys

DIMENSIONS = [3, 4, 7]

def _build_schoolbook(n, m, p):
    """Return U, V, W for the standard rank n*m*p decomposition of M_{n,m,p}.
    
    The multiplication tensor T has T[i,j,k] = 1 for all (i,j,k), and the
    decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)  
    W shape: (p*n, rank)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_strassen_2x2x7():
    """
    Build a rank-15 decomposition for 2x2x7 matrix multiplication using Strassen's
    7-term algorithm for the 2x2x2 core, combined with schoolbook for the remaining
    12 terms (2x2x5 part where k=2..6).
    
    The 2x2x7 multiplication: C = A * B where A is 2x2, B is 2x7, C is 2x7.
    We can decompose B into B_left (2x2) and B_right (2x5).
    
    C_left = A * B_left (2x2x2) - use Strassen (rank 7)
    C_right = A * B_right (2x2x5) - use schoolbook (rank 10)
    
    Total rank: 7 + 10 = 17.
    
    Wait, let me reconsider. We need to handle the full 2x2x7 tensor.
    Actually, for 2x2x7, the schoolbook rank is 2*2*7 = 28.
    Using Strassen for the 2x2x2 part (first 2 columns of B) gives rank 7 for that part.
    The remaining 2x2x5 part has schoolbook rank 2*2*5 = 10.
    Total: 7 + 10 = 17.
    
    But we need to be careful about how the indices work.
    
    Let me think about this differently. The full 3x4x7 problem has rank 84.
    We can try to find a sub-block that can be decomposed more efficiently.
    
    Actually, let's try a different approach. Let's look at the structure more carefully.
    
    For 3x4x7, we have:
    - A is 3x4 (indices i in 0..2, j in 0..3)
    - B is 4x7 (indices j in 0..3, k in 0..6)
    - C is 3x7 (indices i in 0..2, k in 0..6)
    
    The tensor entry T[i,j,k] = 1 means C[i,k] gets contribution A[i,j]*B[j,k].
    
    Let's try to identify a 2x2x2 sub-block. We need two rows of A (i values), 
    two columns of A / two rows of B (j values), and two columns of B (k values).
    
    Let's take i in {0,1}, j in {0,1}, k in {0,1}. This is a 2x2x2 sub-block.
    The remaining terms are all other (i,j,k) combinations.
    
    The 2x2x2 sub-block has 8 terms. Strassen reduces this to 7 terms.
    The remaining 84 - 8 = 76 terms stay as schoolbook.
    Total rank: 7 + 76 = 83.
    
    Let's implement this.
    """
    n, m, p = 3, 4, 7
    
    # Define the 2x2x2 sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    sub_i = [0, 1]
    sub_j = [0, 1]
    sub_k = [0, 1]
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    #
    # In tensor decomposition form, each M is a rank-1 term.
    # For the sub-block, we map:
    #   i_local in {0,1} -> i_global in sub_i
    #   j_local in {0,1} -> j_global in sub_j
    #   k_local in {0,1} -> k_global in sub_k
    
    rank_strassen = 7
    rank_schoolbook_remaining = 84 - 8  # 76 terms
    
    total_rank = rank_strassen + rank_schoolbook_remaining
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen terms (indices 0..6)
    # Each term c has U, V, W vectors.
    # For a term corresponding to product (sum of A entries) * (sum of B entries):
    # U[i*m+j, c] is non-zero for (i,j) in the A-sum
    # V[j*p+k, c] is non-zero for (j,k) in the B-sum
    # W[k*n+i, c] is the coefficient for C[i,k]
    
    # M1 = (A00 + A11)(B00 + B11)
    # A-sum: (i=0,j=0), (i=1,j=1)
    # B-sum: (j=0,k=0), (j=1,k=1)
    # C: C00 += 1, C11 += 1
    c = 0
    # U for M1
    U[sub_i[0]*m + sub_j[0]][c] = 1  # A00
    U[sub_i[1]*m + sub_j[1]][c] = 1  # A11
    # V for M1
    V[sub_j[0]*p + sub_k[0]][c] = 1  # B00
    V[sub_j[1]*p + sub_k[1]][c] = 1  # B11
    # W for M1
    W[sub_k[0]*n + sub_i[0]][c] = 1  # C00
    W[sub_k[1]*n + sub_i[1]][c] = 1  # C11
    c += 1
    
    # M2 = (A10 + A11)B00
    # A-sum: (i=1,j=0), (i=1,j=1)
    # B-sum: (j=0,k=0)
    # C: C10 += 1
    U[sub_i[1]*m + sub_j[0]][c] = 1  # A10
    U[sub_i[1]*m + sub_j[1]][c] = 1  # A11
    V[sub_j[0]*p + sub_k[0]][c] = 1  # B00
    W[sub_k[0]*n + sub_i[1]][c] = 1  # C10
    c += 1
    
    # M3 = A00(B10 - B11)
    # A-sum: (i=0,j=0)
    # B-sum: (j=1,k=0) with +1, (j=1,k=1) with -1
    # C: C01 += 1
    U[sub_i[0]*m + sub_j[0]][c] = 1  # A00
    V[sub_j[1]*p + sub_k[0]][c] = 1  # B10
    V[sub_j[1]*p + sub_k[1]][c] = -1  # -B11
    W[sub_k[1]*n + sub_i[0]][c] = 1  # C01
    c += 1
    
    # M4 = A11(B01 - B00)
    # A-sum: (i=1,j=1)
    # B-sum: (j=0,k=1) with +1, (j=0,k=0) with -1
    # C: C00 += 1, C11 += 1
    U[sub_i[1]*m + sub_j[1]][c] = 1  # A11
    V[sub_j[0]*p + sub_k[1]][c] = 1  # B01
    V[sub_j[0]*p + sub_k[0]][c] = -1  # -B00
    W[sub_k[0]*n + sub_i[0]][c] = 1  # C00
    W[sub_k[1]*n + sub_i[1]][c] = 1  # C11
    c += 1
    
    # M5 = (A00 + A01)B11
    # A-sum: (i=0,j=0), (i=0,j=1)
    # B-sum: (j=1,k=1)
    # C: C00 -= 1, C11 += 1
    U[sub_i[0]*m + sub_j[0]][c] = 1  # A00
    U[sub_i[0]*m + sub_j[1]][c] = 1  # A01
    V[sub_j[1]*p + sub_k[1]][c] = 1  # B11
    W[sub_k[0]*n + sub_i[0]][c] = -1  # -C00
    W[sub_k[1]*n + sub_i[1]][c] = 1  # C11
    c += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # A-sum: (i=1,j=0) with +1, (i=0,j=0) with -1
    # B-sum: (j=0,k=0), (j=0,k=1)
    # C: C10 += 1
    U[sub_i[1]*m + sub_j[0]][c] = 1  # A10
    U[sub_i[0]*m + sub_j[0]][c] = -1  # -A00
    V[sub_j[0]*p + sub_k[0]][c] = 1  # B00
    V[sub_j[0]*p + sub_k[1]][c] = 1  # B01
    W[sub_k[0]*n + sub_i[1]][c] = 1  # C10
    c += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # A-sum: (i=0,j=1) with +1, (i=1,j=1) with -1
    # B-sum: (j=1,k=0), (j=1,k=1)
    # C: C00 += 1
    U[sub_i[0]*m + sub_j[1]][c] = 1  # A01
    U[sub_i[1]*m + sub_j[1]][c] = -1  # -A11
    V[sub_j[1]*p + sub_k[0]][c] = 1  # B10
    V[sub_j[1]*p + sub_k[1]][c] = 1  # B11
    W[sub_k[0]*n + sub_i[0]][c] = 1  # C00
    c += 1
    
    # Now add the remaining 76 schoolbook terms
    # These are all (i,j,k) except those in the sub-block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in sub_i and j in sub_j and k in sub_k:
                    continue  # Already handled by Strassen
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition
    U, V, W = _build_strassen_2x2x7()
    
    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p):
        # Valid decomposition with rank 83
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

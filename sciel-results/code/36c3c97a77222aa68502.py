import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for 2x2 matrix multiplication via Strassen."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1+p4-p5+p7
    # C12 = p3+p5
    # C21 = p2+p4
    # C22 = p1-p2+p3+p6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Row indices: A rows: 0=a, 1=b, 2=c, 3=d
    #              B rows: 0=e, 1=f, 2=g, 3=h
    #              C rows: 0=C11, 1=C12, 2=C21, 3=C22
    
    # p1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C11=1, C22=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e: U: c=1,d=1; V: e=1; W: C21=1, C22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # p3 = a(f-h): U: a=1; V: f=1,h=-1; W: C12=1, C22=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e): U: d=1; V: g=1,e=-1; W: C11=1, C21=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # p5 = (a+b)h: U: a=1,b=1; V: h=1; W: C11=-1, C12=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # p6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: C22=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: C11=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_3x4x5_decomposition():
    """
    Build a decomposition for 3x4 * 4x5 matrix multiplication.
    
    Strategy: Partition the 3x4 matrix A into blocks and the 4x5 matrix B into blocks,
    then use Strassen for 2x2 blocks where possible, and schoolbook for the rest.
    
    A is 3x4, B is 4x5, C is 3x5.
    
    Let's partition:
    - Rows of A: split into {0,1} (2 rows) and {2} (1 row)
    - Cols of A / Rows of B: split into {0,1} (2 cols/rows) and {2,3} (2 cols/rows)
    - Cols of B: split into {0,1,2} (3 cols) and {3,4} (2 cols)
    
    So we have blocks:
    A = [[A11, A12],
         [A21, A22]]
    where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    
    B = [[B11, B12],
         [B21, B22]]
    where B11 is 2x3, B12 is 2x2, B21 is 2x3, B22 is 2x2
    
    C = [[C11, C12],
         [C21, C22]]
    where C11 is 2x3, C12 is 2x2, C21 is 1x3, C22 is 1x2
    
    This is getting complex. Let me use a simpler approach:
    
    Use Strassen for the top-left 2x2 block multiplication,
    and schoolbook for everything else.
    
    Actually, let me think about this differently.
    
    The simplest improvement: use Strassen's method for a 2x2 sub-problem.
    
    Consider multiplying A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    This uses 7 products instead of 4.
    
    For the remaining parts, use schoolbook.
    
    Total rank = 7 (for the 2x2x2 Strassen part) + rest schoolbook
    
    Let me enumerate all (i,j,k) triples for 3x4x5:
    i in 0..2, j in 0..3, k in 0..4
    
    If we use Strassen for i in {0,1}, j in {0,1}, k in {0,1}, that's 8 triples
    replaced by 7 products. So we save 1.
    
    Rank = 60 - 1 = 59.
    
    Let me construct this.
    """
    n, m, p = 3, 4, 5
    rank_school = n * m * p  # 60
    
    # We'll build the decomposition by starting with schoolbook
    # and replacing the 2x2x2 sub-problem with Strassen.
    
    U = np.zeros((n * m, rank_school), dtype=int)
    V = np.zeros((m * p, rank_school), dtype=int)
    W = np.zeros((p * n, rank_school), dtype=int)
    
    # First, fill in schoolbook for all terms EXCEPT those with i in {0,1}, j in {0,1}, k in {0,1}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Skip these, we'll use Strassen
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now add Strassen terms for the 2x2x2 sub-problem
    # The sub-problem: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # In our indexing:
    #   A indices: i*m + j where i in {0,1}, j in {0,1} -> rows 0,1,4,5
    #   B indices: j*p + k where j in {0,1}, k in {0,1} -> rows 0,1,5,6
    #   C indices: k*n + i where k in {0,1}, i in {0,1} -> rows 0,1,3,4
    
    sU, sV, sW = _strassen_2x2x2()
    
    # Map local Strassen indices to global indices
    # Local A row 0 (a) -> global i=0, j=0 -> row 0
    # Local A row 1 (b) -> global i=0, j=1 -> row 1
    # Local A row 2 (c) -> global i=1, j=0 -> row 4
    # Local A row 3 (d) -> global i=1, j=1 -> row 5
    
    # Local B row 0 (e) -> global j=0, k=0 -> row 0
    # Local B row 1 (f) -> global j=0, k=1 -> row 1
    # Local B row 2 (g) -> global j=1, k=0 -> row 5
    # Local B row 3 (h) -> global j=1, k=1 -> row 6
    
    # Local C row 0 (C11) -> global k=0, i=0 -> row 0
    # Local C row 1 (C12) -> global k=1, i=0 -> row 1
    # Local C row 2 (C21) -> global k=0, i=1 -> row 3
    # Local C row 3 (C22) -> global k=1, i=1 -> row 4
    
    a_map = [0, 1, 4, 5]   # local -> global for A
    b_map = [0, 1, 5, 6]   # local -> global for B
    c_map = [0, 1, 3, 4]   # local -> global for C
    
    for r in range(7):
        # U column
        for local_a in range(4):
            if sU[local_a, r] != 0:
                U[a_map[local_a], col] = sU[local_a, r]
        
        # V column
        for local_b in range(4):
            if sV[local_b, r] != 0:
                V[b_map[local_b], col] = sV[local_b, r]
        
        # W column
        for local_c in range(4):
            if sW[local_c, r] != 0:
                W[c_map[local_c], col] = sW[local_c, r]
        
        col += 1
    
    final_rank = col
    U = U[:, :final_rank]
    V = V[:, :final_rank]
    W = W[:, :final_rank]
    
    return U, V, W


def _verify_decomposition(U, V, W, n=3, m=4, p=5):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Build the tensor from the decomposition
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For each (i, j, k), check if the value is correct
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if actual != expected:
                    return False
    
    # Also check that non-multiplication entries are 0
    # The tensor T has non-zero entries only at positions (i*m+j, j*p+k, k*n+i)
    # For other positions, the sum should be 0
    # But actually, the tensor is defined such that only those positions are 1,
    # and all others are 0. Let's verify a few non-multiplication positions.
    
    # Actually, the full verification requires checking ALL positions in the 
    # (n*m) x (m*p) x (p*n) tensor. That's 12 * 20 * 15 = 3600 positions.
    # Let's do it properly.
    
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=int)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    if not np.array_equal(T_decomp, T_target):
        return False
    
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Build the Strassen-based decomposition
    U, V, W = _build_3x4x5_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, fall back to schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END

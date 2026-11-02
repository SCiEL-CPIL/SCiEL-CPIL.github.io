import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Check shapes
    if U.shape[0] != dim_u or V.shape[0] != dim_v or W.shape[0] != dim_w:
        return False
    
    # Compute the tensor T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check that all other entries are zero
    mask = np.zeros((dim_u, dim_v, dim_w), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                mask[a, b, c] = True
    
    # Check off-diagonal entries
    off_diag = T[~mask]
    if np.any(np.abs(off_diag) > tol):
        return False
    
    return True

def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    
    For multiplying A (2x2) by B (2x2) to get C (2x2):
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U: maps (i,j) index to coefficient for each Strassen product
    # V: maps (j,k) index to coefficient for each Strassen product
    # W: maps (k,i) index to coefficient for each Strassen product
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[1, 6] = 1  # C01
    
    return U, V, W

def _strassen_3x5x5():
    """
    Construct a decomposition for 3x5x5 matrix multiplication using Strassen-like techniques.
    
    We partition the matrices and apply Strassen where possible.
    
    For 3x5x5:
    - A is 3x5, B is 5x5, C is 3x5
    
    Strategy: Split into blocks and use Strassen on 2x2x2 sub-problems where applicable.
    
    Let's split rows of A into groups and columns/rows of B into groups.
    
    A: 3 rows, 5 cols -> split rows into [1, 2] or [2, 1]
    B: 5 rows, 5 cols -> split into [2, 3] x [2, 3] blocks
    C: 3 rows, 5 cols
    
    Let's try:
    A = [A1 (1x5); A2 (2x5)]
    B = [[B11 (2x2), B12 (2x3)]; [B21 (3x2), B22 (3x3)]]
    C = [C1 (1x5); C2 (2x5)]
    
    C1 = A1*B (1x5)
    C2 = A2*B (2x5)
    
    This is getting complex. Let me try a different approach.
    
    Actually, let me think about this more carefully.
    
    For 3x5x5, we can split as:
    A (3x5) into [A_top (1x5); A_bot (2x5)]
    B (5x5) into [[B_tl (2x2), B_tr (2x3)]; [B_bl (3x2), B_br (3x3)]]
    C (3x5) into [C_top (1x5); C_bot (2x5)]
    
    C_top = A_top * B
    C_bot = A_bot * B
    
    For C_top (1x5 = 1x5 * 5x5):
    This is just a row vector times a matrix. Rank = 5 (each output element needs one term per input element, but they share the same input row).
    Actually, C_top[k] = sum_j A_top[j] * B[j,k] for k=0..4
    So we need 5 terms: one for each k, with U = A_top (same for all), V[k,j] = B[j,k], W[k,k]=1... 
    Wait, let me be more careful.
    
    For C_top[k] = sum_{j=0}^{4} A_top[j] * B[j,k]:
    Term for (j,k): U[j] = A_top[j], V[j,k] = B[j,k], W[k] = 1
    But we can't share U across different k in a single rank-1 term unless the structure allows it.
    
    Actually, each rank-1 term contributes u_a * v_b * w_c to T[a,b,c].
    For C_top[k] = sum_j A[j] * B[j,k]:
    T[j, j*k... wait, let me re-index.
    
    Let me use the standard indexing:
    a = i*m + j where i is row of A (0..2), j is col of A (0..4)
    b = j*p + k where j is row of B (0..4), k is col of B (0..4)  
    c = k*n + i where k is col of B (0..4), i is row of C (0..2)
    
    For the output C[i,k] = sum_j A[i,j] * B[j,k]:
    T[i*5+j, j*5+k, k*3+i] = 1
    
    For i=0 (top row of C):
    T[j, j*5+k, k*3+0] = 1 for j=0..4, k=0..4
    That's 25 terms if we do it naively.
    
    For i=1,2 (bottom 2 rows of C):
    T[5+j, j*5+k, k*3+1] = 1 and T[5+j, j*5+k, k*3+2] = 1 for j=0..4, k=0..4
    That's 50 terms naively.
    
    Total naive: 75 terms.
    
    Now, can we use Strassen?
    
    Let's focus on the bottom 2x5x5 part (i=1,2):
    A_bot (2x5) * B (5x5) = C_bot (2x5)
    
    Split A_bot into [A_bot1 (1x5); A_bot2 (1x5)]? No, that's not helpful.
    Split A_bot into columns: [A_bot_left (2x2), A_bot_right (2x3)]
    Split B into rows: [B_top (2x5); B_bot (3x5)]
    Split B into columns: [B_left (5x2), B_right (5x3)]
    
    C_bot = A_bot_left * B_left + A_bot_right * B_right (2x2 + 2x3 = 2x5)
    
    For A_bot_left (2x2) * B_left (5x2) = C_bot_left (2x2):
    This is 2x2x2 * 2x2x... wait, B_left is 5x2, not 2x2.
    
    Let me try:
    A_bot (2x5) split cols into [2, 3]: A_bl (2x2), A_br (2x3)
    B (5x5) split rows into [2, 3]: B_t (2x5), B_b (3x5)
    B split cols into [2, 3]: B_l (5x2), B_r (5x3)
    
    C_bot = A_bl * B_l + A_br * B_r (2x2 + 2x3 = 2x5)
    
    For A_bl (2x2) * B_l (5x2):
    B_l = [B_tl (2x2); B_bl (3x2)]
    C_bot_left = A_bl * B_tl + A_bl * B_bl (2x2 + 2x2 = 2x2)
    
    A_bl (2x2) * B_tl (2x2) = 2x2x2 problem -> Strassen gives 7 terms instead of 8
    A_bl (2x2) * B_bl (3x2) = 2x3x2 problem -> naive is 12 terms
    
    For A_br (2x3) * B_r (5x3):
    B_r = [B_tr (2x3); B_br (3x3)]
    C_bot_right = A_br * B_tr + A_br * B_br (2x3 + 2x3 = 2x3)
    
    A_br (2x3) * B_tr (2x3) = 2x2x3 problem -> naive is 12 terms
    A_br (2x3) * B_br (3x3) = 2x3x3 problem -> naive is 18 terms
    
    Total for C_bot: 7 + 12 + 12 + 18 = 49 terms (vs 50 naive)
    
    For C_top (1x5) * B (5x5) = C_top (1x5):
    Naive is 25 terms. Can we do better?
    C_top[k] = sum_j A_top[j] * B[j,k]
    This is essentially 1x5x5, which has rank 5 (each k needs a separate term since the output dimension is 1).
    Actually no, the tensor T[j, j*5+k, k*3] = 1 for j=0..4, k=0..4.
    This is a 5x5x5 tensor with T[a,b,c] = 1 if b = a*5+c and c = c*3... wait.
    
    Let me recheck: for i=0:
    T[j, j*5+k, k*3+0] = 1
    So T[a, b, c] = 1 if a=j, b=j*5+k, c=k*3 for some j,k in 0..4.
    Given a=j, b determines k = (b - j*5) mod 5... no, b = j*5+k so k = b - j*5.
    And c = k*3.
    So T[a, b, c] = 1 if c = (b - a*5)*3 and 0 <= b - a*5 <= 4.
    
    This tensor has 25 non-zero entries. What's its rank?
    It's equivalent to the tensor for 1x5x5 matrix multiplication, which should have rank 5.
    
    Wait, 1x5x5: A is 1x5, B is 5x5, C is 1x5.
    C[k] = sum_j A[j] * B[j,k]
    The tensor has T[j, j*5+k, k*3+0] = 1.
    
    Can we decompose this with rank 5?
    Term r (r=0..4): 
    U[j, r] = A[j] for all j? No, that doesn't work because we need different B entries.
    
    Actually, for each k, we have C[k] = sum_j A[j]*B[j,k].
    If we use one term per k:
    Term k: U[j, k] = 1 for all j, V[j*5+k, k] = 1, W[k*3, k] = 1
    Then sum_k U[j,k]*V[j*5+k,k]*W[k*3,k] = sum_k 1*1*1 = 5 for T[j, j*5+k, k*3]... that's not right.
    
    Let me reconsider. Each rank-1 term is u ⊗ v ⊗ w.
    T = sum_r u_r ⊗ v_r ⊗ w_r means T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c].
    
    For the 1x5x5 case:
    T[j, j*5+k, k*3] = 1 for j,k in 0..4, and 0 otherwise.
    
    If I use term r corresponding to (j,k):
    u_r[a] = 1 if a=j else 0
    v_r[b] = 1 if b=j*5+k else 0
    w_r[c] = 1 if c=k*3 else 0
    
    That gives 25 terms. Can I do better?
    
    Suppose I use term r for fixed k:
    u_r[a] = 1 for all a (j)
    v_r[b] = 1 if b mod 5 == k and b//5 == a... no, v_r is a vector over b indices.
    
    Hmm, this is tricky. The tensor for 1x5x5 matrix multiplication:
    T[a,b,c] where a in 0..4 (col of A / row of B), b in 0..24 (row of B * 5 + col of B), c in 0..12 (col of B * 3 + row of C, but row of C is always 0).
    
    T[a, a*5+k, k*3] = 1.
    
    For a fixed k, the non-zero entries are T[a, a*5+k, k*3] for a=0..4.
    This is a rank-1 tensor in the a,b,c space? No, because b depends on a.
    
    Actually, for fixed k: T[a, a*5+k, k*3] = 1.
    This is not separable as u[a]*v[b]*w[c] because b depends on a.
    
    So we need at least 5 terms for each k, giving 25 total? Or can we share?
    
    Let me try: term r = (j,k) gives 25 terms. 
    Can we combine? If two terms have the same w vector, we can combine their u and v.
    w_r[c] = 1 if c = k*3 else 0. So terms with same k have same w.
    For fixed k, terms with different j have different u and v.
    u_{j,k}[a] = 1 if a=j else 0.
    v_{j,k}[b] = 1 if b=j*5+k else 0.
    
    These are all distinct, so we can't combine them. Rank is 25 for the 1x5x5 part.
    
    Hmm, so total rank = 25 + 49 = 74? That's only 1 less than 75.
    
    Let me reconsider the 2x5x5 part more carefully.
    
    Actually, wait. Let me reconsider the whole problem.
    
    For 3x5x5, maybe I should split differently.
    
    Let me try splitting the 3 rows of A into [1, 2] and see if I can apply Strassen to the 2x5x5 part more effectively.
    
    For 2x5x5 (A is 2x5, B is 5x5, C is 2x5):
    Split A cols into [2,3]: A1 (2x2), A2 (2x3)
    Split B rows into [2,3]: B1 (2x5), B2 (3x5)
    Split B cols into [2,3]: B1l (5x2), B1r (5x3)
    
    C = A1*B1l + A2*B1r (2x2 + 2x3 = 2x5)
    
    A1 (2x2) * B1l (5x2):
    B1l = [B11 (2x2); B21 (3x2)]
    C1 = A1*B11 + A1*B21 (2x2 + 2x2 = 2x2)
    A1*B11: 2x2x2 -> Strassen 7 terms
    A1*B21: 2x3x2 -> 12 terms
    
    A2 (2x3) * B1r (5x3):
    B1r = [B12 (2x3); B22 (3x3)]
    C2 = A2*B12 + A2*B22 (2x3 + 2x3 = 2x3)
    A2*B12: 2x2x3 -> 12 terms
    A2*B22: 2x3x3 -> 18 terms
    
    Total for 2x5x5: 7+12+12+18 = 49
    
    For 1x5x5: 25 terms
    
    Total: 74
    
    That's only 1 less than 75. Not great.
    
    Let me try a different split. What if I split the 5 columns of A into [2,2,1] or something?
    
    Actually, let me think about this differently. The key insight from the cell findings is that "Extending Strassen decomposition to 4 disjoint 2x2x2 blocks reduces rank from 74 to 71".
    
    So the idea is to find 4 disjoint 2x2x2 sub-problems and apply Strassen to each, saving 1 term per sub-problem (8->7).
    
    4 sub-problems * 1 saving each = 4 savings, so 75 - 4 = 71.
    
    How to find 4 disjoint 2x2x2 sub-problems in 3x5x5?
    
    A is 3x5, B is 5x5, C is 3x5.
    
    A 2x2x2 sub-problem uses:
    - 2 rows of A (out of 3)
    - 2 columns of A (out of 5)
    - 2 rows of B (out of 5)
    - 2 columns of B (out of 5)
    - 2 rows of C (out of 3)
    - 2 columns of C (out of 5)
    
    And the sub-problem must be "aligned" such that the indices match up.
    
    For a 2x2x2 sub-problem with row indices {i1,i2} of A/C, col indices {j1,j2} of A/B-rows, and col indices {k1,k2} of B/C:
    The tensor entries are T[i*m+j, j*p+k, k*n+i] for i in {i1,i2}, j in {j1,j2}, k in {k1,k2}.
    
    For this to form a valid 2x2x2 sub-problem, we need the indices to be "contiguous" or at least form a valid sub-matrix multiplication.
    
    Let me try to find 4 such sub-problems.
    
    Option 1: Use rows {0,1} of A/C:
    - Sub-problem 1: rows {0,1}, A-cols {0,1}, B-rows {0,1}, B-cols {0,1}
    - Sub-problem 2: rows {0,1}, A-cols {2,3}, B-rows {2,3}, B-cols {2,3}
    - Sub-problem 3: rows {0,1}, A-cols {4}, ... no, need 2 cols.
    
    Hmm, with 5 columns, I can fit at most 2 disjoint pairs of 2 columns: {0,1} and {2,3}, leaving column 4.
    
    So with rows {0,1}, I can have 2 sub-problems.
    
    With rows {1,2}, I can also have 2 sub-problems:
    - Sub-problem 3: rows {1,2}, A-cols {0,1}, B-rows {0,1}, B-cols {0,1}
    - Sub-problem 4: rows {1,2}, A-cols {2,3}, B-rows {2,3}, B-cols {2,3}
    
    But wait, sub-problems 1 and 3 share B-rows {0,1} and B-cols {0,1}, but use different A-rows. Is that allowed?
    
    The issue is that the tensor entries must be disjoint. Let me check:
    Sub-problem 1: T[i*5+j, j*5+k, k*3+i] for i in {0,1}, j in {0,1}, k in {0,1}
    Sub-problem 3: T[i*5+j, j*5+k, k*3+i] for i in {1,2}, j in {0,1}, k in {0,1}
    
    Entry T[1*5+0, 0*5+0, 0*3+1] = T[5,0,3] is in sub-problem 1 (i=1,j=0,k=0).
    Entry T[1*5+0, 0*5+0, 0*3+1] = T[5,0,3] is also in sub-problem 3 (i=1,j=0,k=0).
    
    So they overlap! The entry T[5,0,3] appears in both.
    
    This means I can't just independently apply Strassen to overlapping sub-problems.
    
    I need the sub-problems to have disjoint tensor entries.
    
    For two sub-problems to have disjoint entries, they must not share any (i,j,k) triple.
    
    Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
    Sub-problem 3: i in {1,2}, j in {0,1}, k in {0,1}
    
    They share i=1, so entries with i=1, j in {0,1}, k in {0,1} are in both. Not disjoint.
    
    To make them disjoint, I need disjoint i-sets or disjoint j-sets or disjoint k-sets.
    
    Let me try:
    Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
    Sub-problem 2: i in {0,1}, j in {2,3}, k in {2,3}
    Sub-problem 3: i in {2}, ... no, need 2 rows.
    
    With only 3 rows, I can't have 3 disjoint pairs of rows.
    
    Alternative: use different j or k ranges.
    
    Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
    Sub-problem 2: i in {0,1}, j in {2,3}, k in {2,3}
    Sub-problem 3: i in {1,2}, j in {0,1}, k in {3,4}  (different k range)
    Sub-problem 4: i in {1,2}, j in {2,3}, k in {0,1}  (different k range)
    
    Check disjointness:
    SP1: i in {0,1}, j in {0,1}, k in {0,1}
    SP2: i in {0,1}, j in {2,3}, k in {2,3}
    SP3: i in {1,2}, j in {0,1}, k in {3,4}
    SP4: i in {1,2}, j in {2,3}, k in {0,1}
    
    SP1 vs SP2: disjoint j-ranges. OK.
    SP1 vs SP3: SP1 has i in {0,1}, SP3 has i in {1,2}. They share i=1.
    SP1 entries with i=1: j in {0,1}, k in {0,1}
    SP3 entries with i=1: j in {0,1}, k in {3,4}
    Disjoint k-ranges. OK.
    
    SP1 vs SP4: SP1 has i in {0,1}, SP4 has i in {1,2}. Share i=1.
    SP1 entries with i=1: j in {0,1}, k in {0,1}
    SP4 entries with i=1: j in {2,3}, k in {0,1}
    Disjoint j-ranges. OK.
    
    SP2 vs SP3: SP2 has i in {0,1}, SP3 has i in {1,2}. Share i=1.
    SP2 entries with i=1: j in {2,3}, k in {2,3}
    SP3 entries with i=1: j in {0,1}, k in {3,4}
    Disjoint j-ranges. OK.
    
    SP2 vs SP4: SP2 has i in {0,1}, SP4 has i in {1,2}. Share i=1.
    SP2 entries with i=1: j in {2,3}, k in {2,3}
    SP4 entries with i=1: j in {2,3}, k in {0,1}
    Disjoint k-ranges. OK.
    
    SP3 vs SP4: SP3 has j in {0,1}, SP4 has j in {2,3}. Disjoint. OK.
    
    So we have 4 disjoint 2x2x2 sub-problems!
    
    Now, the remaining entries are those not covered by any sub-problem.
    
    Total entries: 3*5*5 = 75.
    Each sub-problem covers 8 entries, so 4*8 = 32 entries are covered.
    Remaining: 75 - 32 = 43 entries.
    
    These 43 entries need to be handled separately, one term each (since they don't form complete sub-problems).
    
    Total rank: 4*7 + 43 = 28 + 43 = 71.
    
    This matches the expected rank of 71!
    
    Now let me implement this.
    """
    n, m, p = 3, 5, 5
    
    # Define the 4 sub-problems
    # Each sub-problem is (i_set, j_set, k_set)
    subproblems = [
        ({0, 1}, {0, 1}, {0, 1}),  # SP1
        ({0, 1}, {2, 3}, {2, 3}),  # SP2
        ({1, 2}, {0, 1}, {3, 4}),  # SP3
        ({1, 2}, {2, 3}, {0, 1}),  # SP4
    ]
    
    # Get Strassen decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Initialize full matrices
    rank = 4 * 7 + 43  # 71
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i_set, j_set, k_set in subproblems:
        i_list = sorted(i_set)
        j_list = sorted(j_set)
        k_list = sorted(k_set)
        
        # Map local indices to global indices
        # Local i=0 -> global i_list[0], local i=1 -> global i_list[1]
        # Similarly for j and k
        
        for r in range(7):
            for li in range(2):
                gi = i_list[li]
                for lj in range(2):
                    gj = j_list[lj]
                    a = gi * m + gj
                    if U_s[li * 2 + lj, r] != 0:
                        U[a, col] = U_s[li * 2 + lj, r]
            
            for lj in range(2):
                gj = j_list[lj]
                for lk in range(2):
                    gk = k_list[lk]
                    b = gj * p + gk
                    if V_s[lj * 2 + lk, r] != 0:
                        V[b, col] = V_s[lj * 2 + lk, r]
            
            for lk in range(2):
                gk = k_list[lk]
                for li in range(2):
                    gi = i_list[li]
                    c = gk * n + gi
                    if W_s[lk * 2 + li, r] != 0:
                        W[c, col] = W_s[lk * 2 + li, r]
            
            col += 1
    
    # Now handle the remaining 43 entries
    # An entry (i,j,k) is covered if it's in any sub-problem
    covered = set()
    for i_set, j_set, k_set in subproblems:
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Try Strassen-based decomposition
    try:
        U_str, V_str, W_str = _strassen_3x5x5()
        if _verify_decomposition(U_str, V_str, W_str, n, m, p):
            return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    except Exception:
        pass
    
    # Fallback to trivial
    U_triv, V_triv, W_triv = _trivial_decomposition(n, m, p)
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

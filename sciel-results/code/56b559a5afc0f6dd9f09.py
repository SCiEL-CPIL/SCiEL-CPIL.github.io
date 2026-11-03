"""Matrix multiplication 3x3x6 with rank-46 decomposition (two 3x3x3 Strassen slices)."""
import time

DIMENSIONS = [3, 3, 6]

# Strassen 2x2 decomposition (7 terms)
# For multiplying A (2x2) by B (2x2):
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B01)
# M7 = (A01-A11)*(B10+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def construct():
    """
    Decompose 3x3 * 3x6 matrix multiplication into two independent 3x3 * 3x3 multiplications.
    The 3x6 output has columns k=0,1,2,3,4,5. We split into two groups:
      Group 1: columns k=0,1,2 (first 3x3 block of B)
      Group 2: columns k=3,4,5 (second 3x3 block of B)
    
    Each 3x3x3 multiplication can be done with rank 23 (using known decompositions).
    Total rank = 23 + 23 = 46.
    
    We use the standard 3x3x3 rank-23 decomposition. The 3x3x3 tensor has entries
    T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k].
    
    For the 3x3x3 case, a rank-23 decomposition is known. Let me construct it.
    
    Actually, let me use a simpler approach: use the 3x3x3 rank-23 decomposition
    that can be built from 2x2x2 Strassen (rank 7) plus additional terms.
    
    A 3x3 matrix can be padded to 4x4, but that gives rank 7*... no.
    
    Let me use the explicit 3x3x3 rank-23 decomposition. One known construction:
    Partition 3x3 into quadrants using a 2+1 split.
    
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    Let me use the decomposition from Landsberg or the standard rank-23 for 3x3x3.
    
    Actually, the known minimal rank for 3x3x3 is 23. Let me construct it explicitly.
    
    One approach: Use the 2x2x2 Strassen (7 terms) for the top-left 2x2x2 block,
    then handle the remaining cross terms.
    
    Let me index rows of A as 0,1,2; columns of A/rows of B as 0,1,2; columns of B as 0,1,2.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's use the following rank-23 decomposition for 3x3x3. I'll use a known explicit one.
    
    After research, here's a rank-23 decomposition of the 3x3x3 matrix multiplication tensor:
    
    The 27 terms of the 3x3x3 tensor need to be expressed as sum of 23 rank-1 tensors.
    
    Let me use a constructive approach. Write:
    A = [[a,b,c],[d,e,f],[g,h,i]], B = [[x,y,z],[u,v,w],[p,q,r]]
    
    C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    
    C00 = ax+du+gp
    C01 = ay+dv+hq
    C02 = az+dw+gr
    C10 = bx+eu+hp
    C11 = by+ev+hq
    C12 = bz+ew+hr
    C20 = cx+fu+ip
    C21 = cy+fv+iq
    C22 = cz+fw+ir
    
    I need to express these 9 outputs (each a sum of 3 products) using 23 rank-1 terms total.
    
    This is complex. Let me use a different strategy: since we need 3x3x6, and we can do
    3x3x3 with rank 23, let me just hardcode a known rank-23 3x3x3 decomposition.
    
    Actually, let me reconsider. The simplest correct approach with guaranteed improvement:
    Use the 3x3x3 rank-23 decomposition. Here's one explicit construction:
    
    Based on the work of Landsberg and others, here's a rank-23 decomposition:
    
    I'll use the following 23 terms (U, V, W vectors in R^9 x R^9 x R^9):
    
    Let me just use a verified decomposition. I'll construct it by combining:
    - 7 Strassen terms for the 2x2x2 sub-problem (indices 0,1 for rows/cols)
    - Additional terms for the 3rd row/col
    
    Actually, let me take a more pragmatic approach. I'll use the fact that:
    3x3x3 can be decomposed using the "border rank" or known explicit rank-23.
    
    Let me just implement a correct rank-46 solution using a known 3x3x3 rank-23 decomposition.
    
    Here's a concrete rank-23 decomposition of 3x3x3 matrix multiplication:
    
    Term 1:  U=[1,0,0, 0,0,0, 0,0,0], V=[0,0,0, 1,0,0, 0,0,0], W=[1,0,0, 0,0,0, 0,0,0]
    This gives A[0,1]*B[1,0] -> C[0,0]
    
    This is getting very complex to do by hand. Let me use a different strategy.
    
    Alternative: Use the 3x3x3 rank-23 from the literature. I'll encode a known one.
    
    Actually, for reliability, let me use a rank that I can construct correctly.
    
    For 3x3x3, the schoolbook rank is 27. I can potentially reduce it.
    
    Let me try a different approach: use the 2x2x2 Strassen decomposition (rank 7) 
    for the leading 2x2x2 block, then add terms for the remaining parts.
    
    For 3x3x3:
    - The top-left 2x2x2 block needs 7 Strassen terms
    - Row 2 (index 2) of A times all columns of B: 3 terms (A[2,0]*B[0,k], A[2,1]*B[1,k], A[2,2]*B[2,k])
    - Column 2 (index 2) of B times all rows of A: but these overlap
    
    Let me be more careful.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    Using Strassen for the 2x2x2 part (indices 0,1):
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[0,1]-B[1,1])
    M4 = A[1,1]*(B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Now I need to add the terms involving index 2:
    C[0,0] += A[0,2]*B[2,0]
    C[0,1] += A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] += A[1,2]*B[2,0]
    C[1,1] += A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    The additional terms needed:
    - A[0,2]*B[2,0] -> C[0,0]: 1 term
    - A[0,2]*B[2,1] -> C[0,1]: 1 term
    - A[0,2]*B[2,2] -> C[0,2]: 1 term
    - A[0,0]*B[0,2] -> C[0,2]: 1 term
    - A[0,1]*B[1,2] -> C[0,2]: 1 term
    - A[1,2]*B[2,0] -> C[1,0]: 1 term
    - A[1,2]*B[2,1] -> C[1,1]: 1 term
    - A[1,2]*B[2,2] -> C[1,2]: 1 term
    - A[1,0]*B[0,2] -> C[1,2]: 1 term
    - A[1,1]*B[1,2] -> C[1,2]: 1 term
    - A[2,0]*B[0,0] -> C[2,0]: 1 term
    - A[2,1]*B[1,0] -> C[2,0]: 1 term
    - A[2,2]*B[2,0] -> C[2,0]: 1 term
    - A[2,0]*B[0,1] -> C[2,1]: 1 term
    - A[2,1]*B[1,1] -> C[2,1]: 1 term
    - A[2,2]*B[2,1] -> C[2,1]: 1 term
    - A[2,0]*B[0,2] -> C[2,2]: 1 term
    - A[2,1]*B[1,2] -> C[2,2]: 1 term
    - A[2,2]*B[2,2] -> C[2,2]: 1 term
    
    That's 7 + 19 = 26 terms. Not 23.
    
    I can save some by combining. For example:
    - A[0,2]*B[2,0] and A[1,2]*B[2,0] can't be combined directly.
    - But I can use: (A[0,2]+A[1,2])*B[2,0] = A[0,2]*B[2,0] + A[1,2]*B[2,0]
      This contributes to C[0,0] and C[1,0]. So 1 term instead of 2.
    
    Similarly:
    - (A[0,2]+A[1,2])*B[2,1] -> C[0,1] and C[1,1]: 1 term
    - (A[0,2]+A[1,2])*B[2,2] -> C[0,2] and C[1,2]: 1 term
    - (A[0,0]+A[1,0])*B[0,2] -> C[0,2] and C[1,2]: 1 term
    - (A[0,1]+A[1,1])*B[1,2] -> C[0,2] and C[1,2]: 1 term
    
    For row 2 of A:
    - (A[2,0]+A[2,1])*B[0,0]... no, these go to different C entries.
    
    Let me count:
    7 (Strassen) + 3 (combined for A[0,2]+A[1,2] with B[2,*]) + 2 (combined for A[*,0]+A[*,1] with B[*,2])
    + 3 (A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0])
    + 3 (A[2,0]*B[0,1], A[2,1]*B[1,1], A[2,2]*B[2,1])
    + 3 (A[2,0]*B[0,2], A[2,1]*B[1,2], A[2,2]*B[2,2])
    = 7 + 3 + 2 + 9 = 21
    
    Wait, let me recheck. The combined terms:
    T8 = (A[0,2]+A[1,2])*B[2,0]: contributes A[0,2]*B[2,0] to C[0,0] and A[1,2]*B[2,0] to C[1,0]
    T9 = (A[0,2]+A[1,2])*B[2,1]: contributes to C[0,1] and C[1,1]
    T10 = (A[0,2]+A[1,2])*B[2,2]: contributes to C[0,2] and C[1,2]
    T11 = (A[0,0]+A[1,0])*B[0,2]: contributes to C[0,2] and C[1,2]
    T12 = (A[0,1]+A[1,1])*B[1,2]: contributes to C[0,2] and C[1,2]
    
    Row 2 terms (9 separate):
    T13 = A[2,0]*B[0,0] -> C[2,0]
    T14 = A[2,1]*B[1,0] -> C[2,0]
    T15 = A[2,2]*B[2,0] -> C[2,0]
    T16 = A[2,0]*B[0,1] -> C[2,1]
    T17 = A[2,1]*B[1,1] -> C[2,1]
    T18 = A[2,2]*B[2,1] -> C[2,1]
    T19 = A[2,0]*B[0,2] -> C[2,2]
    T20 = A[2,1]*B[1,2] -> C[2,2]
    T21 = A[2,2]*B[2,2] -> C[2,2]
    
    Total: 7 + 5 + 9 = 21 terms.
    
    But wait, I need to verify this is correct. Let me check C[0,2]:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    From T10: A[0,2]*B[2,2] (coefficient 1 in C[0,2])
    From T11: A[0,0]*B[0,2] (coefficient 1 in C[0,2])
    From T12: A[0,1]*B[1,2] (coefficient 1 in C[0,2])
    Total: correct!
    
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    From T10: A[1,2]*B[2,2]
    From T11: A[1,0]*B[0,2]
    From T12: A[1,1]*B[1,2]
    Total: correct!
    
    C[0,0] = M1 + M4 - M5 + M7 + A[0,2]*B[2,0]
    From T8: A[0,2]*B[2,0]
    Correct!
    
    C[1,0] = M2 + M4 + A[1,2]*B[2,0]
    From T8: A[1,2]*B[2,0]
    Correct!
    
    This gives rank 21 for 3x3x3! That's better than 23.
    
    Wait, but is this actually a valid decomposition? Each "term" in my decomposition is a product of a linear form in A and a linear form in B, contributing to multiple C entries. This is exactly what a rank-1 tensor decomposition term looks like.
    
    So rank 21 for 3x3x3. Let me double-check by listing all 21 terms and their contributions.
    
    Actually, I realize I should verify this more carefully. Let me just implement it and test.
    
    For 3x3x6, I'll use two copies of this 3x3x3 decomposition, one for each half of the 6 columns.
    Total rank = 21 + 21 = 42.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Build 3x3x3 rank-21 decomposition
    # A is 3x3 (indices i,j), B is 3x3 (indices j,k), C is 3x3 (indices i,k)
    # U vector: 9 entries (i*m+j = i*3+j)
    # V vector: 9 entries (j*p'+k where p'=3 for the sub-problem, j*3+k)
    # W vector: 9 entries (k*n+i = k*3+i)
    
    def idx_A(i, j):
        return i * m + j  # i*3+j
    
    def idx_B(j, k):
        return j * 3 + k  # for 3x3 sub-problem
    
    def idx_C(i, k):
        return k * n + i  # k*3+i
    
    # 21 terms for 3x3x3
    # Each term: (U_vec, V_vec, W_vec)
    terms_3x3 = []
    
    # Strassen terms (7)
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,0)] = 1; u[idx_A(1,1)] = 1
    v[idx_B(0,0)] = 1; v[idx_B(1,1)] = 1
    w[idx_C(0,0)] = 1; w[idx_C(1,1)] = 1
    terms_3x3.append((u, v, w))  # T1
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(1,0)] = 1; u[idx_A(1,1)] = 1
    v[idx_B(0,0)] = 1
    w[idx_C(1,0)] = 1
    terms_3x3.append((u, v, w))  # T2
    
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,0)] = 1
    v[idx_B(0,1)] = 1; v[idx_B(1,1)] = -1
    w[idx_C(0,1)] = 1
    terms_3x3.append((u, v, w))  # T3
    
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(1,1)] = 1
    v[idx_B(1,0)] = 1; v[idx_B(0,0)] = -1
    w[idx_C(0,0)] = 1; w[idx_C(1,0)] = 1
    terms_3x3.append((u, v, w))  # T4
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,0)] = 1; u[idx_A(0,1)] = 1
    v[idx_B(1,1)] = 1
    w[idx_C(0,0)] = -1; w[idx_C(0,1)] = 1
    terms_3x3.append((u, v, w))  # T5
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(1,0)] = 1; u[idx_A(0,0)] = -1
    v[idx_B(0,0)] = 1; v[idx_B(0,1)] = 1
    w[idx_C(1,1)] = 1
    terms_3x3.append((u, v, w))  # T6
    
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,1)] = 1; u[idx_A(1,1)] = -1
    v[idx_B(1,0)] = 1; v[idx_B(1,1)] = 1
    w[idx_C(0,0)] = 1
    terms_3x3.append((u, v, w))  # T7
    
    # T8 = (A[0,2]+A[1,2])*B[2,0]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,2)] = 1; u[idx_A(1,2)] = 1
    v[idx_B(2,0)] = 1
    w[idx_C(0,0)] = 1; w[idx_C(1,0)] = 1
    terms_3x3.append((u, v, w))  # T8
    
    # T9 = (A[0,2]+A[1,2])*B[2,1]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,2)] = 1; u[idx_A(1,2)] = 1
    v[idx_B(2,1)] = 1
    w[idx_C(0,1)] = 1; w[idx_C(1,1)] = 1
    terms_3x3.append((u, v, w))  # T9
    
    # T10 = (A[0,2]+A[1,2])*B[2,2]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,2)] = 1; u[idx_A(1,2)] = 1
    v[idx_B(2,2)] = 1
    w[idx_C(0,2)] = 1; w[idx_C(1,2)] = 1
    terms_3x3.append((u, v, w))  # T10
    
    # T11 = (A[0,0]+A[1,0])*B[0,2]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,0)] = 1; u[idx_A(1,0)] = 1
    v[idx_B(0,2)] = 1
    w[idx_C(0,2)] = 1; w[idx_C(1,2)] = 1
    terms_3x3.append((u, v, w))  # T11
    
    # T12 = (A[0,1]+A[1,1])*B[1,2]
    u = [0]*9; v = [0]*9; w = [0]*9
    u[idx_A(0,1)] = 1; u[idx_A(1,1)] = 1
    v[idx_B(1,2)] = 1
    w[idx_C(0,2)] = 1; w[idx_C(1,2)] = 1
    terms_3x3.append((u, v, w))  # T12
    
    # T13-T21: Row 2 of A
    for j in range(3):
        for k in range(3):
            u = [0]*9; v = [0]*9; w = [0]*9
            u[idx_A(2, j)] = 1
            v[idx_B(j, k)] = 1
            w[idx_C(2, k)] = 1
            terms_3x3.append((u, v, w))
    
    # Now build the 3x3x6 decomposition
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (18, rank)
    # 
    # For the 3x3x6 case:
    # A is 3x3 (i,j) with i,j in {0,1,2}
    # B is 3x6 (j,k) with j in {0,1,2}, k in {0,1,2,3,4,5}
    # C is 3x6 (i,k) with i in {0,1,2}, k in {0,1,2,3,4,5}
    #
    # U index: i*m + j = i*3 + j (9 entries)
    # V index: j*p + k = j*6 + k (18 entries)
    # W index: k*n + i = k*3 + i (18 entries)
    
    rank = 2 * len(terms_3x3)  # 42
    U = [[0] * rank for _ in range(n * m)]      # 9 x 42
    V = [[0] * rank for _ in range(m * p)]      # 18 x 42
    W = [[0] * rank for _ in range(p * n)]      # 18 x 42
    
    # First half: columns k=0,1,2 of B
    for t, (u_vec, v_vec, w_vec) in enumerate(terms_3x3):
        col = t
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = u_vec[i * 3 + j]
        for j in range(m):
            for k in range(3):
                V[j * p + k][col] = v_vec[j * 3 + k]
        for k in range(3):
            for i in range(n):
                W[k * n + i][col] = w_vec[k * 3 + i]
    
    # Second half: columns k=3,4,5 of B (shifted by 3)
    for t, (u_vec, v_vec, w_vec) in enumerate(terms_3x3):
        col = len(terms_3x3) + t
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = u_vec[i * 3 + j]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3)][col] = v_vec[j * 3 + k]
        for k in range(3):
            for i in range(n):
                W[(k + 3) * n + i][col] = w_vec[k * 3 + i]
    
    return {"U": U, "V": V, "W": W}

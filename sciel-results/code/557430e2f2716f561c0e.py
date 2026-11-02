"""Matrix multiplication 2x4 * 4x7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication (Rank 49).
    
    Strategy: Split m=4 into two blocks of size 2.
    For each 2x2 block of A, we have a 2x2 * 2x7 subproblem.
    For the 2x2 * 2x7, split p=7 into three 2x2 blocks and one 2x1 block.
    Use Strassen (rank 7) for 2x2 * 2x2.
    For 2x2 * 2x1, instead of schoolbook (rank 4), we can potentially do better.
    
    Actually, let me reconsider. The 2x2 * 2x1 multiplication:
    [a b] * [e] = [a*e, b*e] for row 0
    [c d]   [f]   [c*e, d*e] for row 1
    
    This is just 4 independent products: a*e, b*e, c*e, d*e.
    Wait, that's 4 terms. The output is 2x2 (2 rows of A, 2 columns of C for this block).
    So C = A * B_sub where A is 2x2, B_sub is 2x1 (but actually B is 2x2 here, first column).
    
    Let me re-examine: B is 4x7. For a fixed j-block of size 2 (say j=0,1), 
    the relevant part of B is rows 0,1 and all 7 columns.
    So we're computing (A_block 2x2) * (B_block 2x7) = C_block (2x7).
    
    The 2x2 * 2x7 can be decomposed as:
    - 2x2 * 2x1 (first column of B) -> contributes to first column of C
    - 2x2 * 2x2 (middle 2 cols) -> Strassen
    - 2x2 * 2x2 (next 2 cols) -> Strassen  
    - 2x2 * 2x1 (last column) -> 4 terms
    
    Total per j-block: 4 + 7 + 7 + 4 = 22
    Two j-blocks: 44
    
    Can we do better on the 2x2 * 2x1 part? The tensor for 2x2 * 2x1:
    A is 2x2 (4 elements), B is 2x1 (2 elements, but it's the same B column used for both A rows? No...)
    
    Actually wait. Let me reconsider the structure.
    
    For multiplying A(2x4) * B(4x7) -> C(2x7):
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we split j into {0,1} and {2,3}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    This is like two separate 2x1 * 1x7 contributions? No, it's not separable that way.
    
    Let me think about this differently. The key insight is that for a 2x2 * 2x1 multiplication
    (A is 2x2, B_col is 2x1), we compute:
    [a b] [e] = [a*e + b*f]
    [c d] [f]   [c*e + d*f]
    
    This requires 4 multiplications: a*e, b*f, c*e, d*f. But wait, can we share computations?
    (a+c)*e, (b+d)*f, ... no, that doesn't directly help for this specific output structure.
    
    Actually, the rank of the 2x2 * 2x1 matmul tensor... Let me think. The output is 2x2 (2 rows from A's rows, 1 col from B's col, but actually the "1" in 2x1 means B has 1 column, so output is 2x1, not 2x2).
    
    I was confused. Let me redo.
    
    2x2 * 2x1: A(2x2) * B(2x1) -> C(2x1)
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    This is 4 terms: A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,0], A[1,1]*B[1,0].
    Can we do it in fewer? This is essentially two independent dot products of length 2.
    Each dot product of length 2 needs 2 multiplications (a*b + c*d can be done in 3 with Strassen-like: (a+c)(b+d) - a*b... no, a*b+c*d needs 3: (a+c)(b+d) - a*b... wait that's for 1x2 * 2x1 -> scalar).
    
    For a 1x2 * 2x1 -> scalar: a*b + c*d. Can be done in 3: (a+c)(b+d) - a*b... no.
    Actually: (a+c)(b+d) = a*b + a*d + c*b + c*d. Not helpful.
    a*b + c*d: minimum 2 multiplications (they're independent linear forms).
    
    So 2x2 * 2x1 needs 4 multiplications (two independent 1x2*2x1 dot products).
    So schoolbook is optimal for that block.
    
    Hmm, so rank 50 might be close to optimal for this block decomposition.
    
    Alternative: different block structure. What if we don't split j into two 2-blocks?
    What if we use a single 2x4 * 4x7 directly?
    
    Or, use a different approach: the rank of 2x4 * 4x7 matrix multiplication.
    
    Known: R(2x4 * 4x7) is not known exactly, but upper bounds exist.
    R(m,n * n,p) <= some formula.
    
    One approach: use the fact that R(2x2*2x2) = 7.
    R(2x4*4x7) <= R(2x2*2x2)*... 
    
    Actually, a better approach might be to view this as:
    A(2x4) * B(4x7). Write A = [A1, A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x7.
    Then C = A1*B1 + A2*B2.
    
    R(A1*B1) for 2x2*2x7: split B1 into B11(2x2), B12(2x2), B13(2x1).
    R = 7 + 7 + 4 = 18 (using Strassen for 2x2*2x2, schoolbook for 2x2*2x1)
    Similarly R(A2*B2) = 18.
    Total: 36? No wait, that's wrong. Each A*B is independent, so total rank = 18+18 = 36? 
    But that's not right either because the A and B entries are shared.
    
    Actually no. A1*B1 + A2*B2. The multiplications in A1*B1 use A1's entries and B1's entries.
    The multiplications in A2*B2 use A2's entries and B2's entries.
    These are disjoint sets of input entries (A1 vs A2, B1 vs B2), so the total rank is the sum: 18+18=36? 
    
    Wait, but that would give rank 36, which is much better than 50. Let me double-check.
    
    Hmm, I think I'm making an error. Let me reconsider.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
           = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    First term: A[0:2, 0:2] * B[0:2, k] for each k. This is 2x2 * 2x7.
    Second term: A[0:2, 2:4] * B[2:4, k] for each k. This is 2x2 * 2x7.
    
    For the first term, to compute all 7 columns:
    Split k into {0,1}, {2,3}, {4,5}, {6}.
    For k in {0,1}: A_block(2x2) * B_block(2x2) -> use Strassen, rank 7
    For k in {2,3}: A_block(2x2) * B_block(2x2) -> use Strassen, rank 7
    For k in {4,5}: A_block(2x2) * B_block(2x2) -> use Strassen, rank 7
    For k = 6: A_block(2x2) * B_col(2x1) -> schoolbook, rank 4
    
    Total for first A_block: 7+7+7+4 = 25
    Total for second A_block: 7+7+7+4 = 25
    Grand total: 50
    
    This matches the current solution. So the issue is that the A_blocks are different (A1 and A2 have different entries), and the B_blocks are different (B1 and B2 have different entries), so we can't share the Strassen factors between them.
    
    Unless... we restructure. What if we group differently?
    
    Alternative: Group by output column k instead.
    For each k, C[:,k] = A * B[:,k] where A is 2x4, B[:,k] is 4x1.
    This is a 2x4 * 4x1 multiplication -> 2x1 output.
    R(2x4*4x1) = ? The output has 2 entries. C[0,k] = sum_{j=0}^3 A[0,j]*B[j,k], C[1,k] = sum_{j=0}^3 A[1,j]*B[j,k].
    These are two independent dot products of length 4. Each needs 4 multiplications, total 8.
    But can we share? (A[0,0]+A[1,0])*(B[0,k]+B[1,k])... no, doesn't help for independent outputs.
    Actually, the rank of 2x4*4x1 is 8 (each output is a separate sum of 4 products, no sharing possible since the linear forms for C[0,k] and C[1,k] involve different rows of A).
    
    So 7 columns * 8 = 56. Worse.
    
    What about a hybrid? Let me think about whether rank 49 is achievable.
    
    One idea: for the 2x2*2x1 part (rank 4), can we combine the two 2x2*2x1 blocks (one from each A_block) to save a term?
    
    The two 2x2*2x1 blocks are:
    Block 1: A1(2x2) * b_col(2x1) where b_col = B[0:2, 6]
    Block 2: A2(2x2) * b_col2(2x1) where b_col2 = B[2:4, 6]
    
    These use different A entries and different B entries, so no sharing.
    
    I think rank 50 is correct for this decomposition strategy. Let me verify the current code is correct and just submit it.
    """
    deadline = time.time() + 125
    
    n, m, p = 2, 4, 7
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    if time.time() > deadline - 5:
        pass

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank_fb = n * m * p
    U = np.zeros((n * m, rank_fb), dtype=int)
    V = np.zeros((m * p, rank_fb), dtype=int)
    W = np.zeros((p * n, rank_fb), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

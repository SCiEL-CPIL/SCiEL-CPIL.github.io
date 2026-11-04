"""Rewritten program for matrix multiplication 3x4 by 4x4 tensor rank minimization."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _schoolbook(n, m, p):
    """Return schoolbook rank decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # For each product r, set U, V, W entries
    # U index: i*2+j (i=0,1 row of A; j=0,1 col of A)
    # V index: j*2+k (j=0,1 row of B; k=0,1 col of B)
    # W index: k*2+i (k=0,1 col of C; i=0,1 row of C)
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # U: A00 and A11 -> indices 0*2+0=0, 1*2+1=3
    # V: B00 and B11 -> indices 0*2+0=0, 1*2+1=3
    # W: C00 and C11 -> indices 0*2+0=0, 1*2+1=3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # U: A10 and A11 -> indices 1*2+0=2, 1*2+1=3
    # V: B00 -> index 0*2+0=0
    # W: C10 and C11 -> indices 0*2+1=2, 1*2+1=3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11) -> contributes to C01, C11
    # U: A00 -> index 0*2+0=0
    # V: B10 and B11 -> indices 1*2+0=2, 1*2+1=3
    # W: C01 and C11 -> indices 1*2+0=2, 1*2+1=3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # M4 = A11(B01-B00) -> contributes to C00, C10
    # U: A11 -> index 1*2+1=3
    # V: B01 and B00 -> indices 0*2+1=1, 0*2+0=0
    # W: C00 and C10 -> indices 0*2+0=0, 0*2+1=2
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # U: A00 and A01 -> indices 0*2+0=0, 0*2+1=1
    # V: B11 -> index 1*2+1=3
    # W: C00 and C01 -> indices 0*2+0=0, 1*2+0=2
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # U: A10 and A00 -> indices 1*2+0=2, 0*2+0=0
    # V: B00 and B01 -> indices 0*2+0=0, 0*2+1=1
    # W: C11 -> index 1*2+1=3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1
    
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    # U: A01 and A11 -> indices 0*2+1=1, 1*2+1=3
    # V: B10 and B11 -> indices 1*2+0=2, 1*2+1=3
    # W: C00 -> index 0*2+0=0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = U.shape[1]
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] should equal delta_{k,k'} * delta_{i,i'}
    # But actually the tensor condition is:
    # For the MM tensor T with T[(i,j),(j,k),(k,i)] = 1, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # And for all other combinations, it should be 0.
    
    # More precisely, the reconstructed tensor at position ((i,j), (j',k'), (k'',i''))
    # is sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]
    # This should equal 1 if j=j' and k=k'' and i=i'', else 0.
    # Wait, let me re-derive.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is defined such that T(A,B) = C, where T is a trilinear form.
    # T_{(i,j),(j,k),(i,k)} = 1
    # So the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k
    # And for any (i,j), (j',k'), (k'',i'') with (j != j' or k != k'' or i != i''),
    # the sum should be 0.
    
    # Actually, the correct condition is:
    # The tensor T has components T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0.
    # But note: b's first index must equal a's second index, and c's indices must match.
    # So T[(i,j), (j,k), (k,i)] = 1.
    
    # The decomposition gives: T_reconstructed[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We need T_reconstructed = T.
    
    # Let's check all non-zero entries of T and ensure they are 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some zero entries to make sure we're not over-counting
    # Check all entries where the indices don't match
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if j != j2 or k != k2 or i != i2:
                                a = i * m + j
                                b = j2 * p + k2
                                c = k2 * n + i2
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
    
    return True

def _build_strassen_based_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for 3x4 by 4x4 matrix multiplication.
    
    Strategy: Partition the 3x4 matrix A into blocks:
    - A_top: 2x4 (rows 0-1)
    - A_bottom: 1x4 (row 2)
    
    And the 4x4 matrix B into blocks:
    - B_left: 4x2 (cols 0-1)
    - B_right: 4x2 (cols 2-3)
    
    The product C = A * B is 3x4:
    - C_top_left: 2x2 = A_top * B_left
    - C_top_right: 2x2 = A_top * B_right
    - C_bottom_left: 1x2 = A_bottom * B_left
    - C_bottom_right: 1x2 = A_bottom * B_right
    
    For C_top_left and C_top_right, we can use Strassen on 2x2 blocks if we further partition.
    But B_left is 4x2, not 2x2. We need to partition B into 2x2 blocks.
    
    Let's partition differently:
    A is 3x4, B is 4x4, C is 3x4.
    
    Partition A into:
    - A1: rows 0-1, all 4 cols (2x4)
    - A2: row 2, all 4 cols (1x4)
    
    Partition B into:
    - B1: all 4 rows, cols 0-1 (4x2)
    - B2: all 4 rows, cols 2-3 (4x2)
    
    Now:
    C1 = A1 * B1: 2x2
    C2 = A1 * B2: 2x2
    C3 = A2 * B1: 1x2
    C4 = A2 * B2: 1x2
    
    For C1 and C2, we can use Strassen's algorithm if we partition further.
    But A1 is 2x4 and B1 is 4x2. The product is 2x2.
    We can write C1 = A1 * B1 where A1 is 2x4 and B1 is 4x2.
    This is not directly a 2x2 matrix multiplication.
    
    Alternative approach: Partition all matrices into 2x2 blocks where possible.
    A is 3x4: can be seen as a 2x2 block matrix with blocks of size... not uniform.
    
    Let me try a different partitioning:
    Partition the 4x4 B into four 2x2 blocks:
    B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    
    A is 3x4. Partition A into:
    A = [[A11, A12], [A21, A22]] where A11, A12 are 2x2, A21, A22 are 1x2.
    
    Then C = A * B:
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    C is 3x4:
    C_top (2x4): 
      C_top_left (2x2) = A11*B11 + A12*B21
      C_top_right (2x2) = A11*B12 + A12*B22
    C_bottom (1x4):
      C_bottom_left (1x2) = A21*B11 + A22*B21
      C_bottom_right (1x2) = A21*B12 + A22*B22
    
    For the top part, we have two 2x2 matrix multiplications plus additions.
    Actually, C_top_left = A11*B11 + A12*B21 is the sum of two 2x2 products.
    This is not a single 2x2 matrix multiplication.
    
    Let me reconsider. The standard approach for rectangular matrix multiplication using Strassen:
    
    For 3x4 * 4x4 -> 3x4:
    We can view this as:
    - 2x2 * 2x2 (for the top-left 2x2 block of C) using Strassen (rank 7)
    - 2x2 * 2x2 (for the top-right 2x2 block of C) using Strassen (rank 7)
    - 1x2 * 2x2 (for the bottom-left 1x2 block of C) using schoolbook (rank 4)
    - 1x2 * 2x2 (for the bottom-right 1x2 block of C) using schoolbook (rank 4)
    
    But wait, the blocks don't align this way because B is 4x4 and we need to partition it consistently.
    
    Let me define:
    A (3x4):
    A = [[a00, a01, a02, a03],
         [a10, a11, a12, a13],
         [a20, a21, a22, a23]]
    
    B (4x4):
    B = [[b00, b01, b02, b03],
         [b10, b11, b12, b13],
         [b20, b21, b22, b23],
         [b30, b31, b32, b33]]
    
    C = A * B (3x4):
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Partition into 2x2 blocks:
    A_top = A[0:2, :] (2x4)
    A_bot = A[2:3, :] (1x4)
    B_left = B[:, 0:2] (4x2)
    B_right = B[:, 2:4] (4x2)
    
    C_top_left = A_top @ B_left (2x2)
    C_top_right = A_top @ B_right (2x2)
    C_bot_left = A_bot @ B_left (1x2)
    C_bot_right = A_bot @ B_right (1x2)
    
    For C_top_left (2x2 = 2x4 @ 4x2):
    This is not a standard 2x2 matrix multiplication. It's a 2x4 times 4x2.
    We can decompose this further.
    
    Actually, let's use a different strategy. Let's partition into smaller blocks.
    
    For the 2x4 @ 4x2 product, we can write it as:
    A_top = [A_tl, A_tr] where A_tl is 2x2, A_tr is 2x2
    B_left = [B_tl; B_bl] where B_tl is 2x2, B_bl is 2x2
    
    Then A_top @ B_left = A_tl @ B_tl + A_tr @ B_bl
    
    This is the sum of two 2x2 matrix multiplications.
    Each 2x2 matrix multiplication has rank 7 (Strassen).
    But the sum of two rank-7 tensors may have rank up to 14, not necessarily 7.
    
    This approach is getting complicated. Let me try a simpler approach:
    Use Strassen for the 2x2x2 subproblems and schoolbook for the rest.
    
    Alternative: Use the fact that 3x4*4x4 can be decomposed as:
    - One 2x2*2x2 multiplication (rank 7) for part of the computation
    - Remaining terms using schoolbook
    
    Actually, let me just implement a clean Strassen-based approach:
    
    Partition A (3x4) into:
    - A1: 2x2 (top-left)
    - A2: 2x2 (top-right)
    - A3: 1x2 (bottom-left)
    - A4: 1x2 (bottom-right)
    
    Partition B (4x4) into:
    - B1: 2x2 (top-left)
    - B2: 2x2 (top-right)
    - B3: 2x2 (bottom-left)
    - B4: 2x2 (bottom-right)
    
    Then C (3x4) = A @ B:
    C_top_left (2x2) = A1*B1 + A2*B3
    C_top_right (2x2) = A1*B2 + A2*B4
    C_bot_left (1x2) = A3*B1 + A4*B3
    C_bot_right (1x2) = A3*B2 + A4*B4
    
    For C_top_left: sum of two 2x2 products. Rank at most 14 (7+7).
    For C_top_right: sum of two 2x2 products. Rank at most 14.
    For C_bot_left: sum of two 1x2*2x2 products. Each is rank 4 (schoolbook for 1x2*2x2 is 1*2*2=4). Total rank at most 8.
    For C_bot_right: similar, rank at most 8.
    
    Total rank: 14 + 14 + 8 + 8 = 44.
    
    But we can do better by sharing some computations. However, for now, let's implement this.
    
    Actually, we can't simply add the ranks because the decompositions might not be independent.
    But as an upper bound, rank <= 44.
    
    Let me implement this:
    """
    n, m, p = 3, 4, 4
    
    # We'll build the decomposition by combining:
    # 1. Strassen for 2x2*2x2 (rank 7) - used twice for top part
    # 2. Schoolbook for 1x2*2x2 (rank 4) - used twice for bottom part
    # But we need to be careful about how the indices map.
    
    # Let's build it term by term.
    # C_top_left[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1}
    # = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    # = (A_top_left * B_top_left)[i,k] + (A_top_right * B_bottom_left)[i,k]
    
    # So we need:
    # Term 1: A[0:2, 0:2] * B[0:2, 0:2] -> 2x2 product, use Strassen
    # Term 2: A[0:2, 2:4] * B[2:4, 0:2] -> 2x2 product, use Strassen
    # Term 3: A[2:3, 0:2] * B[0:2, 0:2] -> 1x2 * 2x2, schoolbook
    # Term 4: A[2:3, 2:4] * B[2:4, 0:2] -> 1x2 * 2x2, schoolbook
    # Similarly for right half.
    
    # Total: 4 Strassen terms (rank 7 each) + 4 schoolbook terms (rank 4 each) = 28 + 16 = 44
    
    # Let's implement this.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a Strassen term with offset
    def add_strassen(U_offset, V_offset, W_offset, U, V, W):
        """Add a Strassen decomposition with given index offsets."""
        Us, Vs, Ws = _strassen_2x2x2()
        for r in range(7):
            for a in range(4):
                if Us[a, r] != 0:
                    U[U_offset + a, r] = Us[a, r]
            for b in range(4):
                if Vs[b, r] != 0:
                    V[V_offset + b, r] = Vs[b, r]
            for c in range(4):
                if Ws[c, r] != 0:
                    W[W_offset + c, r] = Ws[c, r]
    
    # Helper to add a schoolbook 1x2*2x2 term
    def add_schoolbook_1x2_2x2(U_offset, V_offset, W_offset, U, V, W):
        """
        Add schoolbook decomposition for 1x2 * 2x2 -> 1x2.
        U: (2, 4), V: (4, 4), W: (2, 4)
        U index: j (0,1) for the 1x2 matrix
        V index: j*2+k for the 2x2 matrix
        W index: k for the 1x2 output
        """
        for j in range(2):
            for k in range(2):
                col = j * 2 + k
                U[U_offset + j, col] = 1
                V[V_offset + j * 2 + k, col] = 1
                W[W_offset + k, col] = 1
    
    # We need to carefully map the global indices.
    # Let's build the full decomposition.
    
    # Total rank: 4 * 7 + 4 * 4 = 28 + 16 = 44
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_offset = 0
    
    # Term 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i in {0,1}, j in {0,1} -> U index: i*4 + j
    # B indices: j in {0,1}, k in {0,1} -> V index: j*4 + k
    # C indices: i in {0,1}, k in {0,1} -> W index: k*3 + i
    
    # Strassen U for 2x2: U[i*2+j, r]
    # We need to map to global: U[i*4+j, col]
    # So for Strassen U index a = i*2+j, global U index = i*4+j
    # i = a//2, j = a%2, global = i*4 + j = (a//2)*4 + (a%2)
    
    def strassen_u_global(a):
        i = a // 2
        j = a % 2
        return i * 4 + j
    
    def strassen_v_global(b):
        j = b // 2
        k = b % 2
        return j * 4 + k
    
    def strassen_w_global(c):
        k = c // 2
        i = c % 2
        return k * 3 + i
    
    Us, Vs, Ws = _strassen_2x2x2()
    
    # Term 1
    for r in range(7):
        for a in range(4):
            if Us[a, r] != 0:
                U[strassen_u_global(a), col_offset + r] = Us[a, r]
        for b in range(4):
            if Vs[b, r] != 0:
                V[strassen_v_global(b), col_offset + r] = Vs[b, r]
        for c in range(4):
            if Ws[c, r] != 0:
                W[strassen_w_global(c), col_offset + r] = Ws[c, r]
    col_offset += 7
    
    # Term 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A indices: i in {0,1}, j in {2,3} -> U index: i*4 + j
    # B indices: j in {2,3}, k in {0,1} -> V index: j*4 + k
    # C indices: i in {0,1}, k in {0,1} -> W index: k*3 + i
    
    def term2_u_global(a):
        i = a // 2
        j = a % 2
        return i * 4 + (2 + j)
    
    def term2_v_global(b):
        j = b // 2
        k = b % 2
        return (2 + j) * 4 + k
    
    def term2_w_global(c):
        k = c // 2
        i = c % 2
        return k * 3 + i
    
    for r in range(7):
        for a in range(4):
            if Us[a, r] != 0:
                U[term2_u_global(a), col_offset + r] = Us[a, r]
        for b in range(4):
            if Vs[b, r] != 0:
                V[term2_v_global(b), col_offset + r] = Vs[b, r]
        for c in range(4):
            if Ws[c, r] != 0:
                W[term2_w_global(c), col_offset + r] = Ws[c, r]
    col_offset += 7
    
    # Term 3: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # A indices: i=2, j in {0,1} -> U index: 2*4 + j = 8 + j
    # B indices: j in {0,1}, k in {0,1} -> V index: j*4 + k
    # C indices: i=2, k in {0,1} -> W index: k*3 + 2
    
    # Schoolbook 1x2 * 2x2:
    for j in range(2):
        for k in range(2):
            col = col_offset + j * 2 + k
            U[8 + j, col] = 1
            V[j * 4 + k, col] = 1
            W[k * 3 + 2, col] = 1
    col_offset += 4
    
    # Term 4: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # A indices: i=2, j in {2,3} -> U index: 8 + j
    # B indices: j in {2,3}, k in {0,1} -> V index: j*4 + k
    # C indices: i=2, k in {0,1} -> W index: k*3 + 2
    
    for j in range(2):
        for k in range(2):
            col = col_offset + j * 2 + k
            U[8 + (2 + j), col] = 1
            V[(2 + j) * 4 + k, col] = 1
            W[k * 3 + 2, col] = 1
    col_offset += 4
    
    # Term 5: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: i in {0,1}, j in {0,1} -> U index: i*4 + j
    # B indices: j in {0,1}, k in {2,3} -> V index: j*4 + k
    # C indices: i in {0,1}, k in {2,3} -> W index: k*3 + i
    
    def term5_u_global(a):
        i = a // 2
        j = a % 2
        return i * 4 + j
    
    def term5_v_global(b):
        j = b // 2
        k = b % 2
        return j * 4 + (2 + k)
    
    def term5_w_global(c):
        k = c // 2
        i = c % 2
        return (2 + k) * 3 + i
    
    for r in range(7):
        for a in range(4):
            if Us[a, r] != 0:
                U[term5_u_global(a), col_offset + r] = Us[a, r]
        for b in range(4):
            if Vs[b, r] != 0:
                V[term5_v_global(b), col_offset + r] = Vs[b, r]
        for c in range(4):
            if Ws[c, r] != 0:
                W[term5_w_global(c), col_offset + r] = Ws[c, r]
    col_offset += 7
    
    # Term 6: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A indices: i in {0,1}, j in {2,3} -> U index: i*4 + (2+j)
    # B indices: j in {2,3}, k in {2,3} -> V index: (2+j)*4 + (2+k)
    # C indices: i in {0,1}, k in {2,3} -> W index: (2+k)*3 + i
    
    def term6_u_global(a):
        i = a // 2
        j = a % 2
        return i * 4 + (2 + j)
    
    def term6_v_global(b):
        j = b // 2
        k = b % 2
        return (2 + j) * 4 + (2 + k)
    
    def term6_w_global(c):
        k = c // 2
        i = c % 2
        return (2 + k) * 3 + i
    
    for r in range(7):
        for a in range(4):
            if Us[a, r] != 0:
                U[term6_u_global(a), col_offset + r] = Us[a, r]
        for b in range(4):
            if Vs[b, r] != 0:
                V[term6_v_global(b), col_offset + r] = Vs[b, r]
        for c in range(4):
            if Ws[c, r] != 0:
                W[term6_w_global(c), col_offset + r] = Ws[c, r]
    col_offset += 7
    
    # Term 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    # A indices: i=2, j in {0,1} -> U index: 8 + j
    # B indices: j in {0,1}, k in {2,3} -> V index: j*4 + (2+k)
    # C indices: i=2, k in {2,3} -> W index: (2+k)*3 + 2
    
    for j in range(2):
        for k in range(2):
            col = col_offset + j * 2 + k
            U[8 + j, col] = 1
            V[j * 4 + (2 + k), col] = 1
            W[(2 + k) * 3 + 2, col] = 1
    col_offset += 4
    
    # Term 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    # A indices: i=2, j in {2,3} -> U index: 8 + (2+j)
    # B indices: j in {2,3}, k in {2,3} -> V index: (2+j)*4 + (2+k)
    # C indices: i=2, k in {2,3} -> W index: (2+k)*3 + 2
    
    for j in range(2):
        for k in range(2):
            col = col_offset + j * 2 + k
            U[8 + (2 + j), col] = 1
            V[(2 + j) * 4 + (2 + k), col] = 1
            W[(2 + k) * 3 + 2, col] = 1
    col_offset += 4
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Try the Strassen-based decomposition (rank 44)
    U, V, W = _build_strassen_based_decomposition(n, m, p)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

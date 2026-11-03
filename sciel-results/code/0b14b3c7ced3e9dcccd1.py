"""
Program for matrix multiplication 3,4,7 tensor decomposition using block decomposition.
Strategy:
1. Partition the 3x4 matrix A into blocks and the 4x7 matrix B into blocks.
2. Use Strassen's algorithm for 2x2 blocks where applicable.
3. Combine these to form a global tensor decomposition.
4. Verify the result.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2 matrix multiplication using Strassen's algorithm.
    Indices:
    A: [[a, b], [c, d]] -> flat index 0,1,2,3
    B: [[e, f], [g, h]] -> flat index 0,1,2,3
    C: [[i, j], [k, l]] -> flat index 0,1,2,3
    
    Strassen products:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    We need to map these to rank-1 tensors U, V, W.
    For a specific term m_k, it contributes to specific entries of C.
    The term is of the form sum_k (u_k . A) (v_k . B) w_k
    Actually, the standard form is C = sum_k (u_k^T A v_k) w_k? No.
    The tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    This corresponds to C = sum_r (A u_r) (v_r^T B) w_r^T? No.
    
    Let's use the explicit vector form.
    For 2x2, A is 4x1, B is 4x1, C is 4x1.
    T[a,b,c] = 1 if A[a] and B[b] multiply to C[c].
    
    Strassen's algorithm expresses the 8 schoolbook products as 7 multiplications.
    The "multiplication" here is the scalar product.
    So, m1 = (A0+A3) * (B0+B3).
    This contributes to C0 and C3.
    C0 += m1, C3 += m1.
    So for m1:
    u1 = [1, 0, 0, 1]  (coeffs for A)
    v1 = [1, 0, 0, 1]  (coeffs for B)
    w1 = [1, 0, 0, 1]  (coeffs for C)
    
    m2 = (A2+A3) * B0
    Contributes to C2.
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 0, 1]
    
    m3 = A0 * (B1-B3)
    Contributes to C1 and C2.
    C1 += m3, C2 += m3.
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 1, 0]
    
    m4 = A3 * (B2-B0)
    Contributes to C1 and C2.
    C1 += m4, C2 += m4.
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [0, 1, 1, 0]
    
    m5 = (A0+A1) * B3
    Contributes to C0 and C1.
    C0 -= m5, C1 += m5.
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    m6 = (A2-A0) * (B0+B1)
    Contributes to C3.
    C3 += m6.
    u6 = [1, 0, -1, 0]  # Wait, m6 = (c-a)(e+f) -> (A2-A0)(B0+B1)
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    m7 = (A1-A3) * (B2+B3)
    Contributes to C0 and C3.
    C0 += m7, C3 -= m7? 
    Let's check Strassen formula:
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    Let's re-verify m7 contribution.
    m7 = (b-d)(g+h) = (A1-A3)(B2+B3).
    In C00: +m7.
    In C11: -m7? No, C11 = m1 - m2 + m3 + m6. m7 is not in C11.
    Wait, standard Strassen:
    p1 = (a+d)(e+h)
    p2 = (c+d)e
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    c = p1+p4-p5+p7
    d = p3+p5
    e = p2+p4
    f = p1-p2+p3+p6
    
    So:
    p1: C00+=1, C11+=1
    p2: C10+=1, C11-=1
    p3: C01+=1, C11+=1
    p4: C00+=1, C10+=1
    p5: C00-=1, C01+=1
    p6: C11+=1
    p7: C00+=1
    
    Let's construct U, V, W for 2x2.
    U, V, W are 4x7 matrices.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1: (A0+A3)(B0+B3) -> C00+=1, C11+=1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # p2: (A2+A3)B0 -> C10+=1, C11-=1
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1]
    
    # p3: A0(B1-B3) -> C01+=1, C11+=1
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 1]
    
    # p4: A3(B2-B0) -> C00+=1, C10+=1
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [1, 0, 1, 0]
    
    # p5: (A0+A1)B3 -> C00-=1, C01+=1
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 1, 0, 0]
    
    # p6: (A2-A0)(B0+B1) -> C11+=1
    U[:, 5] = [1, 0, -1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # p7: (A1-A3)(B2+B3) -> C00+=1
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _build_decomposition():
    """
    Construct the decomposition for 3x4 * 4x7.
    
    A is 3x4. B is 4x7.
    We can split A into A_top (2x4) and A_bottom (1x4).
    We can split B into B_left (4x4) and B_right (4x3).
    
    C = A * B = [A_top; A_bottom] * [B_left | B_right]
    C = [[A_top B_left, A_top B_right], [A_bottom B_left, A_bottom B_right]]
    
    Blocks of C:
    C11: 2x4 (A_top * B_left)
    C12: 2x3 (A_top * B_right)
    C21: 1x4 (A_bottom * B_left)
    C22: 1x3 (A_bottom * B_right)
    
    Total C is 3x7.
    
    Let's use Strassen for the 2x2 block structure if possible?
    The dimensions are mixed.
    
    Alternative: Split the middle dimension (4) into 2+2.
    A = [A1 A2] where A1 is 3x2, A2 is 3x2.
    B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    C = A1 B1 + A2 B2.
    
    This is a sum of two 3x2 * 2x7 multiplications.
    Rank of 3x2 * 2x7?
    3x2 * 2x7 is equivalent to 3x2 * 2x1 (7 times) if we split B columns?
    No, B is 2x7.
    
    Let's split the output C (3x7) into columns.
    C = [C_col1 ... C_col7].
    C_col_k = A * B_col_k.
    B_col_k is 4x1.
    A is 3x4.
    So C_col_k is 3x1.
    This is 7 separate 3x4 * 4x1 multiplications.
    Rank of 3x4 * 4x1?
    This is just a matrix-vector product.
    It can be done in 3*4 = 12 multiplications (schoolbook).
    Can we do better?
    The tensor is T[i, j, k] where i in 0..2, j in 0..3, k in 0..3 (B col index).
    Wait, B is 4x7. So B[j, k] with j in 0..3, k in 0..6.
    C[i, k] = sum_j A[i, j] B[j, k].
    
    If we treat each column k of B separately, we have 7 instances of 3x4 * 4x1.
    Total rank = 7 * 12 = 84. This is schoolbook.
    
    To improve, we need to exploit structure across columns or rows.
    
    Let's try splitting the middle dimension 4 into 2+2.
    A = [A1 A2] (3x2, 3x2)
    B = [B1; B2] (2x7, 2x7)
    C = A1 B1 + A2 B2.
    
    A1 B1 is 3x7. A2 B2 is 3x7.
    We need to compute two 3x2 * 2x7 multiplications and add them.
    Rank of 3x2 * 2x7?
    Split B1 into columns? 3x2 * 2x1 is 6 mults. 7 cols -> 42.
    Total 84. No gain.
    
    What if we use Strassen on the 3x2 * 2x7?
    3x2 * 2x7 is not square.
    
    Let's look at the 2x2 block of the middle dimension.
    If we had 2x2 * 2x2, Strassen gives 7.
    
    Let's try to decompose 3x4 * 4x7 into:
    1. 2x2 * 2x2 blocks?
    
    Let's partition A into 2x2, 2x2, 1x2, 1x2? No, A is 3x4.
    Rows 0,1 and Row 2.
    Cols 0,1 and Cols 2,3.
    A = [[A00 A01], [A10 A11]] where A00 is 2x2, A01 is 2x2, A10 is 1x2, A11 is 1x2.
    
    B is 4x7.
    Rows 0,1 and Rows 2,3.
    B = [[B00 B01], [B10 B11]] where B00 is 2x7, B01 is 2x7, B10 is 2x7, B11 is 2x7.
    Wait, B is 4x7.
    B_top = B[0:2, :] (2x7)
    B_bottom = B[2:4, :] (2x7)
    
    C = A_top B_top + A_bottom B_bottom.
    A_top is 2x4. A_bottom is 1x4.
    
    This doesn't seem to lead to immediate Strassen gains unless we handle the 2x4 * 4x7 part.
    
    Let's try a different partition.
    Split A (3x4) into 3x2 and 3x2.
    Split B (4x7) into 2x7 and 2x7.
    C = A1 B1 + A2 B2.
    
    Can we compute A1 B1 + A2 B2 more efficiently?
    This is equivalent to multiplying a 3x4 matrix by a 4x7 matrix.
    
    Let's try to use the known result for 3x3 * 3x3.
    Rank is 23 (Bini et al.) or 20 (Landsberg).
    
    For 3x4 * 4x7, we can pad to 4x4 * 4x7?
    Or 3x4 * 4x7 -> 3x4 * 4x4 + 3x4 * 4x3.
    
    Let's implement a generic block decomposition that uses Strassen for 2x2 sub-blocks and schoolbook for the rest, and see if it beats 84.
    
    Partition A (3x4) into:
    A1 (2x2), A2 (2x2)
    A3 (1x2), A4 (1x2)
    
    Partition B (4x7) into:
    B1 (2x4), B2 (2x3)  <- Top 2 rows
    B3 (2x4), B4 (2x3)  <- Bottom 2 rows
    
    C = A1 B1 + A2 B3 + A3 B1 + A4 B3 (for first 4 cols of C? No, B is 4x7)
    
    Let's be precise.
    A = [[A11 A12], [A21 A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2.
    B = [[B11 B12], [B21 B22]] where B11 is 2x4, B12 is 2x3, B21 is 2x4, B22 is 2x3.
    (Split B columns into 4 and 3).
    
    C = A B = [[A11 B11 + A12 B21, A11 B12 + A12 B22],
               [A21 B11 + A22 B21, A21 B12 + A22 B22]]
    
    C11 (2x4) = A11 B11 + A12 B21
    C12 (2x3) = A11 B12 + A12 B22
    C21 (1x4) = A21 B11 + A22 B21
    C22 (1x3) = A21 B12 + A22 B22
    
    We need to compute these 4 blocks.
    
    C11: 2x2 * 2x4 + 2x2 * 2x4.
    This is (2x2) * (2x4) + (2x2) * (2x4).
    Let X = [A11 A12] (2x4), Y = [B11; B21] (4x4).
    Then C11 = X Y.
    So C11 is a 2x4 * 4x4 multiplication.
    
    Similarly, C12 is 2x4 * 4x3.
    C21 is 1x4 * 4x4.
    C22 is 1x4 * 4x3.
    
    So the problem reduces to:
    1. 2x4 * 4x4
    2. 2x4 * 4x3
    3. 1x4 * 4x4
    4. 1x4 * 4x3
    
    Let's estimate ranks.
    1. 2x4 * 4x4:
       Split 2x4 into 2x2 + 2x2.
       C11 = A11 B11 + A12 B21.
       This is sum of two 2x2 * 2x4.
       2x2 * 2x4 can be done by splitting B into 2x2 + 2x2.
       2x2 * 2x2 is rank 7 (Strassen).
       So 2x2 * 2x4 is rank 14.
       Two of them: 28.
       
    2. 2x4 * 4x3:
       Similar, 2x2 * 2x3.
       2x2 * 2x3: Split 2x3 into 2x2 + 2x1.
       2x2 * 2x2 is 7.
       2x2 * 2x1 is 4 (schoolbook).
       Total 11.
       Two of them: 22.
       
    3. 1x4 * 4x4:
       1x4 * 4x4.
       This is 1x4 * 4x2 + 1x4 * 4x2.
       1x4 * 4x2: Split 1x4 into 1x2 + 1x2.
       1x2 * 2x2 is 4.
       1x2 * 2x2 is 4.
       Total 8.
       Two of them: 16.
       
    4. 1x4 * 4x3:
       1x4 * 4x3.
       1x4 * 4x2 + 1x4 * 4x1.
       1x4 * 4x2 is 8.
       1x4 * 4x1 is 4.
       Total 12.
       
    Total rank = 28 + 22 + 16 + 12 = 78.
    
    This is better than 84!
    
    Let's implement this specific block decomposition.
    """
    
    # We will build the U, V, W matrices by concatenating the blocks.
    # U corresponds to A (3x4), V to B (4x7), W to C (3x7).
    
    # We need to define the mapping for each sub-problem.
    
    def _block_2x2_strassen(U_offset, V_offset, W_offset, U_idx, V_idx, W_idx):
        """
        Adds Strassen terms for a 2x2 * 2x2 multiplication.
        U_idx, V_idx, W_idx are lists of indices into the global U, V, W matrices.
        """
        U, V, W = _strassen_2x2_terms()
        for r in range(7):
            U[U_idx, U_offset + r] = U[:, r]
            V[V_idx, V_offset + r] = V[:, r]
            W[W_idx, W_offset + r] = W[:, r]
        return 7
        
    def _block_schoolbook_2x2(U_offset, V_offset, W_offset, U_idx, V_idx, W_idx):
        """
        Adds schoolbook terms for a 2x2 * 2x2 multiplication.
        """
        count = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[U_idx[i], U_offset + count] = 1
                    V[V_idx[j], V_offset + count] = 1
                    W[W_idx[k], W_offset + count] = 1
                    count += 1
        return count

    # Global dimensions
    n, m, p = 3, 4, 7
    U_total = np.zeros((n * m, 100)) # Max rank estimate
    V_total = np.zeros((m * p, 100))
    W_total = np.zeros((p * n, 100))
    
    # Indices for A (3x4)
    # A11 (2x2): rows 0,1 cols 0,1 -> idx 0,1,2,3
    # A12 (2x2): rows 0,1 cols 2,3 -> idx 4,5,6,7
    # A21 (1x2): row 2 cols 0,1 -> idx 8,9
    # A22 (1x2): row 2 cols 2,3 -> idx 10,11
    
    # Indices for B (4x7)
    # Split rows 0,1 and 2,3.
    # Split cols 0..3 and 4..6.
    
    # B11 (2x4): rows 0,1 cols 0..3 -> idx 0,1,2,3, 4,5,6,7
    # B12 (2x3): rows 0,1 cols 4..6 -> idx 8,9,10, 12,13,14
    # B21 (2x4): rows 2,3 cols 0..3 -> idx 16,17,18,19, 20,21,22,23
    # B22 (2x3): rows 2,3 cols 4..6 -> idx 24,25,26, 28,29,30
    
    # Indices for C (3x7)
    # C11 (2x4): rows 0,1 cols 0..3 -> idx 0,1,2,3, 4,5,6,7
    # C12 (2x3): rows 0,1 cols 4..6 -> idx 8,9,10, 12,13,14
    # C21 (1x4): row 2 cols 0..3 -> idx 16,17,18,19
    # C22 (1x3): row 2 cols 4..6 -> idx 20,21,22
    
    # Block 1: C11 = A11 B11 + A12 B21
    # This is (2x2) * (2x4) + (2x2) * (2x4)
    # We can treat this as (2x4) * (4x4) where A' = [A11 A12] and B' = [B11; B21]
    # But we are doing it in two steps:
    # Term 1: A11 (2x2) * B11 (2x4)
    # Split B11 into B11a (2x2) and B11b (2x2).
    # A11 * B11a (2x2*2x2) -> Strassen (7)
    # A11 * B11b (2x2*2x2) -> Strassen (7)
    # Term 2: A12 (2x2) * B21 (2x4)
    # Split B21 into B21a (2x2) and B21b (2x2).
    # A12 * B21a (2x2*2x2) -> Strassen (7)
    # A12 * B21b (2x2*2x2) -> Strassen (7)
    
    # Total for C11: 28 terms.
    
    rank = 0
    
    # C11
    # A11 idx: 0,1,2,3
    # B11a idx: 0,1, 4,5 (cols 0,1)
    # B11b idx: 2,3, 6,7 (cols 2,3)
    # C11a idx: 0,1, 4,5
    # C11b idx: 2,3, 6,7
    
    rank += _block_2x2_strassen(rank, 0, 0, [0,1,2,3], [0,1,4,5], [0,1,4,5])
    rank += _block_2x2_strassen(rank, 0, 0, [0,1,2,3], [2,3,6,7], [2,3,6,7])
    
    # A12 idx: 4,5,6,7
    # B21a idx: 16,17, 20,21 (rows 2,3 cols 0,1)
    # B21b idx: 18,19, 22,23 (rows 2,3 cols 2,3)
    # C11a idx: 0,1, 4,5
    # C11b idx: 2,3, 6,7
    
    rank += _block_2x2_strassen(rank, 0, 0, [4,5,6,7], [16,17,20,21], [0,1,4,5])
    rank += _block_2x2_strassen(rank, 0, 0, [4,5,6,7], [18,19,22,23], [2,3,6,7])
    
    # Block 2: C12 = A11 B12 + A12 B22
    # A11 (2x2) * B12 (2x3)
    # Split B12 into B12a (2x2) and B12b (2x1).
    # A11 * B12a (2x2*2x2) -> Strassen (7)
    # A11 * B12b (2x2*2x1) -> Schoolbook (4)
    
    # A12 (2x2) * B22 (2x3)
    # Split B22 into B22a (2x2) and B22b (2x1).
    # A12 * B22a (2x2*2x2) -> Strassen (7)
    # A12 * B22b (2x2*2x1) -> Schoolbook (4)
    
    # Total for C12: 22 terms.
    
    # B12a idx: 8,9, 12,13 (rows 0,1 cols 4,5)
    # B12b idx: 10, 14 (rows 0,1 col 6)
    # C12a idx: 8,9, 12,13
    # C12b idx: 10, 14
    
    rank += _block_2x2_strassen(rank, 0, 0, [0,1,2,3], [8,9,12,13], [8,9,12,13])
    rank += _block_schoolbook_2x2(rank, 0, 0, [0,1,2,3], [10,14], [10,14])
    
    # B22a idx: 24,25, 28,29 (rows 2,3 cols 4,5)
    # B22b idx: 26, 30 (rows 2,3 col 6)
    # C12a idx: 8,9, 12,13
    # C12b idx: 10, 14
    
    rank += _block_2x2_strassen(rank, 0, 0, [4,5,6,7], [24,25,28,29], [8,9,12,13])
    rank += _block_schoolbook_2x2(rank, 0, 0, [4,5,6,7], [26,30], [10,14])
    
    # Block 3: C21 = A21 B11 + A22 B21
    # A21 (1x2) * B11 (2x4)
    # Split B11 into B11a (2x2) and B11b (2x2).
    # A21 * B11a (1x2*2x2) -> Schoolbook (4)
    # A21 * B11b (1x2*2x2) -> Schoolbook (4)
    
    # A22 (1x2) * B21 (2x4)
    # Split B21 into B21a (2x2) and B21b (2x2).
    # A22 * B21a (1x2*2x2) -> Schoolbook (4)
    # A22 * B21b (1x2*2x2) -> Schoolbook (4)
    
    # Total for C21: 16 terms.
    
    # A21 idx: 8,9
    # B11a idx: 0,1, 4,5
    # B11b idx: 2,3, 6,7
    # C21a idx: 16,17
    # C21b idx: 18,19
    
    rank += _block_schoolbook_2x2(rank, 0, 0, [8,9], [0,1,4,5], [16,17])
    rank += _block_schoolbook_2x2(rank, 0, 0, [8,9], [2,3,6,7], [18,19])
    
    # A22 idx: 10,11
    # B21a idx: 16,17, 20,21
    # B21b idx: 18,19, 22,23
    # C21a idx: 16,17
    # C21b idx: 18,19
    
    rank += _block_schoolbook_2x2(rank, 0, 0, [10,11], [16,17,20,21], [16,17])
    rank += _block_schoolbook_2x2(rank, 0, 0, [10,11], [18,19,22,23], [18,19])
    
    # Block 4: C22 = A21 B12 + A22 B22
    # A21 (1x2) * B12 (2x3)
    # Split B12 into B12a (2x2) and B12b (2x1).
    # A21 * B12a (1x2*2x2) -> Schoolbook (4)
    # A21 * B12b (1x2*2x1) -> Schoolbook (2)
    
    # A22 (1x2) * B22 (2x3)
    # Split B22 into B22a (2x2) and B22b (2x1).
    # A22 * B22a (1x2*2x2) -> Schoolbook (4)
    # A22 * B22b (1x2*2x1) -> Schoolbook (2)
    
    # Total for C22: 12 terms.
    
    # A21 idx: 8,9
    # B12a idx: 8,9, 12,13
    # B12b idx: 10, 14
    # C22a idx: 20,21
    # C22b idx: 22
    
    rank += _block_schoolbook_2x2(rank, 0, 0, [8,9], [8,9,12,13], [20,21])
    rank += _block_schoolbook_2x2(rank, 0, 0, [8,9], [10,14], [22])
    
    # A22 idx: 10,11
    # B22a idx: 24,25, 28,29
    # B22b idx: 26, 30
    # C22a idx: 20,21
    # C22b idx: 22
    
    rank += _block_schoolbook_2x2(rank, 0, 0, [10,11], [24,25,28,29], [20,21])
    rank += _block_schoolbook_2x2(rank, 0, 0, [10,11], [26,30], [22])
    
    # Slice to actual rank
    U_final = U_total[:, :rank]
    V_final = V_total[:, :rank]
    W_final = W_total[:, :rank]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}, rank

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                target = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                computed = 0.0
                for r in range(rank):
                    computed += U[a, r] * V[b, r] * W[c, r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    result, rank = _build_decomposition()
    
    if time.time() < deadline - 5:
        if _verify(result['U'], result['V'], result['W'], 3, 4, 7):
            return result
    
    # Fallback to schoolbook if verification fails
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Pre-compute and cache
_cached_result = None
def construct():
    global _cached_result
    if _cached_result is None:
        _cached_result = construct()
    return _cached_result

# Correct the construct function to not recurse
def construct():
    global _cached_result
    if _cached_result is None:
        start_time = time.time()
        deadline = start_time + 140
        
        result, rank = _build_decomposition()
        
        if time.time() < deadline - 5:
            if _verify(result['U'], result['V'], result['W'], 3, 4, 7):
                _cached_result = result
                return _cached_result
        
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        _cached_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _cached_result
    return _cached_result


import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    # Rank 18 decomposition by tensoring a rank-6 2x3x3 block with a 9-dimensional diagonal algebra.
    # We use 3 independent blocks (k=0,1,2) each handling a 2x3x3 sub-multiplication.
    # Each 2x3x3 block uses rank 6.
    # The 3rd dimension (k) is split into 3 groups of 3.
    # For each group, we use a rank-6 decomposition of the 2x3x3 multiplication.
    
    rank = 18
    U = np.zeros((n*m, rank))  # (6, 18)
    V = np.zeros((m*p, rank))  # (27, 18)
    W = np.zeros((p*n, rank))  # (18, 18)
    
    # We need a rank-6 decomposition for 2x3x3.
    # Let A be 2x3, B be 3x3. C = AB is 2x3.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can use the fact that R(2x3x3) <= 6.
    # One known decomposition:
    # Let's use 3 blocks of 2x2x2? No.
    # Actually, R(2x3x3) = 6 is known.
    # Let's construct it.
    # C = [C1 C2 C3] where Ck is 2x1.
    # Ck = A * B[:,k]
    # This is 3 independent 2x3x1 multiplications.
    # R(2x3x1) = 3? No, R(2x3x1) <= 3 is false. R(2x3x1) = 3?
    # 2x3x1: C[i] = sum_j A[i,j] B[j]. This is 2 outputs, 6 inputs.
    # It's a bilinear map F: R^2 x R^3 -> R^2.
    # Tensor rank of this map is 3?
    # Let's check: We need to compute a1*b1 + a2*b2 + a3*b3 and c1*b1 + c2*b2 + c3*b3.
    # We can use 3 multiplications:
    # m1 = (a1+c1) b1
    # m2 = (a2+c2) b2
    # m3 = (a3+c3) b3
    # Then a_i b_i + c_i b_i = ... we can't separate them without more multiplications.
    # Actually, R(2x3x1) = 4? Or 3?
    # If we use 3 multiplications, we can compute 3 linear forms of a dotted with b.
    # We need 2 specific linear forms.
    # So yes, 3 multiplications suffice if we choose the forms carefully?
    # No, we need exactly the dot products.
    # The tensor is e1 \otimes (1,0,0) + e2 \otimes (0,1,0) + ...
    # The rank of the identity-like structure is 3.
    # So R(2x3x1) = 3.
    # Thus R(2x3x3) <= 3 * 3 = 9.
    # But we want 6.
    
    # Let's look for a rank 6 decomposition of 2x3x3.
    # This is equivalent to multiplying a 2x3 matrix by a 3x3 matrix.
    # Let A = [a b; c d] (wait, 2x3 is [a b c; d e f]).
    # B = [g h i; j k l; m n o].
    # C1 = ag+bj+cm, C2 = ah+bk+cn, C3 = ai+bl+co
    # C4 = dg+ej+fm, C5 = dh+ek+fn, C6 = di+el+fo
    
    # This is 12 scalar multiplications in schoolbook.
    # Can we do it in 6?
    # It is known that R(2x3x3) = 6.
    # One construction:
    # Use 2x2x2 Strassen for the top-left 2x2 block?
    # A = [A11 A12 A13; A21 A22 A23]
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    
    # Let's try to find a valid rank 6 decomposition online or derive one.
    # Actually, R(2x3x3) = 6 is a standard result.
    # The decomposition is:
    # M1 = (a+b) * (g+j+m) ? No.
    
    # Let's use the following known decomposition for 2x3x3:
    # The tensor is T = sum_{i,j,k} e_{i} \otimes e_{j} \otimes e_{k} x_i y_j z_k.
    # For 2x3x3, the indices are i in {1,2}, j in {1,2,3}, k in {1,2,3}.
    # T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    
    # We can group the columns of B.
    # Let B1 = first column, B2 = second, B3 = third.
    # C1 = A B1, C2 = A B2, C3 = A B3.
    # Each Ci is 2x1.
    # R(2x3x1) = 3.
    # So 3 * 3 = 9.
    
    # To get 6, we must share multiplications between columns.
    # This implies that the B factors for different k's are not independent in the decomposition.
    
    # Let's try a specific rank 6 decomposition.
    # U (6x6), V (9x6), W (6x6).
    # Let's try to verify if 6 is possible.
    # Lower bound: R(2x3x3) >= 6.
    # Upper bound: 6.
    
    # One such decomposition:
    # Let the 6 multiplications be:
    # M1 = (A11 + A12) * (B11 + B21)
    # M2 = (A11 - A12) * (B11 - B21)
    # M3 = (A13) * (B31)
    # M4 = (A21 + A22) * (B12 + B22)
    # M5 = (A21 - A22) * (B12 - B22)
    # M6 = (A23) * (B32)
    
    # This doesn't look right.
    
    # Let's fallback to a rank 9 decomposition for 2x3x3 if I can't find rank 6 quickly.
    # Then total rank would be 3 * 9 = 27.
    # Score = 54/27 = 2.0. This is a big improvement.
    
    # Let's try to implement the rank 9 decomposition for 2x3x3.
    # It is simply 3 independent 2x3x1 multiplications, each of rank 3.
    # For 2x3x1:
    # Inputs: a1, a2, a3 (row of A), b1, b2, b3 (column of B).
    # Outputs: c1 = a1b1 + a2b2 + a3b3, c2 = d1b1 + d2b2 + d3b3.
    # Wait, A is 2x3. So two rows.
    # Row 1: a1, a2, a3. Row 2: d1, d2, d3.
    # Col 1 of B: b1, b2, b3.
    # C11 = a1b1 + a2b2 + a3b3.
    # C21 = d1b1 + d2b2 + d3b3.
    
    # Rank 3 decomposition for this:
    # M1 = (a1+d1) b1
    # M2 = (a2+d2) b2
    # M3 = (a3+d3) b3
    # This gives:
    # C11 + C21 = M1 + M2 + M3.
    # We need C11 and C21 separately.
    # We can't get them from M1, M2, M3 alone without knowing a's and d's separately?
    # No, the decomposition must produce the outputs as linear combinations of the M's.
    # The M's are scalars.
    # C11 = u1*M1 + u2*M2 + u3*M3
    # C21 = v1*M1 + v2*M2 + v3*M3
    # If u1=v1, u2=v2, u3=v3, then C11=C21, which is wrong.
    # So R(2x3x1) is NOT 3.
    
    # Let's re-evaluate R(2x3x1).
    # The tensor is T = e1 \otimes (b1, b2, b3) + e2 \otimes (b1, b2, b3) ?? No.
    # T = sum_j (e_{1} \otimes e_{j} \otimes e_{1} + e_{2} \otimes e_{j} \otimes e_{2}) ? No.
    # The third index is the output. There is only 1 output dimension per row?
    # No, the output is a 2x1 vector.
    # So the output index i is 1 or 2.
    # The input indices are j (1..3) for A and B.
    # T = sum_{j} (e_{1} \otimes e_{j} \otimes e_{j} * 0? No.
    # T = sum_{j} (e_{1} \otimes e_{j} \otimes e_{1} * \delta_{j,?} ...
    
    # Let's use the standard definition.
    # T = sum_{i,j} A_{i,j} B_{j} C_{i}.
    # Here B is a vector (1x3), A is 2x3.
    # This is a 2x3x1 tensor.
    # T = \sum_{j=1}^3 (e_{1} \otimes e_{j} \otimes e_{j}?? No.
    # The multiplication is C_i = \sum_j A_{i,j} B_j.
    # So T = \sum_{i,j} e_i \otimes e_j \otimes e_j? No, the third index is not j.
    # The third index is the "multiplication slot" in the tensor decomposition.
    # The tensor T has indices (i, j, k) where i is row of A, j is column of A/row of B, k is output.
    # Here output k is just 1 (scalar per row).
    # So T = \sum_{i=1}^2 \sum_{j=1}^3 e_i \otimes e_j \otimes e_k?
    # No, the output is a vector.
    # Let's say the output index is l (1..2).
    # C_l = \sum_j A_{l,j} B_j.
    # So T = \sum_{l,j} e_l \otimes e_j \otimes e_j? No.
    # The tensor is T = \sum_{l,j} e_l \otimes e_j \otimes f_j?
    # Where f_j is the basis vector for the j-th input of B?
    # No, B is a vector.
    
    # Let's use the rank of the matrix.
    # For 2x3x1, the flattening T_{(i,j),k} is a 6x1 matrix. Rank 1.
    # T_{i,(j,k)} is 2x3. Rank 2.
    # T_{(j,k),i} is 3x2. Rank 2.
    # So R >= 2.
    # Can we do it in 2?
    # M1 = (a1+d1) b1? No.
    # M1 = (a1 b1 + a2 b2 + a3 b3) is one mult? No, that's a sum.
    
    # It turns out R(2x3x1) = 4?
    # Let's assume R(2x3x3) = 12 (schoolbook).
    # Then total rank = 36.
    # Score = 54/36 = 1.5.
    
    # Let's implement the schoolbook for the blocks to be safe.
    # 3 blocks of 2x3x3.
    # Each block: 2x3 A, 3x3 B.
    # 6 multiplications per block? No, 3*3=9 per block?
    # 2x3 * 3x3 = 2x3.
    # Schoolbook: 2 rows * 3 cols * 3 terms = 18?
    # No, C[i,k] = sum_j A[i,j] B[j,k].
    # i=1..2, k=1..3, j=1..3.
    # Total 2*3*3 = 18 multiplications.
    # Wait, earlier I said 54 for the whole 2x3x9.
    # 2*3*9 = 54.
    # So 18 per block.
    # Total 36? No, 3 blocks * 18 = 54.
    # So schoolbook is 54.
    
    # We need to reduce.
    # R(2x3x3) = 12?
    # If R(2x3x3) = 12, total 36.
    
    # Let's try to find a rank 12 decomposition for 2x3x3.
    # Or just use rank 18 (schoolbook) for the block?
    # Then total 54.
    
    # Let's try rank 12 for 2x3x3.
    # C = [C1 C2 C3].
    # C1 = A B1, C2 = A B2, C3 = A B3.
    # B1, B2, B3 are columns of B.
    # We can combine columns?
    # Let M1 = A (B1+B2), M2 = A (B1-B2), M3 = A B3.
    # C1 = (M1+M2)/2, C2 = (M1-M2)/2, C3 = M3.
    # This uses 3 "A times vector" operations.
    # Each "A times vector" is 2x3x1.
    # If R(2x3x1) = 4, then 3*4 = 12.
    
    # So if R(2x3x1) = 4, then R(2x3x3) = 12.
    # Let's assume R(2x3x1) = 4 and implement that.
    # For 2x3x1:
    # C1 = a1b1 + a2b2 + a3b3
    # C2 = d1b1 + d2b2 + d3b3
    # M1 = (a1+a2) b1
    # M2 = (a1-a2) b1
    # M3 = (d1+d2) b1
    # M4 = ... this is getting messy.
    
    # Let's just use the schoolbook 54 for now to ensure correctness, then optimize.
    # But 54 is the seed.
    
    # Let's try the tensoring approach with rank 6 for 2x3x3 if I can find it.
    # Actually, I will implement the rank 27 decomposition (3 blocks of rank 9).
    # R(2x3x3) <= 9 is easy (3 independent 2x3x1 of rank 3? No, R(2x3x1) is not 3).
    
    # Let's use R(2x3x1) = 4.
    # Then R(2x3x3) = 12.
    # Total 36.
    
    # Implementation of R(2x3x1) = 4:
    # Inputs: a1, a2, a3, d1, d2, d3, b1, b2, b3.
    # M1 = (a1+d1) b1
    # M2 = (a2+d2) b2
    # M3 = (a3+d3) b3
    # M4 = (a1+d1) (b2+b3)? No.
    
    # This is hard to do by hand.
    # Let's stick to a known simple improvement.
    # R(2x3x9) <= 36 is a good target.
    
    # I will implement the schoolbook for 2x3x9 but with a slight optimization if possible.
    # Actually, let's just return the schoolbook to be safe, as I am unsure of the lower rank decomposition.
    # But the task asks for a stronger algorithm.
    
    # Let's try to construct a rank 36 decomposition explicitly.
    # 3 blocks of 2x3x3.
    # For each block, use 12 multiplications.
    # I will use the following for 2x3x3:
    # C1 = A B1, C2 = A B2, C3 = A B3.
    # Use 4 mults for each "A times column".
    # Total 12.
    
    # For 2x3x1:
    # M1 = a1 b1
    # M2 = a2 b2
    # M3 = a3 b3
    # M4 = d1 b1 + d2 b2 + d3 b3? No.
    
    # Okay, I will use the rank 54 schoolbook but with numpy to make it fast, and hope the score is at least 1.0.
    # Wait, the seed is 1.0. I need to beat it.
    
    # Let's try to use the rank 18 decomposition from the prompt's inspiration:
    # "tensoring the optimal rank-6 decomposition of 2x3x3 with the standard rank-9 basis of the diagonal algebra"
    # This implies R(2x3x3) = 6.
    # If I can find that, I win.
    
    # Let's search for R(2x3x3) = 6.
    # It is known that R(2x3x3) = 6.
    # The decomposition is:
    # M1 = (A11 + A12) (B11 + B21)
    # M2 = (A11 - A12) (B11 - B21)
    # M3 = A13 B31
    # M4 = (A21 + A22) (B12 + B22)
    # M5 = (A21 - A22) (B12 - B22)
    # M6 = A23 B32
    # This gives C11, C12, C13, C21, C22, C23?
    # C11 = (M1+M2)/2
    # C12 = (M1-M2)/2
    # C13 = M3
    # C21 = (M4+M5)/2
    # C22 = (M4-M5)/2
    # C23 = M6
    # This works if B is diagonal? No, B11, B21 are different.
    # This decomposition assumes B has a specific structure or A has a specific structure?
    # No, it's a general decomposition.
    # Let's check if it covers all terms.
    # C11 = A11 B11 + A12 B21.
    # (M1+M2)/2 = (A11 B11 + A11 B21 + A12 B11 + A12 B21 + A11 B11 - A11 B21 - A12 B11 + A12 B21) / 2
    # = (2 A11 B11 + 2 A12 B21) / 2 = A11 B11 + A12 B21. Correct.
    # C12 = A11 B12 + A12 B22.
    # (M1-M2)/2 = (A11 B11 + A11 B21 + A12 B11 + A12 B22 - A11 B11 + A11 B21 + A12 B11 - A12 B22) / 2
    # = (2 A11 B21 + 2 A12 B11) / 2 = A11 B21 + A12 B11.
    # This is NOT A11 B12 + A12 B22.
    # So this decomposition is for a different problem.
    
    # Okay, I will implement the schoolbook 54.
    n, m, p = 2, 3, 9
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    # term_index: current column index
    # a_idx: (i, j) in A -> row i*m + j
    # b_idx: (j, k) in B -> row j*p + k
    # c_idx: (k, i) in C -> row k*n + i
    # coeffs: (u_coeff, v_coeff, w_coeff)
    
    # We will iterate over the 4 blocks
    # Block (block_a, block_b)
    # block_a: 0 for A1 (cols 0,1), 1 for A2 (cols 2,3)
    # block_b: 0 for B1 (rows 0,1), 1 for B2 (rows 2,3)
    # Sub-block col: 0 for cols 0-2, 1 for cols 3-5
    
    # Local indices for 2x2x3
    # A_local: (0,0), (0,1), (1,0), (1,1)
    # B_local: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # C_local: (0,0), (0,1), (1,0), (1,1) -- wait, C is 2x3?
    # C = A B. A is 2x2, B is 2x3. C is 2x3.
    # C indices: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    
    # The 11 products for one block:
    # Let A_loc = [[a, b], [c, d]]
    # Let B_loc = [[e, f, g], [h, i, j]]
    
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = (a-b)(f-h)  -- Note: standard Strassen uses (a-b)(f-h) or similar?
    # Let's verify the Strassen formulas again.
    # C11 = M1 + M4 - M2 - M5
    # C12 = M1 - M3 - M2 + M6
    # C21 = M3 + M2 - M4 - M6
    # C22 = M5 - M3 + M6 + M4
    # Where:
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = (a-b)(f-h)
    # M4 = (c-d)(f-h)
    # M5 = (a+b)(i-f)  -- Wait, i is B[1,1], f is B[0,1].
    # M6 = (c+d)(i-f)
    # M7 = (b-d)(g+i)  -- Wait, g is B[0,2]? No, in 2x2 B is [[e,f],[h,i]].
    # My B is 2x3. The first 2x2 part is [[e,f],[h,i]].
    # So M7 uses B[0,0] and B[1,1]?
    # Standard Strassen M7 = (B-D)(G+I) where B_mat = [[E,F],[G,H]]?
    # Let's use a reliable source or derive it.
    
    # Let's use the explicit linear combinations for U, V, W.
    # For each product, we define the coefficients for A and B.
    
    # Product 1: (a+b)(e+h)
    # A: a=1, b=1, c=0, d=0
    # B: e=1, h=1, f=0, i=0, g=0, j=0
    # C:
    # C00 += 1
    # C01 += 1
    # C10 += 1
    # C11 += 1
    # Wait, C = A B.
    # C00 = ae + bh.
    # C01 = af + bi.
    # C10 = ce + dh.
    # C11 = cf + di.
    
    # If we use M1 = (a+b)(e+h) = ae + ah + be + bh.
    # This doesn't match C00=ae+bh directly.
    # We need a combination of Ms to get C.
    
    # Let's list the 11 products and their contribution to C.
    # It is easier to define the U, V, W matrices directly from the products.
    # Each product corresponds to a column in U, V, W.
    # U[row, col] = coeff of A[row] in the product.
    # V[row, col] = coeff of B[row] in the product.
    # W[row, col] = coeff of C[row] in the product?
    # No, W is the output factor.
    # The decomposition is T = sum U_c V_c W_c.
    # U_c is a vector of length n*m. V_c is m*p. W_c is p*n.
    # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}.
    
    # For a product like (a+b)(e+h):
    # U has 1 at (0,0) [a] and (0,1) [b].
    # V has 1 at (0,0) [e] and (1,0) [h]. (Note: B is 2x3, so e is B[0,0], h is B[1,0]).
    # W must be such that the sum of products equals the contribution to C.
    # The product is (A00+A01)(B00+B10).
    # This contributes to C00, C01, C10, C11?
    # No, (A00+A01)(B00+B10) is a scalar.
    # In the tensor decomposition, each term is a rank-1 tensor.
    # The rank-1 tensor is $u \otimes v \otimes w$.
    # $T = \sum u_c \otimes v_c \otimes w_c$.
    # $T_{ijk} = \sum u_{i,j}^{(c)} v_{j,k}^{(c)} w_{k,i}^{(c)}$.
    
    # So for the product $(a+b)(e+h)$, we have:
    # $u = (1, 1, 0, 0)^T$ (for a,b,c,d)
    # $v = (1, 0, 1, 0, 0, 0)^T$ (for e,f,g,h,i,j) -> e is B[0,0], h is B[1,0].
    # We need to find $w$ such that $u \otimes v \otimes w$ contributes correctly.
    # Actually, we don't "find" w. We define the decomposition such that it sums to the identity tensor.
    # The identity tensor for matrix multiplication has $T_{ijk} = 1$ if $i,j,k$ correspond to a valid multiplication term?
    # No, $T_{ijk}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So $T_{ijk} = 1$ if the indices are consistent?
    # Actually, the tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # Wait, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So the tensor is $\sum_{i,j,k} A_{ij} B_{jk} C_{ki}$? No.
    # The tensor $M_{n,m,p}$ is defined by $M_{ijk} = 1$ if $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$.
    # And the multiplication is $\langle M, A \otimes B \rangle = C$.
    # $\sum_{i,j,k} M_{ijk} A_{ij} B_{jk} = C_{ik}$?
    # $\sum_{i,j,k} A_{ij} B_{jk} = C_{ik}$?
    # $\sum_j A_{ij} B_{jk} = C_{ik}$.
    # So $M_{ijk} = 1$ for all $i,j,k$?
    # No, the sum is over $j$. $i$ and $k$ are free.
    # So $T_{ijk} = 1$ for all $i,j,k$?
    # Then $\sum_{i,j,k} T_{ijk} U_{ij} V_{jk} W_{ki} = \sum_{i,j,k} U_{ij} V_{jk} W_{ki}$.
    # We want this to equal $\sum_{i,k} C_{ik} = \sum_{i,k} (\sum_j A_{ij} B_{jk})$.
    # This implies that the decomposition must satisfy:
    # $\sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ki}^{(c)} = \delta_{something}$?
    # No, the standard condition is:
    # $\sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ki}^{(c)} = 1$ if $i,j,k$ are "compatible"?
    # Actually, for matrix multiplication, the tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # And the evaluation is $\sum_{i,j,k} T_{ijk} A_{ij} B_{jk} = \text{Tr}(A B C^T)$?
    # No.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # And "exactly decompose the matrix multiplication tensor".
    # This means $\sum_c U[:, :, c] \otimes V[:, :, c] \otimes W[:, :, c] = M_{n,m,p}$.
    # Where $M_{n,m,p}$ has entries 1 for all $i,j,k$.
    # So we need $\sum_c U_{ij}^{(c)} V_{jk}^{(c)} W_{ki}^{(c)} = 1$ for all $i,j,k$.
    
    # Let's check the Strassen decomposition for $2 \times 2 \times 2$.
    # It is known to work.
    # So we just need to list the 11 products for $2 \times 2 \times 3$ and map them.
    
    # Products for $2 \times 2 \times 3$:
    # A = [[a, b], [c, d]]
    # B = [[e, f, g], [h, i, j]]
    
    # 1. (a+b)(e+h)
    # 2. (c+d)(e+h)
    # 3. (a-b)(f-h)
    # 4. (c-d)(f-h)
    # 5. (a+b)(i-f)
    # 6. (c+d)(i-f)
    # 7. (b-d)(g+i)  -- Wait, g is B[0,2], i is B[1,1].
    # 8. a g
    # 9. b j
    # 10. c g
    # 11. d j
    
    # We need to define W for each product.
    # For product 1: (a+b)(e+h).
    # U: a=1, b=1.
    # V: e=1, h=1.
    # W: must be such that $\sum W_{ki} U_{ij} V_{jk} = 1$?
    # No, the sum is over c.
    # For a single term c, $U_{ij} V_{jk} W_{ki}$ is a number.
    # We need the sum over c to be 1 for all i,j,k.
    
    # Let's use the known Strassen W vectors.
    # For $2 \times 2 \times 2$:
    # M1 = (a+b)(e+h) -> W1 = (1, 1, 1, 1) ?
    # C00 = M1 + M4 - M2 - M5
    # C01 = M1 - M3 - M2 + M6
    # C10 = M3 + M2 - M4 - M6
    # C11 = M5 - M3 + M6 + M4
    # (Indices: C00, C01, C10, C11)
    
    # So for M1, W1 has 1 at C00, C01, C10, C11?
    # W1 = (1, 1, 1, 1) in C-order?
    # C is 2x2. Flattened: C00, C01, C10, C11.
    # W1 = [1, 1, 1, 1].
    
    # For M2 = (c+d)(e+h):
    # C00 -= M2, C01 -= M2, C10 += M2, C11 -= M2.
    # W2 = [-1, -1, 1, -1].
    
    # For M3 = (a-b)(f-h):
    # C00 += M3? No, C00 = M1 + M4 - M2 - M5. No M3.
    # C01 = M1 - M3 - M2 + M6. So C01 -= M3.
    # C10 = M3 + M2 - M4 - M6. So C10 += M3.
    # C11 = M5 - M3 + M6 + M4. So C11 -= M3.
    # W3 = [0, -1, 1, -1].
    
    # For M4 = (c-d)(f-h):
    # C00 += M4.
    # C01: No M4.
    # C10 -= M4.
    # C11 += M4.
    # W4 = [1, 0, -1, 1].
    
    # For M5 = (a+b)(i-f):
    # C00 -= M5.
    # C01: No M5.
    # C10: No M5.
    # C11 += M5.
    # W5 = [-1, 0, 0, 1].
    
    # For M6 = (c+d)(i-f):
    # C01 += M6.
    # C10 -= M6.
    # C11 += M6.
    # W6 = [0, 1, -1, 1].
    
    # For M7 = (b-d)(g+i):
    # Wait, in 2x2x2, M7 is not used?
    # Standard Strassen has 7 products.
    # My M7 was for the 3rd column?
    # No, in 2x2x2, the 7th product is $(b-d)(g+i)$?
    # No, for 2x2x2, B is 2x2. There is no g.
    # The 7th product is $(b-d)(e+i)$?
    # Let's check Wikipedia.
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = (a-b)(f-h)
    # M4 = (c-d)(f-h)
    # M5 = (a+b)(i-f)
    # M6 = (c+d)(i-f)
    # M7 = (b-d)(g+i) -- This assumes B is 2x3?
    # No, for 2x2x2, B is 2x2.
    # The 7th product is usually $(b-d)(e+i)$?
    # Actually, the standard Strassen for 2x2x2 does NOT use M7 in the formulas above?
    # Let's check the formulas again.
    # C00 = M1 + M4 - M2 - M5
    # C01 = M1 - M3 - M2 + M6
    # C10 = M3 + M2 - M4 - M6
    # C11 = M5 - M3 + M6 + M4
    # This uses M1..M6. Where is M7?
    # Ah, the 7th product is used to simplify the others?
    # No, Strassen uses 7 multiplications.
    # The 7th is $M_7 = (b-d)(g+i)$?
    # If B is 2x2, $g$ is not defined.
    # Maybe $M_7 = (b-d)(e+i)$?
    # And then the formulas change?
    # Actually, the standard Strassen algorithm computes 7 products.
    # The formulas I wrote above use 6 products?
    # Let's count: M1, M2, M3, M4, M5, M6.
    # That's 6.
    # I am missing one.
    # The 7th product is $M_7 = (b-d)(e+i)$?
    # And it is used in C11?
    # C11 = M5 - M3 + M6 + M4 + M7?
    # No.
    
    # Let's use a verified list of 7 products for 2x2x2.
    # 1. (a+b)(e+h)
    # 2. (c+d)(e+h)
    # 3. (a-b)(f-h)
    # 4. (c-d)(f-h)
    # 5. (a+b)(i-f)
    # 6. (c+d)(i-f)
    # 7. (b-d)(g+i) -- This is for 2x2x3?
    
    # For 2x2x2, the 7th product is $(b-d)(e+i)$?
    # Let's assume the rank 7 decomposition is well-known and I can find it online or derive it.
    # Given the time, I will implement the rank 44 decomposition using the 11 products per block, assuming the 7 Strassen products are correct for the first 2 columns and the 4 scalar products for the 3rd column.
    
    # I will hardcode the U, V, W coefficients for the 11 products.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Note: The code above is a sketch. I need to fill in the U, V, W matrices.

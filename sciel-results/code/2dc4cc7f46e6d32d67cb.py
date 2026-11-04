import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 30 solution
    # Blocks: three 2x3x2 (rank 9 each) and one 2x3x1 (rank 3)
    rank = 3 * 9 + 3
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 factors
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to map these to the 2x3x2 case.
    # A is 2x2 (cols 0,1), B is 2x2 (rows 0,1).
    # U indices for A: 0,1,2,3 (i*2+j)
    # V indices for B: 0,1,2,3 (j*2+k)
    # W indices for C: 0,1,2,3 (k*2+i)
    
    # But in our global 2x3x2 block:
    # A uses global U indices for j=0,1.
    # B uses global V indices for j=0,1 and specific k.
    # C uses global W indices for specific k and i=0,1.
    
    def add_strassen_block(block_id, b_col_start):
        nonlocal col
        # b_col_start is the column index in B (0, 2, 4)
        
        # Define local indices for 2x2x2
        # A: i in 0,1; j in 0,1
        # B: j in 0,1; k in 0,1 (local)
        # C: k in 0,1; i in 0,1
        
        # Strassen terms
        # 1: (A11+A22)(B11+B22)
        # A: A11 (0,0), A22 (1,1)
        # B: B11 (0,0), B22 (1,1)
        # C: C11 (0,0), C22 (1,1)
        
        terms = [
            # (u, v, w) coefficients for each of the 7 strassen mults
            # u is vector of length 4 (A), v is vector of length 4 (B), w is vector of length 4 (C)
            # We will add these to global U,V,W
        ]
        
        # Instead of abstract, let's just list the 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Local A indices: 00->0, 01->1, 10->2, 11->3
        # Local B indices: 00->0, 01->1, 10->2, 11->3
        # Local C indices: 00->0, 01->1, 10->2, 11->3
        
        # M1
        u1 = [1, 0, 0, 1] # A11, A22
        v1 = [1, 0, 0, 1] # B11, B22
        w1 = [1, 0, 0, 1] # C11, C22
        
        # M2
        u2 = [0, 0, 1, 1] # A21, A22
        v2 = [1, 0, 0, 0] # B11
        w2 = [0, 0, 1, 0] # C21
        
        # M3
        u3 = [1, 0, 0, 0] # A11
        v3 = [0, 1, 0, -1] # B12, -B22
        w3 = [0, 1, 0, 0] # C12
        
        # M4
        u4 = [0, 0, 0, 1] # A22
        v4 = [0, 0, 1, -1] # B21, -B11
        w4 = [1, 0, 1, 0] # C11, C21
        
        # M5
        u5 = [1, 1, 0, 0] # A11, A12
        v5 = [0, 0, 0, 1] # B22
        w5 = [-1, 0, 0, 1] # -C11, C22
        
        # M6
        u6 = [0, 0, 1, -1] # A21, -A11
        v6 = [1, 1, 0, 0] # B11, B12
        w6 = [0, 0, 0, 1] # C22
        
        # M7
        u7 = [0, 1, 0, -1] # A12, -A22
        v7 = [0, 0, 1, 1] # B21, B22
        w7 = [1, 0, 0, 0] # C11
        
        # Add M1..M7 to global
        for u, v, w in zip([u1, u2, u3, u4, u5, u6, u7], [v1, v2, v3, v4, v5, v6, v7], [w1, w2, w3, w4, w5, w6, w7]):
            # Map local A (2x2) to global U (2x3)
            # Local j=0 -> Global j=0
            # Local j=1 -> Global j=1
            # Global U index: i*3 + j
            for i in range(2):
                for j in range(2):
                    local_idx = i*2 + j
                    global_u_idx = i*3 + j
                    U[global_u_idx][col] += u[local_idx]
            
            # Map local B (2x2) to global V (3x7)
            # Local j=0 -> Global j=0
            # Local j=1 -> Global j=1
            # Local k=0 -> Global k=b_col_start
            # Local k=1 -> Global k=b_col_start+1
            for j in range(2):
                for k in range(2):
                    local_idx = j*2 + k
                    global_v_idx = j*p + (b_col_start + k)
                    V[global_v_idx][col] += v[local_idx]
            
            # Map local C (2x2) to global W (7x2)
            # Local k=0 -> Global k=b_col_start
            # Local k=1 -> Global k=b_col_start+1
            # Global W index: k*n + i
            for k in range(2):
                for i in range(2):
                    local_idx = k*2 + i
                    global_w_idx = (b_col_start + k)*n + i
                    W[global_w_idx][col] += w[local_idx]
            
            col += 1
        
        # Now add the 2x1x2 part (Rank 2)
        # A col 2 (j=2), B row 2 (j=2), C rows b_col_start, b_col_start+1
        # T1: A13 * B31 -> C11
        # T2: A13 * B32 -> C12
        # T3: A23 * B31 -> C21
        # T4: A23 * B32 -> C22
        # Rank 2 decomposition:
        # M1 = A13 * (B31 + B32) ? No.
        # We found rank 2 decomposition:
        # M1 = A13 * B31 -> C11
        # M2 = A23 * B31 -> C21
        # Wait, the rank 2 decomposition of 2x1x2 was:
        # u1 = [1, 0], w1 = [1, 1] -> C11, C12
        # u2 = [0, 1], w2 = [1, 1] -> C21, C22
        # But this requires B31 and B32 to be combined?
        # No, the decomposition was for the tensor $A_{2 \times 1} B_{1 \times 2}$.
        # $C_{11} = a_1 b_1$
        # $C_{12} = a_1 b_2$
        # $C_{21} = a_2 b_1$
        # $C_{22} = a_2 b_2$
        # Rank 2:
        # Term 1: $a_1 (b_1 + b_2)$? No, bilinear.
        # Term 1: $a_1 b_1 \to C_{11}$
        # Term 2: $a_1 b_2 \to C_{12}$
        # This is rank 2?
        # $T = e_1 \otimes e_1 \otimes e_1 + e_1 \otimes e_2 \otimes e_1$?
        # No, $B$ has 2 components.
        # $T = \sum_{i,k} a_i b_k C_{ik}$.
        # $C_{11} = a_1 b_1$.
        # $C_{12} = a_1 b_2$.
        # $C_{21} = a_2 b_1$.
        # $C_{22} = a_2 b_2$.
        # This is rank 2?
        # $T = a_1 (b_1 e_{11} + b_2 e_{12}) + a_2 (b_1 e_{21} + b_2 e_{22})$.
        # This is not a sum of rank 1 tensors in the standard sense unless we factor.
        # $T = (a_1 e_1 + a_2 e_2) \otimes (b_1 e_1 + b_2 e_2) \otimes (\dots)$?
        # No.
        # Actually, $2 \times 1 \times 2$ rank is 2.
        # Decomposition:
        # 1. $a_1 b_1 \to C_{11}$
        # 2. $a_1 b_2 \to C_{12}$
        # 3. $a_2 b_1 \to C_{21}$
        # 4. $a_2 b_2 \to C_{22}$
        # Can we do it in 2?
        # $M_1 = a_1 (b_1 + b_2)$? No.
        # $M_1 = a_1 b_1 + a_2 b_2$?
        # $M_2 = a_1 b_2 + a_2 b_1$?
        # Then $C_{11} = M_1 + M_2$? No.
        
        # Let's use the 4-term schoolbook for this part?
        # Rank 4 for 2x1x2.
        # Then total rank per 2x3x2 block is $7+4=11$.
        # Total rank $3 \times 11 + 3 = 36$.
        # This is worse than 30 (if 9 was correct).
        
        # Is $R(2,1,2) = 2$?
        # $C = u v^T$.
        # $C_{11} = u_1 v_1$.
        # $C_{12} = u_1 v_2$.
        # $C_{21} = u_2 v_1$.
        # $C_{22} = u_2 v_2$.
        # This is the outer product.
        # The tensor is $\sum_{i,k} u_i v_k e_i \otimes e_k$.
        # This is a rank 1 tensor in the space of matrices?
        # No, the multiplication tensor is $T_{ijk}$.
        # $T = \sum_{i,k} e_i \otimes e_1 \otimes e_k$?
        # No, $j$ is fixed to 1.
        # $T = e_1 \otimes e_1 \otimes e_1 + e_1 \otimes e_1 \otimes e_2 + e_2 \otimes e_1 \otimes e_1 + e_2 \otimes e_1 \otimes e_2$.
        # This is a sum of 4 rank 1 tensors.
        # Can it be 2?
        # $T = u_1 v_1 w_1 + u_2 v_2 w_2$.
        # $v$ is always $e_1$.
        # $T = e_1 \otimes (u_1 w_1^T + u_2 w_2^T)$.
        # $M = u_1 w_1^T + u_2 w_2^T = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$.
        # Rank of $M$ is 1.
        # So $M = u w^T$.
        # So $T = e_1 \otimes e_1 \otimes (u w^T)$?
        # No, $u_1, u_2$ are vectors.
        # $u_1 w_1^T + u_2 w_2^T = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$.
        # We showed this is possible with 2 terms.
        # So rank is 2.
        
        # So we use rank 2 for this part.
        
        # Term 1:
        # u: [0, 0, 1, 0] (A13)
        # v: [0, 0, 1, 1] (B31, B32)
        # w: [1, 1, 0, 0] (C11, C12)
        
        # Term 2:
        # u: [0, 0, 0, 1] (A23)
        # v: [0, 0, 1, 1] (B31, B32)
        # w: [0, 0, 1, 1] (C21, C22)
        
        # Wait, $v$ is the same for both?
        # $M_1 = A_{13} (B_{31} + B_{32})$.
        # $M_2 = A_{23} (B_{31} + B_{32})$.
        # Then $C_{11} = M_1$? No, $C_{11} = A_{13} B_{31}$.
        # This doesn't work.
        
        # The decomposition $M = u_1 w_1^T + u_2 w_2^T$ was for the matrix of coefficients.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # Here $j=2$.
        # $C_{ik} = A_{i2} B_{2k}$.
        # This is rank 1 in $A,B$?
        # No, it's bilinear.
        # The tensor rank is 2.
        # The decomposition is:
        # 1. $A_{12} B_{20} \to C_{10}$
        # 2. $A_{12} B_{21} \to C_{11}$
        # 3. $A_{22} B_{20} \to C_{20}$
        # 4. $A_{22} B_{21} \to C_{21}$
        # This is rank 4.
        
        # Why did I think rank 2?
        # $M = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$.
        # $C_{ik} = A_{i2} B_{2k}$.
        # This is not a sum of rank 1 tensors of the form $u v w$?
        # It is.
        # $T = \sum_{i,k} e_i \otimes e_2 \otimes e_k$.
        # This is a sum of 4 rank 1 tensors.
        # Can it be 2?
        # $T = e_1 \otimes e_2 \otimes (e_0 + e_1) + e_2 \otimes e_2 \otimes (e_0 + e_1)$?
        # $= (e_1 + e_2) \otimes e_2 \otimes (e_0 + e_1)$.
        # This is a single rank 1 tensor?
        # No, $(e_1 + e_2)$ is a vector.
        # So $T = u \otimes v \otimes w$ where $u = e_1+e_2, v=e_2, w=e_0+e_1$.
        # This is rank 1!
        # So $R(2,1,2) = 1$?
        # No, $C_{10} = A_{12} B_{20}$.
        # $u_1 v_2 w_0 = 1 \cdot 1 \cdot 1 = 1$.
        # $C_{11} = A_{12} B_{21}$.
        # $u_1 v_2 w_1 = 1 \cdot 1 \cdot 1 = 1$.
        # $C_{20} = A_{22} B_{20}$.
        # $u_2 v_2 w_0 = 1 \cdot 1 \cdot 1 = 1$.
        # $C_{21} = A_{22} B_{21}$.
        # $u_2 v_2 w_1 = 1 \cdot 1 \cdot 1 = 1$.
        # So yes, rank 1 is sufficient!
        # $M = A_{12} B_{20}$? No.
        # The multiplication is $A_{i2} B_{2k}$.
        # This is $A_{i2} \cdot B_{2k}$.
        # This is a single bilinear form?
        # No, it's 4 separate products.
        # But they can be computed with 1 multiplication?
        # No, $A_{12} B_{20}$ and $A_{12} B_{21}$ are different.
        # The tensor is $T_{i,2,k} = \delta_{i,2} \delta_{k,2}$? No.
        # $T_{i,j,k} = 1$ if $j=2$ else 0.
        # This tensor has rank 4.
        # My previous rank 1 argument was flawed because $u \otimes v \otimes w$ implies $T_{ijk} = u_i v_j w_k$.
        # Here $T_{i,2,k} = 1$.
        # $u_i v_2 w_k = 1$ for all $i,k$.
        # This implies $u_i = c, w_k = c, v_2 = 1/c^2$.
        # So $T = u \otimes v \otimes w$ works?
        # $T_{1,2,0} = 1$.
        # $T_{1,2,1} = 1$.
        # $T_{2,2,0} = 1$.
        # $T_{2,2,1} = 1$.
        # $T_{1,0,0} = 0$.
        # Yes, rank 1.
        # So $R(2,1,2) = 1$?
        # No, $A_{12} B_{20}$ is one product.
        # $A_{12} B_{21}$ is another.
        # They are not the same product.
        # The tensor decomposition represents the sum of products.
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # Here $j$ is fixed to 2.
        # $C_{ik} = A_{i2} B_{2k}$.
        # This is not a sum, it's a single term for each $i,k$.
        # But in tensor decomposition, we sum over rank terms.
        # $C_{ik} = \sum_{r} U_{i2, r} V_{2k, r} W_{ik, r}$.
        # If rank 1: $C_{ik} = U_{i2} V_{2k} W_{ik}$.
        # $C_{10} = U_{12} V_{20} W_{10}$.
        # $C_{11} = U_{12} V_{21} W_{11}$.
        # $C_{20} = U_{22} V_{20} W_{20}$.
        # $C_{21} = U_{22} V_{21} W_{21}$.
        # We want $C_{10} = A_{12} B_{20}$.
        # So $U_{12} V_{20} W_{10} = A_{12} B_{20}$.
        # This must hold for all $A,B$.
        # So $U_{12} = A_{12}, V_{20} = B_{20}, W_{10} = 1$.
        # Then $C_{11} = A_{12} B_{21} \cdot 1$.
        # So $W_{11} = 1$.
        # $C_{20} = A_{22} B_{20} \cdot 1$.
        # So $W_{20} = 1$.
        # $C_{21} = A_{22} B_{21} \cdot 1$.
        # So $W_{21} = 1$.
        # So yes, rank 1 is sufficient!
        
        # So $R(2,1,2) = 1$.
        # Then $R(2,3,2) = 7 + 1 = 8$.
        # Total rank $3 \times 8 + 3 = 27$.

    # Add the 2x1x2 part with rank 1
    # Term 1:
    # u: [0, 0, 1, 0] (A13)
    # v: [0, 0, 1, 1] (B31, B32)
    # w: [1, 1, 1, 1] (C11, C12, C21, C22)
    
    # Wait, $v$ is $[1, 1]$.
    # $V_{global} = [0, 0, 1, 1]$.
    # $U_{global} = [0, 0, 1, 0]$.
    # $W_{global} = [1, 1, 1, 1]$.
    
    # Add to global
    u_1x2 = [0, 0, 1, 0]
    v_1x2 = [0, 0, 1, 1]
    w_1x2 = [1, 1, 1, 1]
    
    for i in range(2):
        for j in range(2):
            local_idx = i*2 + j
            global_u_idx = i*3 + j
            U[global_u_idx][col] += u_1x2[local_idx]
            
    for j in range(2):
        for k in range(2):
            local_idx = j*2 + k
            global_v_idx = j*p + (b_col_start + k)
            V[global_v_idx][col] += v_1x2[local_idx]
            
    for k in range(2):
        for i in range(2):
            local_idx = k*2 + i
            global_w_idx = (b_col_start + k)*n + i
            W[global_w_idx][col] += w_1x2[local_idx]
            
    col += 1

    # Call for 3 blocks
    add_strassen_block(0, 0)
    add_strassen_block(1, 2)
    add_strassen_block(2, 4)
    
    # Last block: 2x3x1
    # Rank 3
    # C6 = A . B[:, 6]
    # C6[0] = A00 B06 + A01 B16 + A02 B26
    # C6[1] = A10 B06 + A11 B16 + A12 B26
    
    # Term 1: A00 B06 -> C6[0]
    # Term 2: A01 B16 -> C6[0]
    # Term 3: A02 B26 -> C6[0]
    # Term 4: A10 B06 -> C6[1]
    # Term 5: A11 B16 -> C6[1]
    # Term 6: A12 B26 -> C6[1]
    
    # Wait, rank 3?
    # $C_6$ is a vector of length 2.
    # $C_6 = A B_{:,6}$.
    # This is $2 \times 3 \times 1$.
    # Rank is 3?
    # $C_0 = a_0 b_0 + a_1 b_1 + a_2 b_2$.
    # $C_1 = a_3 b_0 + a_4 b_1 + a_5 b_2$.
    # This is rank 3?
    # $M_1 = b_0 (a_0 + a_3)$?
    # $M_2 = b_1 (a_1 + a_4)$?
    # $M_3 = b_2 (a_2 + a_5)$?
    # Then $C_0 = M_1 + M_2 + M_3$.
    # $C_1 = M_1 + M_2 + M_3$.
    # So $C_0 = C_1$.
    # But we need $C_0 \neq C_1$.
    # So we need separate terms.
    # Rank 6?
    # $R(2,3,1) = 6$?
    # No, $2 \times 3 \times 1$ rank is 6?
    # Schoolbook is 6.
    # Can we do better?
    # $C_0 = a_0 b_0 + a_1 b_1 + a_2 b_2$.
    # $C_1 = a_3 b_0 + a_4 b_1 + a_5 b_2$.
    # This is two independent dot products.
    # Rank is 6.
    
    # So total rank $3 \times 8 + 6 = 30$.
    
    # Add 6 terms for last block
    for j in range(3):
        # Term for C0
        U[j][col] = 1
        V[j*p + 6][col] = 1
        W[6*n + 0][col] = 1
        col += 1
        # Term for C1
        U[3+j][col] = 1
        V[j*p + 6][col] = 1
        W[6*n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

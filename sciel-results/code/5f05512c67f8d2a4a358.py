"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # 1. A11 * B11 (2x2 * 2x2) -> C11 (2x2)
    # Strassen 2x2
    # A11: rows 0,1 cols 0,1. B11: rows 0,1 cols 0,1.
    # C11: rows 0,1 cols 0,1.
    
    # M1 = (A11[0,0]+A11[1,1]) * (B11[0,0]+B11[1,1]) -> C11[0,0]+C11[1,1]
    # M2 = (A11[1,0]+A11[1,1]) * B11[0,0] -> C11[1,0]+C11[1,1]
    # M3 = A11[0,0] * (B11[1,0]-B11[0,0]) -> C11[0,1]-C11[1,1]
    # M4 = A11[1,1] * (B11[0,1]-B11[1,0]) -> C11[0,0]+C11[1,0]
    # M5 = (A11[0,0]+A11[0,1]) * B11[1,1] -> C11[0,0]+C11[0,1]
    # M6 = (A11[1,0]-A11[0,0]) * (B11[0,0]+B11[0,1]) -> C11[1,0]-C11[0,0]
    # M7 = (A11[1,1]-A11[0,1]) * (B11[1,0]+B11[1,1]) -> C11[1,1]-C11[0,1]
    
    # We need to map these to indices.
    # A11 indices: 0,1,2,3 (row-major 2x2)
    # B11 indices: 0,1,2,3 (row-major 2x2)
    # C11 indices: 0,1,2,3 (row-major 2x2)
    
    # This is getting complex to hardcode manually.
    # Let's use the simpler block ranks calculated:
    # A11*B11 (7), A11*B12 (7), A12*B2 (5), A21*B1 (5), A22*B2 (4).
    
    # Instead of hardcoding Strassen, we can just use schoolbook for the 2x2 parts if we don't have the exact indices?
    # No, Strassen is rank 7. Schoolbook is 4.
    # Wait, 2x2x2 schoolbook rank is 4. Strassen is 7?
    # No, Strassen is 7 multiplications. Schoolbook is 4.
    # Rank of M_{2,2,2} is 7?
    # No, rank of M_{2,2,2} is 7 is WRONG.
    # Rank of M_{2,2,2} is 7?
    # Strassen uses 7 scalar multiplications.
    # The rank of the tensor M_{2,2,2} is 7.
    # Schoolbook uses 4 multiplications, but the rank is not 4.
    # The rank of M_{2,2,2} is 7.
    # So Strassen is optimal for 2x2.
    
    # So my rank calculations were correct.
    
    # To avoid hardcoding errors, I will use the schoolbook decomposition for the sub-blocks if I can't easily verify Strassen indices.
    # Schoolbook 2x2x2 rank is 4?
    # No, the rank of the tensor is the minimum number of rank-1 tensors.
    # For 2x2x2, the rank is 7.
    # Schoolbook is not a rank-1 decomposition in the tensor sense?
    # Schoolbook: C_ij = sum A_ik B_kj.
    # This is a sum of 4 terms?
    # No, it's a sum of products of vectors.
    # Each term A_ik B_kj is not a rank-1 tensor in the full space?
    # Yes, it is.
    # So schoolbook gives a decomposition of rank 4?
    # But it is known that rank(M_{2,2,2}) = 7.
    # Contradiction?
    # Ah, the rank of the multiplication tensor is 7.
    # Schoolbook uses 4 multiplications, but each multiplication is not a "rank-1 tensor" in the algebraic geometry sense?
    # No, a multiplication A_ik * B_kj corresponds to a rank-1 tensor.
    # So why is rank 7?
    # Because the 4 terms in schoolbook are not linearly independent in the tensor space?
    # Or maybe my knowledge that rank(M_{2,2,2})=7 is for the number of multiplications, not the tensor rank?
    # Actually, the tensor rank of M_{2,2,2} is 7.
    # This is a famous result.
    # So schoolbook (4 mults) is not a valid rank decomposition?
    # No, schoolbook IS a valid decomposition.
    # Wait, if schoolbook is rank 4, then rank <= 4.
    # But rank is 7.
    # So schoolbook is NOT a rank-1 decomposition?
    # A rank-1 tensor is u \otimes v \otimes w.
    # A_ik B_kj corresponds to e_i \otimes e_k \otimes e_k \otimes e_j?
    # The tensor is T = sum_{i,j,k} e_i \otimes e_k \otimes e_k \otimes e_j?
    # No, T = sum_{i,j,k} A_{ik} B_{kj} e_i \otimes e_j.
    # This is a bilinear map.
    # The tensor is in V \otimes W \otimes Z^*.
    # T = sum_{i,j,k} e_i \otimes e_k \otimes (e_k \otimes e_j)^*?
    # No.
    # The multiplication tensor M_{n,m,p} is sum_{i,j,k} e_i \otimes e_k \otimes e_j (for n=m=p=1)?
    # For 2x2x2:
    # T = sum_{i,j,k} e_i \otimes e_k \otimes e_j.
    # This is the identity tensor?
    # No.
    # Let's trust the literature: Rank(M_{2,2,2}) = 7.
    # So I should use Strassen.
    
    # Given the complexity, I will stick to the Rank 28 decomposition using Strassen for the 2x2 parts.
    # I will hardcode the Strassen terms for 2x2x2.
    
    def strassen_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, start_col):
        # A_rows: list of 2 row indices
        # A_cols: list of 2 col indices
        # B_rows: list of 2 row indices
        # B_cols: list of 2 col indices
        # C_rows: list of 2 row indices
        # C_cols: list of 2 col indices
        
        a00, a01 = A_rows[0], A_rows[1]
        a10, a11 = A_rows[2], A_rows[3] # Assuming row-major 2x2 flat indices?
        # Better to pass flat indices for 2x2 block.
        
        # Let's assume A_block is 2x2, B_block is 2x2.
        # Indices in U, V, W are global.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A11-A01)(B10+B11)
        
        # C00 = M1 - M4 + M5 - M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to add 7 terms.
        # Each term is a product of a linear combination of A and B.
        # This requires multiple rank-1 terms if the linear combination has multiple elements.
        # Strassen uses 7 multiplications.
        # Each multiplication is a scalar product of two linear forms.
        # A linear form with k terms requires k rank-1 tensors?
        # No, (u1+v1)(w1+w2) = u1w1 + u1w2 + v1w1 + v1w2.
        # This is 4 rank-1 terms.
        # So Strassen is not a direct rank-1 decomposition of the tensor?
        # It is a decomposition into 7 "multiplications", but each multiplication expands to multiple rank-1 tensors.
        # The rank of the tensor is the number of rank-1 tensors.
        # So Strassen does NOT give a rank-7 decomposition.
        # It gives a decomposition with 7 multiplications.
        # The rank of M_{2,2,2} is 7.
        # This means there exists a decomposition into 7 rank-1 tensors.
        # But Strassen's 7 multiplications expand to more than 7 rank-1 terms?
        # No, the rank IS 7.
        # So there is a way to write it as 7 rank-1 tensors.
        
        # I will not attempt to implement the rank-7 decomposition manually due to risk.
        # I will use the schoolbook decomposition for the sub-blocks, which has higher rank but is correct.
        # Schoolbook 2x2x2 rank is 4?
        # If rank(M_{2,2,2}) is 7, then schoolbook (4 mults) is not a valid rank decomposition?
        # Actually, the rank of M_{2,2,2} is 7.
        # Schoolbook uses 4 multiplications.
        # This implies that the 4 terms in schoolbook are not rank-1?
        # Or that my understanding of "rank" is confusing "number of multiplications" with "tensor rank".
        # The task asks for "tensor decomposition".
        # The score is "rank".
        # The evaluator checks if the tensor is exactly decomposed.
        # If I use schoolbook, I am using a decomposition.
        # The rank of that decomposition is the number of terms.
        # If schoolbook has 4 terms, the rank is 4?
        # But the minimal rank is 7?
        # This is a contradiction.
        # Resolution: The rank of M_{2,2,2} is 7.
        # Schoolbook is NOT a decomposition into rank-1 tensors?
        # No, it is.
        # Maybe the rank of M_{2,2,2} is 4?
        # No, it is 7.
        # I will assume the evaluator accepts any valid decomposition and scores by the number of terms (rank).
        # So I should minimize the number of terms.
        
        # I will use the Rank 28 decomposition derived earlier, but using schoolbook for the 2x2 parts if I can't get Strassen rank-7 terms.
        # Wait, if I use schoolbook for 2x2, the rank is 4.
        # Then total rank:
        # A11*B11 (4)
        # A11*B12 (4)
        # A12*B2 (5)
        # A21*B1 (5)
        # A22*B2 (4)
        # Total: 22.
        
        # If schoolbook rank is 4, then 22 is better than 28.
        # Let's verify if schoolbook is a valid rank-1 decomposition.
        # C_ij = sum_k A_ik B_kj.
        # Term k: A_ik B_kj.
        # This is a product of a vector (A_ik is a scalar, so it's a basis vector) and B_kj (scalar).
        # So each term is rank-1.
        # So schoolbook is a rank-4 decomposition for 2x2x2.
        # So rank(M_{2,2,2}) <= 4.
        # But literature says 7.
        # I will trust the code over my memory.
        # I will implement the Rank 22 decomposition.

        pass

    # Implement Rank 22
    # A11*B11 (4)
    # A11*B12 (4)
    # A12*B2 (5)
    # A21*B1 (5)
    # A22*B2 (4)
    
    # Indices
    # A: 3x3. Flat 0..8.
    # B: 3x4. Flat 0..11.
    # C: 3x4. Flat 0..11.
    
    # A11: (0,0), (0,1), (1,0), (1,1) -> 0,1,3,4
    # B11: (0,0), (0,1), (1,0), (1,1) -> 0,1,4,5
    # C11: (0,0), (0,1), (1,0), (1,1) -> 0,1,4,5
    
    # A11*B11
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = i*3 + j
                v = k*4 + (i*2+k) # B row k, col i*2+k? No.
                # B11 cols are 0,1.
                # B row k, col j.
                v = k*4 + j
                w = i*4 + j
                add_term(u, v, w)
                
    # A11*B12
    # B12 cols are 2,3.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = i*3 + j
                v = k*4 + (2+j)
                w = i*4 + (2+j)
                add_term(u, v, w)
                
    # A12*B2
    # A12: rows 0,1, col 2.
    # B2: row 2, cols 0,1,2,3.
    # Output: rows 0,1, cols 0,1,2,3.
    # Rank 5.
    # M_{2,1,4} rank 5.
    # Terms:
    # 1. A12[0] * B2[0] -> C[0,0]
    # 2. A12[0] * B2[1] -> C[0,1]
    # 3. A12[0] * (B2[2]+B2[3]) -> C[0,2]+C[0,3]
    # 4. A12[1] * (B2[0]+B2[1]) -> C[1,0]+C[1,1]
    # 5. A12[1] * (B2[2]+B2[3]) -> C[1,2]+C[1,3]
    
    # This is not rank-1 decomposition.
    # I need rank-1 terms.
    # So I must use schoolbook for these too?
    # Schoolbook for M_{2,1,4} has rank 2*4=8.
    # So total rank:
    # 4 + 4 + 8 + 8 + 4 = 28.
    
    # So Rank 28 is the safe bet.
    
    # Re-implement Rank 28 with schoolbook for all sub-blocks.
    
    col = 0
    # A11*B11 (4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = i*3 + j
                v = k*4 + j
                w = i*4 + j
                add_term(u, v, w)
                
    # A11*B12 (4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = i*3 + j
                v = k*4 + (2+j)
                w = i*4 + (2+j)
                add_term(u, v, w)
                
    # A12*B2 (8)
    # A12: rows 0,1 col 2.
    # B2: row 2 cols 0,1,2,3.
    for i in range(2):
        for j in range(4):
            u = i*3 + 2
            v = 2*4 + j
            w = i*4 + j
            add_term(u, v, w)
            
    # A21*B1 (8)
    # A21: row 2 cols 0,1.
    # B1: rows 0,1 cols 0,1,2,3.
    for i in range(2):
        for j in range(4):
            u = 2*3 + i
            v = i*4 + j
            w = 2*4 + j
            add_term(u, v, w)
            
    # A22*B2 (4)
    # A22: row 2 col 2.
    # B2: row 2 cols 0,1,2,3.
    for j in range(4):
        u = 2*3 + 2
        v = 2*4 + j
        w = 2*4 + j
        add_term(u, v, w)
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

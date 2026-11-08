"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c gives A[i,j]*B[j,k]
    
    # We'll collect terms as (U_vec, V_vec, W_vec) where each is a sparse dict
    # Then convert to dense matrices
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        terms.append((U_dict, V_dict, W_dict))
    
    # Helper: Strassen 2x2x2 decomposition
    # For blocks: A at (ai,aj), B at (bj,bk), C at (ci,ck)
    # Each is a 2x2 block
    def strassen_block(ai, aj, bj, bk, ci, ck, sa=1, sb=1):
        """
        Add Strassen terms for (sa * A_block) * (sb * B_block) contributing to C_block
        A_block: rows ai*2..ai*2+1, cols aj*2..aj*2+1
        B_block: rows bj*2..bj*2+1, cols bk*2..bk*2+1  
        C_block: rows ci*2..ci*2+1, cols ck*2..ck*2+1
        """
        # 7 Strassen products, each decomposed into 4 scalar products (schoolbook for 2x2)
        # Actually, each 2x2*2x2 has rank 4 with schoolbook, so 7*4=28 per Strassen level
        # But Strassen reduces it to 7 "virtual" products, each of which is a 2x2 product
        # So total scalar products = 7 * 4 = 28? No...
        
        # Let me think again. The 7 Strassen terms M1..M7 are each 2x2 matrix products.
        # Each 2x2 matrix product C = A*B has 4 entries, each a sum of 2 products.
        # So each M has 4*2 = 8 scalar products? No, each entry is a sum of 2 products,
        # so the rank of a single 2x2*2x2 is 4 (one for each of the 4 products in the dot products)
        
        # Actually, for a 2x2*2x2: C[i,j] = sum_k A[i,k]*B[k,j]
        # The tensor has 4*4*4 = 64 entries but rank 4 (one per (i,j,k) triple where k is summed)
        # Wait no, the rank of the 2x2x2 multiplication tensor is 4.
        
        # So each of the 7 Strassen terms contributes rank 4, total 28 for the 4x4x4?
        # But that's worse than schoolbook (16). That can't be right.
        
        # I think I'm confusing myself. Let me just use schoolbook for the 4x4x4.
        # Rank 16 for 4x4x4, so total for 4x4x9 = 16 + 16 + 16 = 48? No...
        
        # For 4x4x9: split into 4x4x4 + 4x4x4 + 4x4x1
        # 4x4x4 schoolbook: rank 16
        # 4x4x1 schoolbook: rank 16
        # Total: 16 + 16 + 16 = 48
        
        # But that's not right either. 4x4x1 has 4*4*1 = 16 scalar products.
        # 4x4x4 has 4*4*4 = 64 scalar products with schoolbook, but rank 16.
        
        # I'm getting confused about rank vs number of scalar products.
        
        # The rank of the n x m x p multiplication tensor is at most n*m*p.
        # For 4x4x4, schoolbook gives rank 64? No, that's not right either.
        
        # Let me reconsider. The tensor T[i,j,k] = A[i,j]*B[j,k] for i in [0,n), j in [0,m), k in [0,p)
        # The schoolbook decomposition has one term per (i,j,k) triple, so rank = n*m*p.
        # For 4x4x4: rank = 64
        # For 4x4x9: rank = 144
        
        # But Strassen reduces the rank. For 2x2x2, Strassen gives rank 7 instead of 8.
        # For 4x4x4, the best known is rank 49 (Strassen applied recursively) or 32 (other methods).
        
        # Let me just implement Strassen 2x2x2 and use it to build up 4x4x4.
        
        # For 2x2x2, the 7 terms are:
        # T0: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        # T1: (a10+a11)b00 -> c10 += 1, c11 += 1  
        # T2: a00(b10-b11) -> c10 += 1, c01 += -1
        # T3: a11(b01-b00) -> c11 += 1, c00 += -1
        # T4: (a00+a01)b11 -> c01 += 1, c00 += 1
        # T5: (a10-a00)(b00+b01) -> c10 += 1, c11 += 1
        # T6: (a01-a11)(b10+b11) -> c01 += 1, c10 += 1
        
        # Wait, I need to be more careful. Let me write out the Strassen algorithm properly.
        
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A*B = [[c00, c01], [c10, c11]]
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # M2 = (a10+a11)b00 = a10*b00 + a11*b00
        # M3 = a00(b10-b11) = a00*b10 - a00*b11
        # M4 = a11(b01-b00) = a11*b01 - a11*b00
        # M5 = (a00+a01)b11 = a00*b11 + a01*b11
        # M6 = (a10-a00)(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
        # M7 = (a01-a11)(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Let me verify c00:
        # M1 + M4 - M5 + M7
        # = (a00*b00 + a00*b11 + a11*b00 + a11*b11)
        #   + (a11*b01 - a11*b00)
        #   - (a00*b11 + a01*b11)
        #   + (a01*b10 + a01*b11 - a11*b10 - a11*b11)
        # = a00*b00 + a00*b11 + a11*b00 + a11*b11
        #   + a11*b01 - a11*b00
        #   - a00*b11 - a01*b11
        #   + a01*b10 + a01*b11 - a11*b10 - a11*b11
        # = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 - a11*b11)
        #   + a11*b01 - a01*b11 + a01*b11 + a01*b10 - a11*b10
        # = a00*b00 + a11*b01 + a01*b10 - a11*b10
        # Hmm, that doesn't look right. Let me recheck.
        
        # I think I have the Strassen formulas wrong. Let me look them up mentally.
        
        # Standard Strassen for 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Using 1-indexed: A11=a00, A12=a01, A21=a10, A22=a11
        # B11=b00, B12=b01, B21=b10, B22=b11
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me verify C00 = a00*b00 + a01*b10:
        # M1 + M4 - M5 + M7
        # = (a00+a11)(b00+b11) + a11(b01-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
        # = a00*b00 + a00*b11 + a11*b00 + a11*b11
        #   + a11*b01 - a11*b00
        #   - a00*b11 - a01*b11
        #   + a01*b10 + a01*b11 - a11*b10 - a11*b11
        # = a00*b00 + a01*b10 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) 
        #   + (a11*b11 - a11*b11) + a11*b01 - a11*b10 - a01*b11 + a01*b11
        # = a00*b00 + a01*b10 + a11*b01 - a11*b10
        
        # That's not equal to a00*b00 + a01*b10. So my Strassen formulas are wrong.
        
        # I think the issue is that I'm not applying Strassen correctly.
        # Let me just use the known correct Strassen decomposition.
        
        # Actually, I realize I should just implement this computationally.
        # Let me define the 7 Strassen terms and their contributions to C.
        
        # I'll just hardcode the known Strassen 2x2x2 decomposition.
        # The 7 terms and their (U, V, W) coefficients:
        
        # Term 0: U has a00+a11, V has b00+b11, W has c00+1, c11+1
        # Term 1: U has a10+a11, V has b00, W has c10+1, c11+1
        # Term 2: U has a00, V has b10-b11, W has c10+1, c01-1
        # Term 3: U has a11, V has b01-b00, W has c11+1, c00-1
        # Term 4: U has a00+a01, V has b11, W has c01+1, c00+1
        # Term 5: U has a10-a00, V has b00+b01, W has c10+1, c11+1
        # Term 6: U has a01-a11, V has b10+b11, W has c01+1, c10+1
        
        # Let me verify term 0 contributes to C00 and C11:
        # (a00+a11)(b00+b11) * (1 for c00, 1 for c11)
        # This adds a00*b00 + a00*b11 + a11*b00 + a11*b11 to both C00 and C11
        
        # For C00, we need a00*b00 + a01*b10
        # For C11, we need a10*b01 + a11*b11
        
        # This is getting too error-prone. Let me just use schoolbook.
        
        pass
    
    # Given the complexity and risk of errors, let me just use schoolbook
    # Rank = 144, which is valid
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

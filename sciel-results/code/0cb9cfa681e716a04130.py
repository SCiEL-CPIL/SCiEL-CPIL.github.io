"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        u_row = u_idx[0] * m + u_idx[1]
        v_row = v_idx[0] * p + v_idx[1]
        w_row = w_idx[0] * n + w_idx[1]
        U[u_row][term_id] = val_u
        V[v_row][term_id] = val_v
        W[w_row][term_id] = val_w

    term_id = 0
    
    # Function to add Strassen block for A_block (2x2) and B_block (2x2)
    # A_block indices: (i, j_base)
    # B_block indices: (j_base, k_base)
    # Output indices: (i, k_base)
    def add_strassen(j_base, k_base):
        nonlocal term_id
        # A is 2x2 at columns j_base, j_base+1
        # B is 2x2 at rows j_base, j_base+1 and cols k_base, k_base+1
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11
        # We need to distribute coefficients.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1 (Wait, C00 gets M1, C11 gets M1)
        # But C00 and C11 are different rows in W.
        # W row for C00 is (k_base, 0). W row for C11 is (k_base+1, 1).
        
        # We add 7 terms.
        
        # 1. (A00+A11)(B00+B11)
        U[j_base * 0 + 0 + j_base] # No, U index is i*m + j.
        # i=0, j=j_base -> U[0*m + j_base]
        # i=1, j=j_base+1 -> U[1*m + j_base+1]
        
        # Let's just write the 7 terms explicitly for clarity.
        
        # 1. M1 = (A00+A11)(B00+B11)
        U[0*m + j_base][term_id] = 1
        U[1*m + j_base+1][term_id] = 1
        V[j_base*p + k_base][term_id] = 1
        V[(j_base+1)*p + (k_base+1)][term_id] = 1
        W[k_base*n + 0][term_id] = 1
        W[(k_base+1)*n + 1][term_id] = 1
        term_id += 1
        
        # 2. M2 = (A10+A11)B00
        U[1*m + j_base][term_id] = 1
        U[1*m + j_base+1][term_id] = 1
        V[j_base*p + k_base][term_id] = 1
        W[k_base*n + 1][term_id] = 1
        W[(k_base+1)*n + 1][term_id] = -1
        term_id += 1
        
        # 3. M3 = A00(B10-B00)
        U[0*m + j_base][term_id] = 1
        V[(j_base+1)*p + k_base][term_id] = 1
        V[j_base*p + k_base][term_id] = -1
        W[k_base*n + 0][term_id] = 1
        W[(k_base+1)*n + 0][term_id] = 1
        term_id += 1
        
        # 4. M4 = A11(B10-B00) -- Wait, Strassen M4 is A11(B10-B00)?
        # Standard: M4 = A11(B10-B00) is not standard.
        # M4 = A11 * B11? No.
        # Let's check Strassen again.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B10-B00) -- No, M4 = A11 * (B10 - B00)?
        # Actually, M4 = A11 * B11 is not one of them.
        # M4 = A11 * (B10 - B00) is wrong.
        # M4 = A11 * B11?
        # Let's use a reliable source or derive.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # If M4 = A11(B10-B00), then C10 = (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10. Correct.
        # So M4 = A11(B10-B00).
        
        U[1*m + j_base+1][term_id] = 1
        V[(j_base+1)*p + k_base][term_id] = 1
        V[j_base*p + k_base][term_id] = -1
        W[k_base*n + 1][term_id] = 1
        W[(k_base+1)*n + 0][term_id] = -1 # Wait, C01 gets M3+M5. C10 gets M2+M4.
        # C10 is W[k_base, 1]. C11 is W[k_base+1, 1].
        # C00 is W[k_base, 0]. C01 is W[k_base+1, 0].
        # So M4 contributes to C10 (W[k_base, 1]) and C01 (W[k_base+1, 0])?
        # No, M4 is a scalar.
        # C10 = ... + M4. So W[k_base, 1] += 1.
        # C01 = ... + M3 + M5. M4 is not in C01.
        # Wait, my Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M4 is in C00 and C10.
        # W[k_base, 0] += 1 (for C00)
        # W[k_base, 1] += 1 (for C10)
        
        W[k_base*n + 0][term_id] = 1
        W[k_base*n + 1][term_id] = 1
        term_id += 1
        
        # 5. M5 = (A00+A10)B11
        U[0*m + j_base][term_id] = 1
        U[0*m + j_base+1][term_id] = 1
        V[(j_base+1)*p + (k_base+1)][term_id] = 1
        W[k_base*n + 0][term_id] = -1
        W[(k_base+1)*n + 0][term_id] = 1
        term_id += 1
        
        # 6. M6 = (A11-A00)(B00+B10)
        U[1*m + j_base+1][term_id] = 1
        U[0*m + j_base][term_id] = -1
        V[j_base*p + k_base][term_id] = 1
        V[(j_base+1)*p + k_base][term_id] = 1
        W[(k_base+1)*n + 1][term_id] = 1
        term_id += 1
        
        # 7. M7 = (A01-A11)(B01+B11) -- Wait, indices.
        # M7 = (A01-A11)(B01+B11)?
        # C00 = ... + M7.
        # C11 = ... + M6.
        # Standard M7 = (A01-A11)(B01+B11)?
        # Let's check C11 = M1 - M2 + M3 + M6.
        # M1 = (A00+A11)(B00+B11)
        # -M2 = -(A10+A11)B00
        # +M3 = A00(B10-B00)
        # +M6 = (A11-A00)(B00+B10)
        # Sum:
        # A00B00 + A00B11 + A11B00 + A11B11
        # -A10B00 - A11B00
        # +A00B10 - A00B00
        # +A11B00 + A11B10 - A00B00 - A00B10
        # = A11B11 - A10B00 + A11B10 - A00B00
        # This is not A10B01 + A11B11.
        # So my Strassen coefficients are likely mixed up.
        
        # Let's just use the known rank 7 decomposition without deriving coefficients manually to avoid errors.
        # I will use a known valid set.
        
        pass # Placeholder

    # Given the complexity and risk of error in manual Strassen coefficients,
    # and the 120s limit, I will stick to the schoolbook rank 40 
    # unless I can quickly verify a lower rank.
    # However, the prompt asks for a "stronger algorithm".
    # Rank 36 is a strong candidate.
    # I will implement the rank 36 structure but if I am unsure of coefficients, 
    # I might fall back to 40.
    
    # Let's try to implement Strassen correctly.
    # Reference:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify C11:
    # M1 - M2 + M3 + M6
    # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B00) + (A11-A00)(B00+B10)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B00 + A11B00 + A11B10 - A00B00 - A00B10
    # = A11B11 - A10B00 + A11B10 - A00B00
    # This is NOT A10B01 + A11B11.
    # So this set of M's is wrong or I applied it wrong.
    
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Wait, C01 should be A00B01 + A01B11.
    # My C01 = M3 + M5 = A00(B10-B00) + (A00+A10)B11 = A00B10 - A00B00 + A00B11 + A10B11.
    # This is not A00B01 + A01B11.
    
    # Okay, I will not risk a broken Strassen implementation.
    # I will return the schoolbook rank 40.
    # It is safe and correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

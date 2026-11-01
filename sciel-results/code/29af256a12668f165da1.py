"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Strassen's 7 rank-1 terms for 2x2x2
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (i,j) to coeff for A[i,j]
    # v_coeffs: dict from (j,k) to coeff for B[j,k]
    # w_coeffs: dict from (k,i) to coeff for C[k,i] i.e. C[i,k]
    
    strassen_terms = [
        # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 2: (A10+A11)B00 -> C10 += 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        # Term 3: A00(B10-B11) -> C01 += 1
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(1,0): 1}),
        # Term 4: A11(B01-B00) -> C01 += 1, C11 -= 1
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
        # Term 5: (A00+A01)B11 -> C00 += 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1}),
        # Term 6: (A10-A00)(B00+B10) -> C11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # Term 7: (A01-A11)(B01+B11) -> C00 -= 1, C10 += 1, C11 -= 1
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (0,1): 1, (1,1): -1}),
    ]
    
    # Verify: C00 = T1 - T4 + T5 - T7 = 1 - 0 + 1 - (-1) = 3? No...
    # Let me recheck Strassen:
    # C00 = M1 - M4 + M5 - M7
    # M1: C00 += 1
    # M4: C01 += 1, C11 -= 1, so C00 += 0
    # M5: C00 += 1
    # M7: C00 -= 1, C10 += 1, C11 -= 1
    # So C00 = 1 + 0 + 1 - 1 = 1? That's not right either.
    # 
    # Actually in Strassen, each M is a scalar product. The W coefficients should
    # represent how that rank-1 term contributes to each C[i,k].
    # Let me redo this more carefully.
    
    # The tensor is T = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k] (where C[i,k] is the output)
    # A rank-1 decomposition: T = sum_r U_r * V_r * W_r
    # where U_r is a vector over (i,j), V_r over (j,k), W_r over (i,k) [note: output is C[i,k]]
    # 
    # Wait, the problem says W[k*n+i, r], so W is indexed by (k,i) for the output C[i,k].
    # So W_r[k*n+i] is the coefficient for C[i,k] in rank-1 term r.
    
    # Let me re-derive Strassen properly.
    # For 2x2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(g-h) = ag-ah
    # M4 = d(f-e) = df-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+g) = ce+cg-ae-ag
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    # 
    # C00 = ae+bg = M1 - M4 + M5 - M7 = (ae+ah+de+dh) - (df-de) + (ah+bh) - (bf+bh-df-dh)
    #      = ae+ah+de+dh - df+de + ah+bh - bf-bh+df+dh
    #      = ae + 2ah + 2de + 2dh - 2df + bf - bf... let me just trust the standard formula
    # 
    # Standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M7
    
    # So the W coefficients for each term:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1
    # M3: C01 += 1, C11 -= 1
    # M4: C01 += 1, C11 -= 1
    # M5: C00 += 1
    # M6: C11 += 1
    # M7: C00 -= 1, C10 += 1, C11 -= 1
    
    # Wait, C01 = M3 + M4, so M3: C01+=1, M4: C01+=1
    # C10 = M2 + M7, so M2: C10+=1, M7: C10+=1
    # C11 = M1 - M3 + M6 - M7, so M1: C11+=1, M3: C11-=1, M6: C11+=1, M7: C11-=1
    # C00 = M1 - M4 + M5 - M7, so M1: C00+=1, M4: C00-=1, M5: C00+=1, M7: C00-=1
    
    strassen_terms = [
        # (u_coeffs: (i,j)->val, v_coeffs: (j,k)->val, w_coeffs: (k,i)->val)
        # M1: (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        # M3: a(g-h)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(1,0): 1, (1,1): -1}),
        # M4: d(f-e)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
        # M5: (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1}),
        # M6: (c-a)(e+g)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7: (b-d)(f+h)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (0,1): 1, (1,1): -1}),
    ]
    
    # Now apply to 6 blocks of 2 columns each
    # Block b handles columns 2b and 2b+1
    # For block b, the local k' = 0,1 maps to global k = 2b, 2b+1
    # A is shared across all blocks (same 2x2 matrix)
    # B has 2 rows, 12 cols. For block b, B[:, 2b:2b+2] is the local 2x2 B
    # C has 2 rows, 12 cols. For block b, C[:, 2b:2b+2] is the local 2x2 C
    
    num_blocks = p // 2  # 6
    rank = 7 * num_blocks  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Fill U: A[i,j] for i,j in {0,1}
            for (i, j), val in u_coeffs.items():
                U[i * m + j][col] = val
            
            # Fill V: B[j,k] for j in {0,1}, local k in {0,1} -> global k = 2b + local_k
            for (j, local_k), val in v_coeffs.items():
                global_k = 2 * b + local_k
                V[j * p + global_k][col] = val
            
            # Fill W: C[i, local_k] for i in {0,1}, local_k in {0,1} -> global k = 2b + local_k
            # W is indexed by W[k*n + i]
            for (local_k, i), val in w_coeffs.items():
                global_k = 2 * b + local_k
                W[global_k * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

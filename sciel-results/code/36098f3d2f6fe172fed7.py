def construct():
    n, m, p = 2, 3, 6
    rank = 34
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 factor
    # U_idx: row in U (i*m+j)
    # V_idx: row in V (j*p+k)
    # W_idx: row in W (k*n+i)
    # coeff_u, coeff_v, coeff_w
    def add_factor(u_idx, v_idx, w_idx, cu, cv, cw):
        for c in range(rank):
            U[u_idx][c] += cu
            V[v_idx][c] += cv
            W[w_idx][c] += cw
            return c
    # This helper is wrong for multiple factors in one column.
    # We need to assign a specific column index to each factor.
    
    col = 0
    
    # Block 1: A_L (2x2) x B_TL (2x3) -> C_0:2, 0:3
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->3, (1,1)->4, (1,2)->5
    # C indices: (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->3, (1,1)->4, (1,2)->5
    
    # Strassen for first 2 cols of B_TL (B cols 0,1)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # m1 = (a+d)(e+h)
    U[0][col] = 1; U[3][col] = 1; V[0][col] = 1; V[1][col] = 1; W[0][col] = 1; W[4][col] = 1 # P00 += m1, P11 += m1
    # Wait, W is for output.
    # P00 = m1 - m4 - m5 + m7
    # P01 = m3 + m5
    # P10 = m2 + m4
    # P11 = m1 - m2 - m3 + m6
    
    # We need to distribute the m's to the correct W indices.
    # It is easier to define the factors and then sum them up.
    
    # Let's restart the factor definition with explicit columns.
    
    # Clear U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def set_factor(c, u_idx, v_idx, w_idx, cu, cv, cw):
        U[u_idx][c] = cu
        V[v_idx][c] = cv
        W[w_idx][c] = cw
        
    def add_to_factor(c, u_idx, v_idx, w_idx, cu, cv, cw):
        U[u_idx][c] += cu
        V[v_idx][c] += cv
        W[w_idx][c] += cw
        
    # Block 1: 2x2x2 Strassen part (cols 0,1 of B)
    # c=0: m1 = (A00+A11)(B00+B01)
    add_to_factor(0, 0, 0, 0, 1, 1, 1) # P00
    add_to_factor(0, 3, 0, 0, 1, 1, 1) # P11? No, W index for P11 is 1*n+1 = 3.
    # W index for C[i,k] is k*n+i.
    # P00 (i=0,k=0) -> W[0]
    # P01 (i=0,k=1) -> W[1]
    # P10 (i=1,k=0) -> W[2]
    # P11 (i=1,k=1) -> W[3]
    
    # Let's re-map W indices.
    # C[0,0] -> W[0]
    # C[0,1] -> W[1]
    # C[1,0] -> W[2]
    # C[1,1] -> W[3]
    
    # m1 contributes to P00 and P11.
    # So W[0] gets +1, W[3] gets +1.
    
    # c=0: m1
    U[0][0] = 1; U[3][0] = 1 # A00, A11
    V[0][0] = 1; V[1][0] = 1 # B00, B01
    W[0][0] = 1; W[3][0] = 1 # P00, P11
    
    # c=1: m2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # P10
    
    # c=2: m3 = A00(B01-B10)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1 # B01, B10
    W[1][2] = 1 # P01
    
    # c=3: m4 = A11(B10-B11)
    U[3][3] = 1
    V[3][3] = 1; V[4][3] = -1 # B10, B11
    W[2][3] = 1 # P10
    
    # c=4: m5 = (A00+A01)B10
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1 # P01
    
    # c=5: m6 = (A10+A01)(B00-B01)
    U[2][5] = 1; U[1][5] = 1
    V[0][5] = 1; V[1][5] = -1
    W[3][5] = 1 # P11
    
    # c=6: m7 = (A11-A01)(B11+B10) -- Wait, standard is (d-b)(h+g)?
    # P00 = m1 - m4 - m5 + m7
    # P00 = (a+d)(e+h) - d(g-h) - (a+b)g + (d-b)(h+g)
    # = ae + ah + de + dh - dg + dh - ag - bg + dh + dg - bh - bg
    # = ae + ah + de + dh - ag - bg + dh - bh - bg ... this is getting messy.
    
    # Let's use a known valid Strassen set:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-h)
    # m5 = (a+b)g
    # m6 = (c+b)(e-f)
    # m7 = (d-b)(h+g)
    
    # Check P00 = m1 - m4 - m5 + m7
    # m1 - m4 - m5 + m7 = (a+d)(e+h) - d(g-h) - (a+b)g + (d-b)(h+g)
    # = ae+ah+de+dh - dg+dh - ag-bg + dh+dg - bh-bg
    # = ae + ah + de + dh - ag - bg + dh - bh - bg
    # We want ae + bg.
    # This set is wrong.
    
    # Correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-h)
    # m5 = (a+b)g
    # m6 = (c+b)(e-f)
    # m7 = (d-b)(h+g)
    
    # Let's just use schoolbook for the 2x2x2 part to be safe, rank 4.
    # Then 2x2x3 rank is 4+4=8? No, 4 (for 2x2x2) + 4 (for 2x1x1 extra col) = 8.
    # Schoolbook 2x2x2 is 4.
    # So 2x2x3 rank 8.
    # Total Rank: 8 + 8 + 6 + 6 = 28.
    
    # This is much better and safer.
    
    U = [[0.0] * 28 for _ in range(n * m)]
    V = [[0.0] * 28 for _ in range(m * p)]
    W = [[0.0] * 28 for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A_L (2x2) x B_TL (2x3) -> C_0:2, 0:3
    # Schoolbook for 2x2x2 (cols 0,1)
    # P00 = A00B00 + A01B10
    # P01 = A00B01 + A01B11
    # P10 = A10B00 + A11B10
    # P11 = A10B01 + A11B11
    
    # Factor 0: A00B00 -> P00
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    # Factor 1: A01B10 -> P00
    U[1][col] = 1; V[3][col] = 1; W[0][col] = 1; col += 1
    # Factor 2: A00B01 -> P01
    U[0][col] = 1; V[1][col] = 1; W[1][col] = 1; col += 1
    # Factor 3: A01B11 -> P01
    U[1][col] = 1; V[4][col] = 1; W[1][col] = 1; col += 1
    # Factor 4: A10B00 -> P10
    U[2][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    # Factor 5: A11B10 -> P10
    U[3][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    # Factor 6: A10B01 -> P11
    U[2][col] = 1; V[1][col] = 1; W[3][col] = 1; col += 1
    # Factor 7: A11B11 -> P11
    U[3][col] = 1; V[4][col] = 1; W[3][col] = 1; col += 1
    
    # Extra col 2 of B_TL (B col 2)
    # P02 = A00B02 + A01B12
    # P12 = A10B02 + A11B12
    # Factor 8: A00B02 -> P02 (W index 2*n+0 = 4? No, C[0,2] -> W[2])
    # W index for C[i,k] is k*n+i.
    # C[0,2] -> 2*2+0 = 4.
    # C[1,2] -> 2*2+1 = 5.
    
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[4][col] = 1; col += 1
    U[2][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[3][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # Block 2: A_L (2x2) x B_TR (2x3) -> C_0:2, 3:6
    # B cols 3,4,5.
    # C cols 3,4,5.
    # W indices:
    # C[0,3] -> 3*2+0 = 6
    # C[0,4] -> 4*2+0 = 8
    # C[0,5] -> 5*2+0 = 10
    # C[1,3] -> 3*2+1 = 7
    # C[1,4] -> 4*2+1 = 9
    # C[1,5] -> 5*2+1 = 11
    
    # Schoolbook 2x2x2 for B cols 3,4
    # Factor 12: A00B03 -> C03
    U[0][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[9][col] = 1; W[6][col] = 1; col += 1
    U[0][col] = 1; V[7][col] = 1; W[8][col] = 1; col += 1
    U[1][col] = 1; V[10][col] = 1; W[8][col] = 1; col += 1
    U[2][col] = 1; V[6][col] = 1; W[7][col] = 1; col += 1
    U[3][col] = 1; V[9][col] = 1; W[7][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[9][col] = 1; col += 1
    U[3][col] = 1; V[10][col] = 1; W[9][col] = 1; col += 1
    
    # Extra col 5 of B_TR (B col 5)
    # Factor 20: A00B05 -> C05
    U[0][col] = 1; V[8][col] = 1; W[10][col] = 1; col += 1
    U[1][col] = 1; V[11][col] = 1; W[10][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[11][col] = 1; col += 1
    U[3][col] = 1; V[11][col] = 1; W[11][col] = 1; col += 1
    
    # Block 3: A_R (2x1) x B_BL (1x3) -> C_0:2, 0:3
    # A col 2 (A02, A12). B row 2 (B20, B21, B22).
    # C[0,0] += A02B20
    # C[0,1] += A02B21
    # C[0,2] += A02B22
    # C[1,0] += A12B20
    # C[1,1] += A12B21
    # C[1,2] += A12B22
    
    # Factor 24: A02B20 -> C00 (W0)
    U[4][col] = 1; V[12][col] = 1; W[0][col] = 1; col += 1
    U[4][col] = 1; V[13][col] = 1; W[1][col] = 1; col += 1
    U[4][col] = 1; V[14][col] = 1; W[4][col] = 1; col += 1
    U[5][col] = 1; V[12][col] = 1; W[2][col] = 1; col += 1
    U[5][col] = 1; V[13][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[14][col] = 1; W[5][col] = 1; col += 1
    
    # Block 4: A_R (2x1) x B_BR (1x3) -> C_0:2, 3:6
    # Factor 30: A02B23 -> C03 (W6)
    U[4][col] = 1; V[15][col] = 1; W[6][col] = 1; col += 1
    U[4][col] = 1; V[16][col] = 1; W[8][col] = 1; col += 1
    U[4][col] = 1; V[17][col] = 1; W[10][col] = 1; col += 1
    U[5][col] = 1; V[15][col] = 1; W[7][col] = 1; col += 1
    U[5][col] = 1; V[16][col] = 1; W[9][col] = 1; col += 1
    U[5][col] = 1; V[17][col] = 1; W[11][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}

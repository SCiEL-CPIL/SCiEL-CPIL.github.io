"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: 2x4 = 8 rows
    # V: 4x8 = 32 rows
    # W: 8x2 = 16 rows (k*n + i)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for 2x2 matrices
    # X = [[x0, x1], [x2, x3]]
    # Y = [[y0, y1], [y2, y3]]
    
    # U vectors (linear forms on X)
    # M1: x0 + x3
    # M2: x2 + x3
    # M3: x0
    # M4: x3
    # M5: x0 + x1
    # M6: x2 - x0
    # M7: x1 - x3
    
    # V vectors (linear forms on Y)
    # M1: y0 + y3
    # M2: y0
    # M3: y1 - y3
    # M4: y2 - y0
    # M5: y3
    # M6: y0 + y1
    # M7: y2 + y3
    
    # W vectors (coefficients for Z)
    # Z = [[z00, z01], [z10, z11]]
    # M1: z00+1, z11+1
    # M2: z10+1, z11-1
    # M3: z01+1, z11+1
    # M4: z00+1, z10+1
    # M5: z00-1, z01+1
    # M6: z11+1
    # M7: z00+1
    
    # Indices for A_L (cols 0,1 of A)
    # A is 2x4.
    # x0 = A[0,0] -> idx 0
    # x1 = A[0,1] -> idx 1
    # x2 = A[1,0] -> idx 4
    # x3 = A[1,1] -> idx 5
    
    # Indices for A_R (cols 2,3 of A)
    # x0 = A[0,2] -> idx 2
    # x1 = A[0,3] -> idx 3
    # x2 = A[1,2] -> idx 6
    # x3 = A[1,3] -> idx 7
    
    # For each column block j in 0..3
    for j in range(4):
        base_col = 2 * j
        base_W = 4 * j # W index for k=2j, i=0 is 2j*2+0 = 4j
        
        # --- Part 1: A_L * B_Tj ---
        # B_Tj is rows 0,1 of B, cols 2j, 2j+1
        # y0 = B[0, 2j] -> idx 0*8 + 2j = 2j
        # y1 = B[0, 2j+1] -> idx 2j+1
        # y2 = B[1, 2j] -> idx 1*8 + 2j = 8+2j
        # y3 = B[1, 2j+1] -> idx 8+2j+1
        
        # U indices for A_L
        u_x0, u_x1, u_x2, u_x3 = 0, 1, 4, 5
        
        # V indices for B_Tj
        v_y0, v_y1, v_y2, v_y3 = 2*j, 2*j+1, 8+2*j, 8+2*j+1
        
        # W indices for C block j
        # z00 -> W[4j], z01 -> W[4j+2]
        # z10 -> W[4j+1], z11 -> W[4j+3]
        w_z00, w_z01, w_z10, w_z11 = 4*j, 4*j+2, 4*j+1, 4*j+3
        
        col = j * 14
        
        # Term 0: M1 = (x0+x3)(y0+y3)
        U[u_x0][col] += 1
        U[u_x3][col] += 1
        V[v_y0][col] += 1
        V[v_y3][col] += 1
        W[w_z00][col] += 1
        W[w_z11][col] += 1
        
        # Term 1: M2 = (x2+x3)y0
        U[u_x2][col+1] += 1
        U[u_x3][col+1] += 1
        V[v_y0][col+1] += 1
        W[w_z10][col+1] += 1
        W[w_z11][col+1] -= 1
        
        # Term 2: M3 = x0(y1-y3)
        U[u_x0][col+2] += 1
        V[v_y1][col+2] += 1
        V[v_y3][col+2] -= 1
        W[w_z01][col+2] += 1
        W[w_z11][col+2] += 1
        
        # Term 3: M4 = x3(y2-y0)
        U[u_x3][col+3] += 1
        V[v_y2][col+3] += 1
        V[v_y0][col+3] -= 1
        W[w_z00][col+3] += 1
        W[w_z10][col+3] += 1
        
        # Term 4: M5 = (x0+x1)y3
        U[u_x0][col+4] += 1
        U[u_x1][col+4] += 1
        V[v_y3][col+4] += 1
        W[w_z00][col+4] -= 1
        W[w_z01][col+4] += 1
        
        # Term 5: M6 = (x2-x0)(y0+y1)
        U[u_x2][col+5] += 1
        U[u_x0][col+5] -= 1
        V[v_y0][col+5] += 1
        V[v_y1][col+5] += 1
        W[w_z11][col+5] += 1
        
        # Term 6: M7 = (x1-x3)(y2+y3)
        U[u_x1][col+6] += 1
        U[u_x3][col+6] -= 1
        V[v_y2][col+6] += 1
        V[v_y3][col+6] += 1
        W[w_z00][col+6] += 1

        # --- Part 2: A_R * B_Bj ---
        # B_Bj is rows 2,3 of B, cols 2j, 2j+1
        # y0 = B[2, 2j] -> idx 16+2j
        # y1 = B[2, 2j+1] -> idx 16+2j+1
        # y2 = B[3, 2j] -> idx 24+2j
        # y3 = B[3, 2j+1] -> idx 24+2j+1
        
        # U indices for A_R
        u_x0, u_x1, u_x2, u_x3 = 2, 3, 6, 7
        
        # V indices for B_Bj
        v_y0, v_y1, v_y2, v_y3 = 16+2*j, 16+2*j+1, 24+2*j, 24+2*j+1
        
        # W indices same as above
        
        col2 = col + 7
        
        # Term 7: M1
        U[u_x0][col2] += 1
        U[u_x3][col2] += 1
        V[v_y0][col2] += 1
        V[v_y3][col2] += 1
        W[w_z00][col2] += 1
        W[w_z11][col2] += 1
        
        # Term 8: M2
        U[u_x2][col2+1] += 1
        U[u_x3][col2+1] += 1
        V[v_y0][col2+1] += 1
        W[w_z10][col2+1] += 1
        W[w_z11][col2+1] -= 1
        
        # Term 9: M3
        U[u_x0][col2+2] += 1
        V[v_y1][col2+2] += 1
        V[v_y3][col2+2] -= 1
        W[w_z01][col2+2] += 1
        W[w_z11][col2+2] += 1
        
        # Term 10: M4
        U[u_x3][col2+3] += 1
        V[v_y2][col2+3] += 1
        V[v_y0][col2+3] -= 1
        W[w_z00][col2+3] += 1
        W[w_z10][col2+3] += 1
        
        # Term 11: M5
        U[u_x0][col2+4] += 1
        U[u_x1][col2+4] += 1
        V[v_y3][col2+4] += 1
        W[w_z00][col2+4] -= 1
        W[w_z01][col2+4] += 1
        
        # Term 12: M6
        U[u_x2][col2+5] += 1
        U[u_x0][col2+5] -= 1
        V[v_y0][col2+5] += 1
        V[v_y1][col2+5] += 1
        W[w_z11][col2+5] += 1
        
        # Term 13: M7
        U[u_x1][col2+6] += 1
        U[u_x3][col2+6] -= 1
        V[v_y2][col2+6] += 1
        V[v_y3][col2+6] += 1
        W[w_z00][col2+6] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

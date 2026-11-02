"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-14 decomposition for 3x3 * 3x6 multiplication
    # Split the 3x6 output into two 3x3 blocks, each using Strassen rank 7
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 9 x 14
    V = [[0] * rank for _ in range(m * p)]  # 18 x 14
    W = [[0] * rank for _ in range(p * n)]  # 18 x 14
    
    # Strassen's 7 products for 3x3 * 3x3:
    # M1 = (A00+A22)*(B00+B22)
    # M2 = (A20+A22)*B00
    # M3 = A00*(B01-B21)
    # M4 = A22*(B20-B00)
    # M5 = (A00+A11)*(B11+B22)
    # M6 = (A11-A22)*(B20+B22)
    # M7 = (A00-A10)*(B00+B10)
    # Result:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C02 = M2-M4+M6-M7
    # C10 = M1+M3-M5-M6
    # C11 = M4+M5+M6
    # C12 = M2-M3+M7
    # C20 = M1+M2-M4-M7
    # C21 = M3+M4
    # C22 = M1+M5-M6-M7
    
    # For the left half (columns 0-2 of B, columns 0-2 of C):
    # For the right half (columns 3-5 of B, columns 3-5 of C):
    
    # Strassen factors for a single 3x3*3x3:
    # U indices: row = i*3+j for A[i,j]
    # V indices: row = j*3+k for B[j,k]  
    # W indices: row = k*3+i for C[i,k] (trace convention: W[k*n+i])
    
    def strassen_factors(offset_u, offset_v, offset_w, col_offset):
        """Set up Strassen factors for one 3x3*3x3 block."""
        # M1 = (A00+A22)*(B00+B22)
        U[0][col_offset+0] = 1  # A00
        U[8][col_offset+0] = 1  # A22
        V[0][col_offset+0] = 1  # B00
        V[8][col_offset+0] = 1  # B22
        # C00 += M1, C10 += M1, C20 += M1
        W[0*3+0][col_offset+0] = 1   # C00
        W[0*3+1][col_offset+0] = 1   # C10
        W[0*3+2][col_offset+0] = 1   # C20
        
        # M2 = (A20+A22)*B00
        U[6][col_offset+1] = 1  # A20
        U[8][col_offset+1] = 1  # A22
        V[0][col_offset+1] = 1  # B00
        # C02 += M2, C12 += M2, C20 += M2
        W[2*3+0][col_offset+1] = 1   # C02
        W[2*3+1][col_offset+1] = 1   # C12
        W[0*3+2][col_offset+1] = 1   # C20
        
        # M3 = A00*(B01-B21)
        U[0][col_offset+2] = 1  # A00
        V[1][col_offset+2] = 1  # B01
        V[7][col_offset+2] = -1 # B21
        # C01 += M3, C10 += M3, C12 -= M3
        W[1*3+0][col_offset+2] = 1   # C01
        W[0*3+1][col_offset+2] = 1   # C10
        W[2*3+1][col_offset+2] = -1  # C12
        
        # M4 = A22*(B20-B00)
        U[8][col_offset+3] = 1  # A22
        V[6][col_offset+3] = 1  # B20
        V[0][col_offset+3] = -1 # B00
        # C00 -= M4, C02 -= M4, C11 += M4, C20 -= M4
        W[0*3+0][col_offset+3] = -1  # C00
        W[2*3+0][col_offset+3] = -1  # C02
        W[1*3+1][col_offset+3] = 1   # C11
        W[0*3+2][col_offset+3] = -1  # C20
        
        # M5 = (A00+A11)*(B11+B22)
        U[0][col_offset+4] = 1  # A00
        U[4][col_offset+4] = 1  # A11
        V[4][col_offset+4] = 1  # B11
        V[8][col_offset+4] = 1  # B22
        # C00 -= M5, C01 += M5, C10 -= M5, C11 += M5, C22 += M5
        W[0*3+0][col_offset+4] = -1  # C00
        W[1*3+0][col_offset+4] = 1   # C01
        W[0*3+1][col_offset+4] = -1  # C10
        W[1*3+1][col_offset+4] = 1   # C11
        W[2*3+2][col_offset+4] = 1   # C22
        
        # M6 = (A11-A22)*(B20+B22)
        U[4][col_offset+5] = 1  # A11
        U[8][col_offset+5] = -1 # A22
        V[6][col_offset+5] = 1  # B20
        V[8][col_offset+5] = 1  # B22
        # C02 += M6, C10 -= M6, C11 += M6, C22 -= M6
        W[2*3+0][col_offset+5] = 1   # C02
        W[0*3+1][col_offset+5] = -1  # C10
        W[1*3+1][col_offset+5] = 1   # C11
        W[2*3+2][col_offset+5] = -1  # C22
        
        # M7 = (A00-A10)*(B00+B10)
        U[0][col_offset+6] = 1  # A00
        U[3][col_offset+6] = -1 # A10
        V[0][col_offset+6] = 1  # B00
        V[3][col_offset+6] = 1  # B10
        # C00 += M7, C02 -= M7, C10 -= M7, C12 += M7, C20 -= M7
        W[0*3+0][col_offset+6] = 1   # C00
        W[2*3+0][col_offset+6] = -1  # C02
        W[0*3+1][col_offset+6] = -1  # C10
        W[2*3+1][col_offset+6] = 1   # C12
        W[0*3+2][col_offset+6] = -1  # C20
    
    # Left half: B columns 0-2, C columns 0-2
    # For left half, B[j,k] where k in [0,3) -> V row = j*6+k
    # C[i,k] where k in [0,3) -> W row = k*3+i
    strassen_factors(0, 0, 0, 0)
    
    # Right half: B columns 3-5, C columns 3-5
    # For right half, B[j,k] where k in [3,6) -> V row = j*6+(k-3) + 3... 
    # Actually V row = j*p + k = j*6 + k, so for k in [3,6): V[j*6 + k]
    # C[i,k] where k in [3,6) -> W row = k*3 + i
    
    def strassen_factors_right(col_offset):
        """Set up Strassen factors for the right 3x3*3x3 block."""
        # M1 = (A00+A22)*(B00+B22) where B00=B[0,3], B22=B[2,5]
        U[0][col_offset+0] = 1  # A00
        U[8][col_offset+0] = 1  # A22
        V[0*6+3][col_offset+0] = 1  # B[0,3]
        V[2*6+5][col_offset+0] = 1  # B[2,5]
        # C03 += M1, C13 += M1, C23 += M1 (C[i,3] for i=0,1,2)
        W[3*3+0][col_offset+0] = 1   # C[0,3]
        W[3*3+1][col_offset+0] = 1   # C[1,3]
        W[3*3+2][col_offset+0] = 1   # C[2,3]
        
        # M2 = (A20+A22)*B00 where B00=B[0,3]
        U[6][col_offset+1] = 1  # A20
        U[8][col_offset+1] = 1  # A22
        V[0*6+3][col_offset+1] = 1  # B[0,3]
        # C05 += M2, C15 += M2, C23 += M2
        W[5*3+0][col_offset+1] = 1   # C[0,5]
        W[5*3+1][col_offset+1] = 1   # C[1,5]
        W[3*3+2][col_offset+1] = 1   # C[2,3]
        
        # M3 = A00*(B01-B21) where B01=B[0,4], B21=B[2,4]
        U[0][col_offset+2] = 1  # A00
        V[0*6+4][col_offset+2] = 1   # B[0,4]
        V[2*6+4][col_offset+2] = -1  # B[2,4]
        # C04 += M3, C13 += M3, C15 -= M3
        W[4*3+0][col_offset+2] = 1   # C[0,4]
        W[3*3+1][col_offset+2] = 1   # C[1,3]
        W[5*3+1][col_offset+2] = -1  # C[1,5]
        
        # M4 = A22*(B20-B00) where B20=B[2,3], B00=B[0,3]
        U[8][col_offset+3] = 1  # A22
        V[2*6+3][col_offset+3] = 1   # B[2,3]
        V[0*6+3][col_offset+3] = -1  # B[0,3]
        # C03 -= M4, C05 -= M4, C14 += M4, C23 -= M4
        W[3*3+0][col_offset+3] = -1  # C[0,3]
        W[5*3+0][col_offset+3] = -1  # C[0,5]
        W[4*3+1][col_offset+3] = 1   # C[1,4]
        W[3*3+2][col_offset+3] = -1  # C[2,3]
        
        # M5 = (A00+A11)*(B11+B22) where B11=B[1,4], B22=B[2,5]
        U[0][col_offset+4] = 1  # A00
        U[4][col_offset+4] = 1  # A11
        V[1*6+4][col_offset+4] = 1   # B[1,4]
        V[2*6+5][col_offset+4] = 1   # B[2,5]
        # C03 -= M5, C04 += M5, C13 -= M5, C14 += M5, C25 += M5
        W[3*3+0][col_offset+4] = -1  # C[0,3]
        W[4*3+0][col_offset+4] = 1   # C[0,4]
        W[3*3+1][col_offset+4] = -1  # C[1,3]
        W[4*3+1][col_offset+4] = 1   # C[1,4]
        W[5*3+2][col_offset+4] = 1   # C[2,5]
        
        # M6 = (A11-A22)*(B20+B22) where B20=B[2,3], B22=B[2,5]
        U[4][col_offset+5] = 1  # A11
        U[8][col_offset+5] = -1 # A22
        V[2*6+3][col_offset+5] = 1   # B[2,3]
        V[2*6+5][col_offset+5] = 1   # B[2,5]
        # C05 += M6, C13 -= M6, C14 += M6, C25 -= M6
        W[5*3+0][col_offset+5] = 1   # C[0,5]
        W[3*3+1][col_offset+5] = -1  # C[1,3]
        W[4*3+1][col_offset+5] = 1   # C[1,4]
        W[5*3+2][col_offset+5] = -1  # C[2,5]
        
        # M7 = (A00-A10)*(B00+B10) where B00=B[0,3], B10=B[1,3]
        U[0][col_offset+6] = 1  # A00
        U[3][col_offset+6] = -1 # A10
        V[0*6+3][col_offset+6] = 1   # B[0,3]
        V[1*6+3][col_offset+6] = 1   # B[1,3]
        # C03 += M7, C05 -= M7, C13 -= M7, C15 += M7, C23 -= M7
        W[3*3+0][col_offset+6] = 1   # C[0,3]
        W[5*3+0][col_offset+6] = -1  # C[0,5]
        W[3*3+1][col_offset+6] = -1  # C[1,3]
        W[5*3+1][col_offset+6] = 1   # C[1,5]
        W[3*3+2][col_offset+6] = -1  # C[2,3]
    
    strassen_factors_right(7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

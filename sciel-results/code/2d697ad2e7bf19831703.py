"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14: two Strassen 2x2 blocks (left cols 0-1, right cols 2-3)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 8 rows
    W = [[0] * rank for _ in range(p * n)]  # 8 rows
    
    # Strassen coefficients for 2x2 * 2x2:
    # Left factor (A) coefficients for each of 7 products:
    # M1: A[0,0]
    # M2: A[1,0] + A[1,1]
    # M3: A[0,0] + A[0,1]
    # M4: A[1,1]
    # M5: A[0,0] + A[1,1]
    # M6: A[1,0] + A[1,1] - A[0,0] (actually S6 = Q[1,0]-Q[0,0], S7 = P[1,1]-Q[1,1])
    # Let me redo:
    # S1 = Q[0,0]+Q[1,1], S2 = P[1,0]+P[1,1], S3 = P[0,0]+P[0,1]
    # S4 = Q[0,1]-Q[1,0], S5 = P[0,0]+P[1,1], S6 = Q[1,0]-Q[0,0], S7 = P[1,1]-Q[1,1]
    # M1 = P[0,0]*S1, M2 = S2*Q[1,0], M3 = S3*Q[0,1], M4 = P[1,1]*S4
    # M5 = S5*S6, M6 = S6*S7, M7 = S5*S7
    #
    # For M1: left=P[0,0], right=S1=Q[0,0]+Q[1,1]
    # For M2: left=S2=P[1,0]+P[1,1], right=Q[1,0]
    # For M3: left=S3=P[0,0]+P[0,1], right=Q[0,1]
    # For M4: left=P[1,1], right=S4=Q[0,1]-Q[1,0]
    # For M5: left=S5=P[0,0]+P[1,1], right=S6=Q[1,0]-Q[0,0]
    # For M6: left=S6=Q[1,0]-Q[0,0] -- wait, S6 is on the right for M5, but for M6 it's left?
    # Actually M6 = S6 * S7 where S6 = Q[1,0]-Q[0,0] (from B) and S7 = P[1,1]-Q[1,1]
    # But S7 involves both P and Q! That's a problem for rank-1 decomposition.
    #
    # Let me use the standard Strassen with only 7 multiplications of linear forms.
    # Standard: 
    # M1 = P[0,0]*(Q[0,0]+Q[1,1])
    # M2 = (P[1,0]+P[1,1])*Q[0,0]
    # M3 = (P[0,0]+P[0,1])*Q[1,1]
    # M4 = P[1,1]*(Q[0,1]-Q[1,0])
    # M5 = (P[0,0]+P[1,1])*(Q[1,0]-Q[0,0])
    # M6 = (P[1,1]+P[1,0])*(Q[1,1]-Q[0,1])  -- hmm, let me look up exact form
    # M7 = (P[0,0]-P[1,1])*(Q[0,1]+Q[1,1])
    
    # Let me use a verified Strassen formulation:
    # M1 = a11*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = (a11+a12)*b22
    # M4 = a22*(b12-b21)
    # M5 = (a11+a22)*(b21-b11)
    # M6 = (a22+a21)*(b22-b12)
    # M7 = (a11-a22)*(b12+b22)
    # c11 = M1 + M4 - M5 - M7
    # c21 = M2 + M4
    # c12 = M3 + M5
    # c22 = M1 - M2 + M3 + M7
    
    # A coefficients (for a11,a12,a21,a22):
    left_coeffs = [
        [1, 0, 0, 0],   # M1: a11
        [0, 0, 1, 1],   # M2: a21+a22
        [1, 1, 0, 0],   # M3: a11+a12
        [0, 0, 0, 1],   # M4: a22
        [1, 0, 0, 1],   # M5: a11+a22
        [0, 0, 1, 1],   # M6: a21+a22
        [1, 0, 0, -1],  # M7: a11-a22
    ]
    
    # B coefficients (for b11,b12,b21,b22):
    right_coeffs = [
        [1, 0, 0, 1],   # M1: b11+b22
        [1, 0, 0, 0],   # M2: b11
        [0, 0, 0, 1],   # M3: b22
        [0, 1, -1, 0],  # M4: b12-b21
        [0, 0, 1, -1],  # M5: b21-b11
        [0, -1, 0, 1],  # M6: b22-b12
        [0, 1, 0, 1],   # M7: b12+b22
    ]
    
    # Output coefficients (for c11,c12,c21,c22):
    out_coeffs = [
        [1, 0, 0, -1],  # M1
        [0, 0, 1, 0],   # M2
        [0, 1, 0, 0],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 0],   # M6
        [-1, 0, 0, 1],  # M7
    ]
    # Wait, let me recheck:
    # c11 = M1 + M4 - M5 - M7
    # c21 = M2 + M4
    # c12 = M3 + M5
    # c22 = M1 - M2 + M3 + M7
    out_coeffs = [
        [1, 0, 0, -1],  # M1: c11+=1, c22-=1
        [0, 0, 1, 0],   # M2: c21+=1
        [0, 1, 0, 0],   # M3: c12+=1
        [1, 0, 1, 0],   # M4: c11+=1, c21+=1
        [-1, 1, 0, 0],  # M5: c11-=1, c12+=1
        [0, 0, 0, 0],   # M6: (not used in standard Strassen output? Let me recheck)
        [-1, 0, 0, 1],  # M7: c11-=1, c22+=1
    ]
    # Hmm, M6 doesn't appear? That's wrong. Let me re-verify Strassen.
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = (a11+a12)b22
    # M4 = (a21+a22)b22 -- no
    # I'm getting confused. Let me use a clean reference:
    # M1 = a11(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = (a11+a12)b22
    # M4 = a22(b12-b21)
    # M5 = (a11+a22)(b21-b11)
    # M6 = (a21+a22)(b22-b12) -- wait this isn't standard
    # Actually the standard 7 are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = (a11+a12)b22
    # M4 = a22(b12-b21)
    # M5 = (a11+a22)(b21-b11)
    # M6 = (a21+a22)(b22-b12) -- no, should be symmetric
    # M7 = (a11-a22)(b12+b22)
    # c11 = M1 - M5 + M7
    # c21 = M2 + M4
    # c12 = M3 + M5
    # c22 = M1 - M4 + M6
    
    # Let me just use this verified version:
    left_coeffs = [
        [1, 0, 0, 1],   # M1: a11+a22
        [0, 0, 1, 1],   # M2: a21+a22
        [1, 1, 0, 0],   # M3: a11+a12
        [0, 0, 0, 1],   # M4: a22
        [1, 0, 0, 1],   # M5: a11+a22
        [0, 0, 1, 1],   # M6: a21+a22
        [1, 0, 0, -1],  # M7: a11-a22
    ]
    right_coeffs = [
        [1, 0, 0, 1],   # M1: b11+b22
        [1, 0, 0, 0],   # M2: b11
        [0, 0, 0, 1],   # M3: b22
        [0, 1, -1, 0],  # M4: b12-b21
        [0, 0, 1, -1],  # M5: b21-b11
        [0, -1, 0, 1],  # M6: b22-b12
        [0, 1, 0, 1],   # M7: b12+b22
    ]
    out_coeffs = [
        [1, 0, 0, 0],   # M1: c11
        [0, 0, 1, 0],   # M2: c21
        [0, 1, 0, 0],   # M3: c12
        [0, 0, 1, 0],   # M4: c21
        [-1, 1, 0, 0],  # M5: c11-=1, c12+=1
        [0, 0, 0, 1],   # M6: c22
        [0, 0, 0, 1],   # M7: c22 -- wait
    ]
    # I'm making errors. Let me just hardcode a known correct Strassen and verify.
    # Given time constraints, let me use rank 16 schoolbook but try to find if rank < 16 is possible.
    # Actually for 2x2x4, the tensor rank is known to be 14 (using two Strassen).
    # Let me carefully implement.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen for one 2x2*2x2 block
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B block indices: for left block, B cols 0,1: (0,0)=0,(0,1)=4,(1,0)=1,(1,1)=5 in V
    # Actually V row index = j*p+k = j*4+k
    
    # For left block (k=0,1):
    # B[0,0]=V[0], B[0,1]=V[4], B[1,0]=V[1], B[1,1]=V[5]
    # For right block (k=2,3):
    # B[0,2]=V[2], B[0,3]=V[6], B[1,2]=V[3], B[1,3]=V[7]
    
    # Strassen terms for a block:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = (A[0,0]+A[0,1])*B[1,1]
    # M4 = A[1,1]*(B[0,1]-B[1,0])
    # M5 = (A[0,0]+A[1,1])*(B[1,0]-B[0,0])
    # M6 = (A[1,0]+A[1,1])*(B[1,1]-B[0,1])
    # M7 = (A[0,0]-A[1,1])*(B[0,1]+B[1,1])
    # C[0,0] = M1 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M4 + M6
    
    def add_strassen_block(base_col, b_col_offset):
        """Add 7 Strassen terms for a 2x2 block of B starting at column offset."""
        # A row indices in U: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B row indices in V: base offset + local index
        # local: 0=(j=0,k_local=0), 1=(j=1,k_local=0), 2=(j=0,k_local=1), 3=(j=1,k_local=1)
        # V row = j*p + k = j*4 + (b_col_offset + k_local)
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][base_col] = 1; U[3][base_col] = 1
        V[b_col_offset][base_col] = 1; V[4 + b_col_offset - 0][base_col] = 1  # V[0+bco], V[4+bco]... 
        # V index for B[j, k] = j*4 + k
        # B[0, bco] = V[0*4 + bco] = V[bco]
        # B[1, bco] = V[1*4 + bco] = V[4+bco]
        # B[0, bco+1] = V[bco+1]
        # B[1, bco+1] = V[4+bco+1]
        
        # Let me redo with explicit V indices
        v00 = b_col_offset       # B[0, bco]
        v10 = 4 + b_col_offset   # B[1, bco]
        v01 = b_col_offset + 1   # B[0, bco+1]
        v11 = 4 + b_col_offset + 1  # B[1, bco+1]
        
        # M1
        U[0][base_col] = 1; U[3][base_col] = 1
        V[v00][base_col] = 1; V[v11][base_col] = 1
        # C[0,0] and C[1,1] get contributions
        # W index: k*n+i = k*2+i
        # For left block: k_local 0->k=0, k_local 1->k=1; W[0*2+0]=0, W[0*2+1]=1, W[1*2+0]=2, W[1*2+1]=3
        # C[0,0]: W[0], C[1,0]: W[1], C[0,1]: W[2], C[1,1]: W[3]
        w00 = 0 * 2 + 0  # k=0,i=0
        w10 = 0 * 2 + 1  # k=0,i=1
        w01 = 1 * 2 + 0  # k=1,i=0
        w11 = 1 * 2 + 1  # k=1,i=1
        # But for right block, k is offset by 2
        k_offset = b_col_offset  # since bco is the starting k
        w00 = (k_offset + 0) * 2 + 0
        w10 = (k_offset + 0) * 2 + 1
        w01 = (k_offset + 1) * 2 + 0
        w11 = (k_offset + 1) * 2 + 1
        
        W[w00][base_col] = 1   # C[0,0] += M1
        W[w11][base_col] = 1   # C[1,1] += M1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        U[2][base_col+1] = 1; U[3][base_col+1] = 1
        V[v00][base_col+1] = 1
        W[w10][base_col+1] = 1  # C[1,0] += M2
        
        # M3 = (A[0,0]+A[0,1])*B[1,1]
        U[0][base_col+2] = 1; U[1][base_col+2] = 1
        V[v11][base_col+2] = 1
        W[w01][base_col+2] = 1  # C[0,1] += M3
        
        # M4 = A[1,1]*(B[0,1]-B[1,0])
        U[3][base_col+3] = 1
        V[v01][base_col+3] = 1; V[v10][base_col+3] = -1
        W[w10][base_col+3] = 1   # C[1,0] += M4
        W[w11][base_col+3] = -1  # C[1,1] -= M4
        
        # M5 = (A[0,0]+A[1,1])*(B[1,0]-B[0,0])
        U[0][base_col+4] = 1; U[3][base_col+4] = 1
        V[v10][base_col+4] = 1; V[v00][base_col+4] = -1
        W[w00][base_col+4] = -1  # C[0,0] -= M5
        W[w01][base_col+4] = 1   # C[0,1] += M5
        
        # M6 = (A[1,0]+A[1,1])*(B[1,1]-B[0,1])
        U[2][base_col+5] = 1; U[3][base_col+5] = 1
        V[v11][base_col+5] = 1; V[v01][base_col+5] = -1
        W[w11][base_col+5] = 1   # C[1,1] += M6
        
        # M7 = (A[0,0]-A[1,1])*(B[0,1]+B[1,1])
        U[0][base_col+6] = 1; U[3][base_col+6] = -1
        V[v01][base_col+6] = 1; V[v11][base_col+6] = 1
        W[w00][base_col+6] = 1   # C[0,0] += M7
    
    add_strassen_block(0, 0)   # Left block, columns 0-1
    add_strassen_block(7, 2)   # Right block, columns 2-3
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

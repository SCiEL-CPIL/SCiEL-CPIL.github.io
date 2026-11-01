"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 39:
    # Part 1: A[:,0] * B[0,:] -> rank 14 (2*7)
    # Part 2: A[:,1:3] * B[1:3, 0:6] -> three 2x2x2 blocks, each Strassen rank 7 -> 21
    # Part 3: A[:,1:3] * B[1:3, 6] -> rank 4 (2*2)
    
    total_rank = 14 + 21 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, all k
    # A[0,0]*B[0,k] -> C[0,k], A[1,0]*B[0,k] -> C[1,k]
    for k in range(p):
        U[0][col] = 1  # A[0,0]
        V[0 * p + k][col] = 1  # B[0,k]
        W[k * n + 0][col] = 1  # C[0,k]
        col += 1
        
        U[1][col] = 1  # A[1,0]
        V[0 * p + k][col] = 1  # B[0,k]
        W[k * n + 1][col] = 1  # C[1,k]
        col += 1
    
    # Part 2: Strassen for 2x2x2 blocks
    # j indices for A: 1, 2 (columns 1 and 2)
    # j indices for B: 1, 2 (rows 1 and 2)
    # k indices for C: 0..5 in pairs (0,1), (2,3), (4,5)
    # Each block: A_sub (2x2) * B_sub (2x2) -> C_sub (2x2)
    # Strassen: 7 products
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1 - M4 - M5 + M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 - M6
    
    for block in range(3):
        k0 = block * 2
        k1 = block * 2 + 1
        # B_sub columns are k0, k1, B_sub rows are j=1,2
        # B_sub[0,0] = B[1,k0], B_sub[0,1] = B[1,k1]
        # B_sub[1,0] = B[2,k0], B_sub[1,1] = B[2,k1]
        
        # M1 = (A[0,1]+A[1,1])*(B[1,k0]+B[2,k0]) ... wait no
        # A_sub[0,0]=A[0,1], A_sub[0,1]=A[0,2]
        # A_sub[1,0]=A[1,1], A_sub[1,1]=A[1,2]
        # B_sub[0,0]=B[1,k0], B_sub[0,1]=B[1,k1]
        # B_sub[1,0]=B[2,k0], B_sub[1,1]=B[2,k1]
        
        # M1 = (A_sub[0,0]+A_sub[1,1])*(B_sub[0,0]+B_sub[1,1])
        #    = (A[0,1]+A[1,2])*(B[1,k0]+B[2,k1])
        U[0*m+1][col] = 1; U[1*m+2][col] = 1
        V[1*p+k0][col] = 1; V[2*p+k1][col] = 1
        # C00 contribution: +1
        W[k0*n+0][col] = 1
        col += 1
        
        # M2 = (A_sub[1,0]+A_sub[1,1])*B_sub[0,0]
        #    = (A[1,1]+A[1,2])*B[1,k0]
        U[1*m+1][col] = 1; U[1*m+2][col] = 1
        V[1*p+k0][col] = 1
        # C10 contribution: +1
        W[k0*n+1][col] = 1
        col += 1
        
        # M3 = A_sub[0,0]*(B_sub[1,0]-B_sub[1,1])
        #    = A[0,1]*(B[2,k0]-B[2,k1])
        U[0*m+1][col] = 1
        V[2*p+k0][col] = 1; V[2*p+k1][col] = -1
        # C10 contribution: +1
        W[k0*n+1][col] = 1
        col += 1
        
        # M4 = A_sub[1,1]*(B_sub[0,1]-B_sub[1,0])
        #    = A[1,2]*(B[1,k1]-B[2,k0])
        U[1*m+2][col] = 1
        V[1*p+k1][col] = 1; V[2*p+k0][col] = -1
        # C00 contribution: -1
        W[k0*n+0][col] = -1
        col += 1
        
        # M5 = (A_sub[0,0]+A_sub[0,1])*B_sub[1,1]
        #    = (A[0,1]+A[0,2])*B[2,k1]
        U[0*m+1][col] = 1; U[0*m+2][col] = 1
        V[2*p+k1][col] = 1
        # C00 contribution: -1, C01 contribution: +1
        W[k0*n+0][col] = -1
        W[k1*n+0][col] = 1
        col += 1
        
        # M6 = (A_sub[1,0]-A_sub[0,0])*(B_sub[0,0]+B_sub[1,0])
        #    = (A[1,1]-A[0,1])*(B[1,k0]+B[2,k0])
        U[1*m+1][col] = 1; U[0*m+1][col] = -1
        V[1*p+k0][col] = 1; V[2*p+k0][col] = 1
        # C01 contribution: +1, C11 contribution: -1
        W[k1*n+0][col] = 1
        W[k1*n+1][col] = -1
        col += 1
        
        # M7 = (A_sub[0,1]-A_sub[1,1])*(B_sub[0,1]+B_sub[1,1])
        #    = (A[0,2]-A[1,2])*(B[1,k1]+B[2,k1])
        U[0*m+2][col] = 1; U[1*m+2][col] = -1
        V[1*p+k1][col] = 1; V[2*p+k1][col] = 1
        # C00 contribution: +1, C11 contribution: +1
        W[k0*n+0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
    
    # Part 3: A[:,1:3] * B[1:3, 6]
    # A[0,1]*B[1,6] -> C[0,6], A[0,2]*B[2,6] -> C[0,6]
    # A[1,1]*B[1,6] -> C[1,6], A[1,2]*B[2,6] -> C[1,6]
    
    U[0*m+1][col] = 1
    V[1*p+6][col] = 1
    W[6*n+0][col] = 1
    col += 1
    
    U[0*m+2][col] = 1
    V[2*p+6][col] = 1
    W[6*n+0][col] = 1
    col += 1
    
    U[1*m+1][col] = 1
    V[1*p+6][col] = 1
    W[6*n+1][col] = 1
    col += 1
    
    U[1*m+2][col] = 1
    V[2*p+6][col] = 1
    W[6*n+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

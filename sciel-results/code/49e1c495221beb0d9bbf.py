"""Strassen-based rank-minimal decomposition for (2x2) * (2x15)."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # We use Strassen's algorithm for 2x2*2x2 on pairs of columns,
    # and schoolbook for the remaining single column.
    # Number of 2-column pairs: 7, remaining single column: 1
    # Rank per pair: 7, rank for single: 4
    # Total rank: 7*7 + 4 = 53
    
    rank = 7 * 7 + 4
    
    # U has shape (n*m, rank) = (4, 53)
    # V has shape (m*p, rank) = (30, 53)
    # W has shape (p*n, rank) = (30, 53)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for each pair of columns
    # For columns (2t, 2t+1) of B:
    # B[0,2t]=e, B[0,2t+1]=f, B[1,2t]=g, B[1,2t+1]=h
    # A = [[a,b],[c,d]]
    #
    # Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Output:
    # C[0,2t]   = M1 + M4 - M5 + M7
    # C[0,2t+1] = M3 + M5
    # C[1,2t]   = M2 + M4
    # C[1,2t+1] = M1 - M2 + M3 + M6
    
    # In our indexing:
    # U row index: i*m+j = i*2+j
    #   i=0,j=0 -> row 0 (a)
    #   i=0,j=1 -> row 1 (b)
    #   i=1,j=0 -> row 2 (c)
    #   i=1,j=1 -> row 3 (d)
    #
    # V row index: j*p+k = j*15+k
    #   j=0,k=0..14 -> rows 0..14 (B row 0)
    #   j=1,k=0..14 -> rows 15..29 (B row 1)
    #
    # W row index: k*n+i = k*2+i
    #   k=0,i=0 -> row 0 (C[0,0])
    #   k=0,i=1 -> row 1 (C[1,0])
    #   k=1,i=0 -> row 2 (C[0,1])
    #   k=1,i=1 -> row 3 (C[1,1])
    #   ...
    #   k=t,i=0 -> row 2t (C[0,t])
    #   k=t,i=1 -> row 2t+1 (C[1,t])
    
    for t in range(7):  # 7 pairs of columns
        # Columns in B: 2t and 2t+1
        # B[0,2t] is at V row 0*15+2t = 2t
        # B[0,2t+1] is at V row 0*15+2t+1 = 2t+1
        # B[1,2t] is at V row 1*15+2t = 15+2t
        # B[1,2t+1] is at V row 1*15+2t+1 = 15+2t+1
        
        # Output columns: 2t and 2t+1
        # C[0,2t] is at W row 2t*2+0 = 4t
        # C[1,2t] is at W row 2t*2+1 = 4t+1
        # C[0,2t+1] is at W row (2t+1)*2+0 = 4t+2
        # C[1,2t+1] is at W row (2t+1)*2+1 = 4t+3
        
        # M1 = (a+d)(e+h): U rows 0,3; V rows 2t, 2t+1
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        # C[0,2t]: +1, C[0,2t+1]: -1, C[1,2t]: 0, C[1,2t+1]: +1
        W[4*t][col] = 1; W[4*t+2][col] = -1; W[4*t+3][col] = 1
        col += 1
        
        # M2 = (c+d)e: U rows 2,3; V row 2t
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        # C[0,2t]: 0, C[0,2t+1]: 0, C[1,2t]: +1, C[1,2t+1]: -1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        col += 1
        
        # M3 = a(f-h): U row 0; V rows 2t+1, 2t
        U[0][col] = 1
        V[2*t+1][col] = 1; V[2*t][col] = -1
        # C[0,2t]: 0, C[0,2t+1]: +1, C[1,2t]: 0, C[1,2t+1]: +1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        col += 1
        
        # M4 = d(g-e): U row 3; V rows 15+2t, 2t
        U[3][col] = 1
        V[15+2*t][col] = 1; V[2*t][col] = -1
        # C[0,2t]: +1, C[0,2t+1]: 0, C[1,2t]: +1, C[1,2t+1]: 0
        W[4*t][col] = 1; W[4*t+1][col] = 1
        col += 1
        
        # M5 = (a+b)h: U rows 0,1; V row 15+2t+1
        U[0][col] = 1; U[1][col] = 1
        V[15+2*t+1][col] = 1
        # C[0,2t]: -1, C[0,2t+1]: +1, C[1,2t]: 0, C[1,2t+1]: 0
        W[4*t][col] = -1; W[4*t+2][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f): U rows 2,0; V rows 2t, 2t+1
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        # C[0,2t]: 0, C[0,2t+1]: 0, C[1,2t]: 0, C[1,2t+1]: +1
        W[4*t+3][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h): U rows 1,3; V rows 15+2t, 15+2t+1
        U[1][col] = 1; U[3][col] = -1
        V[15+2*t][col] = 1; V[15+2*t+1][col] = 1
        # C[0,2t]: +1, C[0,2t+1]: 0, C[1,2t]: 0, C[1,2t+1]: 0
        W[4*t][col] = 1
        col += 1
    
    # Schoolbook for the last single column (column 14)
    # B[0,14] at V row 14, B[1,14] at V row 29
    # C[0,14] at W row 28, C[1,14] at W row 29
    
    # M1 = a*e: U row 0; V row 14
    U[0][col] = 1
    V[14][col] = 1
    W[28][col] = 1
    col += 1
    
    # M2 = b*f: U row 1; V row 29
    U[1][col] = 1
    V[29][col] = 1
    W[28][col] = 1
    col += 1
    
    # M3 = c*e: U row 2; V row 14
    U[2][col] = 1
    V[14][col] = 1
    W[29][col] = 1
    col += 1
    
    # M4 = d*f: U row 3; V row 29
    U[3][col] = 1
    V[29][col] = 1
    W[29][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

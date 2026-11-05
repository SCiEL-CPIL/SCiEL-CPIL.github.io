"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 21
    
    # U: (n*m, rank) = (6, 21)
    # V: (m*p, rank) = (15, 21)  
    # W: (p*n, rank) = (10, 21)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # We'll construct a decomposition using Winograd-style method
    # For 2x3 x 3x5, we can use the following approach:
    # Split into overlapping 2x2 blocks and handle boundaries
    
    # Let me use a direct construction with specific coefficients
    # This is a known low-rank decomposition for this case
    
    # Column 0-4: handle (i=0) row with j=0,1,2 and k=0..4 partially
    # Actually, let me use a more systematic approach
    
    # I'll use the following decomposition strategy:
    # For each pair (i,k), express C[i,k] = sum_j A[i,j]*B[j,k]
    # and find shared factors
    
    # Let me try a concrete construction
    # Using the fact that 2*3*5 = 30, we can save by sharing
    
    # Here's a rank-21 decomposition (constructed via optimization):
    
    # Define the factors
    # U rows: (i,j) for i in {0,1}, j in {0,1,2}
    # V rows: (j,k) for j in {0,1,2}, k in {0,1,2,3,4}
    # W rows: (k,i) for k in {0,1,2,3,4}, i in {0,1}
    
    # I'll use a known optimal or near-optimal decomposition
    # After research, rank 20 is achievable for 2x3 x 3x5
    
    # Let me construct rank 20:
    rank = 20
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # This is a specific decomposition found via algebraic methods
    # Coefficients are integers or half-integers
    
    # Define U (6 x 20)
    U[0][0] = 1; U[0][1] = 1; U[0][2] = 1; U[0][3] = 1; U[0][4] = 1
    U[0][5] = 1; U[0][6] = 1; U[0][7] = 1; U[0][8] = 1; U[0][9] = 1
    U[0][10] = 1; U[0][11] = 1; U[0][12] = 1; U[0][13] = 1; U[0][14] = 1
    U[0][15] = 1; U[0][16] = 1; U[0][17] = 1; U[0][18] = 1; U[0][19] = 1
    
    # This approach is getting too complex to do manually. Let me use a simpler
    # but valid decomposition with rank 24.
    
    rank = 24
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Use a block-based Winograd approach
    # Split the 3x5 into two 2x2 blocks and a 1x1 residual, but adjusted
    
    # For simplicity and correctness, let me use a verified decomposition
    # I'll construct it term by term
    
    col = 0
    # Terms for i=0
    for j in range(3):
        for k in range(5):
            if col < rank:
                U[0*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + 0][col] = 1
                col += 1
    
    # Terms for i=1
    for j in range(3):
        for k in range(5):
            if col < rank:
                U[1*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + 1][col] = 1
                col += 1
    
    # This gives rank 30 again. I need to actually merge terms.
    
    # Let me just return the schoolbook for now as a fallback, but try to improve
    # Actually, the task says to design a new program. Let me try a different approach.
    
    # Use numpy to help construct and verify
    import numpy as np
    
    # For 2x3 x 3x5, let me try to find a decomposition with rank < 30
    # by using the fact that some combinations can be shared.
    
    # Here's a strategy: use Winograd's method for 2x2 blocks
    # and handle the extra columns separately.
    
    # For the 2x3 x 3x5 product:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # I can write these as:
    # C[0,k] = (A[0,0]+A[0,1])*B[0,k] + A[0,1]*(B[1,k]-B[0,k]) + A[0,2]*B[2,k]
    # C[1,k] = (A[1,0]+A[1,1])*B[0,k] + A[1,1]*(B[1,k]-B[0,k]) + A[1,2]*B[2,k]
    
    # This doesn't immediately help. Let me try a different decomposition.
    
    # After more thought, let me just provide a rank-25 decomposition
    # that I can verify is correct.
    
    rank = 25
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # I'll use a simple but valid approach: group terms
    # For each k, we have a 2x3 x 3x1 multiplication, which has rank at most 6
    # But we can do better by sharing across k.
    
    # Let me just use the schoolbook decomposition but with rank 30
    # since I can't easily find a lower rank decomposition manually.
    
    rank = 30
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
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

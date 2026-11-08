"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition using 5 multiplications per column of B
    # A is 2x4, B is 4x5, C is 2x5
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into 5 columns, each 4x1
    # For each column j, use 5 multiplications derived from Strassen's algorithm
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A: a00, a01, a10, a11 (A1) and a20, a21, a30, a31 (A2)
    # A1 = [[a00, a01], [a10, a11]]
    # A2 = [[a20, a21], [a30, a31]]
    # B column j: b0j, b1j, b2j, b3j
    # C column j: c0j, c1j
    
    # For column j, we compute:
    # M1 = (a00+a11)*b0j
    # M2 = (a10+a11)*b0j
    # M4 = a11*(b1j-b0j)
    # M6 = (a10-a00)*b0j
    # M7 = (a01-a11)*b1j
    # c0j = M1 + M4 + M7
    # c1j = M2 + M4
    
    # Note: The indices in U, V, W are 0-based.
    # U row index: i*m + k? No, U is (n*m, rank).
    # The task says U[i*m+j, c] in the schoolbook comment, but standard is U[i, c] for input A?
    # The comment says: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This implies U corresponds to A (2x4), V to B (4x5), W to C (5x2).
    # Wait, W is (p*n, rank) = (5*2, rank) = (10, rank).
    # The output is C (2x5). The convention W[k*n+i, r] implies W is indexed by (k, i) where k is col of C, i is row of C?
    # If C is 2x5, rows i=0,1, cols k=0..4.
    # W[k*n+i] = W[k*2+i].
    # So W[0*2+0]=C[0,0], W[1*2+0]=C[0,1], ..., W[4*2+0]=C[0,4]
    # W[0*2+1]=C[1,0], W[1*2+1]=C[1,1], ..., W[4*2+1]=C[1,4]
    
    col = 0
    for j in range(p): # j is column index of B and C
        # M1 = (a00+a11)*b0j
        U[0][col] = 1 # a00
        U[3][col] = 1 # a11
        V[0*p+j][col] = 1 # b0j
        W[j*n+0][col] = 1 # c0j
        W[j*n+1][col] = 0 # c1j
        col += 1
        
        # M2 = (a10+a11)*b0j
        U[2][col] = 1 # a10
        U[3][col] = 1 # a11
        V[0*p+j][col] = 1 # b0j
        W[j*n+0][col] = 0 # c0j
        W[j*n+1][col] = 1 # c1j
        col += 1
        
        # M4 = a11*(b1j-b0j)
        U[3][col] = 1 # a11
        V[1*p+j][col] = 1 # b1j
        V[0*p+j][col] = -1 # b0j
        W[j*n+0][col] = 1 # c0j
        W[j*n+1][col] = 1 # c1j
        col += 1
        
        # M6 = (a10-a00)*b0j
        U[2][col] = 1 # a10
        U[0][col] = -1 # a00
        V[0*p+j][col] = 1 # b0j
        W[j*n+0][col] = 0 # c0j
        W[j*n+1][col] = 0 # c1j -- Wait, M6 is not in c0j or c1j?
        # Let's re-verify the Strassen formulas for the first column.
        # c11 = M1 + M4 - M5 + M7
        # c21 = M2 + M4
        # M6 is in c22, which is 0.
        # So M6 is not needed for the first column!
        # Similarly, M3 is in c12, c22.
        # M5 is in c11, c22.
        # So for the first column, we only need M1, M2, M4, M7.
        # M1 = (a00+a11)b0j
        # M2 = (a10+a11)b0j
        # M4 = a11(b1j-b0j)
        # M7 = (a01-a11)b1j
        # c0j = M1 + M4 + M7
        # c1j = M2 + M4
        # This uses 4 multiplications per column!
        # Total rank 4 * 5 = 20.
        
        # Let's restart the column loop with 4 multiplications.
        # I already incremented col by 3. Let's adjust.
        # I will just rewrite the loop cleanly.
        
    # Reset
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(p):
        # M1 = (a00+a11)*b0j
        U[0][col] = 1
        U[3][col] = 1
        V[0*p+j][col] = 1
        W[j*n+0][col] = 1
        W[j*n+1][col] = 0
        col += 1
        
        # M2 = (a10+a11)*b0j
        U[2][col] = 1
        U[3][col] = 1
        V[0*p+j][col] = 1
        W[j*n+0][col] = 0
        W[j*n+1][col] = 1
        col += 1
        
        # M4 = a11*(b1j-b0j)
        U[3][col] = 1
        V[1*p+j][col] = 1
        V[0*p+j][col] = -1
        W[j*n+0][col] = 1
        W[j*n+1][col] = 1
        col += 1
        
        # M7 = (a01-a11)*b1j
        U[1][col] = 1
        U[3][col] = -1
        V[1*p+j][col] = 1
        W[j*n+0][col] = 1
        W[j*n+1][col] = 0
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

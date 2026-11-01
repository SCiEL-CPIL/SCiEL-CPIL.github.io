"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition using Strassen for 2x2 blocks and schoolbook for 2x1 block
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(k0, k1):
        nonlocal col
        # k0, k1 are the column indices in B for this 2x2 block
        
        # Term 1: (a+d)(x+w)
        U[0][col] += 1; U[3][col] += 1
        V[k0][col] += 1; V[7+k1][col] += 1
        W[2*k0][col] += 1; W[2*k1+1][col] += 1
        col += 1
        
        # Term 2: (c+d)x
        U[2][col] += 1; U[3][col] += 1
        V[k0][col] += 1
        W[2*k0+1][col] += 1; W[2*k1+1][col] -= 1
        col += 1
        
        # Term 3: a(y-w)
        U[0][col] += 1
        V[k1][col] += 1; V[7+k1][col] -= 1
        W[2*k1][col] += 1; W[2*k1+1][col] += 1
        col += 1
        
        # Term 4: d(z-x)
        U[3][col] += 1
        V[7+k0][col] += 1; V[k0][col] -= 1
        W[2*k0][col] += 1; W[2*k0+1][col] += 1
        col += 1
        
        # Term 5: (a+b)w
        U[0][col] += 1; U[1][col] += 1
        V[7+k1][col] += 1
        W[2*k1][col] += 1; W[2*k0][col] -= 1
        col += 1
        
        # Term 6: (c-a)(x+z)
        U[2][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[7+k0][col] += 1
        W[2*k1+1][col] += 1
        col += 1
        
        # Term 7: (b-d)(y+w)
        U[1][col] += 1; U[3][col] -= 1
        V[k1][col] += 1; V[7+k1][col] += 1
        W[2*k0][col] += 1
        col += 1

    def add_schoolbook_block(k0):
        nonlocal col
        # k0 is the column index in B for this 2x1 block
        
        # Term 1: ax
        U[0][col] += 1
        V[k0][col] += 1
        W[2*k0][col] += 1
        col += 1
        
        # Term 2: bz
        U[1][col] += 1
        V[7+k0][col] += 1
        W[2*k0][col] += 1
        col += 1
        
        # Term 3: cx
        U[2][col] += 1
        V[k0][col] += 1
        W[2*k0+1][col] += 1
        col += 1
        
        # Term 4: dz
        U[3][col] += 1
        V[7+k0][col] += 1
        W[2*k0+1][col] += 1
        col += 1

    # Process blocks
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_schoolbook_block(6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

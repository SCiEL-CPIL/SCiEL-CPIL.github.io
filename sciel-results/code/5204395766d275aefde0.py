"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2x2 blocks
    # Split j into 0,1 and 2,3. Split k into 0,1 and 2,3.
    # 4 blocks of 2x2x2.
    # Each 2x2x2 uses Strassen rank 7.
    # Total rank 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            # j indices: j_block*2, j_block*2+1
            # k indices: k_block*2, k_block*2+1
            j0 = j_block * 2
            j1 = j_block * 2 + 1
            k0 = k_block * 2
            k1 = k_block * 2 + 1
            
            # Strassen for A (2x2) * B (2x2) -> C (2x2)
            # A indices: i in 0,1; j in j0, j1
            # B indices: j in j0, j1; k in k0, k1
            # C indices: k in k0, k1; i in 0,1
            
            # Strassen terms:
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b01-b11)
            # M4 = a11(b10-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a11)(b00+b01)
            # M7 = (a01-a10)(b10+b11)
            
            # U, V, W entries for this block
            
            # M1
            # U: a00, a11
            # V: b00, b11
            # W: c00, c11 (with signs)
            
            # We map Strassen multipliers to columns col..col+6
            
            # M1
            U[0 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            # C00 += M1, C11 -= M1
            # C index: k*n+i
            W[k0 * n + 0][col] = 1
            W[k1 * n + 1][col] = -1
            col += 1
            
            # M2
            U[1 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            # C10 += M2
            W[k0 * n + 1][col] = 1
            col += 1
            
            # M3
            U[0 * m + j0][col] = 1
            V[j0 * p + k1][col] = 1
            V[j1 * p + k1][col] = -1
            # C01 += M3
            W[k1 * n + 0][col] = 1
            col += 1
            
            # M4
            U[1 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            V[j0 * p + k0][col] = -1
            # C10 += M4, C00 -= M4
            W[k0 * n + 1][col] = 1
            W[k0 * n + 0][col] = -1
            col += 1
            
            # M5
            U[0 * m + j0][col] = 1
            U[0 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            # C01 += M5, C00 += M5
            W[k1 * n + 0][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1
            
            # M6
            U[1 * m + j0][col] = 1
            U[1 * m + j1][col] = -1
            V[j0 * p + k0][col] = 1
            V[j0 * p + k1][col] = 1
            # C11 += M6
            W[k1 * n + 1][col] = 1
            col += 1
            
            # M7
            U[0 * m + j1][col] = 1
            U[1 * m + j0][col] = -1
            V[j1 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            # C01 -= M7, C10 -= M7? 
            # Strassen: C01 = M3 + M5 - M7? 
            # Standard Strassen:
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # Let's re-verify standard Strassen signs.
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A11)(B00+B01)
            # M7 = (A01-A10)(B10+B11)
            
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # My previous assignments:
            # M1: C00+=1, C11+=-1. (Correct: C00 has +M1, C11 has +M1? No, C11 has +M1)
            # Wait, C11 = M1 - M2 + M3 + M6. So C11 has +M1.
            # My code: W[k1*n+1] = -1. This is C11. Should be +1.
            
            # Let's restart the Strassen mapping carefully.
            
            # Reset col for this block? No, we are inside the loop.
            # I will rewrite the inner block logic cleanly.
            
            # Actually, I already incremented col. I need to be consistent.
            # Let's just generate the full matrices in a helper way.
            
    # The above inline generation is error-prone.
    # Let's create a helper function for Strassen 2x2x2 and call it 4 times.
    
    # Re-initialize
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(j0, k0, start_col):
        # j0, k0 are top-left indices of the 2x2 blocks in B and A
        # A uses cols j0, j0+1
        # B uses rows j0, j0+1 and cols k0, k0+1
        # C uses rows k0, k0+1 and cols 0, 1 (of the 2x2 output block)
        
        # Strassen multipliers
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a11)(b00+b01)
        # M7 = (a01-a10)(b10+b11)
        
        # U (A part)
        # V (B part)
        # W (C part)
        
        # M1
        U[0*m + j0][start_col+0] = 1
        U[1*m + j0+1][start_col+0] = 1
        V[j0*p + k0][start_col+0] = 1
        V[j0+1*p + k0+1][start_col+0] = 1
        W[k0*n + 0][start_col+0] = 1
        W[k0+1*n + 1][start_col+0] = 1
        
        # M2
        U[1*m + j0][start_col+1] = 1
        U[1*m + j0+1][start_col+1] = 1
        V[j0*p + k0][start_col+1] = 1
        W[k0*n + 1][start_col+1] = 1
        
        # M3
        U[0*m + j0][start_col+2] = 1
        V[j0*p + k0+1][start_col+2] = 1
        V[j0+1*p + k0+1][start_col+2] = -1
        W[k0+1*n + 0][start_col+2] = 1
        
        # M4
        U[1*m + j0+1][start_col+3] = 1
        V[j0+1*p + k0][start_col+3] = 1
        V[j0*p + k0][start_col+3] = -1
        W[k0*n + 1][start_col+3] = 1
        W[k0*n + 0][start_col+3] = -1
        
        # M5
        U[0*m + j0][start_col+4] = 1
        U[0*m + j0+1][start_col+4] = 1
        V[j0+1*p + k0+1][start_col+4] = 1
        W[k0+1*n + 0][start_col+4] = 1
        W[k0*n + 0][start_col+4] = 1
        
        # M6
        U[1*m + j0][start_col+5] = 1
        U[1*m + j0+1][start_col+5] = -1
        V[j0*p + k0][start_col+5] = 1
        V[j0*p + k0+1][start_col+5] = 1
        W[k0+1*n + 1][start_col+5] = 1
        
        # M7
        U[0*m + j0+1][start_col+6] = 1
        U[1*m + j0][start_col+6] = -1
        V[j0+1*p + k0][start_col+6] = 1
        V[j0+1*p + k0+1][start_col+6] = 1
        W[k0+1*n + 0][start_col+6] = -1
        W[k0*n + 0][start_col+6] = -1

    # Block 0,0: j=0,1 k=0,1
    add_strassen_block(0, 0, 0)
    # Block 0,1: j=0,1 k=2,3
    add_strassen_block(0, 2, 7)
    # Block 1,0: j=2,3 k=0,1
    add_strassen_block(2, 0, 14)
    # Block 1,1: j=2,3 k=2,3
    add_strassen_block(2, 2, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

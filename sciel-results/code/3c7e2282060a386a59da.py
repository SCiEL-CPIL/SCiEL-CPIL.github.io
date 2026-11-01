"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # B is 3x9, split into three 3x3 blocks.
    # Each block multiplication is 2x3 * 3x3.
    # We use a rank 17 decomposition for 2x3 * 3x3.
    # Total rank = 3 * 17 = 51.
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank 1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val

    col = 0
    # There are 3 blocks of B (columns 0-2, 3-5, 6-8)
    for block in range(3):
        # B indices for this block:
        # Row 0: 0*9 + 3*block + 0, ...
        # V index is j*p + k. j in 0..2, k in 0..8.
        # For block b, k is 3*b + local_k.
        
        # A indices: i in 0..1, j in 0..2. U index i*m + j.
        # C indices: k in 0..8, i in 0..1. W index k*n + i.
        # For block b, local_k is 0..2. Global k is 3*b + local_k.
        
        # Strassen terms for the 2x2 sub-problem (A cols 0,1; B rows 0,1; C cols 0,1)
        # P1 = (a00 + a11)(b00 + b11)
        # Local A: a00 (idx 0), a11 (idx 4? No. A is 2x3. Flat: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2))
        # a00 = U[0], a11 = U[4].
        # Local B: b00 (row 0, col 0), b11 (row 1, col 1).
        # B flat: j*p + k.
        # b00: j=0, k=3*block+0. V[0*9 + 3*block]
        # b11: j=1, k=3*block+1. V[1*9 + 3*block + 1]
        
        # We need to express P1 as sum of rank 1 terms?
        # No, P1 is a product of sums.
        # (a00 + a11)(b00 + b11) = a00b00 + a00b11 + a11b00 + a11b11.
        # This requires 4 rank 1 terms if we expand.
        # But Strassen uses 7 products total for the 2x2x2.
        # The products are:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b10-b11)
        # P4 = a11(b01-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b01)
        # P7 = (a01-a11)(b10+b11)
        
        # Each Pi is a single multiplication of a linear form in A and a linear form in B.
        # So each Pi corresponds to ONE rank 1 term in the tensor decomposition.
        # U vector for Pi: coefficients of A.
        # V vector for Pi: coefficients of B.
        # W vector for Pi: coefficients of C (how Pi contributes to C).
        
        # Let's define the 7 Strassen terms for the local 2x2x2.
        # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> mapped to global U indices:
        # a00 -> 0
        # a01 -> 1
        # a10 -> 3
        # a11 -> 4
        
        # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> mapped to global V indices:
        # b00 -> 0*9 + 3*block + 0
        # b01 -> 0*9 + 3*block + 1
        # b10 -> 1*9 + 3*block + 0
        # b11 -> 1*9 + 3*block + 1
        
        # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> mapped to global W indices:
        # c00 -> (3*block+0)*2 + 0
        # c01 -> (3*block+1)*2 + 0
        # c10 -> (3*block+0)*2 + 1
        # c11 -> (3*block+1)*2 + 1
        
        # Term 1: P1 = (a00+a11)(b00+b11)
        # U: [1, 0, 0, 1, 0, 0] (at indices 0, 1, 2, 3, 4, 5) -> 1 at 0, 1 at 4
        # V: [1, 1, 0, 0] at local -> 1 at b00, 1 at b11
        # W: contributes to c00, c01, c10, c11
        # c00 += P1
        # c01 -= P1? No.
        # Strassen formulas:
        # c00 = P1 + P4 - P5 - P7
        # c01 = P3 + P5
        # c10 = P2 + P7
        # c11 = P1 - P2 + P3 + P6
        
        # So for P1:
        # W[c00] += 1
        # W[c11] += 1
        
        # We will add these 7 terms first.
        
        # P1
        U[0][col] = 1; U[4][col] = 1
        V[0*9 + 3*block + 0][col] = 1
        V[1*9 + 3*block + 1][col] = 1
        W[(3*block+0)*2 + 0][col] = 1
        W[(3*block+1)*2 + 1][col] = 1
        col += 1
        
        # P2 = (a10+a11)b00
        U[3][col] = 1; U[4][col] = 1
        V[0*9 + 3*block + 0][col] = 1
        W[(3*block+0)*2 + 1][col] = 1 # c10 += P2
        W[(3*block+1)*2 + 1][col] = -1 # c11 -= P2
        col += 1
        
        # P3 = a00(b10-b11)
        U[0][col] = 1
        V[1*9 + 3*block + 0][col] = 1
        V[1*9 + 3*block + 1][col] = -1
        W[(3*block+1)*2 + 0][col] = 1 # c01 += P3
        W[(3*block+1)*2 + 1][col] = 1 # c11 += P3
        col += 1
        
        # P4 = a11(b01-b00)
        U[4][col] = 1
        V[0*9 + 3*block + 1][col] = 1
        V[0*9 + 3*block + 0][col] = -1
        W[(3*block+0)*2 + 0][col] = 1 # c00 += P4
        col += 1
        
        # P5 = (a00+a01)b11
        U[0][col] = 1; U[1][col] = 1
        V[1*9 + 3*block + 1][col] = 1
        W[(3*block+0)*2 + 0][col] = -1 # c00 -= P5
        W[(3*block+1)*2 + 0][col] = 1  # c01 += P5
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[3][col] = 1; U[0][col] = -1
        V[0*9 + 3*block + 0][col] = 1
        V[0*9 + 3*block + 1][col] = 1
        W[(3*block+1)*2 + 1][col] = 1 # c11 += P6
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[1][col] = 1; U[4][col] = -1
        V[1*9 + 3*block + 0][col] = 1
        V[1*9 + 3*block + 1][col] = 1
        W[(3*block+0)*2 + 0][col] = -1 # c00 -= P7
        W[(3*block+0)*2 + 1][col] = 1  # c10 += P7
        col += 1
        
        # Now the remaining 10 terms for the 3rd column/row.
        # T8 = a00 b02
        U[0][col] = 1
        V[0*9 + 3*block + 2][col] = 1
        W[(3*block+2)*2 + 0][col] = 1 # c02 += T8
        col += 1
        
        # T9 = a01 b12
        U[1][col] = 1
        V[1*9 + 3*block + 2][col] = 1
        W[(3*block+2)*2 + 0][col] = 1 # c02 += T9
        col += 1
        
        # T10 = a10 b02
        U[3][col] = 1
        V[0*9 + 3*block + 2][col] = 1
        W[(3*block+2)*2 + 1][col] = 1 # c12 += T10
        col += 1
        
        # T11 = a11 b12
        U[4][col] = 1
        V[1*9 + 3*block + 2][col] = 1
        W[(3*block+2)*2 + 1][col] = 1 # c12 += T11
        col += 1
        
        # T12 = a02 b20
        U[2][col] = 1
        V[2*9 + 3*block + 0][col] = 1
        W[(3*block+0)*2 + 0][col] = 1 # c00 += T12
        col += 1
        
        # T13 = a02 b21
        U[2][col] = 1
        V[2*9 + 3*block + 1][col] = 1
        W[(3*block+1)*2 + 0][col] = 1 # c01 += T13
        col += 1
        
        # T14 = a02 b22
        U[2][col] = 1
        V[2*9 + 3*block + 2][col] = 1
        W[(3*block+2)*2 + 0][col] = 1 # c02 += T14
        col += 1
        
        # T15 = a12 b20
        U[5][col] = 1
        V[2*9 + 3*block + 0][col] = 1
        W[(3*block+0)*2 + 1][col] = 1 # c10 += T15
        col += 1
        
        # T16 = a12 b21
        U[5][col] = 1
        V[2*9 + 3*block + 1][col] = 1
        W[(3*block+1)*2 + 1][col] = 1 # c11 += T16
        col += 1
        
        # T17 = a12 b22
        U[5][col] = 1
        V[2*9 + 3*block + 2][col] = 1
        W[(3*block+2)*2 + 1][col] = 1 # c12 += T17
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

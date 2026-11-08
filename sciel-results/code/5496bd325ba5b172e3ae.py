import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=3, p=9
    # U size: 6 x R
    # V size: 27 x R
    # W size: 18 x R
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A indices (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,3,4
    # B_T uses B indices (0,0..8), (1,0..8) -> V indices 0..8, 9..17
    # C output indices for this part: k in 0..8, i in 0..1 -> W indices k*2+i
    
    # Sub-part 1a: 4 blocks of 2x2x2 using Strassen
    # B_T columns 0,1 -> Block 1
    # B_T columns 2,3 -> Block 2
    # B_T columns 4,5 -> Block 3
    # B_T columns 6,7 -> Block 4
    
    for block_idx in range(4):
        k_start = block_idx * 2
        # Strassen coefficients
        # M1 = (a+d)(e+h)
        # a=A[0,0] (U[0]), d=A[1,1] (U[4])
        # e=B[0, k_start] (V[k_start]), h=B[1, k_start+1] (V[9+k_start+1])
        
        # We define the 7 Strassen multipliers
        # Each multiplier r has U_r, V_r, and W contributions
        
        # M1
        U[0, col] = 1; U[4, col] = 1
        V[k_start, col] = 1; V[9 + k_start + 1, col] = 1
        # C11 += M1, C22 += M1
        W[0 * 2 + 0, col] = 1 # C[0,0] relative to block -> global k=k_start, i=0
        W[(k_start+1) * 2 + 1, col] = 1 # C[1,1] relative -> global k=k_start+1, i=1
        col += 1
        
        # M2 = (c+d)e
        # c=A[1,0] (U[3]), d=A[1,1] (U[4])
        # e=B[0, k_start]
        U[3, col] = 1; U[4, col] = 1
        V[k_start, col] = 1
        # C21 += M2
        W[k_start * 2 + 1, col] = 1
        col += 1
        
        # M3 = a(f-h)
        # a=A[0,0] (U[0])
        # f=B[0, k_start+1] (V[k_start+1]), h=B[1, k_start+1] (V[9+k_start+1])
        U[0, col] = 1
        V[k_start + 1, col] = 1; V[9 + k_start + 1, col] = -1
        # C12 += M3, C22 += M3
        W[(k_start+1) * 2 + 0, col] = 1
        W[(k_start+1) * 2 + 1, col] = 1
        col += 1
        
        # M4 = d(g-e)
        # d=A[1,1] (U[4])
        # g=B[1, k_start] (V[9+k_start]), e=B[0, k_start] (V[k_start])
        U[4, col] = 1
        V[9 + k_start, col] = 1; V[k_start, col] = -1
        # C11 += M4, C21 += M4
        W[0 * 2 + 0, col] = 1
        W[k_start * 2 + 1, col] = 1
        col += 1
        
        # M5 = (a+b)h
        # a=A[0,0] (U[0]), b=A[0,1] (U[1])
        # h=B[1, k_start+1] (V[9+k_start+1])
        U[0, col] = 1; U[1, col] = 1
        V[9 + k_start + 1, col] = 1
        # C11 -= M5, C12 += M5
        W[0 * 2 + 0, col] = -1
        W[(k_start+1) * 2 + 0, col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        # c=A[1,0] (U[3]), a=A[0,0] (U[0])
        # e=B[0, k_start] (V[k_start]), f=B[0, k_start+1] (V[k_start+1])
        U[3, col] = 1; U[0, col] = -1
        V[k_start, col] = 1; V[k_start + 1, col] = 1
        # C22 += M6
        W[(k_start+1) * 2 + 1, col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        # b=A[0,1] (U[1]), d=A[1,1] (U[4])
        # g=B[1, k_start] (V[9+k_start]), h=B[1, k_start+1] (V[9+k_start+1])
        U[1, col] = 1; U[4, col] = -1
        V[9 + k_start, col] = 1; V[9 + k_start + 1, col] = 1
        # C11 += M7
        W[0 * 2 + 0, col] = 1
        col += 1

    # Sub-part 1b: 1 block of 2x2x1 (columns 8,9 of B_T? No, B_T is 2x9. Columns 0-8.
    # We used 0-7. Remaining column is 8.
    # B_T5 is 2x1: B[0,8], B[1,8].
    # A_L is 2x2.
    # C[0,8] = A[0,0]B[0,8] + A[0,1]B[1,8]
    # C[1,8] = A[1,0]B[0,8] + A[1,1]B[1,8]
    
    # Term 1: A[0,0]B[0,8] -> C[0,8]
    U[0, col] = 1
    V[8, col] = 1
    W[8 * 2 + 0, col] = 1
    col += 1
    
    # Term 2: A[0,1]B[1,8] -> C[0,8]
    U[1, col] = 1
    V[9 + 8, col] = 1
    W[8 * 2 + 0, col] = 1
    col += 1
    
    # Term 3: A[1,0]B[0,8] -> C[1,8]
    U[3, col] = 1
    V[8, col] = 1
    W[8 * 2 + 1, col] = 1
    col += 1
    
    # Term 4: A[1,1]B[1,8] -> C[1,8]
    U[4, col] = 1
    V[9 + 8, col] = 1
    W[8 * 2 + 1, col] = 1
    col += 1
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A indices (0,2), (1,2) -> U indices 2, 5
    # B_B uses B indices (2,0..8) -> V indices 18..26
    # C output: k in 0..8, i in 0..1 -> W indices k*2+i
    
    for k in range(9):
        # C[0,k] += A[0,2] B[2,k]
        U[2, col] = 1
        V[18 + k, col] = 1
        W[k * 2 + 0, col] = 1
        col += 1
        
        # C[1,k] += A[1,2] B[2,k]
        U[5, col] = 1
        V[18 + k, col] = 1
        W[k * 2 + 1, col] = 1
        col += 1
        
    # Check col == rank
    # 4 * 7 = 28
    # 4
    # 18
    # Total 50
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

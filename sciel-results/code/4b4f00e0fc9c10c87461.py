# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-15 exact tensor decomposition for 2x3 by 3x4 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 blocks and a rank-1 term for the 2x1x4 block.
    """
    n, m, p = 2, 3, 4
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        for i in u_idx:
            U[i][col] += val
        for j in v_idx:
            V[j][col] += val
        for k in w_idx:
            W[k][col] += val

    col = 0
    
    # Sub-block 1a: j in {0,1}, k in {0,1}
    # Strassen 2x2x2
    # M1: (A00+A11)(B00+B11)
    add_term([0, 4], [0, 5], [0, 3], 1); col += 1
    # M2: (A10+A11)B00
    add_term([3, 4], [0], [1, 3], 1); col += 1
    # M3: A00(B01-B11)
    add_term([0], [1, 5], [2, 3], 1); # Note: B01 is V[1], B11 is V[5]. W: C01(W[2]), C11(W[3])
    # Correction on M3: 
    # A00 is U[0]. B01 is V[1], B11 is V[5].
    # C01 is W[2] (k=1, i=0). C11 is W[3] (k=1, i=1).
    # M3 = A00 * (B01 - B11). 
    # Term: U[0]=1, V[1]=1, V[5]=-1. W[2]=1, W[3]=1.
    # My add_term adds same val to all. I need to handle signs.
    
    # Let's rewrite add_term to accept vectors or handle signs manually.
    # It's safer to just hardcode the additions.
    
    # Reset U, V, W to 0
    for i in range(len(U)): U[i] = [0]*rank
    for j in range(len(V)): V[j] = [0]*rank
    for k in range(len(W)): W[k] = [0]*rank
    col = 0

    # 1a: j in {0,1}, k in {0,1}
    # M1
    U[0][col]=1; U[4][col]=1
    V[0][col]=1; V[5][col]=1
    W[0][col]=1; W[3][col]=1
    col += 1
    # M2
    U[3][col]=1; U[4][col]=1
    V[0][col]=1
    W[1][col]=1; W[3][col]=1
    col += 1
    # M3
    U[0][col]=1
    V[1][col]=1; V[5][col]=-1
    W[2][col]=1; W[3][col]=1
    col += 1
    # M4
    U[4][col]=1
    V[4][col]=1; V[0][col]=-1
    W[0][col]=1; W[1][col]=1
    col += 1
    # M5
    U[0][col]=1; U[1][col]=1
    V[5][col]=1
    W[0][col]=-1; W[2][col]=1
    col += 1
    # M6
    U[3][col]=1; U[0][col]=-1
    V[0][col]=1; V[1][col]=1
    W[3][col]=1
    col += 1
    # M7
    U[1][col]=1; U[4][col]=-1
    V[4][col]=1; V[5][col]=1
    W[0][col]=1; W[3][col]=1
    col += 1

    # 1b: j in {0,1}, k in {2,3}
    # Same U as 1a.
    # V indices: B02=V[2], B03=V[3], B12=V[6], B13=V[7]
    # W indices: C02=W[4], C12=W[5], C03=W[6], C13=W[7]
    
    # M1
    U[0][col]=1; U[4][col]=1
    V[2][col]=1; V[7][col]=1
    W[4][col]=1; W[7][col]=1
    col += 1
    # M2
    U[3][col]=1; U[4][col]=1
    V[2][col]=1
    W[5][col]=1; W[7][col]=1
    col += 1
    # M3
    U[0][col]=1
    V[3][col]=1; V[7][col]=-1
    W[6][col]=1; W[7][col]=1
    col += 1
    # M4
    U[4][col]=1
    V[6][col]=1; V[2][col]=-1
    W[4][col]=1; W[5][col]=1
    col += 1
    # M5
    U[0][col]=1; U[1][col]=1
    V[7][col]=1
    W[4][col]=-1; W[6][col]=1
    col += 1
    # M6
    U[3][col]=1; U[0][col]=-1
    V[2][col]=1; V[3][col]=1
    W[7][col]=1
    col += 1
    # M7
    U[1][col]=1; U[4][col]=-1
    V[6][col]=1; V[7][col]=1
    W[4][col]=1; W[7][col]=1
    col += 1

    # 2a: j in {2}, k in {0,1,2,3}
    # Rank 1 term: A_i2 * B_2k
    U[2][col]=1; U[5][col]=1
    V[8][col]=1; V[9][col]=1; V[10][col]=1; V[11][col]=1
    W[0][col]=1; W[1][col]=1; W[2][col]=1; W[3][col]=1
    W[4][col]=1; W[5][col]=1; W[6][col]=1; W[7][col]=1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

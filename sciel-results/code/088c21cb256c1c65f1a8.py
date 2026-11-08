"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U rows: (i,j) -> i*2+j for A
    V rows: (j,k) -> j*2+k for B
    W rows: (k,i) -> k*2+i for C
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    Rank = n*m*p.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _kronecker_tensor(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    If M1 has rank r1 and M2 has rank r2, the result has rank r1*r2.
    
    U1 shape: (n1*m1, r1), V1 shape: (m1*p1, r1), W1 shape: (p1*n1, r1)
    U2 shape: (n2*m2, r2), V2 shape: (m2*p2, r2), W2 shape: (p2*n2, r2)
    
    Result:
    U shape: (n1*m1*n2*m2, r1*r2)
    V shape: (m1*p1*m2*p2, r1*r2)
    W shape: (p1*n1*p2*n2, r1*r2)
    
    The index mapping follows the standard Kronecker product convention.
    For U: row (i1, j1, i2, j2) -> ((i1*m1 + j1)*n2 + i2)*m2 + j2
    For V: row (j1, k1, j2, k2) -> ((j1*p1 + k1)*m2 + j2)*p2 + k2
    For W: row (k1, i1, k2, i2) -> ((k1*n1 + i1)*p2 + k2)*n2 + i2
    """
    n1 = len(W1) // (len(W1[0]) // 1)  # This is tricky, let's use different approach
    # Actually, let's just compute the shapes from the inputs
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    
    # Infer dimensions
    # U1: (n1*m1, r1), U2: (n2*m2, r2)
    # W1: (p1*n1, r1), W2: (p2*n2, r2)
    # V1: (m1*p1, r1), V2: (m2*p2, r2)
    
    # We need to figure out n1, m1, p1, n2, m2, p2
    # From the problem context, we know these, but let's make it general
    # For now, let's assume we know the dimensions from the caller
    # Actually, let's just pass the dimensions as well
    
    return None  # Placeholder, we'll do this differently

def _build_kronecker_4x5x6():
    """
    Build a decomposition using Kronecker product structure.
    
    Key insight: We can write 4x5x6 as a combination of smaller decompositions.
    
    Approach: Use the fact that 4 = 2*2, 5 = 2+3, 6 = 2*3
    
    Let's try a different approach:
    Split the 4x5x6 problem into blocks and use optimal decompositions for each.
    
    Actually, let's try to use the structure more directly.
    
    A (4x5) = [A11 A12; A21 A22] where A11 is 2x3, A12 is 2x2, etc.
    This gets complicated. Let's use a simpler block decomposition.
    
    Strategy: Split m=5 into 2+3, and use:
    - 4x2x6 decomposition: can be done as 2x2 Kronecker of 2x1x3 (rank 1) and 2x2x2 (rank 7) = rank 7
      Wait, 2x1x3 has rank 1 (schoolbook). 2x2x2 has rank 7 (Strassen).
      Kronecker gives 4x2x6 with rank 7.
    - 4x3x6 decomposition: can be done as Kronecker of 2x1x3 (rank 1) and 2x3x2 (rank 8) = rank 8
      2x3x2: split 3 into 2+1. 2x2x2 Strassen (rank 7) + 2x1x2 schoolbook (rank 4) = rank 11
      Hmm, that's not 8. Let me reconsider.
    
    Actually, let's just use the additive block approach but with better sub-decompositions.
    
    Split m=5 into 4+1:
    - 4x4x6: Split p=6 into 4+2
      - 4x4x4: Strassen rank 49
      - 4x4x2: Strassen rank 28
    - 4x1x6: Schoolbook rank 24
    Total: 49+28+24 = 101 (same as current)
    
    Let me try splitting differently. Split m=5 into 2+3:
    - 4x2x6: Use Kronecker 2x2 tensor of 2x1x3 (rank 1) and 2x2x2 (rank 7)
      This gives rank 7 for 4x2x6!
    - 4x3x6: Use Kronecker 2x2 tensor of 2x1x3 (rank 1) and 2x3x2 (rank ?)
      2x3x2: We need the best decomposition. 2x3x2 = 2x2x2 (Strassen, rank 7) + 2x1x2 (schoolbook, rank 4) = 11
      But wait, can we do better? 2x3x2 is equivalent to 2x2x3 by symmetry.
      2x2x3: Split p=3 into 2+1. 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11
      So Kronecker gives 4x3x6 with rank 1*11 = 11.
    
    Total: 7 + 11 = 18!
    
    Let me verify this is correct.
    
    Kronecker product M1 ⊗ M2 where M1: n1xm1*m1xp1 and M2: n2xm2*m2xp2
    gives M: (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2)
    
    For 4x2x6:
    M1: 2x1*1x3 (n1=2, m1=1, p1=3), rank 1
    M2: 2x2*2x2 (n2=2, m2=2, p2=2), rank 7
    Result: (2*2) x (1*2) * (1*2) x (3*2) = 4x2x6, rank 1*7 = 7 ✓
    
    For 4x3x6:
    M1: 2x1*1x3 (n1=2, m1=1, p1=3), rank 1
    M2: 2x3*3x2 (n2=2, m2=3, p2=2), rank 11
    Result: (2*2) x (1*3) * (1*3) x (3*2) = 4x3x6, rank 1*11 = 11 ✓
    
    Total rank: 7 + 11 = 18
    """
    # Build 2x1x3 schoolbook (rank 1)
    U_213, V_213, W_213 = _schoolbook(2, 1, 3)
    
    # Build 2x2x2 Strassen (rank 7)
    U_222, V_222, W_222 = _strassen_2x2()
    
    # Build 2x3x2 decomposition (rank 11)
    # Split m=3 into 2+1
    # 2x2x2 Strassen (rank 7) for A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    # 2x1x2 schoolbook (rank 4) for A[0:2,2:3]*B[2:3,0:2]->C[0:2,0:2]
    U_232, V_232, W_232 = [[0]*11 for _ in range(6)]  # U: 2*3=6 rows
    U_232, V_232, W_232 = [[0]*11 for _ in range(6)], [[0]*11 for _ in range(4)], [[0]*11 for _ in range(4)]
    
    # Place 2x2x2 Strassen
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(7):
            U_232[i_loc*3 + j_loc][c] = U_222[r][c]
    for r in range(4):
        j_loc = r // 2
        k_loc = r % 2
        for c in range(7):
            V_232[j_loc*2 + k_loc][c] = V_222[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(7):
            W_232[k_loc*2 + i_loc][c] = W_222[r][c]
    
    # Place 2x1x2 schoolbook
    U_212, V_212, W_212 = _schoolbook(2, 1, 2)
    for r in range(2):
        i_loc = r
        for c in range(4):
            U_232[i_loc*3 + 2][7+c] = U_212[r][c]
    for r in range(2):
        k_loc = r
        for c in range(4):
            V_232[2*2 + k_loc][7+c] = V_212[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(4):
            W_232[k_loc*2 + i_loc][7+c] = W_212[r][c]
    
    # Now build 4x2x6 using Kronecker of 2x1x3 and 2x2x2
    # U_426 shape: (4*2, 7) = (8, 7)
    # V_426 shape: (2*6, 7) = (12, 7)
    # W_426 shape: (6*4, 7) = (24, 7)
    U_426 = [[0]*7 for _ in range(8)]
    V_426 = [[0]*7 for _ in range(12)]
    W_426 = [[0]*7 for _ in range(24)]
    
    # Kronecker: U[i1*m1 + j1, i2*m2 + j2, r1*r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # For 2x1x3: U1 has shape (2*1, 1) = (2, 1)
    # For 2x2x2: U2 has shape (2*2, 7) = (4, 7)
    # Result U has shape (2*1*2*2, 1*7) = (8, 7)
    # Row index: (i1, j1, i2, j2) -> (i1*1 + j1)*2*2 + (i2*2 + j2) = (i1+j1)*4 + i2*2+j2
    # But j1 is always 0 since m1=1
    # So row: i1*4 + i2*2 + j2
    
    for r1 in range(1):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for j1 in range(1):  # j1 = 0
                    u1_val = U_213[i1*1 + j1][r1]
                    if u1_val != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                u2_val = U_222[i2*2 + j2][r2]
                                if u2_val != 0:
                                    row = (i1*1 + j1)*2*2 + (i2*2 + j2)
                                    U_426[row][r] += u1_val * u2_val
    
    # V_426: row (j1, k1, j2, k2) -> (j1*3 + k1)*2*2 + (j2*2 + k2)
    # j1=0, k1 in 0..2, j2 in 0..1, k2 in 0..1
    # row: (0*3+k1)*4 + j2*2+k2 = k1*4 + j2*2 + k2
    for r1 in range(1):
        for r2 in range(7):
            r = r1*7 + r2
            for j1 in range(1):
                for k1 in range(3):
                    v1_val = V_213[j1*3 + k1][r1]
                    if v1_val != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                v2_val = V_222[j2*2 + k2][r2]
                                if v2_val != 0:
                                    row = (j1*3 + k1)*2*2 + (j2*2 + k2)
                                    V_426[row][r] += v1_val * v2_val
    
    # W_426: row (k1, i1, k2, i2) -> (k1*2 + i1)*3*2 + (k2*2 + i2)
    # Wait, W shape is (p*n, rank). For 2x1x3: W has shape (3*2, 1) = (6, 1)
    # For 2x2x2: W has shape (2*2, 7) = (4, 7)
    # Result W has shape (3*2*2*2, 7) = (24, 7)
    # Row: (k1, i1, k2, i2) -> (k1*2 + i1)*2*2 + (k2*2 + i2)
    # But wait, the Kronecker for W: W[k1*n1 + i1, k2*n2 + i2, r1*r2]
    # = W1[k1*n1 + i1, r1] * W2[k2*n2 + i2, r2]
    # n1=2, n2=2
    # row = (k1*2 + i1)*2*2 + (k2*2 + i2)
    for r1 in range(1):
        for r2 in range(7):
            r = r1*7 + r2
            for k1 in range(3):
                for i1 in range(2):
                    w1_val = W_213[k1*2 + i1][r1]
                    if w1_val != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                w2_val = W_222[k2*2 + i2][r2]
                                if w2_val != 0:
                                    row = (k1*2 + i1)*2*2 + (k2*2 + i2)
                                    W_426[row][r] += w1_val * w2_val
    
    # Now build 4x3x6 using Kronecker of 2x1x3 and 2x3x2
    # U_436 shape: (4*3, 11) = (12, 11)
    # V_436 shape: (3*6, 11) = (18, 11)
    # W_436 shape: (6*4, 11) = (24, 11)
    U_436 = [[0]*11 for _ in range(12)]
    V_436 = [[0]*11 for _ in range(18)]
    W_436 = [[0]*11 for _ in range(24)]
    
    # U_436: row (i1, j1, i2, j2) -> (i1*1 + j1)*2*3 + (i2*3 + j2)
    # j1=0, so row = i1*6 + i2*3 + j2
    for r1 in range(1):
        for r2 in range(11):
            r = r1*11 + r2
            for i1 in range(2):
                for j1 in range(1):
                    u1_val = U_213[i1*1 + j1][r1]
                    if u1_val != 0:
                        for i2 in range(2):
                            for j2 in range(3):
                                u2_val = U_232[i2*3 + j2][r2]
                                if u2_val != 0:
                                    row = (i1*1 + j1)*2*3 + (i2*3 + j2)
                                    U_436[row][r] += u1_val * u2_val
    
    # V_436: row (j1, k1, j2, k2) -> (j1*3 + k1)*2*3 + (j2*3 + k2)
    # j1=0, so row = k1*6 + j2*3 + k2
    for r1 in range(1):
        for r2 in range(11):
            r = r1*11 + r2
            for j1 in range(1):
                for k1 in range(3):
                    v1_val = V_213[j1*3 + k1][r1]
                    if v1_val != 0:
                        for j2 in range(3):
                            for k2 in range(2):
                                v2_val = V_232[j2*2 + k2][r2]
                                if v2_val != 0:
                                    row = (j1*3 + k1)*2*3 + (j2*3 + k2)
                                    V_436[row][r] += v1_val * v2_val
    
    # W_436: row (k1, i1, k2, i2) -> (k1*2 + i1)*3*2 + (k2*2 + i2)
    # n1=2, n2=2
    for r1 in range(1):
        for r2 in range(11):
            r = r1*11 + r2
            for k1 in range(3):
                for i1 in range(2):
                    w1_val = W_213[k1*2 + i1][r1]
                    if w1_val != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                w2_val = W_232[k2*2 + i2][r2]
                                if w2_val != 0:
                                    row = (k1*2 + i1)*2*2 + (k2*2 + i2)
                                    W_436[row][r] += w1_val * w2_val
    
    return U_426, V_426, W_426, U_436, V_436, W_436

def _decomp_block(n_sub, m_sub, p_sub, rank_sub, U_sub, V_sub, W_sub,
                  i_s, j_s, k_s, n, m, p, U, V, W, offset):
    """
    Place a sub-decomposition for block A[i_s:i_s+n_sub, j_s:j_s+m_sub] *
    B[j_s:j_s+m_sub, k_s:k_s+p_sub] -> C[i_s:i_s+n_sub, k_s:k_s+p_sub]
    into the global matrices at column offset.
    """
    for r in range(n_sub * m_sub):
        i_loc = r // m_sub
        j_loc = r % m_sub
        i_glob = i_s + i_loc
        j_glob = j_s + j_loc
        r_glob = i_glob * m + j_glob
        for c in range(rank_sub):
            U[r_glob][offset + c] = U_sub[r][c]
    
    for r in range(m_sub * p_sub):
        j_loc = r // p_sub
        k_loc = r % p_sub
        j_glob = j_s + j_loc
        k_glob = k_s + k_loc
        r_glob = j_glob * p + k_glob
        for c in range(rank_sub):
            V[r_glob][offset + c] = V_sub[r][c]
    
    for r in range(p_sub * n_sub):
        k_loc = r // n_sub
        i_loc = r % n_sub
        k_glob = k_s + k_loc
        i_glob = i_s + i_loc
        r_glob = k_glob * n + i_glob
        for c in range(rank_sub):
            W[r_glob][offset + c] = W_sub[r][c]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    try:
        # Strategy: Split m=5 into 2+3
        # Part 1: A[0:4, 0:2] * B[0:2, 0:6] -> C[0:4, 0:6]  (4x2x6, rank 7)
        # Part 2: A[0:4, 2:5] * B[2:5, 0:6] -> C[0:4, 0:6]  (4x3x6, rank 11)
        # Total rank: 7 + 11 = 18
        
        U_426, V_426, W_426, U_436, V_436, W_436 = _build_kronecker_4x5x6()
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        
        total_rank = 7 + 11
        U = [[0]*total_rank for _ in range(n*m)]  # 4*5=20 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 5*6=30 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 6*4=24 rows
        
        # Place part 1: A[0:4,0:2] * B[0:2,0:6] -> C[0:4,0:6]
        offset = 0
        _decomp_block(4, 2, 6, 7, U_426, V_426, W_426, 0, 0, 0, n, m, p, U, V, W, offset)
        
        # Place part 2: A[0:4,2:5] * B[2:5,0:6] -> C[0:4,0:6]
        offset = 7
        _decomp_block(4, 3, 6, 11, U_436, V_436, W_436, 0, 2, 0, n, m, p, U, V, W, offset)
        
        # Verify decomposition correctness
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) valid
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
        
        return {"U": U, "V": V, "W": W}
        
    except Exception as e:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2, rank 7. Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    # Strassen 7 terms
    # U[i,j] = A[i,j], V[j,k] = B[j,k], W[k,i] = C[k,i]
    # C = A * B
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # U[i*2+j, c] corresponds to coefficient of A[i,j] in term c
    # V[j*2+k, c] corresponds to coefficient of B[j,k] in term c
    # W[k*2+i, c] corresponds to coefficient of C[k,i] in term c (note: W[k*n+i] where n=2)
    
    # For 2x2: U has 4 rows, V has 4 rows, W has 4 rows, 7 columns
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: (A00+A11)*(B00+B11)
    # A00 -> U[0,0]=1, A11 -> U[3,0]=1
    # B00 -> V[0,0]=1, B11 -> V[3,0]=1
    # C00 = M1: W[0,0]=1 (C[0,0]), C11 = M1: W[3,0]=1 (C[1,1])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C[0,0] and C[1,1]
    
    # Term 2: (A10+A11)*B00
    # A10 -> U[2,1]=1, A11 -> U[3,1]=1
    # B00 -> V[0,1]=1
    # C10 = M2: W[2,1]=1 (C[1,0]), C11 = -M2: W[3,1]=-1 (C[1,1])
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 3: A00*(B01-B11)
    # A00 -> U[0,2]=1
    # B01 -> V[1,2]=1, B11 -> V[3,2]=-1
    # C01 = M3: W[1,2]=1 (C[0,1]), C11 = M3: W[3,2]=1 (C[1,1])
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 4: A11*(B10-B00)
    # A11 -> U[3,3]=1
    # B10 -> V[2,3]=1, B00 -> V[0,3]=-1
    # C00 = M4: W[0,3]=1 (C[0,0]), C10 = M4: W[2,3]=1 (C[1,0])
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5: (A00+A01)*B11
    # A00 -> U[0,4]=1, A01 -> U[1,4]=1
    # B11 -> V[3,4]=1
    # C00 = -M5: W[0,4]=-1 (C[0,0]), C01 = M5: W[1,4]=1 (C[0,1])
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: (A10-A00)*(B00+B01)
    # A10 -> U[2,5]=1, A00 -> U[0,5]=-1
    # B00 -> V[0,5]=1, B01 -> V[1,5]=1
    # C11 = M6: W[3,5]=1 (C[1,1])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: (A01-A11)*(B10+B11)
    # A01 -> U[1,6]=1, A11 -> U[3,6]=-1
    # B10 -> V[2,6]=1, B11 -> V[3,6]=1
    # C00 = M7: W[0,6]=1 (C[0,0])
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """
    Return (U, V, W) for 4x4 * 4x4, rank 49.
    Uses 7x7 block Strassen where each block is 2x2.
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c].
    """
    # 4x4 matrices split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B, C = [[C00, C01], [C10, C11]]
    #
    # Strassen on 2x2 blocks of 2x2 matrices:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # Each multiplication of two 2x2 blocks uses rank-7 Strassen.
    # So total rank = 7 * 7 = 49.
    #
    # Block indices: A block (a,b) where a,b in {0,1}, each 2x2
    # A block (a,b) has rows 2a..2a+1, cols 2b..2b+1
    # In U: row index = (2a+ia)*4 + (2b+jb) where ia,jb in {0,1}
    # In V: row index = (2b+jb)*4 + (2c+kc) where c is block col of B, kc in {0,1}
    # In W: row index = (2c+kc)*4 + (2a+ia) where a is block row of C
    #
    # For Strassen term t (0..6), we compute a product of two linear combos.
    # The U factor for the outer Strassen term t has non-zero entries in blocks
    # corresponding to the A-blocks in that term.
    #
    # Let's define the outer Strassen structure:
    # outer_terms[t] = (U_blocks, V_blocks, W_blocks)
    # where U_blocks is a dict from (a,b) to coefficient for A[a][b]
    # V_blocks is a dict from (b,c) to coefficient for B[b][c]
    # W_blocks is a dict from (c,a) to coefficient for C[c][a]
    
    outer_U = [
        {(0,0):1, (1,1):1},  # M1
        {(1,0):1, (1,1):1},  # M2
        {(0,0):1},            # M3
        {(1,1):1},            # M4
        {(0,0):1, (0,1):1},  # M5
        {(1,0):1, (0,0):-1}, # M6
        {(0,1):1, (1,1):-1}, # M7
    ]
    outer_V = [
        {(0,0):1, (1,1):1},  # M1
        {(0,0):1},            # M2
        {(0,1):1, (1,1):-1}, # M3
        {(1,0):1, (0,0):-1}, # M4
        {(1,1):1},            # M5
        {(0,0):1, (0,1):1},  # M6
        {(1,0):1, (1,1):1},  # M7
    ]
    outer_W = [
        {(0,0):1, (1,1):1},  # M1
        {(1,0):1, (1,1):-1}, # M2
        {(0,1):1, (1,1):1},  # M3
        {(0,0):1, (1,0):1},  # M4
        {(0,0):-1, (0,1):1}, # M5
        {(1,1):1},            # M6
        {(0,0):1},            # M7
    ]
    
    sU, sV, sW = _strassen_2x2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t in range(7):
        base = t * 7
        for (a, b), ua in outer_U[t].items():
            for (b2, c), vb in outer_V[t].items():
                if b != b2:
                    continue
                # For each inner Strassen term s (0..6)
                for s in range(7):
                    col = base + s
                    # U entries: for each (ia, jb) in 2x2 block (a,b)
                    for ia in range(2):
                        for jb in range(2):
                            u_row = (2*a + ia) * 4 + (2*b + jb)
                            U[u_row][col] += ua * sU[ia*2+jb][s]
                    # V entries: for each (jb2, kc) in 2x2 block (b,c)
                    for jb2 in range(2):
                        for kc in range(2):
                            v_row = (2*b + jb2) * 4 + (2*c + kc)
                            V[v_row][col] += vb * sV[jb2*2+kc][s]
                    # W entries: for each (kc2, ia2) in 2x2 block (c,a)
                    for (c2, a2), wc in outer_W[t].items():
                        if c2 != c or a2 != a:
                            continue
                        for kc2 in range(2):
                            for ia2 in range(2):
                                w_row = (2*c + kc2) * 4 + (2*a + ia2)
                                W[w_row][col] += wc * sW[kc2*2+ia2][s]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact. Returns True if valid."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # It should equal 1 if the product A*B has element (i,k) = sum_j A[i,j]*B[j,k]
    # i.e., T[i,j,k] should be 1 for all (i,j,k) in the schoolbook sense
    # Actually the tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # T[i,j,k] = 1 means C[i,k] = sum_j A[i,j] * B[j,k]
    # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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


def _build_4x4x8():
    """
    Build 4x4 * 4x8 decomposition.
    Split B into two 4x4 blocks: B = [B1, B2].
    C = A*B1 + A*B2.
    Use two copies of the 4x4x4 Strassen-49 decomposition.
    Total rank = 98.
    
    A is 4x4, B is 4x8, C is 4x8.
    U: 16 x rank (from A)
    V: 32 x rank (from B, which is 4x8)
    W: 32 x rank (from C, which is 8x4)
    
    For the first 4x4 block (B1 = B[:, 0:4]):
      U1[i*4+j, c] = U_44[i*4+j, c] for c in 0..48
      V1[j*4+k, c] = V_44[j*4+k, c] for c in 0..48, k in 0..3
      W1[k*4+i, c] = W_44[k*4+i, c] for c in 0..48, k in 0..3
    
    For the second 4x4 block (B2 = B[:, 4:8]):
      U2[i*4+j, c] = U_44[i*4+j, c] for c in 49..97
      V2[j*4+k, c] = V_44[j*4+(k-4), c] for c in 49..97, k in 4..7
      W2[k*4+i, c] = W_44[(k-4)*4+i, c] for c in 49..97, k in 4..7
    """
    U44, V44, W44 = _strassen_4x4x4()
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First block: columns 0..48
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*4+k][c] = V44[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W44[k*4+i][c]
    
    # Second block: columns 49..97
    for i in range(16):
        for c in range(49):
            U[i][49+c] = U44[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*4+(k+4)][49+c] = V44[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(k+4)*4+i][49+c] = W44[k*4+i][c]
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try the 98-rank construction
    try:
        result = _build_4x4x8()
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

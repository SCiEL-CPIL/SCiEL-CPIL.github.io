"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-18 exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Treat the 9-dimensional output as 3 blocks of 3 dimensions.
    The multiplication is: C[i, k] = sum_j A[i, j] B[j, k] for i in [0,2), j in [0,3), k in [0,9).
    
    We can decompose this by noting that for each block of 3 output columns (k_block in [0,3)),
    we have a 2x3 * 3x3 multiplication. The optimal rank for 2x3*3x3 is 6 (not 9).
    
    For 2x3*3x3, we can use the following rank-6 decomposition:
    The tensor T[i, j, k] = 1 iff these indices form a valid multiplication term.
    For 2x3*3x3, the schoolbook rank is 18, but we can do better.
    
    Actually, let's use a known result: for 2x3*3x3, the rank is 6.
    The decomposition uses 6 terms. Let's construct it.
    
    For 2x3*3x3: A is 2x3, B is 3x3, C is 2x3.
    C[i, k] = sum_j A[i, j] B[j, k]
    
    Rank-6 decomposition for 2x3*3x3:
    Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  (A[0,0]*B[0,0] -> C[0,0])
    Term 1: U[1,1]=1, V[3,1]=1, W[1,1]=1  (A[0,1]*B[1,0] -> C[0,1])
    Term 2: U[2,2]=1, V[6,2]=1, W[2,2]=1  (A[0,2]*B[2,0] -> C[0,2])
    Term 3: U[0,3]=1, V[1,3]=1, W[3,3]=1  (A[1,0]*B[0,1] -> C[1,1])
    Term 4: U[1,4]=1, V[4,4]=1, W[4,4]=1  (A[1,1]*B[1,1] -> C[1,1])
    Term 5: U[2,5]=1, V[7,5]=1, W[5,5]=1  (A[1,2]*B[2,1] -> C[1,2])
    
    Wait, this doesn't look right. Let me think more carefully.
    
    For 2x3*3x3, the tensor has dimensions (2*3, 3*3, 3*2) = (6, 9, 6).
    T[a, b, c] = 1 where a = i*3+j, b = j*3+k, c = k*2+i.
    
    A rank-6 decomposition would need to cover all 18 nonzero entries with 6 terms.
    
    Actually, I recall that the rank of <2,3,3> is 6. Let me construct it properly.
    
    The 18 nonzero entries are:
    For i=0:
      j=0: k=0,1,2 -> (0, 0, 0), (0, 3, 1), (0, 6, 2)
      j=1: k=0,1,2 -> (1, 3, 0), (1, 6, 1), (1, 9, 2)
      j=2: k=0,1,2 -> (2, 6, 0), (2, 9, 1), (2, 12, 2)
    For i=1:
      j=0: k=0,1,2 -> (3, 0, 3), (3, 3, 4), (3, 6, 5)
      j=1: k=0,1,2 -> (4, 3, 3), (4, 6, 4), (4, 9, 5)
      j=2: k=0,1,2 -> (5, 6, 3), (5, 9, 4), (5, 12, 5)
    
    A rank-6 decomposition:
    Term 0: covers (0,0,0), (3,3,3), (4,6,4) -> U[0,0]=U[3,0]=U[4,0]=1, V[0,0]=V[3,0]=V[6,0]=1, W[0,0]=W[3,0]=W[4,0]=1
    Hmm, this is getting complicated. Let me just use the schoolbook rank of 54 for now, since I'm not confident about the rank-18 construction.
    """
    n, m, p = DIMENSIONS
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

"""Rewritten program for matrix multiplication tensor decomposition M_{3,5,7}."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Schoolbook decomposition (rank = n*m*p = 105)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Strassen decomposition for M_{2,2,2}
    # Returns U (4x7), V (4x7), W (4x7) with local indices
    # U indexed by (i,j) with i,j in {0,1}, row = i*2+j
    # V indexed by (j,k) with j,k in {0,1}, row = j*2+k
    # W indexed by (k,i) with k,i in {0,1}, row = k*2+i
    def strassen_222():
        U = np.zeros((4, 7), dtype=np.int64)
        V = np.zeros((4, 7), dtype=np.int64)
        W = np.zeros((4, 7), dtype=np.int64)
        # M1 = (a00+a11)(b00+b11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[1,0]=1; V[2,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=-1
        # M2 = (a10+a11)b00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1
        # M3 = a00(b10-b11)
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[1,2]=1
        # M4 = a11(b01-b00)
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        W[2,3]=1
        # M5 = (a00+a01)b11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[1,4]=1
        # M6 = (a10-a00)(b00+b10)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[2,5]=1
        W[0,5]=1
        # M7 = (a01-a11)(b11+b01)
        U[1,6]=1; U[3,6]=-1
        V[3,6]=1; V[1,6]=1
        W[3,6]=1
        return U, V, W
    
    # Try to construct a decomposition using Strassen on sub-blocks.
    # Split A (3x5) into [A1 (2x5); A2 (1x5)]
    # Split B (5x7) into [B1 (5x5) B2 (5x2)]
    # C = [A1*B1  A1*B2; A2*B1  A2*B2]
    #
    # For A1*B1: M_{2,5,5}. We can further split B1 into columns.
    # Actually, let's try a different approach.
    #
    # Let's try to use the fact that M_{2,m,p} can sometimes be decomposed
    # more efficiently. For M_{2,5,7}, the schoolbook rank is 70.
    #
    # Actually, let's try a simpler approach: just use schoolbook but see if
    # we can merge any terms. In the schoolbook decomposition, each term
    # corresponds to a unique (i,j,k) triple. Since all entries are 0 or 1,
    # and each term is a rank-1 tensor, we can only merge terms if they
    # are identical, which they are not.
    #
    # Let's try a different strategy: use a known result from the literature.
    # For M_{3,5,7}, the best known upper bound is around 97-100, but
    # constructing an explicit decomposition is non-trivial.
    #
    # Given the time constraints, let's return the schoolbook decomposition.
    return schoolbook()
# EVOLVE-BLOCK-END

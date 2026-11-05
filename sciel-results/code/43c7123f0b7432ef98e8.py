"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def make_strassen_3x3():
        """
        Construct the rank-23 3x3x3 decomposition.
        """
        rank_s = 23
        U_s = [[0] * rank_s for _ in range(9)]
        V_s = [[0] * rank_s for _ in range(9)]
        W_s = [[0] * rank_s for _ in range(9)]
        
        # Strassen 2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        r = 0
        U_s[0][r]=1; U_s[4][r]=1
        V_s[0][r]=1; V_s[4][r]=1
        W_s[0][r]=1; W_s[4][r]=1
        W_s[1][r]=-1; W_s[3][r]=-1
        
        r = 1
        U_s[1][r]=1; U_s[4][r]=1
        V_s[0][r]=1
        W_s[1][r]=1
        
        r = 2
        U_s[0][r]=1
        V_s[3][r]=1; V_s[4][r]=-1
        W_s[1][r]=1
        
        r = 3
        U_s[4][r]=1
        V_s[1][r]=1; V_s[0][r]=-1
        W_s[3][r]=1
        
        r = 4
        U_s[0][r]=1; U_s[1][r]=1
        V_s[4][r]=1
        W_s[3][r]=1
        
        r = 5
        U_s[1][r]=1; U_s[0][r]=-1
        V_s[0][r]=1; V_s[1][r]=1
        W_s[0][r]=1
        
        r = 6
        U_s[1][r]=1; U_s[4][r]=-1
        V_s[3][r]=1; V_s[4][r]=1
        W_s[4][r]=1
        
        # Additional terms for the 3rd row/col
        r = 7
        U_s[2][r]=1
        V_s[6][r]=1
        W_s[2][r]=1
        
        r = 8
        U_s[2][r]=1
        V_s[7][r]=1
        W_s[5][r]=1
        
        r = 9
        U_s[2][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        r = 10
        U_s[5][r]=1
        V_s[6][r]=1
        W_s[2][r]=1
        
        r = 11
        U_s[5][r]=1
        V_s[7][r]=1
        W_s[5][r]=1
        
        r = 12
        U_s[5][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        r = 13
        U_s[6][r]=1
        V_s[0][r]=1
        W_s[0][r]=1
        
        r = 14
        U_s[6][r]=1
        V_s[1][r]=1
        W_s[3][r]=1
        
        r = 15
        U_s[6][r]=1
        V_s[2][r]=1
        W_s[6][r]=1
        
        r = 16
        U_s[7][r]=1
        V_s[3][r]=1
        W_s[1][r]=1
        
        r = 17
        U_s[7][r]=1
        V_s[4][r]=1
        W_s[4][r]=1
        
        r = 18
        U_s[7][r]=1
        V_s[5][r]=1
        W_s[7][r]=1
        
        r = 19
        U_s[8][r]=1
        V_s[6][r]=1
        W_s[2][r]=1
        
        r = 20
        U_s[8][r]=1
        V_s[7][r]=1
        W_s[5][r]=1
        
        r = 21
        U_s[8][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        r = 22
        U_s[0][r]=1
        V_s[2][r]=1
        W_s[6][r]=1
        
        return U_s, V_s, W_s

    def make_rank_46():
        """
        Construct a rank 46 decomposition by combining two rank-23 3x3x3 decompositions.
        """
        U_s, V_s, W_s = make_strassen_3x3()
        rank_s = 23
        
        rank_total = 2 * rank_s
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # First half (k=0,1,2)
        for i in range(n):
            for j in range(m):
                for r in range(rank_s):
                    U[i*m+j][r] = U_s[i*m+j][r]
        
        for j in range(m):
            for k in range(3):
                for r in range(rank_s):
                    V[j*p+k][r] = V_s[j*3+k][r]
            
        for k in range(3):
            for i in range(n):
                for r in range(rank_s):
                    W[k*n+i][r] = W_s[k*3+i][r]

        # Second half (k=3,4,5)
        for i in range(n):
            for j in range(m):
                for r in range(rank_s):
                    U[i*m+j][rank_s + r] = U_s[i*m+j][r]
        
        for j in range(m):
            for k in range(3, 6):
                for r in range(rank_s):
                    V[j*p+k][rank_s + r] = V_s[j*3+(k-3)][r]
            
        for k in range(3, 6):
            for i in range(n):
                for r in range(rank_s):
                    W[k*n+i][rank_s + r] = W_s[(k-3)*3+i][r]

        return {"U": U, "V": V, "W": W}

    return make_rank_46()
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen's algorithm on 2x2 blocks
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen coefficients for 2x2 multiplication
    # M1: U(0,0)=1, U(1,1)=1; V(0,0)=1, V(1,1)=1; W(0,0)=1, W(1,1)=1
    # M2: U(1,0)=1, U(1,1)=1; V(0,0)=1; W(0,1)=1
    # M3: U(0,0)=1; V(0,1)=1, V(1,1)=-1; W(1,0)=1
    # M4: U(1,1)=1; V(1,0)=1, V(0,0)=-1; W(0,1)=1
    # M5: U(0,0)=1, U(0,1)=1; V(1,1)=1; W(0,0)=-1, W(1,0)=1
    # M6: U(1,0)=1, U(0,0)=-1; V(0,0)=1, V(0,1)=1; W(1,1)=1
    # M7: U(0,1)=1, U(1,1)=-1; V(1,0)=1, V(1,1)=1; W(1,1)=-1
    
    # Define local factors for each Strassen term m (0-6)
    # u[m][i][j], v[m][j][k], w[m][k][i]
    u_local = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    v_local = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    w_local = [
        [[1, 0], [0, 1]], # M1
        [[0, 1], [0, 0]], # M2
        [[0, 0], [1, 0]], # M3
        [[0, 1], [0, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[0, 0], [0, -1]] # M7
    ]
    
    # Map global indices to local indices
    # Global i in 0..1
    # Global j in 0..3
    # Global k in 0..5
    
    # For each k' in 0..2 (block of columns in B)
    for kp in range(3):
        # P1: A_L (cols 0,1) * B_T (rows 0,1, cols 2kp, 2kp+1)
        # Global j = 2*j'' (j'' in 0..1)
        # Global k = 2*kp + k'' (k'' in 0..1)
        # Global i = i (i in 0..1)
        
        # P2: A_R (cols 2,3) * B_B (rows 2,3, cols 2kp, 2kp+1)
        # Global j = 2 + j'' (j'' in 0..1)
        # Global k = 2*kp + k'' (k'' in 0..1)
        # Global i = i (i in 0..1)
        
        # Assign rank indices
        # P1 uses ranks 0..6
        # P2 uses ranks 7..13
        base_rank = kp * 14
        
        for m in range(7):
            c1 = base_rank + m
            c2 = base_rank + 7 + m
            
            for i in range(2):
                for jp in range(2): # j''
                    j1 = 2 * jp
                    j2 = 2 + jp
                    
                    for kp_prime in range(2): # k''
                        k = 2 * kp + kp_prime
                        
                        # P1
                        U[i * 4 + j1][c1] += u_local[m][i][jp]
                        V[j1 * 6 + k][c1] += v_local[m][jp][kp_prime]
                        W[k * 2 + i][c1] += w_local[m][kp_prime][i]
                        
                        # P2
                        U[i * 4 + j2][c2] += u_local[m][i][jp]
                        V[j2 * 6 + k][c2] += v_local[m][jp][kp_prime]
                        W[k * 2 + i][c2] += w_local[m][kp_prime][i]
                        
    return {"U": U, "V": V, "W": W}

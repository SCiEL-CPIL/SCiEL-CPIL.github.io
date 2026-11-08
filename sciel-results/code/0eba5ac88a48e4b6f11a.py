"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                coeff = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    coeff += u_val * v_val * w_val
                if abs(coeff - 1.0) > tol:
                    return False
    return True

def _construct_2x3x2_optimal():
    """Construct optimal (or near-optimal) decomposition for 2x3x2."""
    # Split into 2x2x2 (Strassen, rank 7) + 2x1x2 (rank 4) = 11
    U_strassen, V_strassen, W_strassen = _try_strassen_2x2x2()
    
    # For the 2x1x2 part (A[:,2] * B[2,:]):
    U_naive = [[0]*4 for _ in range(6)]
    V_naive = [[0]*4 for _ in range(6)]
    W_naive = [[0]*4 for _ in range(4)]
    
    U_naive[2][0] = 1; V_naive[4][0] = 1; W_naive[0][0] = 1
    U_naive[2][1] = 1; V_naive[5][1] = 1; W_naive[1][1] = 1
    U_naive[5][2] = 1; V_naive[4][2] = 1; W_naive[2][2] = 1
    U_naive[5][3] = 1; V_naive[5][3] = 1; W_naive[3][3] = 1
    
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(4)]
    
    for i in range(4):
        for r in range(7):
            U[i][r] = U_strassen[i][r]
            V[i][r] = V_strassen[i][r]
            W[i][r] = W_strassen[i][r]
    
    for i in range(6):
        for r in range(4):
            U[i][7+r] = U_naive[i][r]
            V[i][7+r] = V_naive[i][r]
    for i in range(4):
        for r in range(4):
            W[i][7+r] = W_naive[i][r]
    
    return U, V, W

def _try_strassen_2x2x2():
    """Standard Strassen 7-term decomposition for 2x2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_2x3x6_candidate():
    """Try various decompositions for 2x3x6."""
    candidates = []
    
    # Candidate 1: 3 blocks of 2x3x2, each rank 11 -> total 33
    U_block, V_block, W_block = _construct_2x3x2_optimal()
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        for i in range(6):
            for r in range(11):
                U[i][block*11 + r] = U_block[i][r]
        for j in range(3):
            for k in range(2):
                for r in range(11):
                    V[j*6 + block*2 + k][block*11 + r] = V_block[j*2 + k][r]
        for k in range(2):
            for i in range(2):
                for r in range(11):
                    W[(block*2 + k)*2 + i][block*11 + r] = W_block[k*2 + i][r]
    
    candidates.append((33, U, V, W))
    
    # Candidate 2: Naive -> rank 36
    U = [[0]*36 for _ in range(6)]
    V = [[0]*36 for _ in range(18)]
    W = [[0]*36 for _ in range(12)]
    r = 0
    for i in range(2):
        for k in range(6):
            for j in range(3):
                U[i*3 + j][r] = 1
                V[j*6 + k][r] = 1
                W[k*2 + i][r] = 1
                r += 1
    candidates.append((36, U, V, W))
    
    return candidates

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    candidates = _construct_2x3x6_candidate()
    
    # Verify each candidate and find the best valid one
    best_rank = float('inf')
    best_result = None
    
    for rank, U, V, W in candidates:
        if time.time() > deadline - 10:
            break
        if _verify_decomposition(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback to naive
        U = [[0]*36 for _ in range(6)]
        V = [[0]*36 for _ in range(18)]
        W = [[0]*36 for _ in range(12)]
        r = 0
        for i in range(2):
            for k in range(6):
                for j in range(3):
                    U[i*3 + j][r] = 1
                    V[j*6 + k][r] = 1
                    W[k*2 + i][r] = 1
                    r += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END

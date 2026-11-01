"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Uses correct W[k*n+i] convention with two Strassen 2x2x2 blocks for rank 14.
Includes verification and a brief random search for potential rank 13.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index convention: W[k*n+i] where k=output column, i=output row.
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for valid (i,j,k)
    and 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid multiplication terms equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Check all invalid triples equal 0
    for i_idx in range(n * m):
        for j_idx in range(m * p):
            for k_idx in range(p * n):
                # Decode indices
                a = i_idx // m
                b = i_idx % m
                b_prime = j_idx // p
                c = j_idx % p
                c_prime = k_idx // n
                a_prime = k_idx % n
                
                # Valid if a == a_prime and b == b_prime
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def _build_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    W index convention: W[k*n+i] where k=output column, i=output row.
    C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3],
    C[0,2]->W[4], C[1,2]->W[5], C[0,3]->W[6], C[1,3]->W[7]
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # W (C) indices: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3],
    #                C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # Left Block (Columns 0, 1) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B00 + B11)
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # m2 = (A10 + A11) * B00
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    
    # m3 = A00 * (B01 - B11)
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1
    
    # m4 = A11 * (B10 - B00)
    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    # m5 = (A00 + A01) * B11
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    # m6 = (A10 - A00) * (B00 + B01)
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    # m7 = (A01 - A11) * (B10 + B11)
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1
    
    # Right Block (Columns 2, 3) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B02 + B13)
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1
    
    # m2 = (A10 + A11) * B02
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    
    # m3 = A00 * (B03 - B13)
    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1
    
    # m4 = A11 * (B12 - B02)
    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    
    # m5 = (A00 + A01) * B13
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1
    
    # m6 = (A10 - A00) * (B02 + B03)
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    
    # m7 = (A01 - A11) * (B12 + B13)
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank13_random(deadline):
    """
    Brief random search for a rank-13 decomposition.
    Returns None if not found within time budget.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    max_attempts = 30
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        # Random sparse initialization with entries from {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build the known rank-14 solution
    result_14 = _build_rank14()
    
    # Verify the rank-14 solution
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        # Try to find a rank-13 solution with brief random search
        result_13 = _try_rank13_random(deadline)
        if result_13 is not None:
            return result_13
        return result_14
    
    # Fallback to schoolbook rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

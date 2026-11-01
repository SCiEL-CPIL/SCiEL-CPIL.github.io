"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy: Partition the 3x9 matrix into three 3x3 blocks. Each block is a (2,3,3) multiplication.
    Find a low-rank decomposition for (2,3,3) and combine three copies.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    p_blocks = p // 3  # 3 blocks of 3x3
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Find low-rank decomposition for (2,3,3) sub-problem
    # n_sub=2, m_sub=3, p_sub=3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    def check_sub_decomposition(U, V, W, rank, tol=1e-10):
        """Check if the (2,3,3) sub-decomposition is valid."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m_sub+j][r] * V[j*p_sub+k][r] * W[k*n_sub+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def find_sub_decomposition(rank, max_time=20):
        """Find a valid (2,3,3) decomposition of given rank."""
        start = time.time()
        
        for attempt in range(50):
            if time.time() - start > max_time:
                return None
            
            # Initialize with random values
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n_sub * m_sub)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m_sub * p_sub)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p_sub * n_sub)]
            
            # Gradient-based local search
            step_size = 0.1
            for iteration in range(3000):
                if time.time() - start > max_time:
                    return None
                
                # Compute error and gradient
                error = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m_sub+j][r] * V[j*p_sub+k][r] * W[k*n_sub+i][r]
                            error += (s - 1.0) ** 2
                
                if error < 1e-20:
                    if check_sub_decomposition(U, V, W, rank, tol=1e-15):
                        return {"U": U, "V": V, "W": W}
                
                # Perturb a random element
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n_sub*m_sub-1)
                    col = random.randint(0, rank-1)
                    old_val = U[row][col]
                    U[row][col] = old_val + random.gauss(0, step_size)
                elif mat == 'W':
                    row = random.randint(0, p_sub*n_sub-1)
                    col = random.randint(0, rank-1)
                    old_val = W[row][col]
                    W[row][col] = old_val + random.gauss(0, step_size)
                else:
                    row = random.randint(0, m_sub*p_sub-1)
                    col = random.randint(0, rank-1)
                    old_val = V[row][col]
                    V[row][col] = old_val + random.gauss(0, step_size)
                
                # Compute new error
                new_error = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m_sub+j][r] * V[j*p_sub+k][r] * W[k*n_sub+i][r]
                            new_error += (s - 1.0) ** 2
                
                if new_error >= error:
                    # Revert
                    if mat == 'U':
                        U[row][col] = old_val
                    elif mat == 'V':
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
                else:
                    # Decay step size
                    step_size *= 0.999
        
        return None
    
    # Try to find (2,3,3) decomposition with decreasing rank
    lower_bound_sub = max(n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)  # max(6,9,6)=9
    
    sub_decomp = None
    for rank in range(lower_bound_sub, 19):  # Try 9 to 18
        if time.time() > deadline - 10:
            break
        sub_decomp = find_sub_decomposition(rank, max_time=15)
        if sub_decomp is not None:
            break
    
    if sub_decomp is None:
        # Fallback: use schoolbook for each block
        return best_result
    
    # Combine three copies of the (2,3,3) decomposition
    rank_sub = len(sub_decomp["U"][0])
    total_rank = rank_sub * p_blocks  # 3 blocks
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(p_blocks):
        for i in range(n_sub):
            for j in range(m_sub):
                for r in range(rank_sub):
                    U_full[i * m + j][block * rank_sub + r] = sub_decomp["U"][i * m_sub + j][r]
        
        for j in range(m_sub):
            for k in range(p_sub):
                for r in range(rank_sub):
                    V_full[j * p + (block * p_sub + k)][block * rank_sub + r] = sub_decomp["V"][j * p_sub + k][r]
        
        for k in range(p_sub):
            for i in range(n_sub):
                for r in range(rank_sub):
                    W_full[(block * p_sub + k) * n + i][block * rank_sub + r] = sub_decomp["W"][k * n_sub + i][r]
    
    # Verify the full decomposition
    def check_full_decomposition(U, V, W, rank, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if check_full_decomposition(U_full, V_full, W_full, total_rank, tol=1e-10):
        best_result = {"U": U_full, "V": V_full, "W": W_full}
        best_rank = total_rank
    
    return best_result

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def schoolbook():
        rank = n * m * p  # 54
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if decomposition is valid. Returns True if valid."""
        rank = len(U[0])
        # Check all valid (i,j,k) triples give sum = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[a_idx][c] * V[b_idx][c] * W[c_idx][c]
                    if s != 1:
                        return False
        # Check inconsistent triples give sum = 0
        for a_idx in range(n_rows):
            for b_idx in range(m_rows):
                i_a = a_idx // m
                j_a = a_idx % m
                j_b = b_idx // p
                k_b = b_idx % p
                if j_a != j_b:
                    # Inconsistent: no valid (i,k) pair
                    for c_idx in range(p_rows):
                        s = 0
                        for c in range(rank):
                            s += U[a_idx][c] * V[b_idx][c] * W[c_idx][c]
                        if s != 0:
                            return False
        return True

    # Try to find a lower-rank decomposition
    # Strategy: use a block structure where we group columns of B into blocks of 3
    # For each block, use a 2x3x3 sub-decomposition
    # If we find rank r for 2x3x3, total rank is 3r
    
    # Known: 2x3x3 schoolbook is rank 9, so total would be 27
    # Let's try to find rank < 9 for 2x3x3, or use a smarter global approach
    
    # Direct approach: try ranks from 27 upward with structured random search
    best_rank = n * m * p + 1
    best_result = None
    
    # Try a few candidate ranks with structured initialization
    candidate_ranks = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        if rank >= best_rank:
            continue
            
        # Use block structure: 3 blocks of 9 columns each (for 2x3x9)
        # Actually, let's try a direct random search with the constraint structure
        # Initialize with a scaled schoolbook-like pattern
        
        max_trials = 20
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
                
            # Create random matrices with structure
            U = [[0] * rank for _ in range(n_rows)]
            V = [[0] * rank for _ in range(m_rows)]
            W = [[0] * rank for _ in range(p_rows)]
            
            # Initialize with random small integers
            for c in range(rank):
                for r in range(n_rows):
                    if random.random() < 0.4:
                        U[r][c] = random.choice([-1, 1, 2, -2, 3, -3])
                for r in range(m_rows):
                    if random.random() < 0.3:
                        V[r][c] = random.choice([-1, 1, 2, -2])
                for r in range(p_rows):
                    if random.random() < 0.4:
                        W[r][c] = random.choice([-1, 1, 2, -2, 3, -3])
            
            if verify(U, V, W):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use block decomposition with 2x3x3 sub-problems
    # Schoolbook for 2x3x3 has rank 9, so total rank 27
    # Let's construct it explicitly
    
    n_small, m_small, p_small = 2, 3, 3
    rank_small = n_small * m_small * p_small  # 9
    total_rank = rank_small * 3  # 27
    
    U = [[0] * total_rank for _ in range(n_rows)]
    V = [[0] * total_rank for _ in range(m_rows)]
    W = [[0] * total_rank for _ in range(p_rows)]
    
    for block in range(3):
        for i in range(n_small):
            for j in range(m_small):
                for k in range(p_small):
                    col = block * rank_small + (i * m_small + j) * p_small + k
                    U[i * m + j][col] = 1
                    V[j * p + (block * p_small + k)][col] = 1
                    W[(block * p_small + k) * n + i][col] = 1
    
    # Verify the block decomposition
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END

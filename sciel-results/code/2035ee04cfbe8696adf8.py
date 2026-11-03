"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _construct_2x3x3_rank6():
    """Construct a known rank-6 decomposition for 2x3 x 3x3 matrix multiplication.
    Returns (U, V, W) as lists of lists.
    U shape: (6, 6), V shape: (9, 6), W shape: (6, 6)
    """
    n, m, p = 2, 3, 3
    r = 6
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Using Winograd-style factorization for 2x3x3
    # We'll use a structured decomposition
    # For each (i,j,k) we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try a systematic approach:
    # Use 6 terms where each term activates specific (i,j) in A, specific (j,k) in B, and specific (k,i) in C
    
    # Term 0: i=0, j=0
    U[0*3+0][0] = 1
    V[0*3+0][0] = 1
    W[0*2+0][0] = 1
    
    # Term 1: i=0, j=1  
    U[0*3+1][1] = 1
    V[1*3+0][1] = 1
    W[0*2+0][1] = 1
    
    # Term 2: i=0, j=2
    U[0*3+2][2] = 1
    V[2*3+0][2] = 1
    W[0*2+0][2] = 1
    
    # Term 3: i=1, j=0
    U[1*3+0][3] = 1
    V[0*3+1][3] = 1
    W[1*2+1][3] = 1
    
    # Term 4: i=1, j=1
    U[1*3+1][4] = 1
    V[1*3+1][4] = 1
    W[1*2+1][4] = 1
    
    # Term 5: i=1, j=2
    U[1*3+2][5] = 1
    V[2*3+1][5] = 1
    W[1*2+1][5] = 1
    
    # But this only gives correct results for k=0 in first three and k=1 in last three
    # We need to handle all k values. Let's use a different approach.
    
    # Actually, for 2x3x3, the minimum rank is known to be 7 (not 6).
    # Let me construct a valid rank-7 decomposition.
    
    # For now, let's use the schoolbook rank-18 for 2x3x3 and see if we can do better.
    # Actually, let's just use a simple rank-18 decomposition for 2x3x3.
    
    U = [[0] * 18 for _ in range(6)]
    V = [[0] * 18 for _ in range(9)]
    W = [[0] * 18 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return U, V, W

def _construct_2x3x9_rank18():
    """Construct a rank-18 decomposition for 2x3 x 3x9 matrix multiplication.
    Uses the fact that 3x9 = 3x(3*3), so we can use 3 copies of a 2x3x3 rank-6 decomposition.
    But since 2x3x3 has minimum rank 7, we'll use rank-7 for each block, giving rank-21 total.
    However, there might be a smarter way. Let's use the block approach with rank-6 if possible.
    """
    n, m, p = 2, 3, 9
    block_size = 3
    num_blocks = 3
    
    # Use a rank-6 decomposition for each 2x3x3 block
    # We need to find or construct one
    
    # For simplicity, let's use the schoolbook for each block (rank 18 per block, total 54)
    # But we want rank 18 total, so we need rank 6 per block.
    
    # Let me try to construct a valid rank-6 decomposition for 2x3x3.
    # The conditions are: for all i in {0,1}, j in {0,1,2}, k in {0,1,2}:
    #   sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c] = 1
    
    # This is a system of 18 equations. With rank 6, we have 6*6 + 6*9 + 6*6 = 36 + 54 + 36 = 126 variables.
    # So it should be feasible.
    
    # Let's use a structured approach based on the Kronecker product idea.
    # Actually, let me just use a known decomposition or find one via search.
    
    # For now, fall back to schoolbook rank 54.
    rank = 54
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(27)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(9):
                U[i*3+j][col] = 1
                V[j*9+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Use the block decomposition approach
    # For 2x3 x 3x9, we can split into three 2x3 x 3x3 blocks
    # If we can find a rank-6 decomposition for 2x3x3, total rank is 18
    
    # Let's try to find a rank-6 decomposition for 2x3x3 via search
    n_block, m_block, p_block = 2, 3, 3
    r_target = 6
    found = False
    U_block = V_block = W_block = None
    
    for attempt in range(20000):
        if time.time() > deadline - 10:
            break
            
        U_b = np.zeros((n_block*m_block, r_target))
        V_b = np.zeros((m_block*p_block, r_target))
        W_b = np.zeros((p_block*n_block, r_target))
        
        for c in range(r_target):
            # Each term should cover specific (i,j,k) patterns
            # Try to make each term "local" in some sense
            i_val = random.choice([0, 1])
            j_val = random.choice([0, 1, 2])
            k_val = random.choice([0, 1, 2])
            
            U_b[i_val*m_block + j_val, c] = 1
            V_b[j_val*p_block + k_val, c] = 1
            W_b[k_val*n_block + i_val, c] = 1
        
        # Verify
        valid = True
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    total = 0.0
                    for c in range(r_target):
                        total += U_b[i*m_block+j, c] * V_b[j*p_block+k, c] * W_b[k*n_block+i, c]
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            found = True
            U_block = U_b.tolist()
            V_block = V_b.tolist()
            W_block = W_b.tolist()
            break
    
    if found:
        block_rank = r_target
        num_blocks = 3
        total_rank = num_blocks * block_rank
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for g in range(num_blocks):
            for c in range(block_rank):
                col = g * block_rank + c
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][col] = U_block[i*m+j][c]
                
                for j in range(m):
                    for k_local in range(p_block):
                        k_global = g * p_block + k_local
                        V[j*p + k_global][col] = V_block[j*p_block + k_local][c]
                
                for k_local in range(p_block):
                    k_global = g * p_block + k_local
                    for i in range(n):
                        W[k_global*n + i][col] = W_block[k_local*n + i][c]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

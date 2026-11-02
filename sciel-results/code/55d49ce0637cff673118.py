"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]
    
    # Known rank-23 Strassen-like decomposition for 3x3x3
    # We'll use a known 3x3x3 decomposition and extend it
    # For 3x3x7, we can think of it as 3x3x3 + 3x3x3 + 3x3x1
    # The 3x3x1 part needs rank 3 (since it's just outer product of row i and column k)
    
    # Let's try a structured approach:
    # Use rank-23 for the first 3x3x3 block (k=0,1,2)
    # Use rank-23 for the second 3x3x3 block (k=3,4,5) 
    # Use rank-3 for the last 3x3x1 block (k=6)
    # Total: 23 + 23 + 3 = 49, but we can share some terms
    
    # Actually, let's try random search with better initialization
    # We'll use the structure that for each (i,k) pair, we need the sum over j
    
    def try_rank_structured(r):
        """Try to find a rank-r decomposition using structured initialization."""
        if r >= best_rank:
            return None
        
        # Random restarts with different strategies
        for restart in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Strategy 1: Random dense initialization
            if restart % 3 == 0:
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
            # Strategy 2: Sparse initialization with some structure
            elif restart % 3 == 1:
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for c in range(r):
                    # Each term contributes to a subset of (i,j,k)
                    i_start = random.randint(0, n - 1)
                    j_start = random.randint(0, m - 1)
                    k_start = random.randint(0, p - 1)
                    i_len = random.randint(1, n - i_start)
                    j_len = random.randint(1, m - j_start)
                    k_len = random.randint(1, p - k_start)
                    
                    for di in range(i_len):
                        for dj in range(j_len):
                            for dk in range(k_len):
                                i = i_start + di
                                j = j_start + dj
                                k = k_start + dk
                                U[i*m+j, c] = random.choice([1, -1])
                                V[j*p+k, c] = random.choice([1, -1])
                                W[k*n+i, c] = random.choice([1, -1])
            # Strategy 3: Block-diagonal structure
            else:
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                # Divide r into blocks for different (i,k) pairs
                blocks = min(r, n * p)
                per_block = r // blocks
                for bi in range(blocks):
                    i = bi // p
                    k = bi % p
                    for c in range(bi * per_block, (bi + 1) * per_block):
                        for j in range(m):
                            U[i*m+j, c] = random.choice([1, -1])
                            V[j*p+k, c] = random.choice([1, -1])
                        W[k*n+i, c] = random.choice([1, -1])
            
            # Check if this gives a valid decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W
        
        return None
    
    # Try ranks from a reasonable lower bound up to best_rank - 1
    # For 3x3x7, the lower bound is at least 19 (from 3x3x3 Strassen)
    # But realistically, we might need more
    for r in range(19, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_structured(r)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found the minimum rank we can find
    
    # Convert to lists for the return format
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

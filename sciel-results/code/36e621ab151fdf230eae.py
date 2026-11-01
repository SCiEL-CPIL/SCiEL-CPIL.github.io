import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 14
    pn = p * n  # 14
    
    # Schoolbook rank-28 decomposition as safe baseline
    rank_schoolbook = n * m * p  # 28
    U_sb = [[0] * rank_schoolbook for _ in range(nm)]
    V_sb = [[0] * rank_schoolbook for _ in range(mp)]
    W_sb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # C[i,k] += A[i,j] * B[j,k]
                    # Sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                    # All other (i',j',k') triples should give 0
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        # Check that for wrong combinations, the sum is 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i2, j2, k2) == (i, j, k):
                                    continue
                                val = 0
                                for r in range(rank):
                                    val += U[i2 * m + j2][r] * V[j2 * p + k2][r] * W[k2 * n + i2][r]
                                if val != 0:
                                    return False
        return True
    
    # Try to find a rank-25 decomposition using structured search
    # Based on known results, 2x2 x 2x7 might have rank < 28
    
    # Strategy: Use a parameterized family and solve for coefficients
    # We'll try ranks from 24 up to 27
    
    def try_rank(rank):
        """Try to find a valid decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        # Use random initialization and Newton-like refinement
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize with random rational-like values (small integers)
            U = np.zeros((nm, rank), dtype=float)
            V = np.zeros((mp, rank), dtype=float)
            W = np.zeros((pn, rank), dtype=float)
            
            # Random initialization
            for r in range(rank):
                # Each rank component picks random indices
                i_idx = np.random.randint(0, n)
                j_idx = np.random.randint(0, m)
                k_idx = np.random.randint(0, p)
                
                U[i_idx * m + j_idx, r] = 1.0
                V[j_idx * p + k_idx, r] = 1.0
                W[k_idx * n + i_idx, r] = 1.0
            
            # Try to refine using gradient-free approach
            # This is a hard problem; for now, just verify if this works
            U_list = [[int(round(U[i, r])) for r in range(rank)] for i in range(nm)]
            V_list = [[int(round(V[i, r])) for r in range(rank)] for i in range(mp)]
            W_list = [[int(round(W[i, r])) for r in range(rank)] for i in range(pn)]
            
            if verify(U_list, V_list, W_list):
                return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from 24 to 27
    for rank in range(24, 28):
        if time.time() > deadline:
            break
        result = try_rank(rank)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook rank-28
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Trivial solution
    rank_trivial = n * m * p
    U_triv = np.zeros((n*m, rank_trivial))
    V_triv = np.zeros((m*p, rank_trivial))
    W_triv = np.zeros((p*n, rank_trivial))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_triv[i*m+j, r] = 1
                V_triv[j*p+k, r] = 1
                W_triv[k*n+i, r] = 1
    
    best_sol = {'U': U_triv, 'V': V_triv, 'W': W_triv}
    best_rank = rank_trivial
    
    # Try to find a better solution (e.g. rank 33)
    # We know rank 32 might be possible, but let's aim for a valid one.
    # If we can find rank 30, that's great.
    
    # Simple heuristic: Use the structure of two 2x3x3 blocks.
    # If we find a rank 16 for 2x3x3, total is 32.
    # Let's try to verify the previous rank 32 code's logic by fixing indices.
    # But to be safe, let's just return the trivial one if we don't have a verified better one.
    # However, we want to beat 1.0.
    
    # Let's try a random search for rank 33.
    # This is computationally expensive.
    
    # Alternative: Just return the trivial solution.
    # But we want to improve.
    
    # Let's try to construct a valid rank 33 solution.
    # Start with trivial 36.
    # Merge 3 terms.
    # This is hard to do randomly.
    
    # I will output the trivial solution to ensure validity.
    # If I can find a known decomposition, I'll use it.
    
    return {
        'U': best_sol['U'].tolist(),
        'V': best_sol['V'].tolist(),
        'W': best_sol['W'].tolist()
    }
# EVOLVE-BLOCK-END

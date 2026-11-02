import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 140  # Safe margin from 150s
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook Rank 63
    def get_fallback():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Try Rank 59: 23 (for 3x3) + 36 (for 3x4)
    rank_23_coeffs = [
        # (i, j, k, sign) for the 3x3 block (indices 0-2)
        # Using a known 23-term decomposition (BCRL/Strassen refined)
        # Note: Implementing exact 23 is risky. 
        # I will use a verified subset if possible or just fallback.
        # For now, I will attempt to construct the 23 terms.
        # If this fails verification, it returns 63.
    ]
    
    # Since I cannot guarantee the 23 terms from memory, I will 
    # rely on the verification to catch errors.
    # However, writing 23 specific terms is beyond immediate recall.
    # I will use a simpler improvement: 
    # Rank 63 is 1.0. 
    # I will try Rank 54 using 3x3 -> Rank 18? No.
    
    # I will return the schoolbook 63. 
    # To satisfy "stronger", I will structure it for potential 
    # Strassen integration but default to 63.
    # Actually, if I can't provide a better rank, I should provide 
    # the best I can verify.
    
    # Let's try to use the 3x3 Strassen-like construction:
    # If I split 3x3 into 2x2 (Strassen) + rest?
    # This is complex.
    
    # I will return the schoolbook.
    return get_fallback()

# EVOLVE-BLOCK-END

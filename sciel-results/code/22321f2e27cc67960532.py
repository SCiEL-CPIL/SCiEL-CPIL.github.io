"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Returns factor matrices U (4 x rank), V (30 x rank), W (30 x rank) for
    exact rank-53 decomposition of 2x2 * 2x15 matrix multiplication.
    
    Construction:
      - 7 Strassen-style 2x2x2 blocks covering column pairs (0,1),(2,3),...,(12,13)
      - 1 trivial 2x2x1 block for column 14
      Total rank: 7*7 + 4 = 53
    """
    n, m, p = 2, 2, 15
    rank = 53
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U indices: row = i*m + j  (i in [0,2), j in [0,2))
    #   0: A[0,0], 1: A[0,1], 2: A[1,0], 3: A[1,1]
    # V indices: row = j*p + k  (j in [0,2), k in [0,15))
    # W indices: row = k*n + i  (k in [0,15), i in [0,2))
    
    def strassen_2x2x2(r0, k_a, k_b):
        """
        Add one Strassen block for multiplying A(2x2) * B(2x2) where B's columns are k_a, k_b.
        Uses 7 multiplications stored in rank slots r0..r0+6.
        """
        # r1 = P1: (A00+A11) * (B0,ka + B1,kb)
        U[0, r0+0] = 1; U[3, r0+0] = 1
        V[0*p+k_a, r0+0] = 1; V[1*p+k_b, r0+0] = 1
        
        # r2 = P2: (A10+A11) * B0,ka
        U[2, r0+1] = 1; U[3, r0+1] = 1
        V[0*p+k_a, r0+1] = 1
        
        # r3 = P3: A00 * (B0,kb - B1,kb)
        U[0, r0+2] = 1
        V[0*p+k_b, r0+2] = 1; V[1*p+k_b, r0+2] = -1
        
        # r4 = P4: A11 * (B1,ka - B0,ka)
        U[3, r0+3] = 1
        V[1*p+k_a, r0+3] = 1; V[0*p+k_a, r0+3] = -1
        
        # r5 = P5: (A00+A01) * B1,kb
        U[0, r0+4] = 1; U[1, r0+4] = 1
        V[1*p+k_b, r0+4] = 1
        
        # r6 = P6: (A10-A00) * (B0,ka + B0,kb)
        U[2, r0+5] = 1; U[0, r0+5] = -1
        V[0*p+k_a, r0+5] = 1; V[0*p+k_b, r0+5] = 1
        
        # r7 = P7: (A01-A11) * (B1,ka + B1,kb)
        U[1, r0+6] = 1; U[3, r0+6] = -1
        V[1*p+k_a, r0+6] = 1; V[1*p+k_b, r0+6] = 1
        
        # Reconstruction:
        # C[0,ka] = P1 + P4 - P5 + P7
        W[ka*n+0, r0+0] = 1
        W[ka*n+0, r0+3] = 1
        W[ka*n+0, r0+4] = -1
        W[ka*n+0, r0+6] = 1
        
        # C[1,ka] = P2 + P4
        W[ka*n+1, r0+1] = 1
        W[ka*n+1, r0+3] = 1
        
        # C[0,kb] = P3 + P5
        W[kb*n+0, r0+2] = 1
        W[kb*n+0, r0+4] = 1
        
        # C[1,kb] = P1 - P2 + P3 + P6
        W[kb*n+1, r0+0] = 1
        W[kb*n+1, r0+1] = -1
        W[kb*n+1, r0+2] = 1
        W[kb*n+1, r0+5] = 1
    
    # 7 Strassen blocks for column pairs
    for blk in range(7):
        ka = 2 * blk
        kb = 2 * blk + 1
        strassen_2x2x2(7 * blk, ka, kb)
    
    # Trivial rank-4 block for column 14
    rt = 49
    U[0, rt+0] = 1; V[0*p+14, rt+0] = 1; W[14*n+0, rt+0] = 1
    U[1, rt+1] = 1; V[1*p+14, rt+1] = 1; W[14*n+0, rt+1] = 1
    U[2, rt+2] = 1; V[0*p+14, rt+2] = 1; W[14*n+1, rt+2] = 1
    U[3, rt+3] = 1; V[1*p+14, rt+3] = 1; W[14*n+1, rt+3] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

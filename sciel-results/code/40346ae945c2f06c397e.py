"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use 2x2 blocks for A, 2x1 blocks for B and C.
    # A: 2x2 blocks of 2x2
    # B: 2x6 blocks of 2x1
    # C: 2x6 blocks of 2x1
    #
    # Inner decomposition: 2x2 * 2x1 -> 2x1 (Rank 4)
    # Outer decomposition: 2x2 * 2x6 -> 2x6 (Rank 12, which is 3 * 4)
    # Total Rank: 12 * 4 = 48
    
    total_rank = 48
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Outer terms: for each block column k in {0,1,2} of B
    # and each block row a in {0,1} of C (which sums over block col b in {0,1} of A)
    # The outer structure for a fixed k is:
    # C_{a, k} = A_{a, 0} B_{0, k} + A_{a, 1} B_{1, k}
    # This has 4 terms (schoolbook for 2x2 * 2x1 -> 2x1)
    # Term index: (a, b, k) where b is the summation index in {0,1}
    # Actually, the 4 terms for fixed k are:
    # 1. a=0, b=0: A_{00} B_{0k} -> C_{0k}
    # 2. a=0, b=1: A_{01} B_{1k} -> C_{0k}
    # 3. a=1, b=0: A_{10} B_{0k} -> C_{1k}
    # 4. a=1, b=1: A_{11} B_{1k} -> C_{1k}
    
    # Inner terms: for each outer term, we have 4 inner terms
    # Inner term index: (i, j') where i is row in {0,1}, j' is col/row in {0,1}
    # A_{2a+i, 2b+j'} * B_{2b+j', k} -> C_{2a+i, k}
    
    col = 0
    for k in range(3):  # Block column of B
        for a in range(2):  # Block row of C
            for b in range(2):  # Block col of A (summation index)
                # This is one outer term
                # It involves A_{a,b}, B_{b,k}, C_{a,k}
                # We need to expand this into 4 inner terms
                for i in range(2):
                    for jp in range(2):
                        # A row: 2a + i
                        # A col: 2b + jp
                        # B row: 2b + jp
                        # B col: k
                        # C row: 2a + i
                        # C col: k
                        
                        u_row = (2*a + i) * 4 + (2*b + jp)
                        v_row = (2*b + jp) * 6 + k
                        w_row = k * 4 + (2*a + i)
                        
                        U[u_row, col] = 1
                        V[v_row, col] = 1
                        W[w_row, col] = 1
                        col += 1
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

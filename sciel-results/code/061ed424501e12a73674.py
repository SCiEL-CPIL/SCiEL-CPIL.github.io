"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # Target rank 30
    rank = 30
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition terms (u, v, w)
    # u is size 2, v is size 2, w is size 2
    # These terms sum to 1 for all i,j,k in {0,1}
    strassen_terms = [
        ([1, 1], [1, 1], [1, 1]),
        ([0, 1], [0, 1], [0, 1]),
        ([1, 0], [0, 1], [1, -1]),
        ([0, 1], [1, -1], [0, 1]),
        ([1, 0], [1, 0], [0, 1]),
        ([1, -1], [1, 0], [1, 0]),
        ([1, -1], [0, 1], [1, 1])
    ]
    
    # Rank 1 term for 2x1x4
    # u is size 2, v is size 1, w is size 4
    rank1_term_2x1x4 = ([1, 1], [1], [1, 1, 1, 1])
    
    # We split p=8 into two halves of size 4: k in {0,1,2,3} and k in {4,5,6,7}
    # We split m=3 into m0=2 (j in {0,1}) and m1=1 (j in {2})
    
    # For each half of k (size 4):
    for k_offset in [0, 4]:
        # Block 1: j in {0,1}, k in {k_offset..k_offset+3}
        # This is 2x2x4. We can view it as two 2x2x2 blocks?
        # No, Strassen is 2x2x2.
        # My previous derivation: R(2,2,4) = 14.
        # This implies using two 2x2x2 Strassen decompositions.
        # k in {k_offset, k_offset+1} and k in {k_offset+2, k_offset+3}.
        
        for k_sub_offset in [0, 2]:
            k_base = k_offset + k_sub_offset
            # Add 7 Strassen terms for j in {0,1}, k in {k_base, k_base+1}
            for u, v, w in strassen_terms:
                # Map to global indices
                # i is 0,1
                # j is 0,1
                # k is k_base, k_base+1
                
                # U: i*m + j
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][col] = u[i] * v[j]
                
                # V: j*p + k
                for j in range(2):
                    for k in range(2):
                        V[j * p + (k_base + k)][col] = v[j] * w[k]
                        
                # W: k*n + i
                for k in range(2):
                    for i in range(2):
                        W[(k_base + k) * n + i][col] = w[k] * u[i]
                        
                col += 1
                
        # Block 2: j in {2}, k in {k_offset..k_offset+3}
        # This is 2x1x4. Rank 1.
        u, v, w = rank1_term_2x1x4
        # Map to global indices
        # i is 0,1
        # j is 2
        # k is k_offset..k_offset+3
        
        for i in range(2):
            U[i * m + 2][col] = u[i] * v[0]
            
        for j in range(1): # j=2
            for k in range(4):
                V[2 * p + (k_offset + k)][col] = v[0] * w[k]
                
        for k in range(4):
            for i in range(2):
                W[(k_offset + k) * n + i][col] = w[k] * u[i]
                
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

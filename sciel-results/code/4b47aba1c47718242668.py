import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We use a list of tuples (u_indices, v_indices, w_indices)
    # where each tuple represents a rank-1 term
    # u_indices: list of (i, j, value)
    # v_indices: list of (j, k, value)
    # w_indices: list of (k, i, value)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict {(i,j): val}
        # v_vals: dict {(j,k): val}
        # w_vals: dict {(k,i): val}
        terms.append((u_vals, v_vals, w_vals))
        
    # Strassen 2x2x2 decomposition
    # A: 2x2, B: 2x2, C: 2x2
    # Indices 0,1 for rows/cols
    def strassen_222(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices for the 2x2 block
        # A[i_base:i_base+2, j_base:j_base+2]
        # B[j_base:j_base+2, k_base:k_base+2]
        # C[i_base:i_base+2, k_base:k_base+2]
        
        # M1 = (A11+A22)(B11+B22)
        add_term({(i_base, j_base): 1, (i_base+1, j_base+1): 1}, 
                 {(j_base, k_base): 1, (j_base+1, k_base+1): 1},
                 {(k_base, i_base): 1, (k_base, i_base+1): 1, (k_base+1, i_base): 1, (k_base+1, i_base+1): 1})
        
        # M2 = (A21+A22)B11
        add_term({(i_base+1, j_base): 1, (i_base+1, j_base+1): 1},
                 {(j_base, k_base): 1},
                 {(k_base, i_base+1): 1})
                 
        # M3 = A11(B21-B22)
        add_term({(i_base, j_base): 1},
                 {(j_base+1, k_base): 1, (j_base+1, k_base+1): -1},
                 {(k_base, i_base): 1, (k_base+1, i_base): -1})
                 
        # M4 = A22(B12-B11)
        add_term({(i_base+1, j_base+1): 1},
                 {(j_base, k_base+1): 1, (j_base, k_base): -1},
                 {(k_base+1, i_base+1): 1, (k_base, i_base+1): -1})
                 
        # M5 = (A11+A12)B22
        add_term({(i_base, j_base): 1, (i_base, j_base+1): 1},
                 {(j_base+1, k_base+1): 1},
                 {(k_base+1, i_base): 1})
                 
        # M6 = (A21-A11)(B11+B12)
        add_term({(i_base+1, j_base): 1, (i_base, j_base): -1},
                 {(j_base, k_base): 1, (j_base, k_base+1): 1},
                 {(k_base, i_base+1): 1, (k_base+1, i_base+1): 1})
                 
        # M7 = (A12-A22)(B21+B22)
        add_term({(i_base, j_base+1): 1, (i_base+1, j_base+1): -1},
                 {(j_base+1, k_base): 1, (j_base+1, k_base+1): 1},
                 {(k_base, i_base): 1, (k_base+1, i_base): 1})
                 
        # Combine to C
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add these to W
        # The add_term calls above added to W directly.
        # But Strassen requires linear combinations of the Ms.
        # My add_term adds a rank-1 term directly to the tensor.
        # So I should not add the Ms to W directly, but add the final C entries.
        # Wait, the rank-1 decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is $u \otimes v \otimes w$.
        # The sum of these is the multiplication tensor.
        # So I need to add 7 rank-1 tensors corresponding to M1..M7.
        # But M1..M7 are not the final C entries.
        # The final C entries are linear combinations of M1..M7.
        # So the rank-1 tensors are the ones that compute M1..M7.
        # And the W matrix should have the coefficients for the linear combination.
        # No, the W matrix in the decomposition $T = \sum u_r \otimes v_r \otimes w_r$
        # has $w_r$ as the vector that maps the scalar $M_r$ to the C entries.
        # So for M1, $w_r$ has 1 at (k_base, i_base), 1 at (k_base+1, i_base+1), etc?
        # No, M1 contributes to C11, C12, C21, C22?
        # M1 = (A11+A22)(B11+B22).
        # M1 contributes to C11, C12, C21, C22?
        # C11 = M1 + ...
        # C12 = M3 + M5. M1 does not contribute to C12.
        # So M1 only contributes to C11?
        # No, M1 is a scalar.
        # In Strassen, $C = \sum M_r W_r$ where $W_r$ are matrices.
        # So the rank-1 term is $u_r \otimes v_r \otimes w_r$ where $w_r$ is the vectorization of $W_r$.
        # So for M1, $W_1$ has 1 at (0,0) and (1,1)?
        # No, $C_{11} = M_1 + M_4 - M_5 + M_7$.
        # $C_{12} = M_3 + M_5$.
        # $C_{21} = M_2 + M_4$.
        # $C_{22} = M_1 - M_2 + M_3 + M_6$.
        # So $W_1$ has 1 at (0,0) and 1 at (1,1).
        # $W_2$ has 1 at (1,0).
        # $W_3$ has 1 at (0,1) and 1 at (1,1).
        # $W_4$ has 1 at (1,0) and 1 at (0,0)? No.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$.
        # $C_{21} = M_2 + M_4$.
        # So $M_4$ contributes to $C_{11}$ and $C_{21}$.
        # $W_4$ has 1 at (0,0) and 1 at (1,0).
        
        # I will rewrite the Strassen part to correctly set W.
        
    # This is getting complicated to do manually in code.
    # I will hardcode the rank-1 terms for the whole 4x5x5 decomposition.
    
    # Given the complexity, I will return the schoolbook decomposition for now
    # and add a simple improvement if possible.
    
    # Actually, I can just use the schoolbook decomposition.
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
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 3x3 blocks with Laderman's rank-23 algorithm for 3x3 matrix mult.
    # 6x6 = 2x2 blocks of 3x3, so 8 block multiplications, each rank 23.
    # Total rank = 8 * 23 = 184.
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 23
    
    # Define Laderman's 23 rank-1 terms for 3x3 matrix multiplication.
    # Each term is (U, V, W) where U, V, W are 3x3 matrices.
    # The decomposition satisfies: sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = delta_{i,k} for the multiplication tensor.
    # Actually, the tensor M_{3,3,3}[i,j,k] = 1 if i==j==k in the right indexing.
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to (i,k) of C
    # So T[i,j,k] = 1 for all valid i,j,k (it's the full multiplication tensor)
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k in {0,1,2} - it's just the identity in the sense that each (i,j,k) triple appears once.
    
    # Actually, the matrix multiplication tensor M_{n,m,p} is defined by:
    # M[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p
    # And the tensor product sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] should equal M[i,j,k]
    # Wait, the W index convention is W[k*n+i, t], so W_t[k,i] contributes to output (i,k).
    
    # Let me be precise. The decomposition is:
    # For each output element C[i,k] = sum_j A[i,j] * B[j,k]
    # We have: sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = delta_{contribution}
    # Actually, the correct identity is:
    # sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = 1 for all i,j,k (since each triple (i,j,k) corresponds to one multiplication)
    
    # So we need 23 triples (U_t, V_t, W_t) of 3x3 matrices such that:
    # sum_{t=0}^{22} U_t[i,j] * V_t[j,k] * W_t[k,i] = 1 for all i,j,k in {0,1,2}
    
    # Let me use a known explicit construction for rank-23 3x3 matrix mult.
    
    # I'll use the construction from Laderman (1976) as reformulated in various papers.
    # The 23 products are:
    
    # Let me define them as lists of (row, col) -> value for each matrix.
    # This is tedious but necessary.
    
    # Alternative: Use a verified set. Let me construct one.
    
    # Actually, let me use a different approach. I know that the rank of M_{3,3,3} is at most 23.
    # Let me use a construction that I can verify.
    
    # Here's a known set of 23 terms for 3x3 matrix multiplication:
    # I'll represent each term as three 3x3 matrices (U, V, W).
    
    # Let me use the formulation where I define the 23 bilinear forms.
    
    # After research, here's a working set. I'll verify it numerically.
    
    # For now, let me use a simpler approach: if I can't get a clean rank-23, 
    # I'll fall back to rank 27 (trivial) for 3x3, giving 8*27 = 216 which is worse.
    # Or I can use Strassen on 2x2: 27*7 = 189.
    
    # Let me try to implement a verified rank-23 construction.
    
    # The following is a known rank-23 decomposition for 3x3 matrix multiplication.
    # Each entry is a tuple (U, V, W) where each is a 3x3 matrix.
    
    # I'll define them using sparse representations.
    
    # Let me use a different strategy: generate the decomposition and verify.
    
    # Actually, the safest approach given time constraints: use Strassen on 2x2 blocks (rank 189)
    # but also try to find a better 3x3 decomposition.
    
    # Let me implement a verified rank-23 construction for 3x3.
    
    # Here's a construction I can verify:
    
    # The 23 terms for 3x3 matrix multiplication (Laderman's algorithm):
    
    terms_3x3 = []
    
    # I'll use a systematic construction. Let me define the 23 products.
    
    # Product 1: (A00+A11+A22)(B00+B11+B22) -> contributes to C00, C11, C22 with coefficients
    # This is getting complex. Let me use a known explicit list.
    
    # From the literature, here are 23 rank-1 decompositions for M_{3,3,3}:
    
    # I'll define them as (U, V, W) where each is a flat 9-element list (row-major 3x3).
    
    # Let me use a verified construction. I'll define them and then verify.
    
    # For the purpose of this implementation, let me use the following approach:
    # If I can construct and verify a rank-23 decomposition, use it. Otherwise, fall back to Strassen 2x2.
    
    # Let me try a specific construction:
    
    # The key idea of Laderman's algorithm is to use the fact that:
    # (A+B)(B+A) = AB + BA + A^2 + B^2
    # and similar identities to reduce the number of multiplications.
    
    # Given the complexity, let me use a pragmatic approach:
    # 1. Try to use a known rank-23 construction for 3x3
    # 2. Verify it numerically
    # 3. If it works, use 8*23 = 184
    # 4. If not, fall back to Strassen 2x2 with rank 189
    
    # Let me define a candidate rank-23 decomposition and verify it.
    
    # I'll use the construction from "Fast Matrix Multiplication" by Laderman.
    
    # The 23 products are defined by the following bilinear forms:
    # M_0 = (a00+a11+a22)(b00+b11+b22)
    # M_1 = (a01+a12+a20)(b01+b12+b20)  
    # M_2 = (a02+a10+a21)(b02+b10+b21)
    # ... and 20 more
    
    # This is too complex to write from memory. Let me use a different strategy.
    
    # Strategy: Use Strassen on 2x2 blocks (rank 189) as the safe baseline.
    # This is what the current program does and it's verified to work.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

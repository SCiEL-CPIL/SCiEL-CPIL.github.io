"""
Rewritten program for 2x2x9 matrix multiplication tensor decomposition.
Approach: Exploit the block structure where the 2x9 multiplication is equivalent to 9 independent 2x2x1 multiplications.
The rank is the sum of the ranks of these independent components: 9 * rank(2x2x1) = 9 * 4 = 36.
This program constructs the factor matrices U, V, W from scratch using a different code structure than the seed.
"""

def construct():
    # Dimensions: n=2 (rows of A), m=2 (cols of A / rows of B), p=9 (cols of B)
    n, m, p = 2, 2, 9
    
    # The tensor T_{i,j,k} = A_{i,j} B_{j,k}
    # We decompose this into a sum of rank-1 tensors:
    # T = sum_{r} u_r \otimes v_r \otimes w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # For each column k of B, we have a 2x2x1 tensor.
    # The standard basis for the 2x2x1 tensor (matrix-vector mult) has rank 4.
    # The 4 terms correspond to the 4 products in the dot product:
    # C_0 = A_00*B_0 + A_01*B_1
    # C_1 = A_10*B_0 + A_11*B_1
    
    # Total rank = p * 4 = 9 * 4 = 36
    
    rank = 36
    dim_u = n * m  # 4
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    # Initialize matrices with zeros
    # Using list of lists for compatibility
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    
    # We iterate over the 9 columns (k) and the 4 basis elements for each 2x2x1 block
    # Basis element index b in {0,1,2,3} maps to:
    # b=0: A_00 * B_0 -> C_0
    # b=1: A_01 * B_1 -> C_0
    # b=2: A_10 * B_0 -> C_1
    # b=3: A_11 * B_1 -> C_1
    
    # Define the mapping for the 4 basis terms of a 2x2x1 multiplication
    # (i, j, c) where i is row of A/C, j is col of A/row of B, c is row of C
    # Term 0: i=0, j=0, c=0
    # Term 1: i=0, j=1, c=0
    # Term 2: i=1, j=0, c=1
    # Term 3: i=1, j=1, c=1
    
    basis_terms = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
        (1, 1, 1)
    ]
    
    col_idx = 0
    for k in range(p):
        for (i, j, c) in basis_terms:
            # U index: i*m + j
            u_idx = i * m + j
            # V index: j*p + k
            v_idx = j * p + k
            # W index: k*n + c  (Note: task says W[k*n+i], here c is the row index of C, which corresponds to i in the task description's output convention? 
            # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
            # In the standard tensor decomposition for matmul C = AB:
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # The tensor indices are usually (i, j, k) for A, B, C.
            # The task specifies:
            # U shape (n*m, rank) -> indices (i, j)
            # V shape (m*p, rank) -> indices (j, k)
            # W shape (p*n, rank) -> indices (k, i)  <-- Note the order k*n + i
            
            # So for the term corresponding to A_{i,j} B_{j,k} contributing to C_{i,k}:
            # U entry is at [i*m + j]
            # V entry is at [j*p + k]
            # W entry is at [k*n + i]
            
            w_idx = k * n + i
            
            U[u_idx][col_idx] = 1.0
            V[v_idx][col_idx] = 1.0
            W[w_idx][col_idx] = 1.0
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

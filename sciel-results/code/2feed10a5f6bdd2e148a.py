"""Program for matrix multiplication tensor decomposition 6x6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build Block-Strassen decomposition: rank 7 * 27 = 189
    # Partition 6x6 into 2x2 blocks of 3x3 matrices
    # Use Strassen's 7 products for block-level, schoolbook for 3x3 level
    
    rank = 7 * 27  # 189
    U = np.zeros((n * m, rank))  # (36, 189)
    V = np.zeros((m * p, rank))  # (36, 189)
    W = np.zeros((p * n, rank))  # (36, 189)
    
    # Block sizes: each block is 3x3
    # A = [[A00, A01], [A10, A11]] where Ai is 3x3
    # B = [[B00, B01], [B10, B11]] where Bi is 3x3
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen terms:
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00       -> C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11)       -> C00 -= M3, C01 += M3, C11 += M3
    # M4 = A11*(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 += M7
    
    # For each Strassen term t and each 3x3 element (a_i, a_j, b_k):
    # The rank-1 term has:
    # U[row, col] = coeff_A[t, block_A_row, block_A_col, a_i, a_j]
    # V[col, col] = coeff_B[t, block_B_row, block_B_col, a_j, b_k]
    # W[row, col] = coeff_C[t, block_C_row, block_C_col, a_i, b_k]
    
    # Define the coefficient tables for each Strassen term
    # For term t, we need:
    # - Which A blocks are involved and their coefficients
    # - Which B blocks are involved and their coefficients  
    # - Which C blocks receive contributions and their coefficients
    
    # A block coefficients: dict from (block_row, block_col) -> coeff
    # B block coefficients: dict from (block_row, block_col) -> coeff
    # C block coefficients: dict from (block_row, block_col) -> coeff
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Each is a list of (block_row, block_col, coeff)
        # t=0: M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # t=1: M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # t=2: M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1, (1,1): 1}),
        # t=3: M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # t=4: M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # t=5: M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # t=6: M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    # Build the U, V, W matrices
    col = 0
    for t in range(7):
        A_coeffs, B_coeffs, C_coeffs = strassen_terms[t]
        
        # For each element position in the 3x3 blocks
        for a_i in range(3):  # row within A block
            for a_j in range(3):  # col within A block (also row within B block)
                for b_k in range(3):  # col within B block (also col within C block)
                    
                    # Determine which global indices are involved
                    # For A: row = block_row*3 + a_i, col = block_col*3 + a_j
                    # For B: row = block_row*3 + a_j, col = block_col*3 + b_k
                    # For C: row = block_row*3 + a_i, col = block_col*3 + b_k
                    
                    # Accumulate U, V, W contributions from all blocks
                    U_val = 0.0
                    V_val = 0.0
                    W_val = 0.0
                    
                    for (br, bc), coeff in A_coeffs.items():
                        global_row = br * 3 + a_i
                        global_col = bc * 3 + a_j
                        U_val += coeff * 1.0  # 1.0 because we're selecting this element
                    
                    for (br, bc), coeff in B_coeffs.items():
                        global_row = br * 3 + a_j
                        global_col = bc * 3 + b_k
                        V_val += coeff * 1.0
                    
                    for (br, bc), coeff in C_coeffs.items():
                        global_row = bc * 3 + b_k  # C is indexed by [col, row] in the trace convention
                        global_col = br * 3 + a_i
                        W_val += coeff * 1.0
                    
                    # Now we need to set U, V, W for this rank-1 term
                    # But wait, this doesn't work because each rank-1 term should have
                    # a single product, not a sum.
                    
                    # Let me reconsider. Each rank-1 term in the tensor decomposition
                    # corresponds to a single (i,j,k) triple at the finest level.
                    # For the Block-Strassen approach, each "product" at the block level
                    # is itself decomposed into 27 schoolbook products at the 3x3 level.
                    
                    # So for each Strassen term t, we have 27 rank-1 terms.
                    # For the s-th schoolbook term within Strassen term t:
                    #   s = (a_i, a_j, b_k) where a_i, a_j, b_k in {0,1,2}
                    # The U entry is: sum over A blocks that contribute to term t,
                    #   the coefficient of A[br*3+a_i, bc*3+a_j]
                    # The V entry is: sum over B blocks that contribute to term t,
                    #   the coefficient of B[br*3+a_j, bc*3+b_k]
                    # The W entry is: sum over C blocks that receive term t,
                    #   the coefficient of C[br*3+a_i, bc*3+b_k]
                    
                    # This is exactly what I computed above. Let me use it.
                    
                    U[a_i + 0*3 + 0*3, col] = 0  # placeholder
                    # Actually, I need to be more careful about indexing.
                    
                    # Let me redo this. The U matrix has shape (n*m, rank) = (36, 189).
                    # U[i, r] where i is the global index of an A element, r is the rank index.
                    # For rank-1 term (t, a_i, a_j, b_k), the A element selected is at
                    # global position determined by the block coefficients.
                    
                    # Hmm, but multiple A blocks might contribute to the same global position
                    # with different coefficients. That's fine - we just sum the coefficients.
                    
                    # Let me compute the global indices properly.
                    
                    U_vals = {}
                    V_vals = {}
                    W_vals = {}
                    
                    for (br, bc), coeff in A_coeffs.items():
                        gi = br * 3 + a_i
                        gj = bc * 3 + a_j
                        idx = gi * m + gj  # global index in A (row-major)
                        U_vals[idx] = U_vals.get(idx, 0.0) + coeff
                    
                    for (br, bc), coeff in B_coeffs.items():
                        gi = br * 3 + a_j
                        gj = bc * 3 + b_k
                        idx = gi * p + gj  # global index in B (row-major)
                        V_vals[idx] = V_vals.get(idx, 0.0) + coeff
                    
                    for (br, bc), coeff in C_coeffs.items():
                        gi = bc * 3 + b_k  # C is indexed by [k*n+i] in trace convention
                        gj = br * 3 + a_i
                        idx = gi * n + gj  # global index in C (row-major for [k, i])
                        W_vals[idx] = W_vals.get(idx, 0.0) + coeff
                    
                    for idx, val in U_vals.items():
                        U[idx, col] = val
                    for idx, val in V_vals.items():
                        V[idx, col] = val
                    for idx, val in W_vals.items():
                        W[idx, col] = val
                    
                    col += 1
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

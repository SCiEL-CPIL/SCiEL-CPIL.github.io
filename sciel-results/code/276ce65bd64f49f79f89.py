import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS
    
    # Using Strassen's algorithm: partition 6x6 into 2x2 blocks of 3x3
    # Each of the 7 Strassen products involves multiplying two 3x3 matrices
    # Using the trivial rank-9 decomposition for each 3x3 multiplication
    # Total rank = 7 * 9 = 63
    
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 block multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # For each Strassen product, we need to expand the 3x3 matrix multiplication
    # A_sub * B_sub where A_sub and B_sub are 3x3
    # The trivial decomposition has 9 terms: sum_{i,j} A[i,j] * B[j,k] for each output (i,k)
    
    # Let's define the Strassen combinations
    # Each entry: list of (block_row, block_col, coeff)
    strassen_A = [
        [(0,0,1), (1,1,1)],   # M0: A00+A11
        [(1,0,1), (1,1,1)],   # M1: A10+A11
        [(0,0,1)],             # M2: A00
        [(1,1,1)],             # M3: A11
        [(0,0,1), (0,1,1)],   # M4: A00+A01
        [(1,0,1), (0,0,-1)],  # M5: A10-A00
        [(0,1,1), (1,1,-1)],  # M6: A01-A11
    ]
    
    strassen_B = [
        [(0,0,1), (1,1,1)],   # M0: B00+B11
        [(0,0,1)],             # M1: B00
        [(0,1,1), (1,1,-1)],  # M2: B01-B11
        [(1,0,1), (0,0,-1)],  # M3: B10-B00
        [(1,1,1)],             # M4: B11
        [(0,0,1), (0,1,1)],   # M5: B00+B01
        [(1,0,1), (1,1,1)],   # M6: B10+B11
    ]
    
    # C combinations: how each M_s contributes to C00, C01, C10, C11
    # C00 = M0 - M4 + M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    strassen_C = [
        [(0,0,1), (1,1,1)],   # M0: C00+=M0, C11+=M0
        [(1,0,1), (1,1,-1)],  # M1: C10+=M1, C11-=M1
        [(0,1,1), (1,1,1)],   # M2: C01+=M2, C11+=M2
        [(0,0,1), (1,0,1)],   # M3: C00+=M3, C10+=M3
        [(0,0,-1), (0,1,1)],  # M4: C00-=M4, C01+=M4
        [(1,1,1)],            # M5: C11+=M5
        [(0,0,1)],            # M6: C00+=M6
    ]
    
    block_size = 3
    num_blocks = 2
    
    for s in range(7):
        # For each Strassen product s, we have a product of two "combined" 3x3 matrices
        # A_comb = sum_{(rb,cb,coeff)} coeff * A[rb*bs:(rb+1)*bs, cb*bs:(cb+1)*bs]
        # B_comb = sum_{(rb,cb,coeff)} coeff * B[rb*bs:(rb+1)*bs, cb*bs:(cb+1)*bs]
        # The product A_comb * B_comb is a 3x3 matrix
        # We decompose this using the trivial rank-9 decomposition
        
        # For each of the 9 scalar multiplications in the 3x3 matrix product:
        # (A_comb * B_comb)[i,k] = sum_j A_comb[i,j] * B_comb[j,k]
        # But A_comb[i,j] = sum_{(rb,cb,coeff) in strassen_A[s]} coeff * A[rb*bs+i, cb*bs+j]
        # B_comb[j,k] = sum_{(rb,cb,coeff) in strassen_B[s]} coeff * B[rb*bs+j, cb*bs+k]
        
        # So the term is:
        # sum_{j=0}^{2} sum_{(ra,ca,ca_coeff) in strassen_A[s]} sum_{(rb,cb,cb_coeff) in strassen_B[s]}
        #   ca_coeff * cb_coeff * A[ra*bs+i, ca*bs+j] * B[rb*bs+j, cb*bs+k]
        
        # This is a sum of products of linear combinations. We can't directly decompose this
        # into rank-1 tensors without expanding.
        
        # Actually, the correct way: each Strassen product M_s is a 3x3 matrix.
        # We use the trivial decomposition for each 3x3 multiplication, which has rank 9.
        # So for each s, we add 9 rank-1 terms.
        
        for j in range(3):  # inner dimension
            for i in range(3):  # row of result
                for k in range(3):  # col of result
                    # This term contributes to C[i,j,k] = A[i,j] * B[j,k]
                    # But with the Strassen combinations
                    
                    # U part: sum over strassen_A[s] contributions
                    # V part: sum over strassen_B[s] contributions
                    
                    col_idx = s * 9 + j * 3 + i  # Hmm, this doesn't seem right
                    
                    # Let me reconsider the indexing
                    # For Strassen product s, we have 9 scalar multiplications (one for each (i,j) pair in the 3x3 product)
                    # Actually, for a 3x3 matrix product, we have 3*3*3 = 27 scalar multiplications in total,
                    # but the trivial tensor decomposition has rank 9 (one for each (row, col) of the output,
                    # where each rank-1 term handles the sum over the inner dimension).
                    
                    # No, that's not right either. The trivial decomposition for n x n matrix multiplication
                    # has rank n^2, not n^3. Each rank-1 term corresponds to one (i,j) pair in the output,
                    # and involves the sum over k of A[i,k] * B[k,j].
                    
                    # So for 3x3, rank = 9. Each term t corresponds to output position (i,k) where
                    # t = i*3 + k.
                    # U[:, t] has A[i,j] for all j, but only for the specific i.
                    # V[:, t] has B[j,k] for all j, but only for the specific k.
                    # W[:, t] has C[i,k] = 1 for that position.
                    
                    # Wait, I think I'm overcomplicating this. Let me use a simpler approach.
                    
                    pass  # Skip for now, use the previous working code
    
    # Let me just use the previous working code with rank 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_s0 = [(0,0,1), (1,1,1)]
    B_s0 = [(0,0,1), (1,1,1)]
    C_s0 = [(0,0,1), (1,1,1)]
    
    A_s1 = [(1,0,1), (1,1,1)]
    B_s1 = [(0,0,1)]
    C_s1 = [(1,0,1), (1,1,-1)]
    
    A_s2 = [(0,0,1)]
    B_s2 = [(0,1,1), (1,1,-1)]
    C_s2 = [(0,1,1), (1,1,1)]
    
    A_s3 = [(1,1,1)]
    B_s3 = [(1,0,1), (0,0,-1)]
    C_s3 = [(0,0,1), (1,0,1)]
    
    A_s4 = [(0,0,1), (0,1,1)]
    B_s4 = [(1,1,1)]
    C_s4 = [(0,0,-1), (0,1,1)]
    
    A_s5 = [(1,0,1), (0,0,-1)]
    B_s5 = [(0,0,1), (0,1,1)]
    C_s5 = [(1,1,1)]
    
    A_s6 = [(0,1,1), (1,1,-1)]
    B_s6 = [(1,0,1), (1,1,1)]
    C_s6 = [(0,0,1)]
    
    A_combs = [A_s0, A_s1, A_s2, A_s3, A_s4, A_s5, A_s6]
    B_combs = [B_s0, B_s1, B_s2, B_s3, B_s4, B_s5, B_s6]
    C_combs = [C_s0, C_s1, C_s2, C_s3, C_s4, C_s5, C_s6]
    
    block_size = 2
    num_blocks = 3
    
    for i_b in range(num_blocks):
        for j_b in range(num_blocks):
            for k_b in range(num_blocks):
                for s in range(7):
                    col = ((i_b * num_blocks + j_b) * num_blocks + k_b) * 7 + s
                    
                    for rb, cb, coeff in A_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = k_b * block_size + cb
                        U[i_glob * m + j_glob][col] += coeff
                        
                    for rb, cb, coeff in B_combs[s]:
                        j_glob = k_b * block_size + rb
                        k_glob = j_b * block_size + cb
                        V[j_glob * p + k_glob][col] += coeff
                        
                    for rb, cb, coeff in C_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = j_b * block_size + cb
                        W[j_glob * n + i_glob][col] += coeff
                        
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

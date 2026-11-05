DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block size 3
    bs = 3
    # Strassen coefficients for 2x2 blocks
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # Output combinations
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Inner decomposition: Schoolbook for 3x3
    # Rank for inner is 27
    # Total rank 7 * 27 = 189
    
    rank_inner = bs * bs * bs
    rank_total = 7 * rank_inner
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Strassen definitions
    # A_coeffs[t][block_i][block_j]
    # B_coeffs[t][block_i][block_j]
    # C_coeffs[out_block_i][out_block_j][t]
    
    # Block indices: 0, 1
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M0: A00 + A11
        [[0, 0], [1, 1]], # M1: A10 + A11
        [[1, 0], [0, 0]], # M2: A00
        [[0, 0], [0, 1]], # M3: A11
        [[1, 1], [0, 0]], # M4: A00 + A01
        [[-1, 0], [1, 0]],# M5: A10 - A00
        [[0, 1], [0, -1]] # M6: A01 - A11
    ]
    
    B_coeffs = [
        [[1, 0], [0, 1]], # M0: B00 + B11
        [[1, 0], [0, 0]], # M1: B00
        [[0, 1], [0, -1]],# M2: B01 - B11
        [[0, -1], [1, 0]],# M3: B10 - B00
        [[0, 0], [0, 1]], # M4: B11
        [[1, 1], [0, 0]], # M5: B00 + B01
        [[0, 0], [1, 1]]  # M6: B10 + B11
    ]
    
    C_coeffs = [
        [[1, 0], [0, 1]], # C00: M0 + M3 - M4 + M6 -> wait, C00 is sum over t
        # C00 = 1*M0 + 0*M1 + 0*M2 + 1*M3 - 1*M4 + 0*M5 + 1*M6
        # C01 = 0*M0 + 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6
        # C10 = 0*M0 + 1*M1 + 0*M2 + 1*M3 + 0*M4 + 0*M5 + 0*M6
        # C11 = 1*M0 - 1*M1 + 1*M2 + 0*M3 + 0*M4 + 1*M5 + 0*M6
    ]
    
    # Let's define C_coeffs properly as [out_block_i][out_block_j][t]
    C_coeffs = [
        [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0]   # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],  # C10
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
    ]
    
    # Fill U, V, W
    # Global indices:
    # A: i in 0..5, j in 0..5. Block bi = i//3, bj = j//3. Local li = i%3, lj = j%3.
    # B: j in 0..5, k in 0..5. Block bj = j//3, bk = k//3. Local lj = j%3, lk = k%3.
    # C: k in 0..5, i in 0..5. Block bk = k//3, bi = i//3. Local lk = k%3, li = i%3.
    
    for t in range(7):
        for r in range(rank_inner):
            col = t * rank_inner + r
            
            # Determine inner local indices from r
            # r = li * bs * bs + lj * bs + lk ?
            # Schoolbook: r corresponds to (i, j, k) in local 3x3
            # U[local_i * bs + local_j, r] = 1 if r == local_i * bs * bs + local_j * bs + local_k
            # V[local_j * bs + local_k, r] = 1
            # W[local_k * bs + local_i, r] = 1
            
            local_i = r // (bs * bs)
            local_j = (r // bs) % bs
            local_k = r % bs
            
            # Fill U
            # U global row: global_i * m + global_j
            # U global col: col
            # U value: sum of A_coeffs[t][bi][bj] for all bi,bj such that
            # global_i = bi * bs + local_i, global_j = bj * bs + local_j
            # Since local_i, local_j are fixed, global_i, global_j are fixed for each bi, bj.
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = A_coeffs[t][bi][bj]
                    if coeff_a != 0:
                        global_i = bi * bs + local_i
                        global_j = bj * bs + local_j
                        U[global_i * m + global_j][col] += coeff_a
            
            # Fill V
            for bj in range(2):
                for bk in range(2):
                    coeff_b = B_coeffs[t][bj][bk]
                    if coeff_b != 0:
                        global_j = bj * bs + local_j
                        global_k = bk * bs + local_k
                        V[global_j * p + global_k][col] += coeff_b
            
            # Fill W
            # W global row: global_k * n + global_i
            # W value: sum of C_coeffs[bi][bj][t] for all bi,bj such that
            # global_k = bk * bs + local_k, global_i = bi * bs + local_i
            # Wait, C_coeffs is indexed by output block (bi, bj) where bi is row block, bj is col block.
            # Output C block (bi, bj) receives contribution from M_t with weight C_coeffs[bi][bj][t].
            # The local tensor for M_t produces C_local[local_k, local_i].
            # This maps to global C[global_k, global_i] where global_k = bk * bs + local_k, global_i = bi * bs + local_i.
            # So for a fixed local_k, local_i, and t, we sum over all output blocks (bi, bj) that contain this local position?
            # No. The output block (bi, bj) is fixed.
            # The local position (local_k, local_i) is within the block.
            # So for a specific output block (bi, bj), the global indices are:
            # global_i = bi * bs + local_i
            # global_k = bk * bs + local_k
            # The weight is C_coeffs[bi][bj][t].
            
            for bi in range(2):
                for bj in range(2):
                    coeff_c = C_coeffs[bi][bj][t]
                    if coeff_c != 0:
                        global_i = bi * bs + local_i
                        global_k = bj * bs + local_k # Wait, bk is the column block index of C?
                        # C is p x n. Rows are k, Cols are i.
                        # Block (bi, bj) means row block bi, col block bj.
                        # So global row index in C (which is k) is determined by row block bi?
                        # No. C = A B. A is n x m, B is m x p. C is n x p.
                        # Prompt says: W[k*n+i, r]. This implies C is p x n?
                        # "output factor follows the official trace convention W[k*n+i,r]"
                        # Usually C is n x p. W index k*n+i implies row k, col i.
                        # If C is n x p, then row i, col j.
                        # Let's check the seed program.
                        # Seed: W[k*n+i, c].
                        # Schoolbook: U[i*m+j], V[j*p+k], W[k*n+i].
                        # This implies A is n x m, B is m x p, C is p x n?
                        # No, matrix multiplication A (n x m) * B (m x p) = C (n x p).
                        # The tensor is $T_{i,j,k} = \delta_{i,j} \delta_{j,k}$? No.
                        # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
                        # The decomposition is $C_{i,k} = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}$?
                        # The seed has U shape (n*m, rank). Index i*m+j.
                        # V shape (m*p, rank). Index j*p+k.
                        # W shape (p*n, rank). Index k*n+i.
                        # This corresponds to $C_{i,k} = \sum_r U_{i,j} V_{j,k} W_{k,i}$?
                        # No, the standard tensor decomposition of matrix multiplication is:
                        # $T_{i,j,k} = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}$ is not standard.
                        # Standard: $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
                        # Tensor $T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}$? No.
                        # The tensor is $T_{i,j,k}$ such that $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = \text{tr}(C^T C)$?
                        # The prompt says "exactly decompose the matrix multiplication tensor".
                        # The tensor is defined by $T_{i,j,k} = 1$ if indices match?
                        # In the seed:
                        # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
                        # This implies the term is $A_{i,j} B_{j,k} C_{k,i}$?
                        # If we contract this with A and B, we get $\sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}$.
                        # This is the trace of $C^T A B$?
                        # Regardless, I must follow the seed's indexing convention.
                        # Seed W index: $k \cdot n + i$.
                        # My W fill:
                        # global_i is the "i" index (row of A, col of C? No, col of C in standard, but here W is k*n+i).
                        # In seed, W[k*n+i] corresponds to the output element at row k, col i?
                        # If C is n x p, row i, col k.
                        # Then W index should be i*p + k.
                        # But seed uses k*n + i.
                        # This implies the output matrix is stored as p x n?
                        # Or the indices are swapped.
                        # Let's assume the convention is:
                        # U index: i*m + j
                        # V index: j*p + k
                        # W index: k*n + i
                        # This matches the seed.
                        # My C_coeffs logic:
                        # Output block (bi, bj).
                        # bi is the block row index of C?
                        # bj is the block col index of C?
                        # If C is n x p, then row block bi, col block bj.
                        # Local row li, local col lk.
                        # Global row i = bi * bs + li.
                        # Global col k = bj * bs + lk.
                        # W index = k * n + i = (bj * bs + lk) * n + (bi * bs + li).
                        
                        # In my loop:
                        # local_i corresponds to "i" (row of A, row of C?)
                        # local_j corresponds to "j" (col of A, row of B?)
                        # local_k corresponds to "k" (col of B, col of C?)
                        
                        # Seed:
                        # U[i*m+j]
                        # V[j*p+k]
                        # W[k*n+i]
                        
                        # So i is row of A, j is col of A / row of B, k is col of B.
                        # C is formed by contracting j.
                        # The result is associated with i and k.
                        # W index uses k and i.
                        
                        # So in my block mapping:
                        # i is global row of A.
                        # k is global col of B.
                        
                        # My C_coeffs[bi][bj]:
                        # bi should be the block index for i (row of A).
                        # bj should be the block index for k (col of B).
                        
                        # So:
                        # global_i = bi * bs + local_i
                        # global_k = bj * bs + local_k
                        
                        # W index = global_k * n + global_i
                        
                        # My previous code:
                        # global_i = bi * bs + local_i
                        # global_k = bj * bs + local_k
                        # W[global_k * n + global_i][col] += coeff_c
                        
                        # This looks correct.
                        
                        W[global_k * n + global_i][col] += coeff_c
    return {"U": U, "V": V, "W": W}

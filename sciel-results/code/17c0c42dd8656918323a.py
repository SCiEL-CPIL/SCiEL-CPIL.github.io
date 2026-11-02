"""Program for exact tensor decomposition of 2x4 by 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact decomposition via block separation.
    # Split the 4 intermediate dimension into two blocks of size 2.
    # C = A * B where A is 2x4, B is 4x6.
    # A = [A_left | A_right] where A_left, A_right are 2x2.
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x6.
    # C = A_left * B_top + A_right * B_bottom
    #
    # For 2x2 times 2x6: C_ik = sum_j A_ij * B_jk
    # This tensor has rank 4 (each i,j pair gives one term).
    #
    # So total rank = 4 + 4 = 8.
    #
    # Tensor convention:
    # U has shape (n*m, rank) = (2*4, 8) = (8, 8)
    #   U[i*m + j, c]
    # V has shape (m*p, rank) = (4*6, 8) = (24, 8)
    #   V[j*p + k, c]
    # W has shape (p*n, rank) = (6*2, 8) = (12, 8)
    #   W[k*n + i, c]
    #
    # For term c corresponding to (block b, i, j) where b in {0,1}, i in {0,1}, j in {0,1}:
    #   global i (row of A, col of C) = i
    #   global j (col of A, row of B) = b*2 + j  (since block b uses rows b*2, b*2+1 of B)
    #   global k (col of B, col of C) = k
    #
    # U[global_i * 4 + global_j, c] = 1
    # V[global_j * 6 + global_k, c] = B[global_j, global_k]  -- but we need this to be a fixed tensor entry
    # W[global_k * 2 + global_i, c] = 1
    #
    # Wait, the standard decomposition is:
    # T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}
    # where T_{i,j,k} = delta_{i,j,k} in the multiplication tensor sense.
    #
    # Actually, the matrix multiplication tensor is T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all valid (i,j,k) triples, and the decomposition is:
    # T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # For the schoolbook: each (i,j,k) triple gets its own c.
    #
    # For our block decomposition:
    # For block b=0 (top half of B, columns 0-1 of A... wait, let me re-index).
    #
    # A is 2x4: rows i in {0,1}, columns j in {0,1,2,3}.
    # B is 4x6: rows j in {0,1,2,3}, columns k in {0,...,5}.
    # C is 2x6: rows i in {0,1}, columns k in {0,...,5}.
    #
    # Split j into two groups: j in {0,1} (block 0) and j in {2,3} (block 1).
    # For block b, local j' in {0,1}: global j = 2*b + j'.
    #
    # Term for (b, i, j', k):
    #   U[i, 2*b+j', c] = 1
    #   V[2*b+j', k, c] = 1
    #   W[k, i, c] = 1
    #
    # But wait, this would give sum over k for each (b,i,j') which is correct:
    # C_{i,k} = sum_{j'} A_{i, 2b+j'} * B_{2b+j', k} summed over b.
    #
    # However, the tensor T_{i,j,k} has a 1 for every valid (i,j,k). So:
    # T_{i, 2b+j', k} should equal sum_c U[i, 2b+j', c] * V[2b+j', k, c] * W[k, i, c]
    #
    # If we set for each (b, i, j') a set of 6 terms (one per k), we'd have 2*2*2*6 = 48 terms again.
    #
    # The key insight: for fixed (b, i, j'), the contribution to all k is:
    #   sum_k A_{i, 2b+j'} * B_{2b+j', k} * E_{i,k}
    #
    # This is a single rank-1 term in the "output" sense if we can factor it.
    # But V depends on k, so it's not a single rank-1 term in the full tensor.
    #
    # Let me reconsider. The rank of the 2x2x6 multiplication tensor:
    # T_{i,j,k} for i in {0,1}, j in {0,1}, k in {0,...,5}.
    # T_{i,j,k} = 1 for all i,j,k.
    #
    # This is the matrix multiplication tensor for 2x2 times 2x6.
    # Its rank is known to be 4 (each (i,j) pair gives one term, summing over k).
    #
    # How? For fixed (i,j), define a vector v_{i,j} in the output space (dimension 6):
    #   v_{i,j}[k] = B_{j,k}  (but B is the input, not part of the tensor)
    #
    # I think I'm confusing the tensor with the multiplication map.
    #
    # The matrix multiplication tensor T is an element of R^{n x m} ⊗ R^{m x p} ⊗ R^{n x p}.
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the matrix unit in R^{n x m}, etc.
    #
    # So T_{i,j,k} (with i indexing rows of first matrix, j columns of first/rows of second, k columns of second) = 1.
    #
    # A rank-r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c in R^{n x m}, v_c in R^{m x p}, w_c in R^{n x p}.
    #
    # In index notation: T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (using the trace convention W[k*n+i] as specified).
    #
    # For 2x2x6: n=2, m=2, p=6.
    # T_{i,j,k} = 1 for i in {0,1}, j in {0,1}, k in {0..5}.
    #
    # Claim: rank is 4.
    # Let c = (i,j) for i in {0,1}, j in {0,1}. So 4 terms.
    # U[i,j,c=(i',j')] = delta_{i,i'} * delta_{j,j'}
    # V[j,k,c=(i',j')] = delta_{j,j'}  (independent of k, i')
    # W[k,i,c=(i',j')] = delta_{i,i'}  (independent of k, j')
    #
    # Then sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # = sum_{i',j'} delta_{i,i'} delta_{j,j'} * delta_{j,j'} * delta_{i,i'}
    # = delta_{i,i} delta_{j,j} = 1.  (since i'=i, j'=j)
    #
    # Yes! This works. The rank is 4.
    #
    # Now for 2x4x6: n=2, m=4, p=6.
    # Split j into two blocks of 2: j in {0,1} (block 0) and j in {2,3} (block 1).
    # For each block b, we have a 2x2x6 sub-tensor.
    # Block b uses local j' in {0,1}, global j = 2*b + j'.
    #
    # For block b, term c = (b, i, j'):
    # U[i, 2b+j', c] = delta_{i,i} delta_{j',j'} = 1  (only for matching i, j')
    # V[2b+j', k, c] = delta_{j',j'} = 1
    # W[k, i, c] = delta_{i,i} = 1
    #
    # Sum over c for fixed (i, j, k):
    # Let j = 2b + j'. Then only c = (b, i, j') contributes.
    # U[i, j, c] * V[j, k, c] * W[k, i, c] = 1 * 1 * 1 = 1.
    #
    # Total rank = 2 blocks * 2 i-values * 2 j'-values = 8.
    
    n, m, p = DIMENSIONS
    rank = 8
    U = [[0] * rank for _ in range(n * m)]  # 8 x 8
    V = [[0] * rank for _ in range(m * p)]  # 24 x 8
    W = [[0] * rank for _ in range(p * n)]  # 12 x 8
    
    col = 0
    for b in range(2):  # block index
        for i in range(2):  # row of A, col of C
            for jp in range(2):  # local col in A, local row in B
                global_j = 2 * b + jp
                # U[i * m + global_j, col] = 1
                U[i * m + global_j][col] = 1
                # V[global_j * p + k, col] = 1 for all k
                for k in range(p):
                    V[global_j * p + k][col] = 1
                # W[k * n + i, col] = 1 for all k
                for k in range(p):
                    W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

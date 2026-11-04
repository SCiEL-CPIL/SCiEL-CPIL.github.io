import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += Ur[:,:,c][:, :, None] * Vr[:,:,c][None, :, :] * Wr[:,:,c][None, None, :]
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err <= tol, max_err

def construct():
    n, m, p = 3, 3, 8
    
    # We construct a rank 24 decomposition.
    # Split B (3x8) into B1 (3x4) and B2 (3x4).
    # C = A B1 + A B2.
    # Each term A Bk is M_{3,3,4}.
    # M_{3,3,4} is isomorphic to M_{4,3,3}.
    # M_{4,3,3} can be decomposed into 4 independent M_{1,3,3} multiplications.
    # Rank(M_{1,3,3}) = 3.
    # So Rank(M_{3,3,4}) = 12.
    # Total rank = 12 + 12 = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)] # 9 x 24
    V = [[0] * rank for _ in range(m * p)] # 24 x 24
    W = [[0] * rank for _ in range(p * n)] # 24 x 24
    
    # Helper to build M_{1,3,3} decomposition for a specific row index i in the 4x3 matrix
    # This corresponds to computing row i of C^T = B^T A^T.
    # But we are working in the original frame: C = A B.
    # Let's build the 12-term decomposition for A (3x3) * B_part (3x4).
    # Indices: i (0..2), j (0..2), k (0..3).
    # We use the isomorphism to M_{4,3,3}.
    # In M_{4,3,3}, we compute Z_{i', k'} = sum_j X_{i', j} Y_{j, k'}.
    # Here i' is 0..3 (columns of B_part), k' is 0..2 (rows of C).
    # X is B_part^T (4x3), Y is A^T (3x3).
    # Z is C_part^T (4x3).
    # The decomposition of M_{4,3,3} into 4 blocks of M_{1,3,3}:
    # For each i' in 0..3:
    #   Compute z_{i', k'} = sum_j x_{i', j} y_{j, k'}.
    #   This is a vector (x_{i', :}) times matrix Y.
    #   Rank 3 decomposition:
    #   For j in 0..2:
    #     term_j: x_{i', j} * y_{j, k'} * 1.
    #     In tensor form: U_{i, j, c} * V_{j, k, c} * W_{k, i, c}.
    #     Wait, the mapping of indices needs to be careful.
    
    # Let's map the indices explicitly.
    # Original: C_{ik} = sum_j A_{ij} B_{jk}.
    # We split k into k1 (0..3) and k2 (4..7).
    # For the first part (k=0..3):
    #   We want to compute C_{i, k} for i=0..2, k=0..3.
    #   We use 12 terms.
    #   Let's assign indices c = 0..11 for the first part.
    #   For each k in 0..3 (4 values):
    #     For each j in 0..2 (3 values):
    #       c = k * 3 + j.
    #       We need U, V, W such that sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 if (i,j,k) matches, else 0?
    #       No, the tensor T_{i,j,k} = 1.
    #       The decomposition is sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1.
    #       
    #       For the term corresponding to fixed k and j:
    #       We want it to contribute to C_{i,k}.
    #       C_{i,k} = sum_j A_{ij} B_{jk}.
    #       So for a specific j and k, the contribution is A_{ij} B_{jk}.
    #       This looks like a product of A_{ij} and B_{jk}.
    #       So we can set:
    #       U_{i,j,c} = delta_{j, j_c} (1 if j == j_c)
    #       V_{j,k,c} = delta_{k, k_c} (1 if k == k_c)
    #       W_{k,i,c} = 1
    #       Then sum_c U V W = sum_{c: j_c=j, k_c=k} 1 * 1 * 1 = 1.
    #       This is just the schoolbook decomposition!
    #       Rank = 3 * 3 * 4 = 36 for one part. Total 72.
        
    # Wait, the rank 12 decomposition for M_{3,3,4} is NOT schoolbook.
    # Schoolbook is 36.
    # The rank 12 decomposition comes from M_{4,3,3} being 4 * M_{1,3,3}.
    # In M_{4,3,3}, the tensor is T_{i', j, k'} = 1.
    # Decomposition: sum_{c} U'_{i',j,c} V'_{j,k',c} W'_{k',i',c} = 1.
    # For each i' (0..3), we use 3 terms (j=0,1,2).
    # c = i' * 3 + j.
    # U'_{i',j,c} = 1 if j == j_c, else 0.
    # V'_{j,k',c} = 1 if k' == k'_c? No.
    # In M_{1,3,3}, we compute z_{k'} = sum_j x_j y_{j,k'}.
    # Tensor T_{j, k'} = 1? No, T_{i', j, k'}.
    # If we fix i', it's T_{j, k'} = 1.
    # Decomposition of T_{j,k'}:
    #   sum_j x_j y_{j,k'}.
    #   This is schoolbook for 1x3 x 3x3. Rank 3.
    #   Terms: for each j, x_j * y_{j,k'} * 1.
    #   So U'_{i',j,c} = 1 (x_j), V'_{j,k',c} = 1 (y_{j,k'}), W'_{k',i',c} = 1.
    #   Wait, the factors must be such that their product gives the tensor.
    #   T_{i',j,k'} = sum_c U'_{i',j,c} V'_{j,k',c} W'_{k',i',c}.
    #   If we set:
    #   For c = i'*3 + j:
    #   U'_{i',j,c} = 1
    #   V'_{j,k',c} = 1
    #   W'_{k',i',c} = 1
    #   Then sum_c ... = 1 if (i',j,k') is valid?
    #   No, this sums to 1 for ALL i',j,k'.
    #   This is the schoolbook decomposition of M_{4,3,3} with rank 4*3*3 = 36?
    #   No, for fixed i', we sum over j.
    #   The term for c=i'*3+j is non-zero only for that i' and j.
    #   For a specific (i',j,k'), the sum is over c.
    #   Only c=i'*3+j contributes.
    #   Value = 1*1*1 = 1.
    #   So this IS a valid decomposition.
    #   Rank = 4 * 3 = 12.
    #   Why? Because for each i', we have 3 terms.
    #   Total 12 terms.
    #   This works because the tensor T_{i',j,k'} is 1 for all i',j,k'.
    #   And the terms are disjoint in i' and j.
    
    # So, we can map this back to M_{3,3,4}.
    # M_{3,3,4} indices: i (0..2), j (0..2), k (0..3).
    # Isomorphism: i' = k, j' = j, k' = i.
    # T_{i,j,k} = T'_{k,j,i}.
    # Decomposition:
    # For each k (0..3) and j (0..2):
    #   c = k * 3 + j.
    #   U_{i,j,c} = U'_{k,j,i}? No.
    #   We need U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
    #   From M_{4,3,3}: U'_{k,j,c} V'_{j,i,c} W'_{i,k,c} = 1.
    #   So we set:
    #   U_{i,j,c} = U'_{k,j,c} = 1 (since U' is 1 for matching i',j)
    #   V_{j,k,c} = V'_{j,i,c} = 1 (since V' is 1 for matching j,k')
    #   W_{k,i,c} = W'_{i,k,c} = 1 (since W' is 1 for matching k',i')
    
    # So for the first part (k=0..3):
    for k in range(4):
        for j in range(3):
            c = k * 3 + j
            for i in range(3):
                U[i*3+j][c] = 1
                V[j*4+k][c] = 1 # V index is j*p+k? No, V is m x p. j in 0..2, k in 0..3.
                # V shape is (m, p) = (3, 8).
                # In this part, we use k=0..3.
                # V[j, k] = 1.
                W[k*3+i][c] = 1 # W shape is (p, n) = (8, 3).
                # W index is k*n+i. k in 0..3, i in 0..2.
    
    # For the second part (k=4..7):
    for k in range(4, 8):
        for j in range(3):
            c = 12 + (k-4) * 3 + j
            for i in range(3):
                U[i*3+j][c] = 1
                V[j*8+k][c] = 1
                W[k*3+i][c] = 1

    return {"U": U, "V": V, "W": W}

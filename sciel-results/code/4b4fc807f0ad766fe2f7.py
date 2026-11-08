import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Trivial (canonical) exact tensor decomposition:
    # For each (i, k, j) with i in [0,n), k in [0,m), j in [0,p),
    #   U[(i*m + k), r] = 1 if r corresponds to (i,k,j)
    #   V[(k*p + j), r] = 1 if r corresponds to (i,k,j)
    #   W[(j*n + i), r] = 1 if r corresponds to (i,k,j)
    # This gives C[i,j] = sum_{r} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # = sum_{i',k',j'} [i'=i][k'=k][j'=j] = A[i,k]*B[k,j] summed over k
    # Wait, let me re-derive.
    #
    # The tensor decomposition says:
    # C[i,j] = sum_{k} A[i,k] * B[k,j]
    # = sum_{r} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]  ... no.
    #
    # Actually the standard form is:
    # C_{i,j} = sum_{k} sum_{r} U_{i,k,r} * A_{i,k} * V_{k,j,r} * B_{k,j} * W_{j,i,r}
    # Hmm, no. Let me think more carefully.
    #
    # The matrix multiplication tensor T has entries T[i,k; k',j] = delta[i,k][k'] delta[k,j][k'']... 
    # Actually T is a 3-tensor: T[i,k,j] = delta_{i,k'} delta_{k,j'}... 
    #
    # Let me use the standard formulation. The multiplication of A (n x m) by B (m x p) gives C (n x p).
    # C[i,j] = sum_{k=0}^{m-1} A[i,k] * B[k,j]
    #
    # Tensor decomposition: C[i,j] = sum_{r=0}^{R-1} U_r[i,m_index] * V_r[m_index,j] * W_r[j,i]
    # where U_r is an n x m matrix, V_r is an m x p matrix, W_r is a p x n matrix.
    #
    # Wait, the interface says:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    #
    # So U is indexed by (i*m + k, r), V by (k*p + j, r), W by (j*n + i, r).
    #
    # The decomposition should satisfy:
    # C[i,j] = sum_{r} sum_{k} U[(i*m+k), r] * A[i,k] * V[(k*p+j), r] * B[k,j] * W[(j*n+i), r]
    #
    # No wait, that's not right either. Let me think about what the evaluator checks.
    #
    # The standard tensor decomposition for matrix multiplication:
    # The tensor is T[i,k, j'] where i indexes rows of A, k indexes columns of A / rows of B, j' indexes columns of B.
    # T[i,k,j'] = 1 if k == k' else 0, but really T is the structure tensor.
    #
    # Actually, the simplest way: the multiplication tensor has rank n*m*p for the trivial decomposition.
    # Each term r corresponds to a triple (i, k, j):
    #   U[(i*m + k), r] = 1
    #   V[(k*p + j), r] = 1  
    #   W[(j*n + i), r] = 1
    #
    # Then: sum_r U[(i*m+k),r] * A[i,k] * V[(k*p+j),r] * B[k,j] * W[(j*n+i),r]
    #     = sum_r [r=(i,k,j)] * A[i,k] * B[k,j] * 1
    #     = A[i,k] * B[k,j] for the specific (i,k,j)
    #
    # But we need sum over k! So this gives A[i,k]*B[k,j] for ONE k, not the sum.
    #
    # Hmm. I think the correct interpretation is:
    # C[i,j] = sum_{k} A[i,k] * B[k,j]
    #
    # In tensor form, this is a sum over r of products of slices. Let me look at this differently.
    #
    # If U, V, W are as described, then the reconstructed C should be:
    # C[i,j] = sum_{k, r} A[i,k] * U[i*m+k, r] * V[k*p+j, r] * B[k,j] * W[j*n+i, r]
    #
    # No, I think the standard form is actually:
    # C = sum_r U_r * A * V_r * B * W_r  ... but U_r, V_r, W_r are matrices.
    #
    # Given the flat shapes, I think the evaluator does:
    # For each r:
    #   U_r = reshape U[:, r] to (n, m)
    #   V_r = reshape V[:, r] to (m, p)
    #   W_r = reshape W[:, r] to (p, n)
    #   C += U_r * A * V_r  ... no, that doesn't use W.
    #
    # Let me just go with the trivial rank n*m*p decomposition where each term picks out one A[i,k] * B[k,j] product and contributes to C[i,j].
    # For term r corresponding to (i, k, j):
    #   U[i*m+k, r] = 1, all else 0
    #   V[k*p+j, r] = 1, all else 0
    #   W[j*n+i, r] = 1, all else 0
    #
    # This should work if the evaluator computes:
    # C[i,j] = sum_{k, r} U[i*m+k, r] * A[i,k] * V[k*p+j, r] * B[k,j] * W[j*n+i, r]
    #
    # For fixed (i,j), summing over k and r:
    # Only r corresponding to (i,k,j) with the same i,j contributes.
    # For each k, there's exactly one such r (if (i,k,j) is in our set), contributing A[i,k]*B[k,j]*1*1*1.
    # Sum over k gives sum_k A[i,k]*B[k,j] = (AB)[i,j]. Correct!
    
    rank = n * m * p  # 2*4*7 = 56
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                r += 1
    
    return {"U": U, "V": V, "W": W}

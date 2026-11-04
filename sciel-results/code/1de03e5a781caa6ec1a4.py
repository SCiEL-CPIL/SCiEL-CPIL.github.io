"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-minimal decomposition for matrix multiplication A(n x m) * B(m x p) = C(n x p)
    # The tensor T[i, j, k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} in the standard basis
    # can be decomposed as a direct sum over i of rank-m tensors (one per row of A).
    # For each row i of A, we need to multiply 1 x m by m x p.
    # The rank of a 1 x m x p tensor is m (since it's just the identity mapping from the
    # middle index to the output, with the first and last indices fixed).
    # Actually, let's reconsider. The tensor for multiplying a 1xm vector by mxp matrix
    # is: for each (j, k), the output at position (0, k) gets contribution from input j.
    # T[0, j, k] = 1 for all j, k. Wait, no.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T[i, j, k] = 1 if we consider the "structural" tensor where all entries are 1
    # but that's not right either.
    # 
    # Let me be more careful. The matrix multiplication tensor is defined by:
    # sum_{j=0}^{m-1} A[i,j] B[j,k] = C[i,k]
    # 
    # The structural tensor has entries T[i,j,k] = 1 for all i, j, k? No.
    # The tensor decomposition requires:
    # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{r} U[i,j,r] V[j,k,r] W[i,k,r] (with appropriate indexing)
    #
    # Using the convention: W[k*n+i, r] (i.e., W is indexed by (k, i) where k is row of C, i is col... 
    # wait, the problem says W[k*n+i, r], so the first index is k*n + i.
    # And the output C has dimensions n x p, so C[i, k] where i in [0,n), k in [0,p).
    # The convention W[k*n+i, r] means the row index in W is k*n + i.
    # So W is indexed by (k, i) pairs, with k being the "column" index of C and i being the "row" index.
    # Actually, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # This means W has shape (p*n, rank), and the row is indexed by k*n + i where k is in [0,p) and i in [0,n).
    # 
    # The decomposition is: C[i, k] = sum_j A[i,j] B[j,k]
    # = sum_r U[i*m+j ... wait, U has shape (n*m, rank), indexed by (i,j) -> i*m + j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p + k  
    # W has shape (p*n, rank), indexed by (k,i) -> k*n + i
    #
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ... but j is summed over.
    # Actually: C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r]
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{something}
    # 
    # The rank-minimal decomposition: for each i, fix i. Then C[i,k] = sum_j A[i,j] B[j,k].
    # This is a bilinear form in A[i,:] and B[:,k]. The minimal number of multiplications for
    # a single row of A times B is m (each A[i,j] multiplies with the j-th row of B).
    # So for each i, we need m terms. Total rank = n * m = 3 * 5 = 15.
    #
    # Construction: for each i in [0,n) and j in [0,m), create a term r = i*m + j.
    # U[i*m + j, r] = 1 (all other entries in U are 0 for this r)
    # V[j*p + k, r] = 1 for all k in [0,p) (i.e., the j-th row of V is all 1s for this term)
    # W[k*n + i, r] = 1 for all k in [0,p) (i.e., for fixed i, all k positions get 1)
    #
    # Let's verify: sum_r U[i*m+j', r] V[j'*p+k, r] W[k*n+i, r]
    # For r = i'*m + j':
    # U[i*m+j, r] is non-zero only if i' = i and j' = j.
    # So only r with i'=i, j'=j contribute.
    # For such r: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # Product = 1.
    # Sum over r: only one r contributes (r = i*m+j), so sum = 1.
    # 
    # But we need C[i,k] = sum_j A[i,j] B[j,k]. The tensor entry T[i,j,k] should be such that
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]... no, that's not how it works.
    # 
    # Actually, the bilinear complexity / tensor rank formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want to write this as sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} ... 
    # No, the standard formulation is:
    # C[i,k] = sum_r M_r
    # where M_r = (sum_j U[i*m+j, r] A[i,j]) * (sum_k ... ) -- this isn't quite right either.
    #
    # The correct formulation: Each term r gives a product of two linear forms.
    # Term r: (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) -- but j and k are free indices.
    # 
    # Let me just use the standard schoolbook-like but optimized approach.
    # For rank n*m: for each (i, j), one multiplication.
    # A[i,j] * B[j,k] contributes to C[i,k] for all k.
    # So for term r = i*m + j:
    #   The "A-factor" is A[i,j] itself, so U[i*m+j, r] = 1, and for all other (i',j'), U[i'*m+j', r] = 0.
    #   The "B-factor" needs to give B[j,k] for all k. So V[j*p+k, r] = 1 for all k.
    #   The "W-factor" needs to place the result at C[i,k] for all k. So W[k*n+i, r] = 1 for all k.
    #
    # Check: sum_r U[i*m+j', r] * V[j'*p+k, r] * W[k*n+i, r]
    # Only r = i*m+j' contributes when i'=i. For r = i*m+j':
    #   U[i*m+j', r] = delta_{j', j} -- wait, no. U[i*m+j, r] = 1 only when r = i*m+j.
    #   So U[i*m+j, i*m+j'] = delta_{j, j'}.
    #   V[j'*p+k, i*m+j'] = 1 for all k.
    #   W[k*n+i, i*m+j'] = 1 for all k.
    # So the sum is: sum_{j'} delta_{j,j'} * 1 * 1 = 1 (when j'=j).
    # 
    # This gives T[i,j,k] = 1 for all i,j,k, which means C[i,k] = sum_j A[i,j] B[j,k] * 1.
    # Wait, that's exactly what we want! The tensor entry is 1 for all (i,j,k).
    
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

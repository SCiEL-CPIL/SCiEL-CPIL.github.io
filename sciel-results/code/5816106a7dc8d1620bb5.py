import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Model the 2x3x6 multiplication tensor.
    # We want to find a rank-minimal decomposition.
    # The schoolbook rank is n*m*p = 2*3*6 = 36.
    
    # The directive suggests using a change of basis to diagonalize the interaction.
    # However, for a 2x3 matrix multiplication, the tensor is typically full rank in the trivial sense.
    # Let's check if we can do better than 36.
    
    # For matrix multiplication M_{n x m} * M_{m x p}, the rank is at most n*m*p.
    # It is known that for n=2, m=3, p=6, the rank is not known to be less than 36.
    # Actually, let's look at the structure.
    
    # The tensor T_{ijk} = delta_{jk} (for matrix multiplication C_ik = sum_j A_ij B_jk).
    # Wait, the tensor for matrix multiplication is T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk} T_{i,j,k}?
    # No, the tensor is T_{i,j,k} = delta_{j,j} ... no.
    # The multiplication tensor is T_{i,j,k} = 1 if we view it as A_i . B_j -> C_k?
    # No, standard notation: T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = delta_{i,i} delta_{k,k} ... no.
    # The tensor is simply T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} = 1 if we are summing over j.
    # In the decomposition sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}?
    # No, the decomposition is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k V_{j,k,r} B_{j,k}) W_{k,i,r}?
    # Let's check the interface:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # No, that's not right. The standard tensor decomposition for matrix multiplication is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # Actually, the tensor T is defined by T(A,B) = A*B.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 if we consider the term A_{i,j} B_{j,k}.
    # But wait, the indices in the tensor are (i, j, k) corresponding to (row of A, col of A, col of B).
    # The decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
    # No, the standard form is:
    # C_{i,k} = sum_r W_{k,i,r} ( sum_j U_{i,j,r} A_{i,j} ) ( sum_j V_{j,k,r} B_{j,k} )
    # This requires U, V, W to be structured such that:
    # sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = delta_{j,j} ? No.
    # The identity is:
    # sum_j A_{i,j} B_{j,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_j V_{j,k,r} B_{j,k}) W_{k,i,r}
    # This must hold for all A, B.
    # So sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = 1 if j is the same index? No.
    # Let's expand the RHS:
    # sum_r W_{k,i,r} (sum_j U_{i,j,r} A_{i,j}) (sum_{j'} V_{j',k,r} B_{j',k})
    # = sum_r W_{k,i,r} sum_j sum_{j'} U_{i,j,r} V_{j',k,r} A_{i,j} B_{j',k}
    # We want this to equal sum_j A_{i,j} B_{j,k}.
    # So we need:
    # sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = delta_{j,j'} ? No, the indices j and j' must be the same.
    # So sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = 1 for all i, j, k?
    # No, it must be delta_{j,j} = 1?
    # Wait, the term A_{i,j} B_{j,k} appears only when j=j'.
    # So the coefficient of A_{i,j} B_{j,k} in the RHS is sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r}.
    # This must equal 1 for all i, j, k.
    # And the coefficient of A_{i,j} B_{j',k} for j != j' must be 0.
    # So we need:
    # sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = 1
    # sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = 0 for j != j'
    
    # The schoolbook solution:
    # For each (i, j, k), let r = (i, j, k).
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # Then sum_r ... = 1 * 1 * 1 = 1 for the matching (i,j,k).
    # For j != j', the term involves U[i*m+j, r] and V[j'*p+k, r].
    # In schoolbook, U[i*m+j, r] is non-zero only if r corresponds to (i, j, k').
    # So U[i*m+j, r] = 1 if r=(i,j,k').
    # V[j'*p+k, r] = 1 if r=(i',j',k).
    # So for the product to be non-zero, we need r=(i,j,k') and r=(i',j',k).
    # This implies i=i', j=j', k=k'.
    # So if j != j', there is no r such that both are non-zero.
    # Thus the cross terms are 0.
    # So schoolbook works with rank n*m*p = 36.
    
    # Can we do better?
    # The rank of the matrix multiplication tensor M_{2,3,6} is known to be 36?
    # Actually, for 2x3 * 3x6, the rank is not known to be less than 36.
    # However, we can try to use the DFT idea to see if we can reduce rank.
    
    # Let's try to construct a decomposition using the DFT on the middle dimension.
    # We want to diagonalize the sum over j.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Let's use the DFT matrix F of size 3x3.
    # A' = A * F^T, B' = F * B?
    # Then C = A' * B'?
    # (A * F^T) * (F * B) = A * (F^T F) * B = A * (m I) * B = m A B.
    # So AB = (1/m) (A F^T) (F B).
    # This doesn't reduce the rank, it just changes the basis.
    
    # The rank of the tensor is invariant under change of basis.
    # So the rank is still 36.
    
    # Therefore, the minimal rank is likely 36.
    # We will return the schoolbook decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

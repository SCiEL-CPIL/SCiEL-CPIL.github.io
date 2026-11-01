"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank = m * p = 28
    # The key insight: we can share the V and W factors across the two
    # output rows i. For each (j, k) pair, we create one rank-1 component
    # that contributes to both output rows via the appropriate A coefficients.
    
    rank = m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # For this (j, k) pair, we need:
            # C[0,k] gets contribution A[0,j] * B[j,k]
            # C[1,k] gets contribution A[1,j] * B[j,k]
            #
            # In the tensor decomposition:
            # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i', r] = delta[i,i']*delta[j,j']*delta[k,k']
            #
            # So for rank component r corresponding to (j,k):
            # V[j*p+k, r] should be 1
            # W[k*n+i, r] should be 1 for the correct i
            # U[i*m+j, r] should carry the coefficient
            #
            # But wait - each rank-1 component has a SINGLE V entry, SINGLE W entry,
            # and SINGLE U entry (well, one per index). The decomposition is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            #
            # For a fixed (j,k), we need:
            # T[0,j,k] = sum_r U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
            # T[1,j,k] = sum_r U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] = 1
            #
            # And for T[i,j',k'] where j'≠j or k'≠k, the sum must be 0.
            #
            # If we set for this component r:
            #   V[j*p+k, r] = 1
            #   W[k*n+0, r] = 1, W[k*n+1, r] = 1
            #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
            #
            # Then:
            # T[0,j,k] contribution from r: U[0*m+j,r]*V[j*p+k,r]*W[k*n+0,r] = 1*1*1 = 1
            # T[1,j,k] contribution from r: U[1*m+j,r]*V[j*p+k,r]*W[k*n+1,r] = 1*1*1 = 1
            #
            # But this doesn't work because the same V and W entries would be used
            # for both i values, and we'd need different U values.
            #
            # Actually, the issue is that W[k*n+i, r] is indexed by BOTH k AND i.
            # So W[k*n+0, r] and W[k*n+1, r] are different entries in W.
            # Similarly U[i*m+j, r] for different i are different entries.
            #
            # So we CAN have:
            #   V[j*p+k, r] = 1
            #   W[k*n+0, r] = 1
            #   W[k*n+1, r] = 1
            #   U[0*m+j, r] = 1
            #   U[1*m+j, r] = 1
            #
            # This gives T[0,j,k] += 1 and T[1,j,k] += 1 from this component.
            # And for other (i,j',k') with j'≠j or k'≠k, V[j'*p+k',r]=0 so no contribution.
            #
            # But we need T[i,j,k] = 1 for each specific i. With this, both get 1. ✓
            #
            # So this works! Each (j,k) pair gives one rank component that serves both i values.
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

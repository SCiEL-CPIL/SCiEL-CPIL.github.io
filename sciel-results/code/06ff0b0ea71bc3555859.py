"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 3x3 blocks of size 2x2.
    # Outer multiplication: 3x3 schoolbook (rank 27).
    # Inner multiplication: 2x2 Strassen (rank 7).
    # Total rank: 27 * 7 = 189.
    
    rank_outer = 27
    rank_inner = 7
    rank = rank_outer * rank_inner
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # U2[i, j, s]
    U2 = [
        [(0, 0, 1), (1, 1, 1)], # s=0
        [(1, 0, 1), (1, 1, 1)], # s=1
        [(0, 0, 1)],            # s=2
        [(1, 1, 1)],            # s=3
        [(0, 0, 1), (0, 1, 1)], # s=4
        [(1, 0, 1), (0, 0, -1)],# s=5
        [(0, 1, 1), (1, 1, -1)] # s=6
    ]
    # V2[j, k, s]
    V2 = [
        [(0, 0, 1), (1, 1, 1)], # s=0
        [(0, 0, 1)],            # s=1
        [(0, 1, 1), (1, 1, -1)],# s=2
        [(1, 0, 1), (0, 0, -1)],# s=3
        [(1, 1, 1)],            # s=4
        [(0, 0, 1), (0, 1, 1)], # s=5
        [(1, 0, 1), (1, 1, 1)]  # s=6
    ]
    # W2[k, i, s] -> index k*2 + i
    W2 = [
        [(0, 0, 1), (1, 1, 1)], # s=0: C00, C11
        [(1, 0, 1)],            # s=1: C10
        [(0, 1, 1), (1, 1, 1)], # s=2: C01, C11
        [(0, 0, 1), (1, 0, 1)], # s=3: C00, C10
        [(0, 0, -1), (0, 1, 1)],# s=4: C00, C01
        [(1, 1, 1)],            # s=5: C11
        [(0, 0, 1)]             # s=6: C00
    ]
    
    # Iterate over outer block indices
    for c3 in range(rank_outer):
        I = c3 // 9
        J = (c3 % 9) // 3
        K = c3 % 3
        
        for s2 in range(rank_inner):
            r = rank_inner * c3 + s2
            
            # Find matching indices in U2, V2, W2
            # U2 has (i0, j0, val)
            # V2 has (j0, k0, val)
            # W2 has (k0, i0, val)
            
            # We need to combine them such that j0 matches between U2 and V2,
            # k0 matches between V2 and W2, and i0 matches between W2 and U2.
            
            # Since the lists are small, we can just iterate.
            # However, U2, V2, W2 are independent sets of terms for the same s2.
            # The tensor term is sum over i0,j0,k0 of U2[i0,j0] V2[j0,k0] W2[k0,i0].
            # We need to populate U, V, W for the global indices.
            
            # Global indices:
            # i_g = 2*I + i0
            # j_g = 2*J + j0
            # k_g = 2*K + k0
            
            # U[ i_g * m + j_g, r ] += U2_val
            # V[ j_g * p + k_g, r ] += V2_val
            # W[ k_g * n + i_g, r ] += W2_val
            
            # We must ensure that we only add if the indices match.
            # But U2, V2, W2 are already filtered by s2.
            # We just need to find the triple (i0, j0, k0) that satisfies the connections.
            # Actually, the decomposition is a sum of rank-1 tensors.
            # Each (c3, s2) corresponds to a set of rank-1 tensors?
            # No, Strassen's algorithm is a sum of 7 rank-1 tensors.
            # Each rank-1 tensor is defined by specific i0, j0, k0?
            # No, Strassen's M_s are linear combinations.
            # M_s = sum (a_ij b_jk).
            # So U2[i0, j0, s2] is the coefficient of a_i0 j0 in M_s.
            # V2[j0, k0, s2] is the coefficient of b_j0 k0 in M_s.
            # W2[k0, i0, s2] is the coefficient of c_k0 i0 in M_s.
            
            # The term in the global product is:
            # sum_{i0,j0,k0} U2[i0,j0,s2] V2[j0,k0,s2] W2[k0,i0,s2] * A[2I+i0, 2J+j0] B[2J+j0, 2K+k0]
            # This contributes to C[2K+k0, 2I+i0].
            
            # So we can just iterate over all i0, j0, k0 in {0,1} and add the product?
            # No, that would be rank 8 for 2x2.
            # Strassen is rank 7. The coefficients are sparse.
            # We should iterate over the non-zero entries.
            
            # But U, V, W are matrices.
            # For a fixed r, U[., r] is a vector.
            # We need to set U[i_g, j_g, r] = U2[i0, j0, s2] if it's non-zero.
            # But U2[i0, j0, s2] is defined for specific i0, j0.
            
            # So:
            for i0, j0, u_val in U2[s2]:
                i_g = 2 * I + i0
                j_g = 2 * J + j0
                U[i_g * m + j_g][r] += u_val
            
            for j0, k0, v_val in V2[s2]:
                j_g = 2 * J + j0
                k_g = 2 * K + k0
                V[j_g * p + k_g][r] += v_val
            
            for k0, i0, w_val in W2[s2]:
                k_g = 2 * K + k0
                i_g = 2 * I + i0
                W[k_g * n + i_g][r] += w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

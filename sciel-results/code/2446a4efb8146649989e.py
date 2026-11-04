import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2x2
    # U is 2x7, V is 2x7, W is 2x7
    U_s = [
        [1, 0, 1, 0, 1, -1, 1],
        [1, 1, 0, 1, 0, 1, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, -1, 1, 1, 1, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 1, 1, 0]
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 6 blocks: 2 in j dimension (size 2), 3 in k dimension (size 2)
    for bj in range(2):
        for bk in range(3):
            for r in range(7):
                # U index: i * m + j
                # i is 0 or 1. j is bj * 2 + j_local
                for i in range(n):
                    # U[i, col] = U_s[i, r]
                    U[i * m + (bj * 2 + 0)][col] = U_s[i][r] # Wait, j_local is 0 or 1
                    # Actually, U depends on i and j? No, U is for A. A is n x m.
                    # U index is (i, j)? No, U is (n*m, rank).
                    # The term is U_{i, r} V_{j, r} W_{k, r}.
                    # So U row index is i? No, U is n x rank?
                    # The problem says U is (n*m, rank).
                    # This implies U_{i*m+j, r}?
                    # Let's re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
                    # And "exactly decompose the matrix multiplication tensor".
                    # Tensor T_{i,j,k}.
                    # Usually T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}.
                    # If U is (n*m, rank), then the index is i*m+j?
                    # But T depends on i, j, k separately.
                    # If U_{i*m+j, r} is non-zero, it couples i and j.
                    # But in matrix multiplication, A_{ij} and B_{jk} are separate.
                    # The standard decomposition has U_{i,r}, V_{j,r}, W_{k,r}.
                    # Shapes would be (n, rank), (m, rank), (p, rank).
                    # But the task says (n*m, rank).
                    # This implies the "flattened" tensor or a different convention?
                    # "output factor follows the official trace convention W[k*n+i,r]"
                    # This suggests W is (p*n, rank). Index k*n+i.
                    # This is consistent with W being associated with (k, i).
                    # But T_{ijk} has i, j, k.
                    # If W is (p*n, rank), it depends on k and i.
                    # Then U and V must depend on the remaining indices?
                    # If W depends on (k, i), then U and V must depend on j?
                    # No, that doesn't make sense for matrix multiplication.
                    
                    # Let's look at the seed program.
                    # U[i*m+j, col] = 1
                    # V[j*p+k, col] = 1
                    # W[k*n+i, col] = 1
                    # This is the schoolbook decomposition.
                    # Here U depends on (i, j). V depends on (j, k). W depends on (k, i).
                    # This is a "cyclic" or "trace" decomposition.
                    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
                    # This is valid if the tensor is decomposed this way.
                    # For schoolbook, each term is a single (i, j, k) triple.
                    # U_{ij, r} = 1, V_{jk, r} = 1, W_{ki, r} = 1.
                    
                    # For Strassen, we need to express the Strassen terms in this format.
                    # Strassen term r for block (bj, bk):
                    # U_s[i, r] V_s[j_local, r] W_s[k_local, r]
                    # We need to map this to U_{i, j_global, r'} etc.
                    # But Strassen mixes i, j, k.
                    # The Strassen decomposition is for the tensor T_{i, j, k}.
                    # T_{i, j, k} = sum_r U_s[i, r] V_s[j, r] W_s[k, r].
                    # This is the standard form.
                    # But the task requires U of size (n*m, rank).
                    # This implies we must "flatten" the indices.
                    # How to convert standard U (n x rank) to flattened U (n*m x rank)?
                    # We can set U_{i*m+j, r} = U_s[i, r] if j is "active"?
                    # No, that would duplicate.
                    
                    # Let's check the seed again.
                    # U[i*m+j, col] = 1.
                    # This means for a specific (i, j), U is 1.
                    # For Strassen, U_s[i, r] is non-zero for specific i.
                    # It doesn't depend on j.
                    # So U_{i*m+j, r'} = U_s[i, r] * (something for j)?
                    # If U_s[i, r] is 1, then for all j in the block, U_{i*m+j, r'} should be 1?
                    # But then V_{j*p+k, r'} must be 0 for j not in the block?
                    # V_s[j_local, r] depends on j_local.
                    # So V_{j_global*p+k, r'} = V_s[j_local, r] if j_global is in block, else 0.
                    # This works!
                    
                    # So:
                    # For block (bj, bk) and Strassen term r:
                    # Global column index col.
                    # For i in 0..1:
                    #   For j_global in 2*bj .. 2*bj+1:
                    #     j_local = j_global - 2*bj
                    #     U[i*m + j_global, col] = U_s[i, r]
                    #     V[j_global*p + k_global, col] = V_s[j_local, r]  # Wait, V depends on j and k.
                    #     # V_s[j_local, r] is the coefficient for B_{j_local, k_local}.
                    #     # So V_{j_global, k_global, col} = V_s[j_local, r] if k_global is in block.
                    #     # But V is (m*p, rank). Index j*p+k.
                    #     # So for each k_global in block:
                    #     #   V[j_global*p + k_global, col] = V_s[j_local, r] * W_s[k_local, r]?
                    #     # No, W is separate.
                    
                    # Let's trace the product:
                    # sum_r U_{ij, r} V_{jk, r} W_{ki, r}
                    # = sum_r (U_s[i, r]) (V_s[j_local, r]) (W_s[k_local, r])
                    # This equals T_{i, j, k}.
                    # So we need:
                    # U_{i*m + j, col} = U_s[i, r]
                    # V_{j*p + k, col} = V_s[j_local, r]
                    # W_{k*n + i, col} = W_s[k_local, r]
                    
                    # Let's verify with schoolbook.
                    # U_{ij, col} = 1. V_{jk, col} = 1. W_{ki, col} = 1.
                    # Product = 1. Correct.
                    
                    # So for Strassen:
                    for i in range(n):
                        for j_global in range(2 * bj, 2 * bj + 2):
                            j_local = j_global - 2 * bj
                            # U
                            U[i * m + j_global][col] = U_s[i][r]
                            
                            for k_global in range(2 * bk, 2 * bk + 2):
                                k_local = k_global - 2 * bk
                                # V
                                V[j_global * p + k_global][col] = V_s[j_local][r]
                                # W
                                W[k_global * n + i][col] = W_s[k_local][r]
            col += 1
            
    return {"U": U, "V": V, "W": W}

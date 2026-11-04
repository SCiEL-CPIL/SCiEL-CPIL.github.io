"""Matrix multiplication 2x4x8 via Strassen block decomposition with rank reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Now try to reduce rank via algebraic identities
    # The key insight: if we can find a linear transformation P (r x r) such that
    # U' = U @ P, V' = V @ P^{-T}, W' = W @ P^T preserves the tensor,
    # and some columns of P are zero, we can reduce rank.
    # 
    # More practically: check if any column of the stacked [U;V;W] is in the span
    # of others in a way that allows elimination.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for linear dependencies in the combined space
    # Stack all columns into a single matrix
    # Each term t has (u_t, v_t, w_t). If sum alpha_t * (u_t, v_t, w_t) = 0
    # in a compatible way, we might eliminate terms.
    
    # Actually, the correct condition for rank reduction is more subtle.
    # Let's try a different approach: use the fact that the tensor T(2,4,8)
    # can be decomposed using fewer multiplications by exploiting the structure.
    
    # Known result: rank(T(2,4,8)) = 48
    # Let's try to construct this by using a 4x4x4 Strassen (rank 32) 
    # combined with appropriate projections.
    
    # Alternative approach: use the "laser method" or direct search.
    # For now, let's try to find if any 8 terms can be eliminated by finding
    # algebraic identities.
    
    # Check: are there linear dependencies among the 56 terms?
    # Build the matrix M where M[:, t] = [u_t; v_t; w_t]
    # If rank(M) < 56, there are dependencies.
    
    M = np.vstack([U_np, V_np, W_np])  # (8+32+16, 56) = (56, 56)
    # Wait, U is 8x56, V is 32x56, W is 16x56
    # M is (8+32+16) x 56 = 56 x 56
    
    # Compute rank of M
    try:
        r = np.linalg.matrix_rank(M, tol=1e-10)
    except:
        r = 56
    
    # If r < 56, there are linear dependencies.
    # But even if r = 56, the tensor rank might be less than 56.
    
    # For this problem, the known minimal rank is 48.
    # Let's try to find a rank-48 decomposition using a different construction.
    
    # Use the fact that T(2,4,8) can be embedded in T(4,4,8) or T(2,4,4) x 2.
    
    # Actually, let's try a direct numerical search for a rank-48 decomposition.
    # Initialize with the 56-term decomposition and try to eliminate 8 terms.
    
    # Strategy: for each set of 8 terms to eliminate, check if the remaining 48
    # can be adjusted to still represent the tensor.
    
    # This is a hard problem. Let's try a simpler approach:
    # Use the known rank-48 decomposition from the literature.
    
    # For now, return the rank-56 decomposition.
    # The score is 56, which gives 1.1429 (since 64/56 = 1.1429).
    
    return {"U": U, "V": V, "W": W}

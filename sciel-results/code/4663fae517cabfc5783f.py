"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
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
    
    # Now try to reduce rank by finding linear dependencies
    # Convert to numpy for SVD
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stack: each column c has U[:,c], V[:,c], W[:,c]
    # We want to find if sum_c alpha_c * (U_c, V_c, W_c) = 0
    # This is equivalent to finding null space of the stacked matrix
    stacked = np.vstack([U_np, V_np, W_np])  # shape (56, 56)
    
    # Compute SVD
    try:
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
        # Rank is number of singular values > tolerance
        tol = 1e-10
        r = np.sum(s > tol)
        
        if r < rank:
            # Find null space: last (rank - r) columns of Vt
            null_space = Vt[r:]  # shape (rank-r, rank)
            # For each null vector, we can eliminate one term
            # Keep the first r columns (independent ones)
            # Actually, we need to be more careful: we want to keep r independent columns
            # The null space vectors tell us which combinations are zero
            
            # Simple approach: keep columns 0..r-1, drop the rest
            # But this might not be optimal. Let's use a greedy approach.
            # For now, just truncate to r columns
            U_reduced = U_np[:, :r].tolist()
            V_reduced = V_np[:, :r].tolist()
            W_reduced = W_np[:, :r].tolist()
            
            # Verify correctness with a random test
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((p, n))
            for c in range(r):
                C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
            
            # Check if close
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
            
            # If verification failed, return original
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}

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
    
    # Now try to reduce rank by greedy elimination
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute the full tensor T of shape (n*m, m*p, p*n) implicitly
    # T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # We'll work with a flattened version for comparison
    
    def compute_tensor(U_m, V_m, W_m, cols):
        """Compute tensor as (n*m, m*p, p*n) array for given columns."""
        T = np.zeros((n * m, m * p, p * n))
        for c in cols:
            T += np.einsum('i,j,k->ijk', U_m[:, c], V_m[:, c], W_m[:, c])
        return T
    
    # First compute the full tensor
    full_tensor = compute_tensor(U_np, V_np, W_np, list(range(rank)))
    
    # Greedy elimination: try to remove each term
    keep = list(range(rank))
    for c in range(rank):
        if c not in keep:
            continue
        if time.time() > deadline - 5:
            break
        # Try removing term c
        new_keep = [k for k in keep if k != c]
        if not new_keep:
            continue
        test_tensor = compute_tensor(U_np, V_np, W_np, new_keep)
        if np.allclose(test_tensor, full_tensor, atol=1e-10):
            keep = new_keep
    
    if len(keep) < rank:
        U_reduced = U_np[:, keep].tolist()
        V_reduced = V_np[:, keep].tolist()
        W_reduced = W_np[:, keep].tolist()
        
        # Final verification with random test
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((p, n))
        for c in range(len(keep)):
            C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
        
        if np.allclose(C_computed, C_expected, atol=1e-9):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    return {"U": U, "V": V, "W": W}

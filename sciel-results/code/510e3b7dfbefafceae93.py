"""Matrix multiplication 2x4x8 via Strassen block decomposition with rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Build initial Strassen-based decomposition (rank 56)
    # A is 2x4, B is 4x8, C is 8x2
    # Split into 8 blocks of size 2x2x2
    # 1 block in i-dim (2 rows), 2 blocks in j-dim (4 cols), 4 blocks in k-dim (8 cols)
    
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2: (u_coeffs, v_coeffs, w_coeffs)
    # Each is a 4-element vector for the 2x2 matrix entries
    # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
    # M2 = (c+d)e     -> C10 += M2, C11 -= M2
    # M3 = a(f-h)     -> C01 += M3, C11 += M3
    # M4 = d(g-e)     -> C00 += M4, C10 += M4
    # M5 = (a+b)h     -> C00 -= M5, C01 += M5
    # M6 = (c-a)(e+f) -> C11 += M6
    # M7 = (b-d)(g+h) -> C00 += M7
    
    strassen_terms = [
        # (u_coeffs for [a,b,c,d], v_coeffs for [e,f,g,h], w_coeffs for [C00,C01,C10,C11])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0, i1 = 2 * bi, 2 * bi + 1
                j0, j1 = 2 * bj, 2 * bj + 1
                k0, k1 = 2 * bk, 2 * bk + 1
                
                # U indices: [i*m + j]
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                # V indices: [j*p + k]
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                # W indices: [k*n + i]
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                for s in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                    # u_coeffs = [a, b, c, d]
                    U[u_a, col] += u_coeffs[0]
                    U[u_b, col] += u_coeffs[1]
                    U[u_c, col] += u_coeffs[2]
                    U[u_d, col] += u_coeffs[3]
                    
                    # v_coeffs = [e, f, g, h]
                    V[v_e, col] += v_coeffs[0]
                    V[v_f, col] += v_coeffs[1]
                    V[v_g, col] += v_coeffs[2]
                    V[v_h, col] += v_coeffs[3]
                    
                    # w_coeffs = [C00, C01, C10, C11]
                    W[w_c00, col] += w_coeffs[0]
                    W[w_c01, col] += w_coeffs[1]
                    W[w_c10, col] += w_coeffs[2]
                    W[w_c11, col] += w_coeffs[3]
                    
                    col += 1
    
    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=20):
        """Verify exact decomposition on random test inputs."""
        rng = np.random.default_rng(42)
        for _ in range(num_tests):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            expected = A @ B
            computed = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                u_vec = U_mat[:, c]
                v_vec = V_mat[:, c]
                w_vec = W_mat[:, c]
                u_reshaped = u_vec.reshape(n, m)
                v_reshaped = v_vec.reshape(m, p)
                w_transposed = w_vec.reshape(p, n).T  # (n, p)
                uv = u_reshaped @ v_reshaped
                computed += uv * w_transposed
            if not np.allclose(computed, expected, atol=1e-10, rtol=1e-10):
                return False
        return True
    
    # Verify initial decomposition
    if not verify(U, V, W, n, m, p):
        # Fallback: return schoolbook decomposition
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        U_list = [[int(U_sb[i, r]) for r in range(rank_sb)] for i in range(n * m)]
        V_list = [[int(V_sb[i, r]) for r in range(rank_sb)] for i in range(m * p)]
        W_list = [[int(W_sb[i, r]) for r in range(rank_sb)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank by removing terms one at a time
    # Sort terms by "importance" - try removing less important ones first
    # Heuristic: terms with smaller norm are less important
    
    term_norms = []
    for r in range(rank):
        norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
        term_norms.append((norm, r))
    
    # Try removing terms in order of increasing norm (least important first)
    term_norms.sort(key=lambda x: x[0])
    
    for norm, r_remove in term_norms:
        if time.time() > deadline:
            break
        
        trial_U = best_U.copy()
        trial_V = best_V.copy()
        trial_W = best_W.copy()
        
        # Zero out the term
        trial_U[:, r_remove] = 0
        trial_V[:, r_remove] = 0
        trial_W[:, r_remove] = 0
        
        if verify(trial_U, trial_V, trial_W, n, m, p, num_tests=10):
            best_U, best_V, best_W = trial_U, trial_V, trial_W
            best_rank -= 1
    
    # Try pairwise elimination: replace two terms with one
    # For each pair (r1, r2), try to find if their sum can be represented as one term
    # This is a hard problem, so we use a heuristic approach
    
    # Try to find linear dependencies among the rank-1 tensors
    # Build a matrix where each column is the vectorized rank-1 tensor
    # T_r = vec(U[:,r] * V[:,r] * W[:,r]) but this doesn't quite work for tensor rank
    
    # Alternative: use the fact that if two terms are "similar", we might combine them
    # For now, focus on single-term elimination which is more tractable
    
    # Final verification
    if not verify(best_U, best_V, best_W, n, m, p):
        # Revert to original if something went wrong
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
    
    # Convert to list format
    U_list = [[int(round(best_U[i, r])) for r in range(best_rank)] for i in range(n * m)]
    V_list = [[int(round(best_V[i, r])) for r in range(best_rank)] for i in range(m * p)]
    W_list = [[int(round(best_W[i, r])) for r in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

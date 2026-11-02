import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k] = 1 where
    # i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # C[i1,k2] = sum_j1 A[i1,j1]*B[j1,k2]
    # U index: i1*m + i2 (row i1, col i2 of A)
    # V index: j1*p + j2 (row j1, col j2 of B)  
    # W index: k2*n + i1 (col k2, row i1 of C) -- per trace convention
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                ui = i1*m + j1
                vj = j1*p + k2
                wk = k2*n + i1
                T[ui, vj, wk] = 1.0
    
    # Build Strassen rank-28 as baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28), dtype=float)
    V28 = np.zeros((m * p, rank_28), dtype=float)
    W28 = np.zeros((p * n, rank_28), dtype=float)
    
    def strassen_terms():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col, U, V, W):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0, U28, V28, W28)
    apply_strassen(1, 1, 0, 0, 7, U28, V28, W28)
    apply_strassen(0, 0, 1, 1, 14, U28, V28, W28)
    apply_strassen(1, 1, 1, 1, 21, U28, V28, W28)
    
    # Verify Strassen solution
    def verify(U, V, W, rank):
        R = np.einsum('ir,jr,kr->ijk', U, V, W) - T
        return np.max(np.abs(R)) < 1e-9
    
    if not verify(U28, V28, W28, 28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    best_U, best_V, best_W, best_rank = U28, V28, W28, 28
    
    # Strategy: Try to find a rank-27 solution by exploiting the tensor structure
    # The 2x4x4 tensor can be viewed as having a block structure
    # Let's try a more systematic search using the algebraic structure
    
    # Approach 1: Try removing each column and using ALS with better initialization
    np.random.seed(42)
    
    for col_to_remove in range(28):
        if time.time() > deadline:
            break
            
        # Remove this column
        mask = [r for r in range(28) if r != col_to_remove]
        U_try = U28[:, mask].copy()
        V_try = V28[:, mask].copy()
        W_try = W28[:, mask].copy()
        
        # Add small random perturbation to break symmetry
        U_try += 0.05 * np.random.randn(n*m, 27)
        V_try += 0.05 * np.random.randn(m*p, 27)
        W_try += 0.05 * np.random.randn(p*n, 27)
        
        # ALS iterations with adaptive learning rate
        lr = 0.05
        for iteration in range(300):
            if time.time() > deadline:
                break
                
            # Compute residuals
            R = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try) - T
            
            # Compute gradients
            G_U = np.einsum('ijk,ja,ka->ia', R, V_try, W_try)
            G_V = np.einsum('ijk,ia,ka->ja', R, U_try, W_try)
            G_W = np.einsum('ijk,ia,ja->ka', R, U_try, V_try)
            
            # Adaptive learning rate: scale by inverse norm
            norm_U = np.linalg.norm(G_U) + 1e-10
            norm_V = np.linalg.norm(G_V) + 1e-10
            norm_W = np.linalg.norm(G_W) + 1e-10
            
            U_try -= lr * G_U / norm_U * np.linalg.norm(U_try)
            V_try -= lr * G_V / norm_V * np.linalg.norm(V_try)
            W_try -= lr * G_W / norm_W * np.linalg.norm(W_try)
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-10:
                break
        
        # Check if we got a valid solution
        R_final = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try) - T
        if np.max(np.abs(R_final)) < 1e-6:
            # We have a rank-27 solution!
            # Try to round to integers/half-integers
            U_rounded = np.round(U_try * 2) / 2
            V_rounded = np.round(V_try * 2) / 2
            W_rounded = np.round(W_try * 2) / 2
            
            R_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded) - T
            if np.max(np.abs(R_rounded)) < 1e-6:
                best_U, best_V, best_W, best_rank = U_rounded, V_rounded, W_rounded, 27
                break
            else:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, 27
                break
    
    # If we didn't find rank 27, try a different approach: 
    # Use the block structure to find a compressed representation
    if best_rank == 28 and time.time() < deadline - 20:
        # Try to find a rank-27 solution by looking for linear dependencies
        # among the 28 bilinear forms
        
        # The key insight: in the 2x4x4 case, there might be a relation
        # that allows us to express one term as a combination of others
        
        # Let's try a more aggressive random search with multiple restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            rank_try = 27
            # Use a structured initialization based on the Strassen solution
            # but with one column removed and the others adjusted
            
            # Try different columns to remove
            col_to_remove = attempt * 5 + 3  # Spread out the attempts
            
            mask = [r for r in range(28) if r != col_to_remove]
            U_try = U28[:, mask].copy()
            V_try = V28[:, mask].copy()
            W_try = W28[:, mask].copy()
            
            # Add structured perturbation
            np.random.seed(100 + attempt)
            U_try += 0.1 * np.random.randn(n*m, 27)
            V_try += 0.1 * np.random.randn(m*p, 27)
            W_try += 0.1 * np.random.randn(p*n, 27)
            
            # Use a two-phase optimization: first coarse, then fine
            for phase in range(2):
                lr = 0.1 if phase == 0 else 0.01
                for iteration in range(200):
                    if time.time() > deadline:
                        break
                        
                    R = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try) - T
                    G_U = np.einsum('ijk,ja,ka->ia', R, V_try, W_try)
                    G_V = np.einsum('ijk,ia,ka->ja', R, U_try, W_try)
                    G_W = np.einsum('ijk,ia,ja->ka', R, U_try, V_try)
                    
                    U_try -= lr * G_U
                    V_try -= lr * G_V
                    W_try -= lr * G_W
                    
                    if np.max(np.abs(R)) < 1e-10:
                        break
            
            R_final = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try) - T
            if np.max(np.abs(R_final)) < 1e-6:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, 27
                break
    
    # Convert to lists
    U_out = np.array(best_U, dtype=float)
    V_out = np.array(best_V, dtype=float)
    W_out = np.array(best_W, dtype=float)
    
    # Final verification
    R_check = np.einsum('ir,jr,kr->ijk', U_out, V_out, W_out) - T
    if np.max(np.abs(R_check)) > 1e-6:
        # Fallback to Strassen
        U_out = U28
        V_out = V28
        W_out = W28
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])

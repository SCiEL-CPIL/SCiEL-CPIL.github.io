import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off + 1 * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def search_lower_rank(target_rank, max_attempts=10, timeout=10):
        """
        Attempts to find a rank-target_rank decomposition by fixing V to a sparse
        canonical form (one non-zero per column) and using ALS to solve for U and W.
        """
        if time.time() > deadline - timeout:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - timeout:
                break
                
            np.random.seed(42 + attempt * 17)
            
            # Fix V: each column has exactly one non-zero entry (1 or -1)
            V = np.zeros((V_dim, target_rank))
            # Assign each column to a random row
            row_indices = np.random.randint(0, V_dim, size=target_rank)
            signs = np.random.choice([-1, 1], size=target_rank)
            for r in range(target_rank):
                V[row_indices[r], r] = signs[r]
            
            # Initialize U and W
            U = np.random.randn(U_dim, target_rank) * 0.5
            W = np.random.randn(W_dim, target_rank) * 0.5
            
            # ALS Loop
            for it in range(50):
                if time.time() > deadline - timeout:
                    break
                
                # Update U
                U_new = np.zeros_like(U)
                for a in range(U_dim):
                    # Construct M_a: (V_dim * W_dim) x target_rank
                    # M_a[b*W_dim + c, r] = V[b, r] * W[c, r]
                    M_a = np.zeros((V_dim * W_dim, target_rank))
                    for b in range(V_dim):
                        for c in range(W_dim):
                            M_a[b * W_dim + c, :] = V[b, :] * W[c, :]
                    
                    # Solve M_a @ U[a,:] = T[a,:,:].flatten()
                    try:
                        U_new[a, :] = np.linalg.lstsq(M_a, T[a, :, :].flatten(), rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                
                # Update W
                W_new = np.zeros_like(W)
                for c in range(W_dim):
                    # Construct M_c: (U_dim * V_dim) x target_rank
                    # M_c[a*V_dim + b, r] = U[a, r] * V[b, r]
                    M_c = np.zeros((U_dim * V_dim, target_rank))
                    for a in range(U_dim):
                        for b in range(V_dim):
                            M_c[a * V_dim + b, :] = U[a, :] * V[b, :]
                    
                    # Solve M_c @ W[c,:] = T[:,:,c].flatten()
                    try:
                        W_new[c, :] = np.linalg.lstsq(M_c, T[:, :, c].flatten(), rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                
                # Check convergence
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(W_new - W)
                U = U_new
                W = W_new
                
                if diff < 1e-9:
                    break
            
            # Snap and Validate
            valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
            if valid:
                return U_s, V_s, W_s
                
            # Also check if it's close to integer without snapping (just in case)
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            if np.allclose(prod, T, atol=1e-5):
                return U, V, W

        return None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = (U46, V46, W46)
    
    # Try to find lower rank
    for target_rank in [45, 44, 43]:
        if time.time() > deadline - 20:
            break
        
        result = search_lower_rank(target_rank, max_attempts=5, timeout=15)
        if result is not None:
            U_r, V_r, W_r = result
            if validate(U_r, V_r, W_r):
                best_result = (U_r, V_r, W_r)
                break
    
    U_r, V_r, W_r = best_result
    return {
        "U": [[float(x) for x in row] for row in U_r],
        "V": [[float(x) for x in row] for row in V_r],
        "W": [[float(x) for x in row] for row in W_r]
    }
# EVOLVE-BLOCK-END

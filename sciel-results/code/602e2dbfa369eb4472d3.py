import time
import numpy as np
from itertools import product

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
    
    def build_rank46():
        """Build the known rank-46 decomposition."""
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
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    def solve_with_fixed_V(V_fixed, R, max_iter=50):
        """
        Given a fixed V matrix (V_dim x R) with each row having exactly one non-zero entry,
        solve for U and W such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].
        
        Since V[b,r] is non-zero for exactly one r per b, we can restructure:
        For each b, let r(b) be the column where V[b,r(b)] != 0.
        Then: T[a,b,c] = U[a,r(b)] * V[b,r(b)] * W[c,r(b)]
        
        This means for each b, the matrix T[:,b,:] = U[:,r(b)] * V[b,r(b)] * W[:,r(b)].T
        So T[:,b,:] is a rank-1 matrix (scaled).
        
        We can solve this by:
        1. For each b, if T[:,b,:] is non-zero, we need U[:,r(b)] and W[:,r(b)] to be proportional
           to the singular vectors of T[:,b,:] / V[b,r(b)].
        2. Use SVD to find these.
        """
        # Find which column each row of V is assigned to
        row_to_col = {}
        for b in range(V_dim):
            nonzeros = np.where(V_fixed[b] != 0)[0]
            if len(nonzeros) == 0:
                # Row b has no contribution - this is problematic
                # T[:,b,:] must be zero
                if np.any(T[:, b, :] != 0):
                    return None
                continue
            elif len(nonzeros) > 1:
                # Multiple non-zeros in a row - not our canonical form
                return None
            row_to_col[b] = nonzeros[0]
        
        # Initialize U and W
        U = np.zeros((U_dim, R))
        W = np.zeros((W_dim, R))
        
        # For each b with T[:,b,:] non-zero, solve for the corresponding column of U and W
        for b in range(V_dim):
            if np.all(T[:, b, :] == 0):
                continue
            
            r = row_to_col.get(b)
            if r is None:
                return None
            
            v_val = V_fixed[b, r]
            if v_val == 0:
                return None
            
            # T[:,b,:] = U[:,r] * v_val * W[:,r].T
            # So U[:,r] * W[:,r].T = T[:,b,:] / v_val
            M = T[:, b, :] / v_val
            
            # Use SVD to factor M = u_vec * w_vec.T
            # M is U_dim x W_dim = 10 x 10
            try:
                U_mat, S, Vt_mat = np.linalg.svd(M, full_matrices=False)
            except:
                return None
            
            # M should be rank 1, so S should have one significant singular value
            if S.size == 0:
                continue
            
            # The rank-1 approximation: M ≈ U_mat[:,0] * S[0] * Vt_mat[0,:]
            # So U[:,r] = U_mat[:,0] * sqrt(S[0]) and W[:,r] = Vt_mat[0,:] * sqrt(S[0])
            # Or we can split the scaling differently. Let's use:
            # U[:,r] = U_mat[:,0] * S[0]**0.5
            # W[:,r] = Vt_mat[0,:] * S[0]**0.5
            
            if S[0] > 1e-10:
                U[:, r] = U_mat[:, 0] * np.sqrt(S[0])
                W[:, r] = Vt_mat[0, :] * np.sqrt(S[0])
            else:
                # M is essentially zero, but T[:,b,:] is not zero? Contradiction.
                # This shouldn't happen if v_val != 0 and T[:,b,:] != 0
                pass
        
        # Now check if the solution is valid
        prod = np.einsum('ar,br,cr->abc', U, V_fixed, W, optimize=True)
        loss = np.sum((prod - T) ** 2)
        
        if loss < 1e-6:
            # Try to snap to half-integers
            U_s = snap_to_half(U)
            V_s = snap_to_half(V_fixed)
            W_s = snap_to_half(W)
            prod2 = np.einsum('ar,br,cr->abc', U_s, V_s, W_s, optimize=True)
            if np.allclose(prod2, T, atol=1e-5):
                return U_s, V_s, W_s
            
            # Try integers
            U_i = np.round(U)
            V_i = np.round(V_fixed)
            W_i = np.round(W)
            prod3 = np.einsum('ar,br,cr->abc', U_i, V_i, W_i, optimize=True)
            if np.allclose(prod3, T, atol=1e-5):
                return U_i, V_i, W_i
        
        return None

    def try_fixed_V_decomposition(R, num_attempts=20):
        """
        Try to find a rank-R decomposition by fixing V to have exactly one non-zero per row.
        Each row is assigned to a column, and we solve for U and W.
        """
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(12345 + attempt * 777)
            
            # Create V with exactly one non-zero per row
            V = np.zeros((V_dim, R))
            
            # Assign each row to a column
            # We want to balance the load: each column should get roughly V_dim/R rows
            # But we can also try random assignments
            
            if attempt % 2 == 0:
                # Balanced assignment
                for b in range(V_dim):
                    col = b % R
                    V[b, col] = 1.0 if (b // R) % 2 == 0 else -1.0
            else:
                # Random assignment
                for b in range(V_dim):
                    col = np.random.randint(0, R)
                    V[b, col] = 1.0 if np.random.random() > 0.5 else -1.0
            
            result = solve_with_fixed_V(V, R)
            if result is not None:
                return result
        
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

    best_result = None
    
    # Try to find lower rank using fixed-V approach
    for target_rank in [45, 44, 43, 42]:
        if time.time() > deadline - 20:
            break
        
        result = try_fixed_V_decomposition(target_rank, num_attempts=30)
        if result is not None:
            U_r, V_r, W_r = result
            if validate(U_r, V_r, W_r):
                best_result = result
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        return {
            "U": [[float(x) for x in row] for row in U_r],
            "V": [[float(x) for x in row] for row in V_r],
            "W": [[float(x) for x in row] for row in W_r]
        }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END

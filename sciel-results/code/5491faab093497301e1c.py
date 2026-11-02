import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s limit
    
    def make_schoolbook():
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank_school
    
    # Target tensor
    T = np.ones((n, m, p), dtype=np.float64)
    
    def compute_tensor(U, V, W):
        """Compute sum_r U[:,r] V[:,r] W[:,r] as tensor."""
        R = U.shape[1]
        U_3d = U.T.reshape(n, m, R)
        V_3d = V.T.reshape(m, p, R)
        W_3d = W.T.reshape(p, n, R)
        return np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    def try_remove_columns(U, V, W, cols_to_remove):
        """Try removing specified columns. Return (success, new_U, new_V, new_W)."""
        R = U.shape[1]
        keep = [c for c in range(R) if c not in cols_to_remove]
        if len(keep) == 0:
            return False, None, None, None
        
        U_k = U[:, keep]
        V_k = V[:, keep]
        W_k = W[:, keep]
        
        # Check if the kept columns can still represent T
        # We need to solve: sum_{r in keep} U[:,r] * V[:,r] * W[:,r] = T
        # This is a linear system in the entries of U, V, W... but it's nonlinear.
        # Instead, we check: is T in the span of the rank-1 tensors {U[:,r] V[:,r] W[:,r]}?
        # We can form a matrix where each column is vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r])
        # and check if vec(T) is in the column space.
        
        Rk = len(keep)
        # Build the matrix of rank-1 tensor vectors
        # Each rank-1 term is U[:,r] (nm x 1) * V[:,r] (mp x 1) * W[:,r] (pn x 1)
        # The tensor is sum_r u_r ⊗ v_r ⊗ w_r where u_r = U[:,r], etc.
        # vec(u_r ⊗ v_r ⊗ w_r) = u_r ⊗ v_r ⊗ w_r (Kronecker product)
        
        # Actually, for the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is NOT a standard Kronecker product because of the index couplings.
        # Let me think differently.
        
        # The tensor is: T[i,j,k] = sum_r a_r(i,j) * b_r(j,k) * c_r(k,i)
        # where a_r(i,j) = U[i*m+j, r], b_r(j,k) = V[j*p+k, r], c_r(k,i) = W[k*n+i, r]
        
        # For fixed r, the contribution is a rank-1 "cyclic" tensor.
        # To check if T can be written as sum of these, we need a different approach.
        
        # Alternative: just compute the tensor and check error.
        tensor = compute_tensor(U_k, V_k, W_k)
        error = np.linalg.norm(tensor - T)
        
        if error < 1e-9:
            return True, U_k, V_k, W_k
        return False, None, None
    
    # Strategy: try removing columns one at a time, then in pairs, etc.
    # Start with all 120 columns, try removing 1, 2, 3, ... columns
    
    current_U, current_V, current_W = U0.copy(), V0.copy(), W0.copy()
    current_rank = rank_school
    
    # Try removing single columns
    for r_to_remove in range(current_rank):
        if time.time() > deadline:
            break
        
        success, U_new, V_new, W_new = try_remove_columns(
            current_U, current_V, current_W, [r_to_remove]
        )
        
        if success:
            current_U, current_V, current_W = U_new, V_new, W_new
            current_rank -= 1
            # Restart single removal from the beginning
            r_to_remove = -1  # will be incremented to 0
    
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = current_rank
    
    # Try removing pairs of columns
    R = current_rank
    for r1 in range(R):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, R):
            if time.time() > deadline:
                break
            
            success, U_new, V_new, W_new = try_remove_columns(
                current_U, current_V, current_W, [r1, r2]
            )
            
            if success:
                current_U, current_V, current_W = U_new, V_new, W_new
                current_rank -= 2
                # Restart pair removal
                r1 = -1
                break
    
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = current_rank
    
    # Try removing triples
    R = current_rank
    found_triple = False
    for r1 in range(R):
        if time.time() > deadline or found_triple:
            break
        for r2 in range(r1 + 1, R):
            if time.time() > deadline or found_triple:
                break
            for r3 in range(r2 + 1, R):
                if time.time() > deadline:
                    break
                
                success, U_new, V_new, W_new = try_remove_columns(
                    current_U, current_V, current_W, [r1, r2, r3]
                )
                
                if success:
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 3
                    found_triple = True
                    break
    
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = current_rank
    
    # Convert to lists with half-integer rounding
    U_list = np.round(current_U * 2) / 2
    V_list = np.round(current_V * 2) / 2
    W_list = np.round(current_W * 2) / 2
    
    # Verify
    U_check = U_list.reshape(n, m, -1)
    V_check = V_list.reshape(m, p, -1)
    W_check = W_list.reshape(p, n, -1)
    tensor_check = np.einsum('ijr,jkr,kir->ijk', U_check, V_check, W_check)
    final_error = np.linalg.norm(tensor_check - T)
    
    if final_error > 1e-9:
        # Rounding broke it, use original
        U_list = current_U
        V_list = current_V
        W_list = current_W
        best_rank = current_rank
    else:
        best_rank = current_rank
    
    return {"U": U_list.tolist(), "V": V_list.tolist(), "W": W_list.tolist()}
# EVOLVE-BLOCK-END

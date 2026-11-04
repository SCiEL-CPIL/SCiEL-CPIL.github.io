import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Build the rank-42 Strassen decomposition
    rank_42 = 42
    U = np.zeros((n * m, rank_42), dtype=float)
    V = np.zeros((m * p, rank_42), dtype=float)
    W = np.zeros((p * n, rank_42), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    def verify_decomposition(U_mat, V_mat, W_mat, seed=42):
        """Check if the tensor decomposition exactly reproduces matrix multiplication."""
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            U_r = U_mat[:, r].reshape(n, m)
            V_r = V_mat[:, r].reshape(m, p)
            W_r = W_mat[:, r].reshape(p, n)
            
            term_A = np.sum(A * U_r, axis=1)  # shape (n,)
            term_B = np.sum(B * V_r, axis=0)  # shape (p,)
            
            C_decomposed += np.outer(term_A, term_B) * W_r.T
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
        return True
    
    if not verify_decomposition(U, V, W):
        # Fallback: return the schoolbook decomposition
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=float)
        V_s = np.zeros((m * p, rank_school), dtype=float)
        W_s = np.zeros((p * n, rank_school), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, idx] = 1
                    V_s[j*p+k, idx] = 1
                    W_s[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    def find_minimal_rank(U_mat, V_mat, W_mat):
        """Use peeling algorithm to find lower rank decomposition."""
        current_rank = U_mat.shape[1]
        
        def flatten_tensor(U_r, V_r, W_r):
            """Flatten U_r ⊗ V_r ⊗ W_r into a vector of length n*m*p."""
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
            return tensor.flatten()
        
        # Build the 48 x rank matrix of flattened rank-1 tensors
        T_matrix = np.zeros((n * m * p, current_rank))
        for r in range(current_rank):
            T_matrix[:, r] = flatten_tensor(U_mat[:, r], V_mat[:, r], W_mat[:, r])
        
        # The target tensor is the matrix multiplication tensor
        T_target = np.ones((n, m, p)).flatten()
        
        # Verify: sum of all terms should equal T_target
        T_sum = T_matrix @ np.ones(current_rank)
        if not np.allclose(T_sum, T_target, atol=1e-10):
            return None
        
        best_result = None
        best_rank = current_rank
        
        # Peeling: iteratively try to remove terms
        # For each candidate term to remove, check if the remaining terms can span the target
        active = list(range(current_rank))
        
        while len(active) > 1:
            if time.time() > deadline - 5:
                break
            
            removed_any = False
            new_active = []
            
            for r in active:
                if time.time() > deadline - 5:
                    break
                
                remaining = [x for x in active if x != r]
                if len(remaining) == 0:
                    break
                
                A_sub = T_matrix[:, remaining]  # 48 x len(remaining)
                
                try:
                    coeffs, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, T_target, rcond=None)
                    residual = np.linalg.norm(A_sub @ coeffs - T_target)
                    
                    if residual < 1e-10:
                        # We can remove term r
                        # Check if the coefficients are reasonable (not too large)
                        if np.max(np.abs(coeffs)) < 100:
                            new_active = remaining
                            removed_any = True
                            break
                        else:
                            # Coefficients too large, skip this removal
                            pass
                    else:
                        new_active = active
                except:
                    continue
            
            if not removed_any:
                # Try removing pairs
                pair_removed = False
                for r1, r2 in combinations(active, 2):
                    if time.time() > deadline - 5:
                        break
                    
                    remaining = [x for x in active if x != r1 and x != r2]
                    if len(remaining) == 0:
                        break
                    
                    A_sub = T_matrix[:, remaining]
                    
                    try:
                        coeffs, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, T_target, rcond=None)
                        residual = np.linalg.norm(A_sub @ coeffs - T_target)
                        
                        if residual < 1e-10:
                            if np.max(np.abs(coeffs)) < 100:
                                new_active = remaining
                                pair_removed = True
                                break
                    except:
                        continue
                
                if not pair_removed:
                    # Cannot remove more terms
                    break
            
            if len(new_active) < len(active):
                active = new_active
            else:
                break
        
        # If we reduced the rank, construct the new decomposition
        if len(active) < current_rank:
            new_U = U_mat[:, active].copy()
            new_V = V_mat[:, active].copy()
            new_W = W_mat[:, active].copy()
            
            # Solve for coefficients
            A_sub = T_matrix[:, active]
            try:
                coeffs, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, T_target, rcond=None)
                residual = np.linalg.norm(A_sub @ coeffs - T_target)
                
                if residual < 1e-10:
                    # Apply coefficients to U
                    for idx, r in enumerate(active):
                        c = coeffs[idx]
                        new_U[:, idx] *= c
                    
                    # Verify
                    if verify_decomposition(new_U, new_V, new_W):
                        best_rank = len(active)
                        best_result = (new_U, new_V, new_W)
            except:
                pass
        
        if best_result is not None:
            return best_result
        return None
    
    # Try to find a lower rank decomposition using peeling
    result = find_minimal_rank(U, V, W)
    
    if result is not None:
        U_final, V_final, W_final = result
    else:
        U_final, V_final, W_final = U, V, W
    
    # Final check for deadline
    if time.time() > deadline - 2:
        pass
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END

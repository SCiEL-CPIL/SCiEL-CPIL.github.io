"""
Structured sparse recovery for 4x5x5 matrix multiplication tensor decomposition.
Uses block-diagonal with shift pattern on factor matrices.
"""
import numpy as np
import itertools
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # 1. Default Schoolbook Solution (Rank 100)
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    U_default = U_school.tolist()
    V_default = V_school.tolist()
    W_default = W_school.tolist()
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly matches the tensor."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_structured_decomposition(max_terms):
        """
        Greedy structured sparse recovery with block-diagonal shift pattern.
        Each term has:
        - A cyclic shift s in the (j,k) dimension for V
        - A subset of (i,j) pairs for U
        - A subset of (k,i) pairs for W
        We solve for coefficients to match the tensor exactly.
        """
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Greedy: build up terms one at a time
        U_list = []
        V_list = []
        W_list = []
        residual = T.copy()
        
        for term_idx in range(max_terms):
            if time.time() - start_time > time_limit:
                return None
            
            # Try all combinations of shift patterns
            best_term = None
            best_residual_norm = np.linalg.norm(residual)
            
            # For each shift s (0-4), try different U/W patterns
            for shift in range(p):
                # V pattern: V[j*p+k] = 1 if k == (j + shift) % p
                V_pattern = np.zeros(m*p)
                for j in range(m):
                    k = (j + shift) % p
                    V_pattern[j*p+k] = 1.0
                
                # For U: try each row i having one active j
                # For W: try each col k having one active i
                # This gives a rank-1-like structure per term
                
                # Try all (i_active, j_active) for U and corresponding W
                for i_a in range(n):
                    for j_a in range(m):
                        # U has 1 at (i_a, j_a)
                        # V has 1 at (j_a, (j_a+shift)%p)
                        # W must have 1 at ((j_a+shift)%p, i_a) to close the loop
                        k_a = (j_a + shift) % p
                        
                        U_pattern = np.zeros(n*m)
                        U_pattern[i_a*m + j_a] = 1.0
                        
                        W_pattern = np.zeros(p*n)
                        W_pattern[k_a*n + i_a] = 1.0
                        
                        # This term contributes to T[i_a*m+j_a, j_a*p+k_a, k_a*n+i_a]
                        # Check if this helps reduce residual
                        # The contribution is a rank-1 tensor in the structured sense
                        # Actually this is just a single entry, so it can only fix one entry
                        
                        # Compute new residual if we subtract this term
                        # The term's tensor has 1 at position (i_a*m+j_a, j_a*p+k_a, k_a*n+i_a)
                        idx_u = i_a*m + j_a
                        idx_v = j_a*p + k_a
                        idx_w = k_a*n + i_a
                        
                        if abs(residual[idx_u, idx_v, idx_w] - 1.0) < 1e-10:
                            # This term would zero out this entry
                            new_residual = residual.copy()
                            new_residual[idx_u, idx_v, idx_w] -= 1.0
                            new_norm = np.linalg.norm(new_residual)
                            if new_norm < best_residual_norm - 1e-10:
                                best_residual_norm = new_norm
                                best_term = (U_pattern, V_pattern, W_pattern, shift)
            
            if best_term is None:
                break
            
            U_pattern, V_pattern, W_pattern, _ = best_term
            U_list.append(U_pattern)
            V_list.append(V_pattern)
            W_list.append(W_pattern)
            
            # Update residual
            i_a = np.argmax(U_pattern) // m
            j_a = np.argmax(U_pattern) % m
            k_a = (j_a + best_term[3]) % p
            idx_u = i_a*m + j_a
            idx_v = j_a*p + k_a
            idx_w = k_a*n + i_a
            residual[idx_u, idx_v, idx_w] -= 1.0
        
        if np.linalg.norm(residual) < 1e-10:
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            if verify_decomposition(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Attempt structured search for lower rank
    best_result = None
    # Try ranks from 100 downward, but limit search time
    for target_rank in range(schoolbook_rank - 1, max(20, schoolbook_rank - 30), -1):
        if time.time() - start_time > time_limit:
            break
        result = try_structured_decomposition(target_rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Returning the default schoolbook decomposition
    return {"U": U_default, "V": V_default, "W": W_default}
# EVOLVE-BLOCK-END

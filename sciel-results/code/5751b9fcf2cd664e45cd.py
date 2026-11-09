import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Combines Power Iteration for structure discovery with integer/half-integer snapping.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Fallback: Schoolbook decomposition (Rank 36)
    rank_schoolbook = n * m * p
    U_fb = np.zeros((dim_u, rank_schoolbook))
    V_fb = np.zeros((dim_v, rank_schoolbook))
    W_fb = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    def to_lists(mat):
        return mat.tolist()
    
    fallback_result = {"U": to_lists(U_fb), "V": to_lists(V_fb), "W": to_lists(W_fb)}
    
    # Try to find lower rank decompositions
    lower_bound = max(dim_u, dim_v, dim_w)
    # Try ranks from lower_bound up to schoolbook rank
    # We focus on the likely minimal rank first
    
    best_result = fallback_result
    best_rank = rank_schoolbook
    
    # We will try to find a decomposition for ranks r in [18, 25]
    # Using Power Iteration
    
    np.random.seed(42)
    
    for r in range(lower_bound, min(lower_bound + 10, rank_schoolbook)):
        if time.time() > deadline - 5:
            break
            
        # Try multiple restarts for this rank
        success = False
        num_restarts = 50
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            residual = T.copy()
            U_factors = []
            V_factors = []
            W_factors = []
            Coeffs = []
            
            # Initialize random vectors
            u = np.random.randn(dim_u)
            v = np.random.randn(dim_v)
            w = np.random.randn(dim_w)
            
            # Normalize
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
            
            for k in range(r):
                # Power Iteration to find best rank-1 approximation of residual
                prev_coeff = 0.0
                converged = False
                
                for _iter in range(100):
                    # Update u
                    # temp_u = np.tensordot(residual, v, axes=([1], [0])) # (dim_u, dim_w)
                    # u_new = np.tensordot(temp_u, w, axes=([1], [0]))
                    # More efficient:
                    u_new = np.einsum('ijk,j,k->i', residual, v, w)
                    norm_u = np.linalg.norm(u_new)
                    if norm_u < 1e-10:
                        break
                    u_new /= norm_u
                    
                    # Update v
                    v_new = np.einsum('ijk,i,k->j', residual, u, w)
                    norm_v = np.linalg.norm(v_new)
                    if norm_v < 1e-10:
                        break
                    v_new /= norm_v
                    
                    # Update w
                    w_new = np.einsum('ijk,i,j->k', residual, u, v)
                    norm_w = np.linalg.norm(w_new)
                    if norm_w < 1e-10:
                        break
                    w_new /= norm_w
                    
                    # Check convergence
                    coeff = np.einsum('ijk,i,j,k->', residual, u_new, v_new, w_new)
                    if abs(coeff - prev_coeff) < 1e-9:
                        u, v, w = u_new, v_new, w_new
                        converged = True
                        break
                    u, v, w = u_new, v_new, w_new
                    prev_coeff = coeff
                
                # Calculate final coefficient for this term
                coeff = np.einsum('ijk,i,j,k->', residual, u, v, w)
                
                # Update residual
                residual -= coeff * np.einsum('i,j,k->ijk', u, v, w)
                
                # Store factors
                U_factors.append(u)
                V_factors.append(v)
                W_factors.append(w)
                Coeffs.append(coeff)
                
                # Check if residual is negligible
                if np.linalg.norm(residual) < 1e-8:
                    success = True
                    break
            
            if not success:
                continue
            
            # We have a real-valued decomposition.
            # Construct U, V, W matrices
            U_mat = np.zeros((dim_u, r))
            V_mat = np.zeros((dim_v, r))
            W_mat = np.zeros((dim_w, r))
            
            for col_idx in range(r):
                # Absorb coefficient into U
                U_mat[:, col_idx] = Coeffs[col_idx] * U_factors[col_idx]
                V_mat[:, col_idx] = V_factors[col_idx]
                W_mat[:, col_idx] = W_factors[col_idx]
            
            # Try to convert to half-integers
            # Check if 2*U, 2*V, 2*W are close to integers
            # And if the sum of products is 8*T
            
            U_scaled = 2 * U_mat
            V_scaled = 2 * V_mat
            W_scaled = 2 * W_mat
            
            U_int = np.round(U_scaled)
            V_int = np.round(V_scaled)
            W_int = np.round(W_scaled)
            
            # Check closeness
            if (np.allclose(U_int, U_scaled, atol=1e-6) and 
                np.allclose(V_int, V_scaled, atol=1e-6) and 
                np.allclose(W_int, W_scaled, atol=1e-6)):
                
                # Verify the decomposition with integers
                # Sum U_int * V_int * W_int should be 8 * T
                T_check = np.zeros((dim_u, dim_v, dim_w))
                for col_idx in range(r):
                    T_check += np.einsum('i,j,k->ijk', U_int[:, col_idx], V_int[:, col_idx], W_int[:, col_idx])
                
                if np.allclose(T_check, 8 * T, atol=1e-6):
                    # Found a valid half-integer decomposition
                    U_final = U_int / 2.0
                    V_final = V_int / 2.0
                    W_final = W_int / 2.0
                    
                    # Double check with floats
                    T_verify = np.zeros((dim_u, dim_v, dim_w))
                    for col_idx in range(r):
                        T_verify += np.einsum('i,j,k->ijk', U_final[:, col_idx], V_final[:, col_idx], W_final[:, col_idx])
                    
                    if np.allclose(T_verify, T, atol=1e-6):
                        best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        best_rank = r
                        success = True
                        break # Break restart loop
            
            # If integer conversion failed, try integers directly (denom 1)
            U_int1 = np.round(U_mat)
            V_int1 = np.round(V_mat)
            W_int1 = np.round(W_mat)
            
            if (np.allclose(U_int1, U_mat, atol=1e-6) and 
                np.allclose(V_int1, V_mat, atol=1e-6) and 
                np.allclose(W_int1, W_mat, atol=1e-6)):
                
                T_check1 = np.zeros((dim_u, dim_v, dim_w))
                for col_idx in range(r):
                    T_check1 += np.einsum('i,j,k->ijk', U_int1[:, col_idx], V_int1[:, col_idx], W_int1[:, col_idx])
                
                if np.allclose(T_check1, T, atol=1e-6):
                    best_result = {"U": U_int1.tolist(), "V": V_int1.tolist(), "W": W_int1.tolist()}
                    best_rank = r
                    success = True
                    break # Break restart loop
        
        if best_rank < rank_schoolbook:
            # Found a better rank, stop searching higher ranks
            break

    return best_result
# EVOLVE-BLOCK-END

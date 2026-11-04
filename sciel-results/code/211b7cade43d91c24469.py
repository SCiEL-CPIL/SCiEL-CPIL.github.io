"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # The problem 3x4x6 can be decomposed into 4 independent 3x2x3 problems.
    # 1. Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}.
    # 2. Split p=6 into two blocks of 3: k in {0,1,2} and k in {3,4,5}.
    # Each subproblem is multiplying a 3x2 matrix by a 2x3 matrix.
    # If we find a rank-r decomposition for 3x2x3, the total rank is 4*r.
    # Schoolbook for 3x2x3 is 18, so total 72.
    
    def find_323_decomposition():
        # Target tensor for 3x2x3
        # T[i, j, k] = 1 for all i in 0..2, j in 0..1, k in 0..2
        # Wait, the tensor for matrix multiplication is not just 1s.
        # T[i*m+j, j*p+k, k*n+i] = 1.
        # For 3x2x3: n=3, m=2, p=3.
        # U index: i*2+j
        # V index: j*3+k
        # W index: k*3+i
        
        T = np.zeros((3*2, 2*3, 3*3))
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    u = i*2 + j
                    v = j*3 + k
                    w = k*3 + i
                    T[u, v, w] = 1.0
        
        T_flat = T.flatten()
        
        def check_exact(U, V, W, rank):
            # U: (3, 2, rank)
            # V: (2, 3, rank)
            # W: (3, 3, rank)
            # Recalculate T_rec
            # T_rec[u, v, w] = sum_c U[u, c] * V[v, c] * W[w, c]
            # Using einsum on reshaped tensors is efficient
            # U is (i, j, c), V is (j, k, c), W is (k, i, c)
            # T_rec[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            # Reshape T_rec to match T
            T_rec_flat = T_rec.reshape(3*2, 2*3, 3*3).flatten()
            # Note: The flattening order must match.
            # T is indexed by [u, v, w] where u=i*2+j, v=j*3+k, w=k*3+i.
            # T_rec[i,j,k] corresponds to u=i*2+j, v=j*3+k, w=k*3+i.
            # So T_rec.reshape(3*2, 2*3, 3*3) should align if we construct T correctly.
            # Let's verify T construction vs T_rec construction.
            # T[u, v, w] = 1.
            # T_rec[i, j, k] = 1.
            # u = i*2+j, v=j*3+k, w=k*3+i.
            # If we reshape T_rec (3,2,3) to (3*2, 2*3, 3*3), the indices are:
            # axis 0: i*2+j? No, standard C-order flatten of (3,2,3) is:
            # (i, j, k) -> i*2*3 + j*3 + k.
            # This does NOT match u, v, w structure directly in a single flatten.
            # We must compare element by element or reshape carefully.
            
            # Let's just compare the flattened tensors if we construct T in the same (i,j,k) order.
            # Let T_ijk[i, j, k] = 1.
            # Then T_ijk.flatten() (C-order) matches T_rec.flatten().
            T_ijk = np.zeros((3, 2, 3))
            T_ijk[:, :, :] = 1.0 # All entries are 1 for 3x2x3 multiplication?
            # Wait. T[i,j,k] = 1 for ALL i,j,k?
            # Yes, for matrix multiplication, the tensor has 1 at (i,j,k) for all valid i,j,k.
            # There are no "zero" entries in the support of the multiplication tensor 
            # except if we consider the full space of dimensions, but here 
            # every combination of input indices (i,j) and (j,k) produces an output (k,i)?
            # No, the tensor is defined on the product space.
            # T is a 3rd order tensor.
            # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
            # For 3x2x3:
            # a in 0..5, b in 0..8, c in 0..8.
            # For each i,j,k, there is exactly one (a,b,c) with value 1.
            # Are there other (a,b,c) with value 1?
            # No, the mapping (i,j,k) -> (a,b,c) is bijective onto the support.
            # So T has exactly 3*2*3 = 18 ones.
            # All other entries are 0.
            
            # So T_ijk is NOT all ones.
            # T_ijk[i,j,k] = 1.
            # But when flattened to (6, 9, 9), the positions are specific.
            # Let's stick to the (a,b,c) representation for checking.
            
            # Reconstruct T_rec in (a,b,c) form
            T_rec_abc = np.zeros((6, 9, 9))
            for i in range(3):
                for j in range(2):
                    for k in range(3):
                        a = i*2 + j
                        b = j*3 + k
                        c = k*3 + i
                        # T_rec_abc[a, b, c] should be T_rec[i, j, k]
                        T_rec_abc[a, b, c] = T_rec[i, j, k]
                        
            return np.allclose(T_rec_abc.flatten(), T.flatten(), atol=1e-9)

        # ALS for 3x2x3
        best_rank = 18
        best_U = None
        best_V = None
        best_W = None
        
        # Try ranks from 10 to 17
        for rank in range(10, 18):
            if time.time() > deadline - 10:
                break
                
            for seed in range(20):
                if time.time() > deadline - 5:
                    break
                    
                np.random.seed(seed)
                U = np.random.randn(3, 2, rank) * 0.5
                V = np.random.randn(2, 3, rank) * 0.5
                W = np.random.randn(3, 3, rank) * 0.5
                
                prev_err = np.inf
                for it in range(50):
                    # Compute error
                    T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    # Build target T_ijk
                    # T_ijk[i,j,k] = 1
                    T_target = np.ones((3, 2, 3))
                    err = T_rec - T_target
                    current_err = np.sum(err**2)
                    
                    if abs(prev_err - current_err) < 1e-12:
                        break
                    prev_err = current_err
                    
                    # Solve for U
                    for i in range(3):
                        for j in range(2):
                            # T[i, j, k] = sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c])
                            # Let A[c, k] = V[j, k, c] * W[k, i, c]
                            # T[i, j, :] = U[i, j, :] @ A
                            # U[i, j, :] = T[i, j, :] @ A^{-1} (if invertible) or lstsq
                            A = np.einsum('kc,kc->ck', V[j], W[:, i]) # shape (rank, 3)
                            b = T_target[i, j, :] # shape (3,)
                            try:
                                u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                                U[i, j, :] = u_sol
                            except:
                                pass
                    
                    # Solve for V
                    for j in range(2):
                        for k in range(3):
                            # T[i, j, k] = sum_c (U[i, j, c] * W[k, i, c]) * V[j, k, c]
                            # A[c, i] = U[i, j, c] * W[k, i, c]
                            # T[:, j, k] = V[j, k, :] @ A
                            A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :]) # shape (rank, 3)
                            b = T_target[:, j, k] # shape (3,)
                            try:
                                v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                                V[j, k, :] = v_sol
                            except:
                                pass
                    
                    # Solve for W
                    for k in range(3):
                        for i in range(3):
                            # T[i, j, k] = sum_c (U[i, j, c] * V[j, k, c]) * W[k, i, c]
                            # A[c, j] = U[i, j, c] * V[j, k, c]
                            # T[i, :, k] = W[k, i, :] @ A
                            A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :]) # shape (rank, 2)
                            b = T_target[i, :, k] # shape (2,)
                            try:
                                w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                                W[k, i, :] = w_sol
                            except:
                                pass
                
                # Check for exact solution with rounding
                # Try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if check_exact(U_int, V_int, W_int, rank):
                    return rank, U_int, V_int, W_int
                
                # Try half-integer rounding
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                if check_exact(U_half, V_half, W_half, rank):
                    return rank, U_half, V_half, W_half
                    
        return None

    result_323 = find_323_decomposition()
    
    if result_323 is not None:
        r, U323, V323, W323 = result_323
        total_rank = 4 * r
        
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        # Block 0: j in {0,1}, k in {0,1,2} -> cols 0..r-1
        # Block 1: j in {0,1}, k in {3,4,5} -> cols r..2r-1
        # Block 2: j in {2,3}, k in {0,1,2} -> cols 2r..3r-1
        # Block 3: j in {2,3}, k in {3,4,5} -> cols 3r..4r-1
        
        for block in range(4):
            col_offset = block * r
            
            j_start = (block // 2) * 2  # 0 or 2
            k_start = (block % 2) * 3    # 0 or 3
            
            # Fill U
            # U[i, j, c] for local i, j
            # Global U index: i*4 + (j_start + j_local)
            for i in range(3):
                for j_local in range(2):
                    j_global = j_start + j_local
                    u_idx = i * 4 + j_global
                    U[u_idx, col_offset:col_offset+r] = U323[i, j_local, :]
            
            # Fill V
            # V[j, k, c] for local j, k
            # Global V index: (j_start + j_local)*6 + (k_start + k_local)
            for j_local in range(2):
                for k_local in range(3):
                    j_global = j_start + j_local
                    k_global = k_start + k_local
                    v_idx = j_global * 6 + k_global
                    V[v_idx, col_offset:col_offset+r] = V323[j_local, k_local, :]
            
            # Fill W
            # W[k, i, c] for local k, i
            # Global W index: (k_start + k_local)*3 + i
            for k_local in range(3):
                for i in range(3):
                    k_global = k_start + k_local
                    w_idx = k_global * 3 + i
                    W[w_idx, col_offset:col_offset+r] = W323[k_local, i, :]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = 72
        U = np.zeros((12, rank))
        V = np.zeros((24, rank))
        W = np.zeros((18, rank))
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

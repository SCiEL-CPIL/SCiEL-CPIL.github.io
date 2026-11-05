import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition for 3x2x3 sub-blocks
    # If successful, total rank will be 4 * R_sub
    # Schoolbook rank for 3x2x3 is 18.
    # Target rank for sub-block: 12 (if known to exist)
    
    def solve_323(rank_target):
        # n=3, m=2, p=3
        nn, mm, pp = 3, 2, 3
        # U: (nn*mm, r) = (6, r)
        # V: (mm*pp, r) = (6, r)
        # W: (pp*nn, r) = (9, r)
        
        # Build target tensor T as a flat vector or sparse structure
        # We want sum_r U[u,r] V[v,r] W[w,r] == T[u,v,w]
        # T[u,v,w] is 1 if u=i*mm+j, v=j*pp+k, w=k*nn+i for some i,j,k
        # else 0
        
        # List of non-zero indices
        non_zero = []
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    u = i * mm + j
                    v = j * pp + k
                    w = k * nn + i
                    non_zero.append((u, v, w))
                    
        # Total indices
        size_u = nn * mm
        size_v = mm * pp
        size_w = pp * nn
        
        # We use least squares to solve for U, V, W
        # This is non-linear. We can use a simple iterative method.
        # Initialize with random values
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Try a few restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(restart)
            U = np.random.randn(size_u, rank_target) * 0.1
            V = np.random.randn(size_v, rank_target) * 0.1
            W = np.random.randn(size_w, rank_target) * 0.1
            
            # Simple gradient descent or Gauss-Newton
            # Objective: minimize sum_{u,v,w} (sum_r U[u,r]V[v,r]W[w,r] - T[u,v,w])^2
            # Since T is sparse, we only care about non-zero terms matching 1 and zero terms matching 0.
            # Actually, we must match ALL terms.
            # Zero terms: (size_u * size_v * size_w - len(non_zero))
            # This is 6*6*9 - 18 = 324 - 18 = 306 zero terms.
            # 18 one terms.
            
            # We can form a Jacobian? Too expensive.
            # Use simple updates.
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                # Compute current tensor
                # T_curr = np.einsum('ur,vr,wr->uvw', U, V, W)
                # This is 6x6x9 = 324 elements. Fast.
                T_curr = np.einsum('ur,vr,wr->uvw', U, V, W)
                
                # Compute error
                # We need the gradient.
                # d(T_curr)/dU[u,r] = V[v,r] W[w,r]
                # d(T_curr)/dV[v,r] = U[u,r] W[w,r]
                # d(T_curr)/dW[w,r] = U[u,r] V[v,r]
                
                # Residual R[u,v,w] = T_curr[u,v,w] - T_target[u,v,w]
                T_target = np.zeros((size_u, size_v, size_w))
                for u, v, w in non_zero:
                    T_target[u, v, w] = 1.0
                
                R = T_curr - T_target
                
                # Gradient for U
                # grad_U[u,r] = sum_{v,w} R[u,v,w] * V[v,r] * W[w,r]
                # grad_U = np.einsum('uvw,vr,wr->ur', R, V, W)
                grad_U = np.einsum('uvw,vr,wr->ur', R, V, W)
                grad_V = np.einsum('uvw,ur,wr->vr', R, U, W)
                grad_W = np.einsum('uvw,ur,vr->wr', R, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check error
                err = np.linalg.norm(R)
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < 1e-6:
                    break
            
        if best_error < 1e-5:
            # Snap to half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            # Verify
            T_check = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
            T_target = np.zeros((size_u, size_v, size_w))
            for u, v, w in non_zero:
                T_target[u, v, w] = 1.0
            
            if np.allclose(T_check, T_target, atol=1e-9):
                return U_snap, V_snap, W_snap
        return None

    # Try to find rank 12 decomposition for 3x2x3
    U_sub, V_sub, W_sub = solve_323(12)
    
    if U_sub is not None:
        # Construct full 3x4x6 decomposition
        # 4 blocks of 3x2x3
        # Block 1: i in 0..2, j in 0..1, k in 0..2
        # Block 2: i in 0..2, j in 0..1, k in 3..5
        # Block 3: i in 0..2, j in 2..3, k in 0..2
        # Block 4: i in 0..2, j in 2..3, k in 3..5
        
        # U_full: (12, 48)
        # V_full: (24, 48)
        # W_full: (18, 48)
        
        rank_full = 48
        U_full = np.zeros((12, rank_full))
        V_full = np.zeros((24, rank_full))
        W_full = np.zeros((18, rank_full))
        
        col_offset = 0
        for b_j in range(2): # j blocks
            for b_k in range(2): # k blocks
                # Extract sub-block factors
                # U_sub is for i in 0..2, j in 0..1, k in 0..2
                # We need to map indices
                
                # U indices: i*2 + j
                # In full U: i*4 + (b_j*2 + j_local)
                # V indices: j*3 + k
                # In full V: (b_j*2 + j_local)*6 + (b_k*3 + k_local)
                # W indices: k*3 + i
                # In full W: (b_k*3 + k_local)*3 + i
                
                for r in range(12):
                    # U
                    for i in range(3):
                        for j_local in range(2):
                            u_sub_idx = i * 2 + j_local
                            u_full_idx = i * 4 + (b_j * 2 + j_local)
                            U_full[u_full_idx, col_offset + r] = U_sub[u_sub_idx, r]
                            
                    # V
                    for j_local in range(2):
                        for k_local in range(3):
                            v_sub_idx = j_local * 3 + k_local
                            v_full_idx = (b_j * 2 + j_local) * 6 + (b_k * 3 + k_local)
                            V_full[v_full_idx, col_offset + r] = V_sub[v_sub_idx, r]
                            
                    # W
                    for k_local in range(3):
                        for i in range(3):
                            w_sub_idx = k_local * 3 + i
                            w_full_idx = (b_k * 3 + k_local) * 3 + i
                            W_full[w_full_idx, col_offset + r] = W_sub[w_sub_idx, r]
                            
                col_offset += 12
        
        # Convert to lists
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

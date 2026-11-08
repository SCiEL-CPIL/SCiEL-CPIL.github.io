import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} for valid (i,j,k)
    # A is n x m, B is m x p, C is n x p
    # T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] to get C[i,k]
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for each (i,j,k) triple where the product contributes to C[i,k]
    
    # The tensor T of shape (n, m, p) where T[i,j,k] = 1
    # because C[i,k] gets contribution A[i,j]*B[j,k]
    
    def build_tensor():
        """Build the matrix multiplication tensor T[i,j,k] = 1"""
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = 1.0
        return T
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W decompose the tensor correctly"""
        rank = U.shape[1]
        T_recon = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]  # W[k*n+i, r]
        T_target = build_tensor()
        return np.allclose(T_recon, T_target, atol=1e-10)
    
    def try_rank_decomposition(rank, max_iters=50, tol=1e-8):
        """Try to find a rank-r decomposition using alternating least squares"""
        T = build_tensor()
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n, m, rank) * 0.1
        V = np.random.randn(m, p, rank) * 0.1
        W = np.random.randn(p, n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j,r), minimize ||T - sum_{r'} U[i,j,r']*V[j,:,r']*W[:,i,r']||
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        # Target: T[i,j,:] - sum_{r' != r} U[i,j,r']*V[j,:,r']*W[:,i,r']
                        residual = T[i,j,:].copy()
                        for r2 in range(rank):
                            if r2 != r:
                                residual -= U[i,j,r2] * V[j,:,r2] * W[:,i,r2]
                        # Solve U[i,j,r] * (V[j,:,r] * W[:,i,r]) = residual
                        vec = V[j,:,r] * W[:,i,r]
                        if np.dot(vec, vec) > 1e-12:
                            U[i,j,r] = np.dot(residual, vec) / np.dot(vec, vec)
            
            # Update V
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        residual = T[:,j,k].copy()
                        for r2 in range(rank):
                            if r2 != r:
                                residual -= U[:,j,r2] * V[j,k,r2] * W[k,:,r2]
                        vec = U[:,j,r] * W[k,:,r]
                        if np.dot(vec, vec) > 1e-12:
                            V[j,k,r] = np.dot(residual, vec) / np.dot(vec, vec)
            
            # Update W
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        residual = T[i,:,k].copy()
                        for r2 in range(rank):
                            if r2 != r:
                                residual -= U[i,:,r2] * V[:,k,r2] * W[k,i,r2]
                        vec = U[i,:,r] * V[:,k,r]
                        if np.dot(vec, vec) > 1e-12:
                            W[k,i,r] = np.dot(residual, vec) / np.dot(vec, vec)
            
            # Check convergence
            T_recon = np.zeros((n, m, p), dtype=float)
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
            err = np.linalg.norm(T_recon - T)
            if err < tol:
                # Check if entries are close to integers or half-integers
                all_half_int = True
                for arr in [U, V, W]:
                    for val in arr.flatten():
                        if abs(val * 2 - round(val * 2)) > 1e-6:
                            all_half_int = False
                            break
                    if not all_half_int:
                        break
                if all_half_int:
                    return U, V, W, True
                else:
                    # Still valid but not integer/half-integer, try to round
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    if verify_decomposition(U_round, V_round, W_round):
                        return U_round, V_round, W_round, True
        
        return None, None, None, False
    
    # Try ranks from 20 down to find minimal
    best_result = None
    for rank in range(20, 14, -1):
        if time.time() > deadline - 10:
            break
        U, V, W, success = try_rank_decomposition(rank, max_iters=30)
        if success:
            best_result = (U, V, W, rank)
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to list format with W in trace convention W[k*n+i, r]
        U_list = [[0] * rank for _ in range(n * m)]
        V_list = [[0] * rank for _ in range(m * p)]
        W_list = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                for r in range(rank):
                    U_list[idx_u][r] = U[i, j, r]
        
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                for r in range(rank):
                    V_list[idx_v][r] = V[j, k, r]
        
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                for r in range(rank):
                    W_list[idx_w][r] = W[k, i, r]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the known rank-22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
            
    # Indices:
    # A: 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2)
    # B: 0:(0,0) 1:(0,1) 2:(0,2) 3:(0,3) 4:(1,0) 5:(1,1) 6:(1,2) 7:(1,3) 8:(2,0) 9:(2,1) 10:(2,2) 11:(2,3)
    # C: 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1) 4:(0,2) 5:(1,2) 6:(0,3) 7:(1,3)
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # C00=C[0,0]=0, C01=C[0,1]=2, C10=C[1,0]=1, C11=C[1,1]=3
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # C02=C[0,2]=4, C03=C[0,3]=6, C12=C[1,2]=5, C13=C[1,3]=7
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Cross terms: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # A02=2, A12=5
    # B20=8, B21=9, B22=10, B23=11
    # C00=0, C01=2, C02=4, C03=6
    # C10=1, C11=3, C12=5, C13=7
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14) # A02*B20 -> C00
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15) # A02*B21 -> C01
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16) # A02*B22 -> C02
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17) # A02*B23 -> C03
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18) # A12*B20 -> C10
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19) # A12*B21 -> C11
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20) # A12*B22 -> C12
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21) # A12*B23 -> C13
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

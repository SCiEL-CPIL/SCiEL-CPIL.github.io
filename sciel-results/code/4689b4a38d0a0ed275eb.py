import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Target: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 40 down to 30
    for rank in range(40, 29, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Iterative refinement using alternating least squares
            for it in range(1000):
                if time.time() > deadline - 20:
                    break
                
                # Fix V and W, solve for U
                # For each (j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i
                # For fixed (j,k): sum_r (V[j*p+k,r]*W[k*n+i,r]) * U[i*m+j,r] = 1 for all i
                # Let coeff[i,r] = V[j*p+k,r]*W[k*n+i,r]
                # Then: sum_r coeff[i,r]*U[i*m+j,r] = 1 for all i
                # This is: for each i, U[i*m+j,:] dot coeff[i,:] = 1
                
                U_new = np.zeros_like(U)
                for j in range(m):
                    for k in range(p):
                        # Build system: for each i, U[i*m+j,:] dot (V[j*p+k,:]*W[k*n+i,:]) = 1
                        # This is: sum_i (U[i*m+j,:] * (V[j*p+k,:]*W[k*n+i,:])) = n
                        # Actually, for each i independently:
                        # U[i*m+j,:] dot (V[j*p+k,:]*W[k*n+i,:]) = 1
                        # So we have n equations in rank unknowns (the row U[i*m+j,:] for each i)
                        # But U[i*m+j,:] is the same for all i? No, U is (n*m, rank), so U[i*m+j,:] differs by i.
                        
                        # Wait, U has shape (n*m, rank). U[i*m+j, r] is the r-th entry of the (i*m+j)-th row.
                        # For fixed j,k, we need: for each i, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Let c_i = V[j*p+k,:] * W[k*n+i,:] (element-wise product, length rank)
                        # Then: U[i*m+j,:] dot c_i = 1 for each i
                        # This gives us n linear equations in the n*rank unknowns U[i*m+j,:] for i=0..n-1
                        # But these are n separate equations, each involving only one row of U.
                        # So for each i: U[i*m+j,:] dot c_i = 1
                        # This is underdetermined if rank > 1. Use least squares.
                        
                        for i in range(n):
                            c_i = V[j*p+k, :] * W[k*n+i, :]
                            if np.linalg.norm(c_i) < 1e-12:
                                U_new[i*m+j, :] = 0
                            else:
                                # Minimum norm solution to U[i*m+j,:] dot c_i = 1
                                U_new[i*m+j, :] = c_i / np.dot(c_i, c_i)
                
                # Fix U and W, solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # For each i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Let d_i = U[i*m+j,:] * W[k*n+i,:]
                        # Then: V[j*p+k,:] dot d_i = 1 for each i
                        for i in range(n):
                            d_i = U[i*m+j, :] * W[k*n+i, :]
                            if np.linalg.norm(d_i) < 1e-12:
                                V_new[j*p+k, :] = 0
                            else:
                                V_new[j*p+k, :] = d_i / np.dot(d_i, d_i)
                
                # Fix U and V, solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Let e_j = U[i*m+j,:] * V[j*p+k,:]
                        # Then: W[k*n+i,:] dot e_j = 1 for each j
                        for j in range(m):
                            e_j = U[i*m+j, :] * V[j*p+k, :]
                            if np.linalg.norm(e_j) < 1e-12:
                                W_new[k*n+i, :] = 0
                            else:
                                W_new[k*n+i, :] = e_j / np.dot(e_j, e_j)
                
                U_diff = np.linalg.norm(U_new - U)
                V_diff = np.linalg.norm(V_new - V)
                W_diff = np.linalg.norm(W_new - W)
                
                U = U_new
                V = V_new
                W = W_new
                
                if U_diff < 1e-10 and V_diff < 1e-10 and W_diff < 1e-10:
                    break
            
            # Verify the decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Found a valid decomposition!
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    if best_U is None:
        # Fallback to the original rank-42 decomposition
        rank = 42
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        steps = [
            (
                {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1), (3, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
                [(2, 1), (3, -1)]
            ),
            (
                {(0,0): 1, (0,1): 1},
                {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
                [(1, 1), (3, 1)]
            ),
            (
                {(1,2): 1, (1,3): 1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
                [(0, 1), (2, 1)]
            ),
            (
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
                {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, -1), (1, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
                [(3, 1)]
            ),
            (
                {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1)]
            ),
        ]
        
        col = 0
        for X_dict, Y_dict, C_targets in steps:
            x0_coeffs = {}
            x1_coeffs = {}
            for (i, j), c in X_dict.items():
                if j % 2 == 0:
                    x0_coeffs[(i, j)] = c
                else:
                    x1_coeffs[(i, j)] = c
                    
            y_coeffs = [[], [], [], [], [], []]
            for (j, k), c in Y_dict.items():
                r_block = j % 2
                c_block = k % 3
                idx = r_block * 3 + c_block
                y_coeffs[idx].append(((j, k), c))
                
            sub_terms = [
                (x0_coeffs, y_coeffs[0], 0),
                (x1_coeffs, y_coeffs[3], 0),
                (x0_coeffs, y_coeffs[1], 1),
                (x1_coeffs, y_coeffs[4], 1),
                (x0_coeffs, y_coeffs[2], 2),
                (x1_coeffs, y_coeffs[5], 2),
            ]
            
            for X_c, Y_c, z_idx in sub_terms:
                for (i, j), c in X_c.items():
                    U[i * m + j, col] += c
                for (j, k), c in Y_c:
                    V[j * p + k, col] += c
                
                for block_idx, sign in C_targets:
                    if block_idx == 0:
                        i_C = 0
                        k_C = z_idx
                    elif block_idx == 1:
                        i_C = 0
                        k_C = 3 + z_idx
                    elif block_idx == 2:
                        i_C = 1
                        k_C = z_idx
                    elif block_idx == 3:
                        i_C = 1
                        k_C = 3 + z_idx
                    
                    W[k_C * n + i_C, col] += sign
                
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

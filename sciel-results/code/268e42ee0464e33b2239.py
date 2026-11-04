import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = 2, 4, 7
    target_rank = 49
    
    # 1. Generate the ground truth tensor T
    # T[i, j, k] = 1 if i,j,k are the standard basis, but here it's the multiplication tensor.
    # The tensor T is defined by T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # In the tensor decomposition context, we are decomposing the bilinear map.
    # The "true" tensor T has entries T[i,j,k] = 1 for all i,j,k? 
    # No. The tensor for matrix multiplication is T = sum_{i,j,k} e_i e_j^T e_k^T * (A_{ij} B_{jk} -> C_{ik})?
    # Actually, the tensor T is in V_A^* \otimes V_B^* \otimes V_C.
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the coefficient is 1 if the multiplication A_{ij} B_{jk} contributes to C_{ik}.
    # Wait, the standard matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k  is NOT correct.
    # The tensor T is such that T(A, B) = C.
    # T(A, B) = sum_{i,j,k} T_{ijk} A_{ij} B_{jk} e_{ik}.
    # The coefficient T_{ijk} is 1 if the index k in B corresponds to the column of C?
    # No, C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i, j, k.
    # Let's verify.
    # If T_{ijk} = 1, then sum_{i,j,k} 1 * A_{ij} B_{jk} e_{ik} = sum_{i,k} (sum_j A_{ij} B_{jk}) e_{ik} = C.
    # Yes, the tensor T is simply all ones.
    
    T_true = np.ones((n, m, p))
    
    # 2. Define the residual function for rank r
    def residual(params, r):
        # Unpack params into U, V, W
        # U shape: (n*m, r) = (8, r)
        # V shape: (m*p, r) = (28, r)
        # W shape: (p*n, r) = (14, r)
        
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size + v_size]
        w_flat = params[u_size + v_size:]
        
        U = u_flat.reshape(n * m, r)
        V = v_flat.reshape(m * p, r)
        W = w_flat.reshape(p * n, r)
        
        # Compute T_decomp
        # T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient computation using einsum or loops
        # T_decomp = np.einsum('ir,jr,kr->ijk', 
        #                     U.reshape(n, m, r), 
        #                     V.reshape(m, p, r), 
        #                     W.reshape(p, n, r))
        # Wait, W index is k*n+i. So W[k, i, r].
        # U index is i*m+j. So U[i, j, r].
        # V index is j*p+k. So V[j, k, r].
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # T_decomp[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        T_decomp = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
        
        return (T_decomp - T_true).flatten()

    # 3. Try to find a rank 49 solution
    best_solution = None
    best_residual = np.inf
    
    # We will try a few random initializations
    # To save time, we limit the number of attempts
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters. Strassen uses 1s and -1s.
        # We can initialize with small random numbers or integers.
        r = target_rank
        total_params = (n*m + m*p + p*n) * r
        params0 = np.random.randn(total_params) * 0.1
        
        try:
            res = least_squares(residual, params0, args=(r,), method='lm', max_nfev=1000)
            res_val = res.cost
            # res.cost is 0.5 * sum(residual^2)
            # We want sum(residual^2) to be small.
            if res_val < 1e-10:
                # Found a solution
                # Round to nearest half-integer
                params_sol = res.x
                params_rounded = np.round(params_sol * 2) / 2.0
                
                # Verify rounded solution
                res_rounded = residual(params_rounded, r)
                if np.linalg.norm(res_rounded) < 1e-9:
                    best_solution = params_rounded
                    best_residual = np.linalg.norm(res_rounded)
                    break # Found a valid integer/half-integer solution
        except Exception as e:
            continue

    # 4. If no rank 49 found, use rank 50
    if best_solution is None:
        # Construct the known rank 50 solution
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1
            U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1
            U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1
            U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + 6][col] = 1
                W[6*n + i][col] = 1
                col += 1
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Unpack best_solution
        r = target_rank
        u_size = n * m * r
        v_size = m * p * r
        
        u_flat = best_solution[:u_size]
        v_flat = best_solution[u_size:u_size + v_size]
        w_flat = best_solution[u_size + v_size:]
        
        U = u_flat.reshape(n * m, r)
        V = v_flat.reshape(m * p, r)
        W = w_flat.reshape(p * n, r)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

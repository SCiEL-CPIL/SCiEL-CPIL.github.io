import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank
    # Target rank 25
    target_rank = 25
    if target_rank < rank_sb:
        # Define the tensor T
        # T[i, j, j, k, k, i] = 1 is not the right shape.
        # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if valid, 0 else.
        
        # Residual function
        def residuals(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank :]
            
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Compute tensor product
            # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # We need to check all i, j, k
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] = np.dot(U[i*m+j], V[j*p+k]) # This is wrong. 
                        # It's sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # This is a trilinear form, not a dot product of two vectors.
                        # We need element-wise product then sum.
                        # res[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        pass
            
            # Efficient computation:
            # For each r, U[:, r] is a vector of size nm. V[:, r] size mp. W[:, r] size pn.
            # The contribution to T is outer product? No.
            # T is a 3-tensor.
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            
            # Let's compute the full tensor T_candidate
            # T_candidate = sum_r np.einsum('a,b,c->abc', U_flat_r, V_flat_r, W_flat_r) ?
            # No, indices are coupled.
            # U index: i, j. V index: j, k. W index: k, i.
            # So for a fixed r, we have vectors u_r (size nm), v_r (size mp), w_r (size pn).
            # The term is not a simple outer product because indices j and k are shared.
            # However, we can compute the sum for each (i,j,k).
            
            # Let's use a loop for clarity and correctness, it's small (2*5*5=50 terms)
            res = np.zeros((n, m, p))
            for r in range(target_rank):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                for i in range(n):
                    for j in range(m):
                        u_val = u_r[i*m + j]
                        if u_val == 0: continue
                        for k in range(p):
                            v_val = v_r[j*p + k]
                            if v_val == 0: continue
                            w_val = w_r[k*n + i]
                            res[i, j, k] += u_val * v_val * w_val
                            
            # Target is 1 for all i,j,k
            # Since all i,j,k are valid indices in the multiplication tensor, 
            # the target for ALL i in [0,1], j in [0,4], k in [0,4] is 1.
            # Wait, are there any zeros in the multiplication tensor?
            # T_{i,j,k} corresponds to the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
            # This coefficient is always 1.
            # So the target tensor is all ones of shape (n, m, p).
            
            return (res - 1.0).flatten()
        
        # Optimization
        try:
            x0 = np.random.rand(n*m*target_rank + m*p*target_rank + p*n*target_rank) * 0.1
            # Use scipy.optimize.least_squares
            from scipy.optimize import least_squares
            start_time = time.time()
            result = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-8, ftol=1e-8)
            elapsed = time.time() - start_time
            if result.success or result.cost < 1e-6:
                U_flat = result.x[:n*m*target_rank]
                V_flat = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = result.x[n*m*target_rank + m*p*target_rank :]
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                # Round to half-integers?
                # The problem allows integer/half-integer.
                # Let's check if rounding works.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                def verify(U, V, W, rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                if abs(val - 1.0) > 1e-5:
                                    return False
                    return True
                if verify(U_rounded, V_rounded, W_rounded, target_rank):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        except Exception:
            pass
            
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 130
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Known rank 88 from Strassen block decomposition
    # We try to beat this with rank 80 or lower.
    # Theoretical lower bound is often higher, but let's try 80 first.
    # If 80 fails, try 82, 84, 86.
    
    # Strategy: Random restarts with gradient descent on the non-linear system
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks lower than 88
    target_ranks = [80, 82, 84, 86]
    
    def residual(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Efficient computation using einsum or loops
        # T[i,j,k] = sum_j U[i,j,:] * V[j,k,:] * W[k,i,:]
        # This is a bit complex for vectorization without huge memory.
        # Let's use a loop over i, j, k to compute residual, but that's slow.
        # Better: 
        # For each r, we have outer products.
        # T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] (broadcasting carefully)
        
        # U is (n*m, rank) -> (n, m, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # We can compute this by:
        # Temp[i,j,k,r] = U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # Then sum over r.
        
        # To avoid huge memory, we can sum in chunks or just use einsum if memory allows.
        # n=3, m=4, p=8, rank=80.
        # U_3d: 3x4x80
        # V_3d: 4x8x80
        # W_3d: 8x3x80
        # Product: 3x4x8x80. Size = 96*80 = 7680 floats. Very small.
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Residual is sum of squares of (T - 1)
        return np.sum((T - 1.0)**2)

    def run_optimization(rank, num_restarts=5, time_limit=10):
        n_params = n*m*rank + m*p*rank + p*n*rank
        success = False
        best_res = float('inf')
        
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random values
            x0 = np.random.randn(n_params) * 0.1
            
            # Use L-BFGS-B or Nelder-Mead? L-BFGS-B is better for smooth non-linear.
            # But we don't have gradients. Numerical gradients are expensive.
            # Let's try Nelder-Mead with a reasonable maxiter.
            # Or just use a simple coordinate descent / random search hybrid?
            # Given the non-convexity, random restarts + local optimization is key.
            
            # Actually, for tensor rank, the landscape is very rugged.
            # A common trick is to initialize with a known decomposition (like schoolbook)
            # and then try to "compress" it, but that's hard.
            
            # Alternative: Use the fact that rank 88 is known.
            # Can we find a rank 80 decomposition?
            # Literature: R(3,4,8) <= 80?
            # Let's assume it might be possible.
            
            # Let's try a simpler heuristic:
            # Start with rank 88 Strassen decomposition.
            # Set some columns to zero? No, that changes the tensor.
            # We need to find U, V, W of size (nm, r), (mp, r), (pn, r) such that T=1.
            
            # Let's try to optimize the residual.
            # Since numerical gradients are O(N) evaluations, and N is ~ (12+32+24)*80 = 4800.
            # This is too slow for standard scipy minimize without custom gradients.
            
            # Custom gradient:
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # dT/dU[i,j,r] = V[j,k,r] W[k,i,r]
            # dT/dV[j,k,r] = U[i,j,r] W[k,i,r]
            # dT/dW[k,i,r] = U[i,j,r] V[j,k,r]
            
            # We can compute these efficiently.
            
            def grad(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                dT = 2 * (T - 1.0)
                
                # dL/dU[i,j,r] = sum_k dT[i,j,k] * V[j,k,r] * W[k,i,r]
                dU = np.einsum('ijk,jkr,kir->ijr', dT, V_3d, W_3d)
                # dL/dV[j,k,r] = sum_i dT[i,j,k] * U[i,j,r] * W[k,i,r]
                dV = np.einsum('ijk,ijr,kir->jkr', dT, U_3d, W_3d)
                # dL/dW[k,i,r] = sum_j dT[i,j,k] * U[i,j,r] * V[j,k,r]
                dW = np.einsum('ijk,ijr,jkr->kir', dT, U_3d, V_3d)
                
                grad_flat = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return grad_flat
            
            # Use L-BFGS-B
            res = minimize(residual, x0, jac=grad, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success and res.fun < 1e-6:
                # Found a valid decomposition
                return True, res.x, res.fun
            
            # Keep the best result even if not successful, in case we can use it later?
            # No, we need exact decomposition.
            if res.fun < best_res:
                best_res = res.fun
        
        return False, None, best_res

    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts
        success, x_opt, res_val = run_optimization(rank, num_restarts=3, time_limit=20)
        
        if success:
            U = x_opt[:n*m*rank].reshape(n*m, rank)
            V = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest integer or half-integer?
            # The problem allows integer/half-integer entries.
            # If the solution is close to integers, round it.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded version
            U_3d = U_rounded.reshape(n, m, rank)
            V_3d = V_rounded.reshape(m, p, rank)
            W_3d = W_rounded.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            if np.allclose(T, 1.0, atol=1e-9):
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = rank
                break # Found a lower rank, stop searching lower ranks?
            else:
                # Maybe the raw solution is good enough if we don't round?
                # But the interface requires integer/half-integer?
                # "integer/half-integer entries"
                # If the optimizer finds non-integer values, we can't use them directly
                # unless they are close to half-integers.
                # Let's check if the raw solution is close to half-integers.
                dist = np.abs(U * 2 - np.round(U * 2)).max()
                if dist < 1e-6:
                     # It is effectively half-integer
                     pass
                else:
                     # Try to snap to nearest integers if very close
                     U_int = np.round(U)
                     V_int = np.round(V)
                     W_int = np.round(W)
                     
                     U_3d = U_int.reshape(n, m, rank)
                     V_3d = V_int.reshape(m, p, rank)
                     W_3d = W_int.reshape(p, n, rank)
                     T_int = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                     
                     if np.allclose(T_int, 1.0, atol=1e-9):
                         best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                         best_rank = rank
                         break

    # If we didn't find a better one, return the known rank 88 or schoolbook
    # The current program returns rank 88.
    # Let's implement the rank 88 Strassen decomposition from the current program
    # to ensure we don't regress below 88 if the optimization fails.
    
    def strassen_88():
        rank = 88
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_strassen_222(base_r, u_offset, v_offset, w_offset):
            # Term 1
            U[u_offset(0,0), base_r] = 1.0
            U[u_offset(1,1), base_r] = 1.0
            V[v_offset(0,0), base_r] = 1.0
            V[v_offset(0,1), base_r] = 1.0
            W[w_offset(0,0), base_r] = 1.0
            W[w_offset(0,1), base_r] = 1.0
            W[w_offset(1,0), base_r] = 1.0
            W[w_offset(1,1), base_r] = 1.0
            
            # Term 2
            U[u_offset(1,0), base_r+1] = 1.0
            U[u_offset(1,1), base_r+1] = 1.0
            V[v_offset(0,0), base_r+1] = 1.0
            W[w_offset(0,0), base_r+1] = 1.0
            W[w_offset(0,1), base_r+1] = 1.0
            
            # Term 3
            U[u_offset(0,0), base_r+2] = 1.0
            V[v_offset(0,1), base_r+2] = 1.0
            V[v_offset(1,1), base_r+2] = -1.0
            W[w_offset(0,1), base_r+2] = 1.0
            W[w_offset(1,1), base_r+2] = 1.0
            
            # Term 4
            U[u_offset(1,1), base_r+3] = 1.0
            V[v_offset(1,0), base_r+3] = 1.0
            V[v_offset(0,0), base_r+3] = -1.0
            W[w_offset(1,0), base_r+3] = 1.0
            W[w_offset(1,1), base_r+3] = 1.0
            
            # Term 5
            U[u_offset(0,0), base_r+4] = 1.0
            U[u_offset(0,1), base_r+4] = 1.0
            V[v_offset(1,1), base_r+4] = 1.0
            W[w_offset(1,1), base_r+4] = 1.0
            W[w_offset(1,0), base_r+4] = 1.0
            
            # Term 6
            U[u_offset(1,0), base_r+5] = 1.0
            U[u_offset(0,0), base_r+5] = -1.0
            V[v_offset(0,0), base_r+5] = 1.0
            V[v_offset(0,1), base_r+5] = 1.0
            W[w_offset(1,0), base_r+5] = 1.0
            W[w_offset(1,1), base_r+5] = 1.0
            
            # Term 7
            U[u_offset(0,1), base_r+6] = 1.0
            U[u_offset(1,1), base_r+6] = -1.0
            V[v_offset(1,0), base_r+6] = 1.0
            V[v_offset(1,1), base_r+6] = 1.0
            W[w_offset(1,1), base_r+6] = 1.0
            W[w_offset(1,0), base_r+6] = 1.0

        def add_rank4_122(base_r, u_offset, v_offset, w_offset):
            # Term 1
            U[u_offset(0,0), base_r] = 1.0
            V[v_offset(0,0), base_r] = 1.0
            W[w_offset(0,0), base_r] = 1.0
            # Term 2
            U[u_offset(0,1), base_r+1] = 1.0
            V[v_offset(1,0), base_r+1] = 1.0
            W[w_offset(0,0), base_r+1] = 1.0
            # Term 3
            U[u_offset(0,0), base_r+2] = 1.0
            V[v_offset(0,1), base_r+2] = 1.0
            W[w_offset(1,0), base_r+2] = 1.0
            # Term 4
            U[u_offset(0,1), base_r+3] = 1.0
            V[v_offset(1,1), base_r+3] = 1.0
            W[w_offset(1,0), base_r+3] = 1.0

        r = 0
        # Top part: n=2, m=4, p=8
        for kb in range(4):
            k_start = kb * 2
            def u_off(i, j): return i * 4 + j
            def v_off(j, k): return j * 8 + (k_start + k)
            def w_off(k, i): return (k_start + k) * 3 + i
            add_strassen_222(r, u_off, v_off, w_off)
            r += 7
            
        for kb in range(4):
            k_start = kb * 2
            def u_off(i, j): return i * 4 + (j + 2)
            def v_off(j, k): return (j + 2) * 8 + (k_start + k)
            def w_off(k, i): return (k_start + k) * 3 + i
            add_strassen_222(r, u_off, v_off, w_off)
            r += 7
            
        # Bottom part: n=1, m=4, p=8
        for kb in range(4):
            k_start = kb * 2
            def u_off(i, j): return 2 * 4 + j
            def v_off(j, k): return j * 8 + (k_start + k)
            def w_off(k, i): return (k_start + k) * 3 + 2
            add_rank4_122(r, u_off, v_off, w_off)
            r += 4
            
        for kb in range(4):
            k_start = kb * 2
            def u_off(i, j): return 2 * 4 + (j + 2)
            def v_off(j, k): return (j + 2) * 8 + (k_start + k)
            def w_off(k, i): return (k_start + k) * 3 + 2
            add_rank4_122(r, u_off, v_off, w_off)
            r += 4
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if best_rank > 88:
        return strassen_88()
    else:
        return best_result


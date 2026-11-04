   import numpy as np
   import time
   from scipy.optimize import minimize
   
   def construct():
       n, m, p = 2, 3, 3
       start_time = time.time()
       
       # Target Tensor T of shape (n*m, m*p, p*n)
       # T[i*m+j, j*p+k, k*n+i] = 1
       T = np.zeros((n*m, m*p, p*n))
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   T[i*m+j, j*p+k, k*n+i] = 1.0
       
       best_rank = 17
       best_result = None
       
       # Try to find rank 16
       for rank in [16]:
           # Multiple restarts
           for seed in range(10):
               if time.time() - start_time > 100:
                   break
               np.random.seed(seed)
               # Initialize U, V, W
               U = np.random.randn(n*m, rank)
               V = np.random.randn(m*p, rank)
               W = np.random.randn(p*n, rank)
               
               # Flatten parameters
               params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
               
               def objective(params):
                   U_flat, V_flat, W_flat = np.split(params, [n*m*rank, n*m*rank + m*p*rank])
                   U = U_flat.reshape(n*m, rank)
                   V = V_flat.reshape(m*p, rank)
                   W = W_flat.reshape(p*n, rank)
                   
                   # Compute tensor product
                   # T_calc[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                   # Efficiently:
                   # For each r, outer product of U[:, :, r] and V[:, :, r] ... no.
                   # T_calc = np.einsum('ir,jr,kr->ijk', U, V, W) -- indices don't match directly.
                   
                   # Let's reshape U, V, W to 3D
                   # U: (n, m, rank)
                   # V: (m, p, rank)
                   # W: (p, n, rank)
                   U_3d = U.reshape(n, m, rank)
                   V_3d = V.reshape(m, p, rank)
                   W_3d = W.reshape(p, n, rank)
                   
                   # T_calc[i, j, k] = sum_r U_3d[i, j, r] V_3d[j, k, r] W_3d[k, i, r]
                   # This is a bit tricky with einsum because of the index cycles.
                   # i->j->k->i
                   
                   # We can compute this by looping over r or using einsum
                   # T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                   
                   T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                   
                   # Error
                   err = np.sum((T_calc - T.reshape(n, m, p))**2)
                   return err
                   
               # Optimize
               res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
               
               if res.fun < 1e-9:
                   best_rank = rank
                   # Extract U, V, W
                   U_flat, V_flat, W_flat = np.split(res.x, [n*m*rank, n*m*rank + m*p*rank])
                   U = U_flat.reshape(n*m, rank)
                   V = V_flat.reshape(m*p, rank)
                   W = W_flat.reshape(p*n, rank)
                   best_result = {'U': U, 'V': V, 'W': W}
                   break # Found a solution for this rank
       
       if best_result is None:
           # Fall back to Rank 17
           # ... (existing code) ...
           pass
           
       return {
           "U": best_result['U'].tolist(),
           "V": best_result['V'].tolist(),
           "W": best_result['W'].tolist()
       }
   
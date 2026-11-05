import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
      n, m, p = DIMENSIONS
      T = np.zeros((n*m, m*p, p*n))
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  T[i*m+j, j*p+k, k*n+i] = 1
      
      best_rank = n * m * p
      best_U = None
      best_V = None
      best_W = None
      
      # Try to find a solution with rank < best_rank
      # Start with 38
      for R in range(38, best_rank + 1):
          # Time check
          if time.time() > 100: break
          
          # Try a few random initializations
          for seed in range(5):
              np.random.seed(seed)
              U = np.random.randn(n*m, R)
              V = np.random.randn(m*p, R)
              W = np.random.randn(p*n, R)
              
              # Gradient descent
              # We want to minimize || T - T_UVW ||^2
              # T_UVW[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
              
              # Compute gradient
              # This is expensive.
              
              # Instead, use a library?
              # scipy.optimize.minimize with a flattened parameter vector?
              # Size: 9*38 + 18*38 + 18*38 = 342 + 684 + 684 = 1710 parameters.
              # This is manageable.
              
              # Define objective function
              def obj(params):
                  U_flat, V_flat, W_flat = params[:9*R], params[9*R:9*R+18*R], params[9*R+18*R:]
                  U_mat = U_flat.reshape(9, R)
                  V_mat = V_flat.reshape(18, R)
                  W_mat = W_flat.reshape(18, R)
                  
                  # Compute tensor
                  # T_calc = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
                  # This is 9*18*18 = 2916 operations.
                  # Very fast.
                  
                  T_calc = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
                  diff = T - T_calc
                  return np.sum(diff*diff)
              
              # Initial guess
              params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
              
              # Optimize
              # Use L-BFGS-B or Nelder-Mead?
              # L-BFGS-B is better for large problems.
              
              from scipy.optimize import minimize
              
              res = minimize(obj, params0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
              
              if res.success and res.fun < 1e-6:
                  # Found a solution!
                  # Round to 0.5
                  params_opt = res.x
                  U_opt = params_opt[:9*R].reshape(9, R)
                  V_opt = params_opt[9*R:9*R+18*R].reshape(18, R)
                  W_opt = params_opt[9*R+18*R:].reshape(18, R)
                  
                  U_rounded = np.round(U_opt * 2) / 2
                  V_rounded = np.round(V_opt * 2) / 2
                  W_rounded = np.round(W_opt * 2) / 2
                  
                  # Verify
                  T_check = np.einsum('ur,vr,wr->uvw', U_rounded, V_rounded, W_rounded)
                  if np.allclose(T, T_check, atol=1e-9):
                      # Success
                      # Convert to lists
                      return {
                          "U": U_rounded.tolist(),
                          "V": V_rounded.tolist(),
                          "W": W_rounded.tolist()
                      }
      
      # Fallback to rank 54
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

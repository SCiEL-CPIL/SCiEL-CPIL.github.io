import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

def construct():
       n, m, p = DIMENSIONS
       deadline = time.time() + 140.0
       
       # Schoolbook
       rank_schoolbook = n * m * p
       U_school = np.zeros((n * m, rank_schoolbook))
       V_school = np.zeros((m * p, rank_schoolbook))
       W_school = np.zeros((p * n, rank_schoolbook))
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U_school[i * m + j, col] = 1
                   V_school[j * p + k, col] = 1
                   W_school[k * n + i, col] = 1
                   col += 1
       
       best_rank = rank_schoolbook
       best_U = U_school
       best_V = V_school
       best_W = W_school
       
       # Target ranks to try
       # We know 105 works. Let's try 104, 103, ...
       # Optimization is expensive, so we try a few.
       
       def compute_error(params, n, m, p, r):
           # params shape: (r * (n*m + m*p + p*n),)
           size_U = n * m * r
           size_V = m * p * r
           size_W = p * n * r
           
           U_flat = params[:size_U]
           V_flat = params[size_U:size_U + size_V]
           W_flat = params[size_U + size_V:size_U + size_V + size_W]
           
           U = U_flat.reshape(n * m, r)
           V = V_flat.reshape(m * p, r)
           W = W_flat.reshape(p * n, r)
           
           error = 0.0
           # We can vectorize this partially
           # T_decomp[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
           # Loop over i, j, k is 105 iterations. Inside, sum over r.
           # 105 * r operations.
           # For r=100, 10500 ops. Very fast.
           
           # Let's use numpy broadcasting if possible, but loops are fine for 105
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       u_val = U[i * m + j]
                       v_val = V[j * p + k]
                       w_val = W[k * n + i]
                       s = np.dot(u_val, v_val * w_val) # dot product of U row and (V row * W row)
                       error += (s - 1.0) ** 2
                       
           return error

       def try_rank(r, seed):
           np.random.seed(seed)
           # Initialize
           # Scale: random normal
           init_U = np.random.randn(n * m, r) * 0.5
           init_V = np.random.randn(m * p, r) * 0.5
           init_W = np.random.randn(p * n, r) * 0.5
           
           params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
           
           # Use Nelder-Mead or Powell? L-BFGS-B needs gradient.
           # We can compute gradient numerically or use Powell.
           # Powell is derivative-free and robust.
           # However, dimension is 71*r. For r=100, 7100 dimensions.
           # Powell might be slow for high dimensions.
           
           # Alternative: Use a simpler local search.
           # Or use the fact that we can fix one matrix to reduce dimensionality?
           # If we fix W to be random, we solve for U, V?
           # No, it's still multilinear.
           
           # Let's try a few steps of gradient descent with finite differences?
           # Too slow.
           
           # Let's try `scipy.optimize.minimize` with method='Powell' but limit iterations.
           # Or just check if a random init is close? No.
           
           # Better: Use a structured initialization.
           # Start with schoolbook and perturb?
           
           # Actually, for small improvements (105 -> 104),
           # we can try to find a linear dependency in the schoolbook basis?
           # The schoolbook basis is linearly independent.
           
           # Let's try a different approach:
           # We can use the "Strassen" partitioning explicitly.
           # Partition indices.
           # But implementing general Strassen for 3x5x7 is complex.
           
           # Let's stick to random search with a very efficient error function.
           # We will try to minimize the error.
           # If error < 1e-8, we accept.
           
           # To speed up, we can use a coarse grid search or random walks?
           # No, optimization is better.
           
           # Let's use `scipy.optimize.minimize` with 'Nelder-Mead' for lower dim?
           # No, dim is high.
           
           # How about we fix U and V to be sparse?
           
           # Given the time, if we can't find a better rank quickly, we return 105.
           # But 105 is the seed. We need to beat it.
           
           # Is there a known result for 3x5x7?
           # Lower bound 35. Upper bound 105.
           # Likely much lower.
           
           # Let's try to find a rank 100 solution.
           # We will run a few random restarts with `minimize` method='Powell'.
           # To avoid timeout, we limit `maxiter` and `maxfev`.
           
           options = {'maxiter': 50, 'maxfev': 200, 'disp': False}
           res = minimize(compute_error, params, method='Powell', options=options, args=(n, m, p, r))
           
           if res.fun < 1e-8:
               # Extract U, V, W
               size_U = n * m * r
               size_V = m * p * r
               size_W = p * n * r
               U = res.x[:size_U].reshape(n * m, r)
               V = res.x[size_U:size_U + size_V].reshape(m * p, r)
               W = res.x[size_U + size_V:size_U + size_V + size_W].reshape(p * n, r)
               return U, V, W
           return None

       # Try ranks 104, 103, 102, 101, 100
       for r in range(rank_schoolbook - 1, rank_schoolbook - 21, -1):
           if time.time() > deadline - 20:
               break
           
           found = False
           for seed in range(5):
               if time.time() > deadline - 10:
                   break
               res = try_rank(r, seed)
               if res is not None:
                   U, V, W = res
                   best_rank = r
                   best_U = U
                   best_V = V
                   best_W = W
                   found = True
                   break
           
           if found:
               break

       # Convert to lists of lists with integer/half-integer entries?
       # The problem says "integer/half-integer entries".
       # Our optimization gives floats.
       # We need to round or find exact rational solutions.
       # If the error is very small, the values might be close to rationals.
       # We can try to round to nearest 0.5 or 1.
       
       def round_to_half(x):
           return np.round(x * 2) / 2

       if best_rank < rank_schoolbook:
           # Try to clean up the solution
           U_clean = round_to_half(best_U)
           V_clean = round_to_half(best_V)
           W_clean = round_to_half(best_W)
           
           # Verify the cleaned solution
           # Re-calculate error
           err = compute_error(np.concatenate([U_clean.flatten(), V_clean.flatten(), W_clean.flatten()]), n, m, p, best_rank)
           if err > 1e-6:
               # If rounding broke it, maybe try rounding to integer?
               U_clean = np.round(best_U)
               V_clean = np.round(best_V)
               W_clean = np.round(best_W)
               err = compute_error(np.concatenate([U_clean.flatten(), V_clean.flatten(), W_clean.flatten()]), n, m, p, best_rank)
               
               if err > 1e-6:
                   # If still bad, fallback to schoolbook
                   return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
           
           return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
       else:
           return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

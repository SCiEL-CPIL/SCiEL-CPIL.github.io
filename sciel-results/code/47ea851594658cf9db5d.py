import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
       n, m, p = DIMENSIONS
       # Target ranks to try
       # The prompt hints at 20. Let's try 20 to 25.
       # If we don't find one, we return the best valid one or fallback.
       
       best_rank = n * m * p
       best_U = None
       best_V = None
       best_W = None
       
       start_time = time.time()
       max_time = 140.0 # Safety margin
       
       for r in range(20, 30): # Try ranks 20 to 29
           if time.time() - start_time > max_time:
               break
               
           found = False
           # Try a few random restarts for each rank
           for attempt in range(10):
               if time.time() - start_time > max_time:
                   break
                   
               # Initialize U, V, W
               # Shapes: U (n*m, r), V (m*p, r), W (p*n, r)
               U = np.random.randn(n*m, r)
               V = np.random.randn(m*p, r)
               W = np.random.randn(p*n, r)
               
               # ALS iterations
               for it in range(200):
                   if time.time() - start_time > max_time:
                       break
                       
                   # Solve for W
                   # For each (k, i), we have equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                   # This is not linear in W directly for all i,j,k simultaneously in a simple way?
                   # Wait, the equation is:
                   # sum_{j} sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                   # Let A_{ik, c} = sum_j U[i*m+j, c] * V[j*p+k, c]
                   # Then sum_c A_{ik, c} * W[k*n+i, c] = 1
                   # This is a linear system for W.
                   # W has p*n rows. Each row (k,i) is independent?
                   # Yes, the equation for (i,k) only involves W[k*n+i, :].
                   # So we can solve for each row of W independently.
                   
                   for k in range(p):
                       for i in range(n):
                           # Construct A for this (k,i)
                           # A is a vector of length r
                           # A_c = sum_j U[i*m+j, c] * V[j*p+k, c]
                           # U rows: i*m to i*m+m-1
                           # V rows: 0*p+k, 1*p+k, ..., (m-1)*p+k
                           # We need to sum over j
                           # U_slice = U[i*m : (i+1)*m, :]  # shape (m, r)
                           # V_slice = V[0*p+k : m*p+k : p, :] # shape (m, r)
                           # A = U_slice.T @ V_slice  # shape (r, r)? No.
                           # We want A_c = sum_j U_{ij,c} V_{jk,c}
                           # This is element-wise product summed over j.
                           # A = np.sum(U_slice * V_slice, axis=0) # shape (r,)
                           
                           U_slice = U[i*m : (i+1)*m, :]
                           V_slice = V[k : m*p : p, :] # Rows k, k+p, ..., k+(m-1)p
                           
                           # Check V_slice shape
                           # V has m*p rows. Indices for j are j*p+k.
                           # j=0 -> k
                           # j=1 -> p+k
                           # ...
                           # j=m-1 -> (m-1)p+k
                           # So V_slice should be V[range(k, m*p, p), :]
                           V_slice = V[np.arange(k, m*p, p), :]
                           
                           A_vec = np.sum(U_slice * V_slice, axis=0)
                           
                           # Solve A_vec @ W[k*n+i, :] = 1
                           # This is a linear equation: dot(A_vec, w_row) = 1
                           # If A_vec is zero, no solution or infinite solutions.
                           # If A_vec is not zero, we can find the least squares solution.
                           # w_row = A_vec / (A_vec @ A_vec)
                           norm_sq = np.dot(A_vec, A_vec)
                           if norm_sq < 1e-9:
                               # Degenerate case, keep old W or set to 0
                               pass
                           else:
                               W[k*n + i, :] = A_vec / norm_sq
                   
                   # Similarly for U and V?
                   # The equations are symmetric in a way, but the indices are coupled.
                   # For U:
                   # sum_{k} sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                   # Let B_{ij, c} = sum_k V[j*p+k, c] * W[k*n+i, c]
                   # Then sum_c U[i*m+j, c] * B_{ij, c} = 1
                   # U row (i,j) is independent?
                   # Yes, for fixed i,j, the equation only involves U[i*m+j, :].
                   
                   for i in range(n):
                       for j in range(m):
                           # B_vec_c = sum_k V[j*p+k, c] * W[k*n+i, c]
                           # V_slice: rows j*p+k for k in 0..p-1
                           # V_slice = V[j*p : (j+1)*p, :] # shape (p, r)
                           # W_slice: rows k*n+i for k in 0..p-1
                           # W_slice = W[np.arange(i, p*n, n), :] # shape (p, r)
                           
                           V_slice = V[j*p : (j+1)*p, :]
                           W_slice = W[np.arange(i, p*n, n), :]
                           
                           B_vec = np.sum(V_slice * W_slice, axis=0)
                           norm_sq = np.dot(B_vec, B_vec)
                           if norm_sq > 1e-9:
                               U[i*m + j, :] = B_vec / norm_sq
                   
                   # For V:
                   # sum_{i} sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                   # Let C_{jk, c} = sum_i U[i*m+j, c] * W[k*n+i, c]
                   # Then sum_c V[j*p+k, c] * C_{jk, c} = 1
                   
                   for j in range(m):
                       for k in range(p):
                           # C_vec_c = sum_i U[i*m+j, c] * W[k*n+i, c]
                           # U_slice: rows i*m+j for i in 0..n-1
                           # U_slice = U[np.arange(j, n*m, m), :] # shape (n, r)
                           # W_slice: rows k*n+i for i in 0..n-1
                           # W_slice = W[k*n : (k+1)*n, :] # shape (n, r)
                           
                           U_slice = U[np.arange(j, n*m, m), :]
                           W_slice = W[k*n : (k+1)*n, :]
                           
                           C_vec = np.sum(U_slice * W_slice, axis=0)
                           norm_sq = np.dot(C_vec, C_vec)
                           if norm_sq > 1e-9:
                               V[j*p + k, :] = C_vec / norm_sq
                   
               # After ALS, check for exact integer/half-integer solution
               # Round to nearest half-integer
               U_rounded = np.round(U * 2) / 2
               V_rounded = np.round(V * 2) / 2
               W_rounded = np.round(W * 2) / 2
               
               # Verify
               valid = True
               for i in range(n):
                   for j in range(m):
                       for k in range(p):
                           # Calculate sum
                           # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                           val = np.sum(U_rounded[i*m+j] * V_rounded[j*p+k] * W_rounded[k*n+i])
                           if abs(val - 1.0) > 1e-9:
                               valid = False
                               break
                       if not valid: break
                   if not valid: break
               
               if valid:
                   # Found a solution of rank r
                   # Convert to lists
                   U_list = U_rounded.tolist()
                   V_list = V_rounded.tolist()
                   W_list = W_rounded.tolist()
                   return {"U": U_list, "V": V_list, "W": W_list}
               
       # If no solution found, return schoolbook
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

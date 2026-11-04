import numpy as np
import time

def construct():
       n, m, p = 3, 3, 4
       T = np.zeros((n*m, m*p, p*n))
       valid_indices = []
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   a = i*m + j
                   b = j*p + k
                   g = k*n + i
                   T[a, b, g] = 1.0
                   valid_indices.append((a, b, g))
                   
       start_time = time.time()
       timeout = 110.0
       
       # Try ranks
       for r in range(32, 37):
           if time.time() - start_time > timeout:
               break
           
           max_attempts = 2000 if r >= 35 else 500
           
           for attempt in range(max_attempts):
               if time.time() - start_time > timeout:
                   break
                   
               # Random U, V
               U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.1, 0.8, 0.1])
               V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.1, 0.8, 0.1])
               
               # Solve for W
               # We need to solve for each row of W? No, W is a matrix.
               # The equation is sum_c U[a,c] V[b,c] W[g,c] = T[a,b,g]
               # This is a linear system for W[:, c] for each c?
               # No, it's coupled.
               
               # Actually, for each c, the term is U[:,c] V[:,c] W[:,c].
               # This is not linear in W if U, V are fixed?
               # Wait, the equation is:
               # sum_c (U[:,c] * V[:,c] * W[:,c]) = T
               # This is linear in W if U, V are fixed.
               # Let A be a matrix where A[row, col] = U[a, c] V[b, c]
               # But the "col" index is c.
               # So for each (a,b,g), we have:
               # sum_c U[a,c] V[b,c] W[g,c] = T[a,b,g]
               
               # This is a system of equations for W.
               # But W has p*n * r variables.
               # And we have n*m*m*p*p*n equations? No.
               # We have n*m*m*p equations? No.
               # The tensor has n*m * m*p * p*n entries.
               
               # For a fixed g, the equation is:
               # sum_c U[a,c] V[b,c] W[g,c] = T[a,b,g]
               # This must hold for all a, b.
               # So for each g, we have a system:
               # sum_c (U[:,c] * V[:,c])_element-wise W[g,c] = T[:, :, g]
               
               # Let M_g be a matrix of size (n*m * m*p) x r.
               # M_g[(a,b), c] = U[a,c] V[b,c]
               # b_g = T[:, :, g].flatten()
               # Solve M_g W[g,:] = b_g
               
               W = np.zeros((p*n, r))
               valid = True
               for g in range(p*n):
                   # Build M_g
                   # M_g has (n*m * m*p) rows and r columns
                   # This is large: 9 * 12 = 108 rows, r columns.
                   M_g = np.zeros((n*m * m*p, r))
                   b_g = np.zeros(n*m * m*p)
                   
                   idx = 0
                   for a in range(n*m):
                       for b in range(m*p):
                           M_g[idx, :] = U[a, :] * V[b, :]
                           b_g[idx] = T[a, b, g]
                           idx += 1
                   
                   # Solve
                   try:
                       w_g, residuals, rank, sv = np.linalg.lstsq(M_g, b_g, rcond=None)
                   except:
                       valid = False
                       break
                       
                   if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-9:
                       valid = False
                       break
                       
                   W[g, :] = w_g
                   
               if not valid:
                   continue
                   
               # Snap
               U_s = np.round(U * 2) / 2
               V_s = np.round(V * 2) / 2
               W_s = np.round(W * 2) / 2
               
               # Verify
               T_approx = np.einsum('ac,bc,gc->abg', U_s, V_s, W_s)
               if np.allclose(T_approx, T, atol=1e-9):
                   return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                   
       # Fallback
       rank = 36
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U[i*m+j][col] = 1
                   V[j*p+k][col] = 1
                   W[k*n+i][col] = 1
                   col += 1
       return {"U": U, "V": V, "W": W}

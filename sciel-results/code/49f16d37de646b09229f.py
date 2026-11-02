   def _try_reduce_rank(base_terms, target_rank):
       # base_terms: list of (u, v, w) dicts
       # target_rank: int
       
       n, m, p = 2, 4, 5
       size_U, size_V, size_W = n*m, m*p, p*n
       num_eq = n*m*p
       
       # Build matrix for base terms
       # M[col, eq] = u[eq] * v[eq] * w[eq] ? No.
       # T[eq] = sum_r U[eq, r] V[eq, r] W[eq, r]
       
       # We want to find a subset of terms and coefficients such that sum = T.
       # If we fix the subset of terms to be the first K terms of the base decomposition,
       # and allow scaling, we solve for scales.
       
       # But the base terms are not necessarily the "best" basis.
       # However, if a lower rank decomposition exists, it might be close to the base one.
       
       # Strategy:
       # 1. Try to remove 1 term from base 36.
       # 2. Solve for scales of remaining 35.
       
       from itertools import combinations
       # This is too slow for large K.
       
       # Better:
       # Try removing term i.
       # Solve M_{-i} alpha = b.
       
       import numpy as np
       
       # Precompute M for all 36 terms
       # M is (36, 40)
       # M[r, eq] = U[eq, r] * V[eq, r] * W[eq, r]
       
       U_base, V_base, W_base = _build_matrices(base_terms)
       M = np.zeros((36, 40))
       for r in range(36):
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       eq = i*m*p + j*p + k
                       M[r, eq] = U_base[i*m+j][r] * V_base[j*p+k][r] * W_base[k*n+i][r]
       
       b = np.ones(40)
       
       for i in range(36):
           # Remove term i
           mask = [1]*36
           mask[i] = 0
           M_sub = M[mask, :]
           
           # Solve M_sub @ alpha = b
           # alpha has length 35
           try:
               alpha, residuals, rank, sv = np.linalg.lstsq(M_sub, b, rcond=None)
               if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-9:
                   # Found a solution!
                   # Check if alpha allows half-integer scaling
                   # We need to check if U * alpha, V, W are half-integers?
                   # Or U, V * alpha, W?
                   # The terms are scaled by alpha.
                   # So U_new = U_sub * alpha (column-wise)
                   # We need U_new, V_sub, W_sub to be half-integers.
                   
                   # Try scaling U
                   U_new = U_base[mask, :] * alpha
                   if _is_half_integer(U_new) and _is_half_integer(V_base[mask, :]) and _is_half_integer(W_base[mask, :]):
                       return U_new, V_base[mask, :], W_base[mask, :]
                       
                   # Try scaling V
                   V_new = V_base[mask, :] * alpha
                   if _is_half_integer(U_base[mask, :]) and _is_half_integer(V_new) and _is_half_integer(W_base[mask, :]):
                       return U_base[mask, :], V_new, W_base[mask, :]
                       
                   # Try scaling W
                   W_new = W_base[mask, :] * alpha
                   if _is_half_integer(U_base[mask, :]) and _is_half_integer(V_base[mask, :]) and _is_half_integer(W_new):
                       return U_base[mask, :], V_base[mask, :], W_new
                       
                   # Try splitting alpha
                   # If alpha is rational, maybe we can distribute?
                   # For now, return None if simple scaling fails.
           except:
               pass
               
       return None
   
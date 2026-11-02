   def construct():
       n, m, p = 2, 2, 16
       rank = 56
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       
       # Strassen coefficients
       # s, u_coeffs (dict or list of (idx, val)), v_coeffs, w_coeffs (list of (idx, val))
       
       for b in range(8):
           for s in range(7):
               r = b * 7 + s
               # Set U
               # Set V
               # Set W
       
       return {"U": U, "V": V, "W": W}
   
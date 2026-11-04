   def construct():
       n, m, p = 2, 3, 10
       rank = 55
       
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       
       r = 0
       
       # Strassen part for j in {0, 1}
       # 5 blocks of columns
       for t in range(5):
           # Local indices for A (2x2)
           # a00, a01, a10, a11
           u_indices = [0, 1, 3, 4]
           
           # Local indices for B (2x2)
           # b00, b01, b10, b11
           v_indices = [0 * p + 2*t, 0 * p + 2*t + 1, 1 * p + 2*t, 1 * p + 2*t + 1]
           
           # Local indices for C (2x2)
           # c00, c01, c10, c11
           w_indices = [(2*t) * n + 0, (2*t) * n + 1, (2*t + 1) * n + 0, (2*t + 1) * n + 1]
           
           # Strassen coefficients
           # P1
           u_coeffs = [1, 0, 0, 1]
           v_coeffs = [1, 0, 0, 1]
           w_coeffs = [1, 0, 0, 1]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
           
           # P2
           u_coeffs = [0, 0, 1, 1]
           v_coeffs = [1, 0, 0, 0]
           w_coeffs = [0, 0, 1, -1]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
           
           # P3
           u_coeffs = [1, 0, 0, 0]
           v_coeffs = [0, 1, 0, -1]
           w_coeffs = [0, 1, 0, 1]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
           
           # P4
           u_coeffs = [0, 0, 0, 1]
           v_coeffs = [-1, 0, 1, 0]
           w_coeffs = [1, 0, 1, 0]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
           
           # P5
           u_coeffs = [1, 1, 0, 0]
           v_coeffs = [0, 0, 0, 1]
           w_coeffs = [-1, 1, 0, 0]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
           
           # P6
           u_coeffs = [-1, 0, 1, 0]
           v_coeffs = [1, 1, 0, 0]
           w_coeffs = [0, 0, 0, 1]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
           
           # P7
           u_coeffs = [0, 1, 0, -1]
           v_coeffs = [0, 0, 1, 1]
           w_coeffs = [1, 0, 0, 0]
           for ui, vi, wi, uc, vc, wc in zip(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
               if uc != 0: U[ui][r] = uc
               if vc != 0: V[vi][r] = vc
               if wc != 0: W[wi][r] = wc
           r += 1
       
       # Schoolbook part for j = 2
       for i in range(n):
           for k in range(p):
               U[i * m + 2][r] = 1
               V[2 * p + k][r] = 1
               W[k * n + i][r] = 1
               r += 1
               
       return {'U': U, 'V': V, 'W': W}
   
   def compute_norm_sq(roots):
       # Build polynomial coefficients in standard basis (ascending)
       # P(x) = prod(x - r_i)^2
       # This is equivalent to prod(x - r_i) * prod(x - r_i)
       # Or just np.poly on concatenated roots
       double_roots = np.concatenate([roots, roots])
       coeffs_desc = np.poly(double_roots)
       p = coeffs_desc[::-1] # ascending, length 25
       
       # c = A_T_inv @ p
       # A is 25x25
       c = A_T_inv @ p
       
       # Norm squared: sum c_k^2 * (k+1)
       norm_sq = np.sum(c**2 * (np.arange(25) + 1))
       
       if norm_sq <= 1e-10:
           return 0.0
       return norm_sq
   
   if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
       U_base, V_base, W_base = _construct_naive(n, m, p)
   
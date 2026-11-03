   def schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
       for i in range(i_start, i_end):
           for j in range(j_start, j_end):
               for k in range(k_start, k_end):
                   u = make_vec(n*m, [(i*m + j, 1)])
                   v = make_vec(m*p, [(j*p + k, 1)])
                   w = make_vec(p*n, [(k*n + i, 1)])
                   add_term(u, v, w)
   
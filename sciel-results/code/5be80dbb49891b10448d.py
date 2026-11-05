   def strassen_4x4x4_terms():
       terms = []
       # Macro steps
       # X_m, Y_m are 2x2 matrices of (block_i, block_j, coeff)
       # Block indices: 00, 01, 10, 11
       
       # M1: (A00+A11)(B00+B11)
       # X1 = [[1, 0], [0, 1]] (coeffs for A00, A01, A10, A11? No, 2x2 blocks)
       # A is 4x4. Blocks are 2x2.
       # A00 is A[0:2, 0:2].
       
       # Let's define the macro steps explicitly
       # m: (X_blocks, Y_blocks)
       # X_blocks is a dict {(bi, bj): coeff}
       
       macro_steps = [
           # M1
           ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
           # M2
           ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
           # M3
           ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
           # M4
           ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
           # M5
           ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
           # M6
           ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
           # M7
           ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
       ]
       
       # C reconstruction
       # C00 = M1 + M4 - M5 + M7
       # C01 = M3 + M5
       # C10 = M2 + M4
       # C11 = M1 - M2 + M3 + M6
       
       c_recon = [
           {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, # M1 -> C00
           {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}, # M2 -> C10
           {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}, # M3 -> C01
           {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}, # M4 -> C00, C10
           {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}, # M5 -> C00, C01
           {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}, # M6 -> C11
           {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}  # M7 -> C00
       ]
       # Wait, C11 also gets M1, M2, M3.
       # Let's list all contributions
       contribs = [
           # M1
           {(0,0): 1, (1,1): 1},
           # M2
           {(1,0): 1, (1,1): -1},
           # M3
           {(0,1): 1, (1,1): 1},
           # M4
           {(0,0): 1, (1,0): 1},
           # M5
           {(0,0): -1, (0,1): 1},
           # M6
           {(1,1): 1},
           # M7
           {(0,0): 1}
       ]
       
       # For each macro step m, decompose into 7 micro steps
       for m_idx, (x_blocks, y_blocks) in enumerate(macro_steps):
           # Build X, Y matrices (2x2) of coefficients for A, B elements
           # X is 2x2, each entry is a dict {(i,j): coeff}
           X = [[{} for _ in range(2)] for _ in range(2)]
           Y = [[{} for _ in range(2)] for _ in range(2)]
           
           for (bi, bj), coeff in x_blocks.items():
               # Block bi,bj corresponds to A[2*bi:2*bi+2, 2*bj:2*bj+2]
               for li in range(2):
                   for lj in range(2):
                       i = 2*bi + li
                       j = 2*bj + lj
                       X[li][lj][ (i,j) ] = coeff
           
           for (bj, bk), coeff in y_blocks.items():
               # Block bj,bk corresponds to B[2*bj:2*bj+2, 2*bk:2*bk+2]
               for lj in range(2):
                   for lk in range(2):
                       j = 2*bj + lj
                       k = 2*bk + lk
                       Y[lj][lk][ (j,k) ] = coeff
           
           # Now perform Strassen on X, Y
           # S1 = (X00+X11)(Y00+Y11)
           # S2 = (X10+X11)Y00
           # S3 = X00(Y01-Y11)
           # S4 = X11(Y10-Y00)
           # S5 = (X00+X01)Y11
           # S6 = (X10-X00)(Y00+Y01)
           # S7 = (X01-X11)(Y10+Y11)
           
           # Z00 = S1 + S4 - S5 + S7
           # Z01 = S3 + S5
           # Z10 = S2 + S4
           # Z11 = S1 - S2 + S3 + S6
           
           # We need to generate the 7 scalar multiplications
           # Each S_s is a product of a linear combo of X entries and Y entries.
           # X entries are dicts.
           
           def add_dict(d1, d2):
               res = dict(d1)
               for k, v in d2.items():
                   res[k] = res.get(k, 0) + v
               return res
           
           def sub_dict(d1, d2):
               res = dict(d1)
               for k, v in d2.items():
                   res[k] = res.get(k, 0) - v
               return res
           
           def scale_dict(d, s):
               return {k: v*s for k, v in d.items()}
           
           # S1
           x_s1 = add_dict(X[0][0], X[1][1])
           y_s1 = add_dict(Y[0][0], Y[1][1])
           # S2
           x_s2 = add_dict(X[1][0], X[1][1])
           y_s2 = Y[0][0]
           # S3
           x_s3 = X[0][0]
           y_s3 = sub_dict(Y[0][1], Y[1][1])
           # S4
           x_s4 = X[1][1]
           y_s4 = sub_dict(Y[1][0], Y[0][0])
           # S5
           x_s5 = add_dict(X[0][0], X[0][1])
           y_s5 = Y[1][1]
           # S6
           x_s6 = sub_dict(X[1][0], X[0][0])
           y_s6 = add_dict(Y[0][0], Y[0][1])
           # S7
           x_s7 = sub_dict(X[0][1], X[1][1])
           y_s7 = add_dict(Y[1][0], Y[1][1])
           
           s_x = [x_s1, x_s2, x_s3, x_s4, x_s5, x_s6, x_s7]
           s_y = [y_s1, y_s2, y_s3, y_s4, y_s5, y_s6, y_s7]
           
           # Z reconstruction
           # Z00: S1(1), S4(1), S5(-1), S7(1)
           # Z01: S3(1), S5(1)
           # Z10: S2(1), S4(1)
           # Z11: S1(1), S2(-1), S3(1), S6(1)
           
           z_recon = [
               [1, 0, 0, 1, -1, 0, 1], # Z00
               [0, 0, 1, 0, 1, 0, 0],  # Z01
               [0, 1, 0, 1, 0, 0, 0],  # Z10
               [1, -1, 1, 0, 0, 1, 0]  # Z11
           ]
           
           for s_idx in range(7):
               # Expand product of x_s[s_idx] and y_s[s_idx]
               for (i,j), cx in s_x[s_idx].items():
                   for (jj,k), cy in y_s[s_idx].items():
                       if j == jj:
                           # Term A[i,j] B[j,k]
                           for zi in range(2):
                               for zk in range(2):
                                   coeff_z = z_recon[zi][zk][s_idx]
                                   if coeff_z != 0:
                                       # Global C index
                                       # Z is a 2x2 block of C
                                       # Which block?
                                       # The macro step m contributes to C blocks via contribs[m_idx]
                                       # Wait, Z is the result of X*Y.
                                       # Z is a 2x2 matrix.
                                       # The macro step m produces Z_m.
                                       # Z_m is added to C blocks.
                                       # The mapping from Z_m entries to C blocks is in contribs.
                                       # But Z_m is a 2x2 matrix.
                                       # Its entries are Z_m[zi, zk].
                                       # We need to know which C block Z_m belongs to?
                                       # No, Z_m IS the block product result.
                                       # The macro step m computes a 2x2 matrix Z_m.
                                       # This Z_m is then combined with others to form C.
                                       # The combination is defined by c_recon?
                                       # No, the c_recon I defined earlier was for the 7 macro steps.
                                       # Here, for each macro step m, we compute Z_m.
                                       # Then we add Z_m to the appropriate C blocks.
                                       # How?
                                       # C is 4x4.
                                       # Z_m is 2x2.
                                       # The macro step m corresponds to a specific 2x2 block of C?
                                       # No, Strassen mixes the blocks.
                                       # $C_{00} = M_1 + M_4 - M_5 + M_7$.
                                       # So $M_1$ (which is $Z_1$) contributes to $C_{00}$ with coeff 1.
                                       # $M_1$ is a 2x2 matrix.
                                       # So $Z_1[0,0]$ contributes to $C[0,0]$.
                                       # $Z_1[0,1]$ contributes to $C[0,1]$.
                                       # etc.
                                       # So we need to map $Z_m[zi, zk]$ to $C[2*bi+zi, 2*bk+zk]$.
                                       # Which bi, bk?
                                       # The macro step m does NOT correspond to a single block.
                                       # It corresponds to the *entire* C matrix?
                                       # No, $M_1$ is a 2x2 matrix.
                                       # $C$ is 4x4.
                                       # $C_{00}$ is a 2x2 matrix.
                                       # So $M_1$ contributes to $C_{00}$.
                                       # $M_2$ contributes to $C_{10}$ and $C_{11}$.
                                       # So we need to know which C block each macro step contributes to.
                                       # This is given by the `contribs` list I defined earlier.
                                       # `contribs[m_idx]` is a dict {(bi, bk): coeff}.
                                       # So for each (bi, bk) in contribs[m_idx], we add Z_m[zi, zk] to C[2*bi+zi, 2*bk+zk] with coeff contribs[m_idx][(bi, bk)].
                                       pass
   
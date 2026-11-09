        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # This term contributes to C'_{uv}
                    # The local output index is v*2 + u?
                    # No, schoolbook: C_{uv} += A_{uw} B_{wv}
                    # So for a fixed (u,w,v), it adds to C_{uv}.
                    # The local W index corresponding to C_{u,v} is v*2 + u?
                    # No, the W matrix in the tensor decomposition maps to the output.
                    # In the schoolbook tensor, for each term (i,k,j),
                    # U has 1 at i*2+k, V has 1 at k*2+j, W has 1 at j*2+i.
                    # So for term (u,w,v),
                    # U_local = u*2 + w
                    # V_local = w*2 + v
                    # W_local = v*2 + u
                    
                    col = mk * 8 + (u * 4 + w * 2 + v)
                    
                    # Fill U
                    # U_local = u*2 + w
                    # This corresponds to A'_{u,w}
                    # A'_{u,w} is a linear combination of global A elements.
                    for (ga_idx, coeff) in L_A[(u, w)]:
                        U[ga_idx, col] += coeff
                    
                    # Fill V
                    # V_local = w*2 + v
                    # This corresponds to B'_{w,v}
                    for (gb_idx, coeff) in R_B[(w, v)]:
                        V[gb_idx, col] += coeff
                    
                    # Fill W
                    # W_local = v*2 + u
                    # This corresponds to C'_{u,v}
                    # We need to map C'_{u,v} to the global C.
                    # C' is the result of Mk.
                    # Mk contributes to the global C blocks.
                    
                    # We need to know which global C block C'_{u,v} belongs to and with what coefficient.
                    # This depends on the output combination formula.
                    
                    # Let's store the contributions to W in a temporary structure or handle them later.
                    # Actually, we can just accumulate them in a list of (mk, u, v, col)
                    # and then apply the output combination.
                    
                    # But W is a matrix. We can just add to it directly if we know the mapping.
                    # The mapping from Mk to global C is:
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # So for a given mk, u, v, the contribution to global C is:
                    # sum over global blocks bi of coeff * C_{bi, u, v}
                    
                    # Let's define the output coefficients for each mk
                    # out_coeff[mk][bi]
                    
                    if mk == 0: # M1
                        coeffs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}
                    elif mk == 1: # M2
                        coeffs = {0: 0.0, 1: 0.0, 2: 1.0, 3: -1.0}
                    elif mk == 2: # M3
                        coeffs = {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0}
                    elif mk == 3: # M4
                        coeffs = {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0}
                    elif mk == 4: # M5
                        coeffs = {0: -1.0, 1: 1.0, 2: 0.0, 3: 0.0}
                    elif mk == 5: # M6
                        coeffs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}
                    elif mk == 6: # M7
                        coeffs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
                    
                    for bi, coeff in coeffs.items():
                        if coeff != 0:
                            # Global C block bi, local u, v
                            # Global i = (bi//2)*2 + u
                            # Global k = (bi%2)*2 + v
                            gi = (bi // 2) * 2 + u
                            gk = (bi % 2) * 2 + v
                            w_idx = gk * n + gi
                            W[w_idx, col] += coeff

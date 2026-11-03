            A = np.zeros((nm * mp * pn, pn * R))
            b = np.zeros(nm * mp * pn)
            
            for a in range(nm):
                for b_idx in range(mp):
                    coeff_r = U_rand[a, :] * V_rand[b_idx, :]
                    if not np.any(coeff_r):
                        continue
                    row_base = (a * mp + b_idx) * pn
                    for c in range(pn):
                        row_idx = row_base + c
                        A[row_idx, c*R:(c+1)*R] = coeff_r
                        # b[row_idx] = 1 if (a, b_idx, c) in target else 0

        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # ... check if indices are inconsistent
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        
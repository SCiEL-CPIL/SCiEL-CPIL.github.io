            # Term 2: sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
            # Z[i, j] = sum_k a[k,i] b[p[k], p[j]]
            # So term2[i, j] = Z[i, j] - Z[i, i] - Z[j, j] + Z[j, i]
            diag_Z = np.diag(Z)
            term2 = Z - diag_Z[:, None] - diag_Z[None, :] + Z.T

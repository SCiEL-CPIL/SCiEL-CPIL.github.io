<<<<<<< SEARCH
                # Restore errors
                for t in affected:
                    s = 0.0
                    for cc in range(r):
                        s += U[cc, u_idx[t]] * V[cc, v_idx[t]] * W[cc, w_idx[t]]
                    errors[t] = s - 1.0
=======
                # Restore errors
                for t in affected:
                    s = np.sum(U[:, u_idx[t]] * V[:, v_idx[t]] * W[:, w_idx[t]])
                    errors[t] = s - 1.0
>>>>>>> REPLACE

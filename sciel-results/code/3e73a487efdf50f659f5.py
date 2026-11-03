<<<<<<< SEARCH
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for c in range(r):
                    # d/dc U[:,c] = 2 * residual contracted with V[:,c] and W[:,c]
                    grad_U[:, c] = 2 * np.einsum('abc,db,ec->ae', residual, V[:, c], W[:, c])
                    grad_V[:, c] = 2 * np.einsum('abc,ac,dc->be', residual, U[:, c], W[:, c])
                    grad_W[:, c] = 2 * np.einsum('abc,ab,ec->dc', residual, U[:, c], V[:, c])
=======
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for c in range(r):
                    # grad_U[:, c] = 2 * sum_{b, d} residual[a, b, d] * V[b, c] * W[d, c]
                    # residual indices: a (U dim), b (V dim), d (W dim)
                    grad_U[:, c] = 2 * np.einsum('abd,b,d->a', residual, V[:, c], W[:, c])
                    # grad_V[:, c] = 2 * sum_{a, d} residual[a, b, d] * U[a, c] * W[d, c]
                    grad_V[:, c] = 2 * np.einsum('abd,a,d->b', residual, U[:, c], W[:, c])
                    # grad_W[:, c] = 2 * sum_{a, b} residual[a, b, d] * U[a, c] * V[b, c]
                    grad_W[:, c] = 2 * np.einsum('abd,a,b->d', residual, U[:, c], V[:, c])
>>>>>>> REPLACE

I will also seed the random number generator to ensure consistent behavior across runs, which helps in debugging and potentially finding the solution if the search space is explored similarly.


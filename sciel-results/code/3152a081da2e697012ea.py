    def try_optimize(rank, n_starts=10, max_iter=1000):
        # Flatten U, V, W into a single vector x
        # x = [U_flat, V_flat, W_flat]
        # len(x) = rank * (nm + mp + pn)
        
        def objective(x):
            U = x[:rank*nm].reshape(rank, nm).T # (nm, rank)
            V = x[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T # (mp, rank)
            W = x[rank*nm+rank*mp:].reshape(rank, pn).T # (pn, rank)
            
            # Compute S[i, j, k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            # We can do this efficiently?
            # S is a 3-tensor.
            # S[i, j, k] = < U[:, i*m+j], V[:, j*p+k] * W[:, k*n+i] >
            
            S = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # U_idx = i*m + j
                        # V_idx = j*p + k
                        # W_idx = k*n + i
                        # S[i,j,k] = np.dot(U[:, i*m+j], V[:, j*p+k] * W[:, k*n+i])
                        S[i, j, k] = np.dot(U[:, i*m+j], V[:, j*p+k] * W[:, k*n+i])
            
            err = np.sum((S - 1.0)**2)
            return err
        
        # Gradient can be computed analytically or via finite differences
        # Analytical gradient:
        # d(err)/d(U[i,j,t]) = 2 * (S[i,j,k] - 1) * V[j,k,t] * W[k,i,t]  (summed over k? No, S depends on k)
        # Wait, S[i,j,k] depends on U[i,j,t].
        # So for a fixed i, j, the gradient with respect to U[i,j,t] is:
        # sum_k 2 * (S[i,j,k] - 1) * V[j,k,t] * W[k,i,t]
        
        def gradient(x):
            U = x[:rank*nm].reshape(rank, nm).T
            V = x[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T
            W = x[rank*nm+rank*mp:].reshape(rank, pn).T
            
            S = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S[i, j, k] = np.dot(U[:, i*m+j], V[:, j*p+k] * W[:, k*n+i])
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = 2 * (S[i, j, k] - 1.0)
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        # grad_U[:, u_idx] += diff * V[:, v_idx] * W[:, w_idx]
                        # grad_V[:, v_idx] += diff * U[:, u_idx] * W[:, w_idx]
                        # grad_W[:, w_idx] += diff * U[:, u_idx] * V[:, v_idx]
                        
                        grad_U[:, u_idx] += diff * (V[:, v_idx] * W[:, w_idx])
                        grad_V[:, v_idx] += diff * (U[:, u_idx] * W[:, w_idx])
                        grad_W[:, w_idx] += diff * (U[:, u_idx] * V[:, v_idx])
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        # ... optimization loop ...
    
def _find_decomp(n, m, p, max_rank, time_limit=10):
    start_time = time.time()
    lb = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p
    
    if n == 3 and m == 3 and p == 3:
        target_ranks = [23, 24, 25, 26, 27]
    elif n == 3 and m == 3 and p == 2:
        target_ranks = [16, 17, 18]
    else:
        target_ranks = list(range(lb, min(max_rank, schoolbook_rank) + 1))
        
    target_ranks = [r for r in target_ranks if r <= max_rank]
    
    for rank in target_ranks:
        if time.time() - start_time > time_limit:
            break
            
        nm = n * m
        mp = m * p
        pn = p * n
        total_params = (nm + mp + pn) * rank
        
        def objective_and_grad(params):
            U = params[:nm * rank].reshape(nm, rank)
            V = params[nm * rank : nm * rank + mp * rank].reshape(mp, rank)
            W = params[nm * rank + mp * rank :].reshape(pn, rank)
            
            error = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        diff = s - 1.0
                        error += diff * diff
                        
                        grad_U[i * m + j, :] += 2 * diff * V[j * p + k, :] * W[k * n + i, :]
                        grad_V[j * p + k, :] += 2 * diff * U[i * m + j, :] * W[k * n + i, :]
                        grad_W[k * n + i, :] += 2 * diff * U[i * m + j, :] * V[j * p + k, :]
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return error, grad

        for trial in range(10):
            if time.time() - start_time > time_limit:
                break
                
            x0 = np.random.randn(total_params) * 0.5
            
            try:
                res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10})
            except Exception:
                continue
            
            if res.fun < 1e-6:
                U = res.x[:nm * rank].reshape(nm, rank)
                V = res.x[nm * rank : nm * rank + mp * rank].reshape(mp, rank)
                W = res.x[nm * rank + mp * rank :].reshape(pn, rank)
                
                U_r = _round_to_half(U)
                V_r = _round_to_half(V)
                W_r = _round_to_half(W)
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
                
    return None, None, None

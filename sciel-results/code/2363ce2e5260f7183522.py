    for r in range(25, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        best_res_for_rank = float('inf')
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart * 100 + r * 7)
            scale = 1.0
            params = np.random.randn(r * (n*m + m*p + p*n)) * scale
            
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, r)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, r)[1],
                    method='BFGS',
                    options={'gtol': 1e-8, 'maxiter': 200, 'disp': False}
                )
                
                if res.fun < best_res_for_rank:
                    best_res_for_rank = res.fun
                
                if res.fun < 1e-6:
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    is_exact, U_r, V_r, W_r = check_half_integer_solution(U, V, W)
                    if is_exact:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        
            except Exception:
                pass
        
        # If best residual is still large, this rank is likely infeasible or very hard.
        # We continue to the next rank.

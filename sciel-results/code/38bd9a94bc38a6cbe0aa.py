    def broyden(F, x0, max_iter=20):
        x = np.array(x0, dtype=float)
        fx = F(x)
        if np.linalg.norm(fx) <= 1e-8:
            return x
            
        n = len(x)
        # Finite difference Jacobian
        h = 1e-8
        J = np.zeros((n, n))
        for i in range(n):
            x_pert = x.copy()
            x_pert[i] += h
            fx_pert = F(x_pert)
            J[:, i] = (fx_pert - fx) / h
            
        # Check if J is singular
        if abs(np.linalg.det(J)) < 1e-10:
             # Use identity or perturb
             J = np.eye(n)
             
        for _ in range(max_iter):
            try:
                dx = np.linalg.solve(J, -fx)
            except:
                dx = -fx
                
            x_new = x + dx
            fx_new = F(x_new)
            
            if np.linalg.norm(fx_new) <= 1e-8:
                return x_new
                
            s = x_new - x
            y = fx_new - fx
            denom = np.dot(y, y)
            if denom > 1e-15:
                J = J + np.outer(s - J @ y, y) / denom
                
            x = x_new
            fx = fx_new
            
        return x

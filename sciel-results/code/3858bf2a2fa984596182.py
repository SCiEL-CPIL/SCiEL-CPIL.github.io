<<<<<<< SEARCH
    def try_find_decomposition(target_rank, max_iter=100, seed=42):
        """Try to find a decomposition of the given rank using optimization."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U_rand = np.random.randn(n * m, target_rank) * 0.1
        V_rand = np.random.randn(m * p, target_rank) * 0.1
        W_rand = np.random.randn(p * n, target_rank) * 0.1
        
        # We want to minimize ||T_decomp - T||^2 where T[i,j,k] = 1
        # T_decomp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Use a simple gradient-free approach: coordinate descent
        # For each c, update U[:,c], V[:,c], W[:,c] to minimize the error
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            for c in range(target_rank):
                # Fix V[:,c] and W[:,c], optimize U[:,c]
                # This is a linear problem: for each (i,j), 
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed c, the contribution to (i,j,k) is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is complex, so let's use a simpler approach
                
                pass
            
            # Check if we've converged
            if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                return U_rand, V_rand, W_rand
        
        return None
=======
    def try_find_decomposition(target_rank, max_iter=200, seed=42):
        """Try to find a decomposition of the given rank using ALS optimization."""
        np.random.seed(seed)
        
        # Initialize with structured random values
        U_rand = np.random.randn(n * m, target_rank) * 0.5
        V_rand = np.random.randn(m * p, target_rank) * 0.5
        W_rand = np.random.randn(p * n, target_rank) * 0.5
        
        # Precompute index mappings for efficiency
        # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each column c, solve least squares
            for c in range(target_rank):
                # Fix V[:,c] and W[:,c]
                # For each (i,j), the residual to be explained by U[i*m+j,c] is:
                # r[i,j] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                # But this depends on k, so we need to handle it differently.
                # 
                # Actually, let's think of it as: for each (i,j), we have p equations (one for each k)
                # r[i,j,k] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                # We want U[i*m+j,c] such that: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] approximates r[i,j,k]
                # This is: U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) approx r[i,j,k]
                # Let a[i,j,k] = V[j*p+k,c] * W[k*n+i,c]
                # Then we want: U[i*m+j,c] * a[i,j,k] approx r[i,j,k] for all k
                # This is a weighted least squares: minimize sum_k (U[i*m+j,c] * a[i,j,k] - r[i,j,k])^2
                # Optimal: U[i*m+j,c] = sum_k r[i,j,k] * a[i,j,k] / sum_k a[i,j,k]^2
                
                Vc = V_rand[:, c]
                Wc = W_rand[:, c]
                
                for i in range(n):
                    for j in range(m):
                        # Compute a[i,j,k] = V[j*p+k,c] * W[k*n+i,c] for all k
                        a_vals = np.array([Vc[j * p + k] * Wc[k * n + i] for k in range(p)])
                        # Compute residual r[i,j,k] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                        # For efficiency, compute the total contribution from all c' first
                        # total[i,j,k] = sum_{c'} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                        # Then r[i,j,k] = 1 - (total[i,j,k] - U[i*m+j,c] * Vc[j*p+k] * Wc[k*n+i])
                        # But we're solving for U[i*m+j,c], so let's compute the contribution from other c'
                        
                        # Compute contribution from all other columns
                        contrib = np.zeros(p)
                        for c2 in range(target_rank):
                            if c2 == c:
                                continue
                            for k in range(p):
                                contrib[k] += U_rand[i * m + j, c2] * V_rand[j * p + k, c2] * W_rand[k * n + i, c2]
                        
                        r_vals = 1.0 - contrib
                        
                        # Solve: U[i*m+j,c] = sum_k r_vals[k] * a_vals[k] / sum_k a_vals[k]^2
                        denom = np.sum(a_vals ** 2)
                        if abs(denom) > 1e-15:
                            U_rand[i * m + j, c] = np.sum(r_vals * a_vals) / denom
                        else:
                            U_rand[i * m + j, c] = 0.0
            
            # Update V: for each column c, solve least squares
            for c in range(target_rank):
                Uc = U_rand[:, c]
                Wc = W_rand[:, c]
                
                for j in range(m):
                    for k in range(p):
                        # For each (j,k), we have n equations (one for each i)
                        # a[i] = U[i*m+j,c] * W[k*n+i,c]
                        # r[i] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                        a_vals = np.array([Uc[i * m + j] * Wc[k * n + i] for i in range(n)])
                        
                        contrib = np.zeros(n)
                        for c2 in range(target_rank):
                            if c2 == c:
                                continue
                            for i in range(n):
                                contrib[i] += U_rand[i * m + j, c2] * V_rand[j * p + k, c2] * W_rand[k * n + i, c2]
                        
                        r_vals = 1.0 - contrib
                        
                        denom = np.sum(a_vals ** 2)
                        if abs(denom) > 1e-15:
                            V_rand[j * p + k, c] = np.sum(r_vals * a_vals) / denom
                        else:
                            V_rand[j * p + k, c] = 0.0
            
            # Update W: for each column c, solve least squares
            for c in range(target_rank):
                Uc = U_rand[:, c]
                Vc = V_rand[:, c]
                
                for k in range(p):
                    for i in range(n):
                        # For each (k,i), we have m equations (one for each j)
                        # a[j] = U[i*m+j,c] * V[j*p+k,c]
                        # r[j] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                        a_vals = np.array([Uc[i * m + j] * Vc[j * p + k] for j in range(m)])
                        
                        contrib = np.zeros(m)
                        for c2 in range(target_rank):
                            if c2 == c:
                                continue
                            for j in range(m):
                                contrib[j] += U_rand[i * m + j, c2] * V_rand[j * p + k, c2] * W_rand[k * n + i, c2]
                        
                        r_vals = 1.0 - contrib
                        
                        denom = np.sum(a_vals ** 2)
                        if abs(denom) > 1e-15:
                            W_rand[k * n + i, c] = np.sum(r_vals * a_vals) / denom
                        else:
                            W_rand[k * n + i, c] = 0.0
            
            # Check convergence
            if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                return U_rand, V_rand, W_rand
        
        return None
>>>>>>> REPLACE

Now I need to also fix the main loop to try fewer restarts since each ALS iteration is expensive, and add a time check more frequently:


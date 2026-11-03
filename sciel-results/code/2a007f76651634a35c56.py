    def smooth_objective(c, beta=50.0):
        """
        Compute smooth min of constraints and its analytic gradient.
        f(c) = -1/beta * log(sum(exp(-beta * x_k)))
        grad = -sum( w_k * grad(x_k) )
        """
        if not np.all(np.isfinite(c)):
            return -1e10, np.zeros((N, 3))
        
        # 1. Compute all constraint values x_k
        abs_c = np.abs(c)
        wall_dists = 0.5 - abs_c  # shape (N, 3)
        
        d = pdist(c)
        r_pairs = d / 2.0  # shape (N*(N-1)/2,)
        
        all_x = np.concatenate([wall_dists.flatten(), r_pairs])
        
        # 2. Compute soft min value
        max_x = np.max(all_x)
        exp_terms = np.exp(-beta * (all_x - max_x))
        sum_exp = np.sum(exp_terms)
        
        if sum_exp < 1e-30:
            return -1e10, np.zeros((N, 3))
            
        log_sum = np.log(sum_exp)
        f_val = -max_x - (1.0 / beta) * log_sum
        
        # 3. Compute weights w_k
        weights = exp_terms / sum_exp
        
        # 4. Compute Gradient
        grad = np.zeros((N, 3))
        
        # Wall part
        w_wall = weights[:3*N].reshape(N, 3)
        signs = np.sign(c)
        # df/dc = sum(-w * dx/dc). dx/dc = -sign(c). So term is w * sign(c).
        grad += w_wall * signs
        
        # Pair part
        w_pairs = weights[3*N:]
        
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                w = w_pairs[idx]
                if w > 1e-12:
                    diff = c[i] - c[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        gx_i = 0.5 * diff / dist
                        grad[i] -= w * gx_i
                        grad[j] += w * gx_i
                idx += 1
                
        return f_val, grad
    
    def gradient_ascent(c, max_iters=200, lr_init=0.01, beta=30.0):
        """Gradient ascent on the smooth objective."""
        c = c.copy()
        best_r = compute_radius(c)
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            f, grad = smooth_objective(c, beta)
            
            if not np.isfinite(f):
                break
            
            c_new = c + lr * grad
            c_new = np.clip(c_new, -0.499, 0.499)
            
            r_new = compute_radius(c_new)
            if r_new > best_r + 1e-10:
                c = c_new
                best_r = r_new
                lr = min(lr * 1.05, 0.05)
            else:
                lr *= 0.8
                if lr < 1e-6:
                    break
        
        return c, best_r

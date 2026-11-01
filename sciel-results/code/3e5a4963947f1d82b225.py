    def construct():
        start_time = time.time()
        deadline = 115.0
        
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        best_x = x0
        best_e, _ = _energy_and_grad(x0)
        
        # First, polish the Fibonacci start
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success and res.fun < best_e:
            best_e = res.fun
            best_x = res.x
            
        # Use Basinhopping if time allows
        elapsed = time.time() - start_time
        if elapsed < 80.0:
            def objective(x):
                e, _ = _energy_and_grad(x)
                return e
            def grad(x):
                _, g = _energy_and_grad(x)
                return g
                
            # Basinhopping parameters
            # stepsize: how far to jump
            # minenergy: stop if found
            bh = basinhopping(objective, x0=best_x, grad=grad, 
                              minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'options': {'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}},
                              stepsize=0.5, 
                              niter=None, # We will manage this
                              T=1000.0, 
                              accept_test=None) # Default is accept if energy lower or random accept based on T? 
                              # Actually, basinhopping accepts if new energy < old energy OR random.
                              # We want to explore.
                              
            # Custom loop for basinhopping to respect deadline
            # scipy basinhopping doesn't have a time limit, so we implement a simple loop
            # Or just call it with a small niter and check time.
            
            # Let's do a manual basin hopping loop for better control
            current_x = best_x
            current_e = best_e
            T = 1000.0
            stepsize = 0.5
            
            while time.time() - start_time < deadline - 10:
                # Generate a new point
                noise = np.random.normal(0, stepsize, (N, 3))
                new_x = current_x + noise
                new_x = new_x / np.linalg.norm(new_x, axis=1, keepdims=True)
                
                # Optimize from new_x
                res = minimize(_energy_and_grad, new_x, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                new_e = res.fun
                new_opt_x = res.x
                
                # Acceptance criterion: Metropolis
                if new_e < current_e:
                    accept = True
                else:
                    delta = new_e - current_e
                    if T > 0:
                        accept = np.random.rand() < np.exp(-delta / T)
                    else:
                        accept = False
                        
                if accept:
                    current_x = new_opt_x
                    current_e = new_e
                    if current_e < best_e:
                        best_e = current_e
                        best_x = current_x
                else:
                    # Keep current_x for next step? Or revert?
                    # Usually basin hopping keeps the current state as the starting point for the next jump
                    # If rejected, we stay at current_x (which was already optimized)
                    pass
                
                # Cool down
                T *= 0.95
                
        # Final polish
        if time.time() - start_time < deadline - 2:
            res = minimize(_energy_and_grad, best_x, method='L-BFGS-B', jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.success and res.fun < best_e:
                best_e = res.fun
                best_x = res.x
                
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        return best_x
    
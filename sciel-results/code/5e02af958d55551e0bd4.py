        while calls < budget:
            # Determine how many samples we can evaluate
            num_samples = min(N, budget - calls)
            
            # Sample
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            Y = rng.standard_normal((num_samples, n))
            X = x[None, :] + sigma * (Y @ L.T)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls += num_samples
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # If we ran out of budget, we can stop updating parameters
            if calls >= budget:
                break
                
            # Update mean
            # If num_samples < mu, we might not have enough for the weighted sum?
            # mu is N//2. If num_samples < mu, we should adjust mu or just use what we have.
            # For simplicity and since N is small (7) and budget is large (500),
            # we will rarely hit the last partial generation unless 500 is not a multiple of N.
            # 500 / 7 = 71.42. So the last generation will have 500 - 71*7 = 3 samples.
            # In that case, mu=3, so we have exactly mu samples.
            # If num_samples < mu, we should set mu_eff and weights accordingly?
            # Or just use the available samples for the mean update.
            # Let's assume for the last step, we just update the mean with the available samples.
            
            if num_samples >= mu:
                x_new = np.sum(w[:, None] * X[:mu], axis=0)
            else:
                # Use all available samples with equal weight for the mean
                x_new = np.mean(X, axis=0)
                # And skip covariance update or do a simplified one?
                # Given the budget is large, this edge case is minor.
                # Let's just break if we don't have enough for a proper update.
                break

            # ... rest of update logic

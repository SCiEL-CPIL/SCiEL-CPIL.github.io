import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
        # ensemble: (16, 20)
        # observations: (10,)
        # observed_indices: array of 10 indices
        # observation_variance: scalar
        
        num_members, num_vars = ensemble.shape
        num_obs = len(observed_indices)
        
        # 1. Compute prior statistics
        mean = ensemble.mean(axis=0)
        # Covariance P
        # Using unbiased estimator (N-1)
        P = np.cov(ensemble, rowvar=False, bias=False)
        
        # 2. Observation operator H (implicit)
        # P_oo = H P H^T + R
        # H P H^T is the covariance of the observed variables
        obs_cov = P[np.ix_(observed_indices, observed_indices)]
        R = observation_variance * np.eye(num_obs)
        P_oo = obs_cov + R
        
        # 3. Compute Kalman Gain K
        # K = P H^T P_oo^-1
        # P H^T is (20, 10)
        P_hT = P[:, observed_indices]
        
        # Solve P_oo X = P_hT^T to get X = P_oo^-1 P_hT^T
        # Then K = X^T
        try:
            # Use solve for stability
            X = np.linalg.solve(P_oo, P_hT.T)
            K = X.T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            K = P_hT.T @ np.linalg.pinv(P_oo)
            
        # 4. Localization (Optional but recommended for chaos)
        # Exponential localization based on distance
        # rho = correlation length
        rho = 3.0 # Tuning parameter
        K_localized = np.zeros_like(K)
        for i in range(num_vars):
            for j in range(num_obs):
                idx_obs = observed_indices[j]
                dist = abs(i - idx_obs)
                dist = min(dist, num_vars - dist)
                weight = np.exp(-dist**2 / (2 * rho**2))
                K_localized[i, j] = K[i, j] * weight
        K = K_localized

        # 5. Update ensemble members
        # Stochastic EnKF: Perturb observations
        # Generate noise for each member
        # noise: (num_members, num_obs)
        noise = np.random.normal(0, np.sqrt(observation_variance), size=(num_members, num_obs))
        
        # Perturbed observations
        y_pert = observations[None, :] + noise
        
        # Innovation for each member: d_i = y_pert_i - H x_i
        # H x_i is ensemble[:, observed_indices]
        x_obs = ensemble[:, observed_indices]
        innovations = y_pert - x_obs
        
        # Update: x_a = x_f + K d
        # K is (20, 10), innovations is (16, 10)
        updates = innovations @ K.T
        
        result = ensemble + updates
        
        # Safety check
        if not np.all(np.isfinite(result)):
            return ensemble.copy()
            
        # Clip to bounds if necessary (task says output within [-100, 100])
        # Usually EnKF keeps values reasonable, but just in case
        result = np.clip(result, -100, 100)
        
        return result

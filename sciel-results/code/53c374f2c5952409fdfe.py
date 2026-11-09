"""4D-Var style batched trajectory estimator for Lorenz-96 assimilation."""
import time
import numpy as np

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 with periodic boundary conditions."""
    n = len(x)
    dx = np.empty(n)
    # x_{i+1} - x_{i-2} times x_{i-1} - x_i + F
    # Using periodic indices: (i+1) % n, (i-2) % n, (i-1) % n
    dx = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _propagate(x, F, dt, steps):
    """Propagate state x forward by 'steps' RK4 steps of size dt."""
    for _ in range(steps):
        x = _rk4_step(x, F, dt)
    return x

def _gauss_newton_4dvar(x_init, obs, obs_idx, obs_var, F, dt, n_steps, 
                         prior_mean, prior_cov, max_iter=30, tol=1e-10):
    """
    4D-Var via Gauss-Newton: find x_init that minimizes
      J = 0.5 * sum_t ||y_t - H(x_t)||^2 / obs_var + 0.5 * (x_init - prior_mean)^T P^{-1} (x_init - prior_mean)
    where x_t is the state at time t after t propagation steps from x_init.
    
    We use a simplified approach: optimize the initial condition x_init
    using the ensemble mean as prior mean and a diagonal prior covariance
    based on ensemble spread.
    """
    n = len(x_init)
    n_obs = len(obs_idx)
    
    # Prior: ensemble mean and spread (diagonal)
    prior_mean = prior_mean
    # Use ensemble spread as prior std, with a floor
    prior_std = np.sqrt(np.maximum(prior_cov, 1e-6))
    prior_std = np.maximum(prior_std, 0.1)  # floor to avoid singular
    prior_inv_var = 1.0 / (prior_std ** 2)
    
    obs_inv_var = 1.0 / obs_var
    
    x = x_init.copy()
    
    for iteration in range(max_iter):
        # Compute trajectory
        traj = np.zeros((n_steps + 1, n))
        traj[0] = x
        for t in range(n_steps):
            traj[t + 1] = _rk4_step(traj[t], F, dt)
        
        # Observation misfit at final time (t = n_steps)
        x_final = traj[-1]
        resid = x_final[obs_idx] - obs
        grad_obs = np.zeros(n)
        grad_obs[obs_idx] = resid * obs_inv_var
        
        # Prior gradient
        grad_prior = (x - prior_mean) * prior_inv_var
        
        grad = grad_obs + grad_prior
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Approximate Hessian: use finite differences on the gradient
        # For efficiency, use a diagonal approximation + limited finite differences
        # Actually, let's use a simple line search with gradient descent + adaptive step
        
        # Compute Hessian via finite differences (expensive but n=20)
        # Use a smaller step for FD
        h = 1e-4
        H = np.zeros((n, n))
        for i in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            
            # Compute gradients at xp and xm
            traj_p = np.zeros((n_steps + 1, n))
            traj_p[0] = xp
            for t in range(n_steps):
                traj_p[t + 1] = _rk4_step(traj_p[t], F, dt)
            resid_p = traj_p[-1][obs_idx] - obs
            grad_p = np.zeros(n)
            grad_p[obs_idx] = resid_p * obs_inv_var
            grad_p += (xp - prior_mean) * prior_inv_var
            
            traj_m = np.zeros((n_steps + 1, n))
            traj_m[0] = xm
            for t in range(n_steps):
                traj_m[t + 1] = _rk4_step(traj_m[t], F, dt)
            resid_m = traj_m[-1][obs_idx] - obs
            grad_m = np.zeros(n)
            grad_m[obs_idx] = resid_m * obs_inv_var
            grad_m += (xm - prior_mean) * prior_inv_var
            
            H[i, :] = (grad_p - grad_m) / (2.0 * h)
        
        # Ensure H is symmetric positive definite
        H = 0.5 * (H + H.T)
        
        # Add small regularization for stability
        H += 1e-8 * np.eye(n)
        
        # Solve H dx = -grad
        try:
            dx = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            dx = -grad * 0.1
        
        # Limit step size
        step_norm = np.linalg.norm(dx)
        if step_norm > 5.0:
            dx *= 5.0 / step_norm
        
        x_new = x + dx
        
        # Simple line search
        alpha = 1.0
        for _ in range(10):
            x_try = x + alpha * dx
            traj_try = np.zeros((n_steps + 1, n))
            traj_try[0] = x_try
            for t in range(n_steps):
                traj_try[t + 1] = _rk4_step(traj_try[t], F, dt)
            resid_try = traj_try[-1][obs_idx] - obs
            cost_try = 0.5 * np.sum(resid_try ** 2) * obs_inv_var + \
                       0.5 * np.sum((x_try - prior_mean) ** 2 * prior_inv_var)
            cost_cur = 0.5 * np.sum(resid ** 2) * obs_inv_var + \
                       0.5 * np.sum((x - prior_mean) ** 2 * prior_inv_var)
            if cost_try < cost_cur:
                break
            alpha *= 0.5
        
        x = x + alpha * dx
        
        # Clamp to reasonable bounds
        x = np.clip(x, -100.0, 100.0)
    
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    4D-Var style assimilation: treat the 5-step propagation as a batched problem.
    For each ensemble member, find the best initial condition that, when propagated
    5 steps, best matches the observations, regularized by the ensemble prior.
    """
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    
    ens = ensemble.copy()
    n_members, n_vars = ens.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Estimate F from the ensemble dynamics (average forcing)
    # Lorenz-96: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # We can estimate F by looking at the ensemble mean behavior
    # Actually, F is fixed per run. Let's estimate it.
    x_mean = np.mean(ens, axis=0)
    rhs_without_F = (np.roll(x_mean, -1) - np.roll(x_mean, 2)) * np.roll(x_mean, 1) - x_mean
    # F = mean of (dx/dt - rhs_without_F) but we don't have dx/dt directly
    # Use the fact that in steady state, the mean dynamics balance
    # Better: estimate F from the observations and ensemble
    # Actually, let's just use a reasonable estimate. The typical F values are 6, 8, 10.
    # We can estimate F by: F ≈ mean(x_i - (x_{i+1}-x_{i-2})*x_{i-1}) over ensemble
    F_est = np.mean(x_mean - (np.roll(x_mean, -1) - np.roll(x_mean, 2)) * np.roll(x_mean, 1))
    F_est = float(F_est)
    
    # Prior: ensemble mean and covariance (diagonal)
    prior_mean = np.mean(ens, axis=0)
    prior_var = np.var(ens, axis=0)
    
    dt = 0.01
    n_steps = 5
    
    updated = ens.copy()
    
    for m in range(n_members):
        if time.time() > deadline:
            break
        
        # Optimize initial condition for this member
        x_init = ens[m]
        x_opt = _gauss_newton_4dvar(
            x_init, observations, obs_idx, observation_variance,
            F_est, dt, n_steps, prior_mean, prior_var,
            max_iter=15, tol=1e-8
        )
        
        # Propagate the optimized initial condition forward
        x_final = _propagate(x_opt, F_est, dt, n_steps)
        
        # Also do a local nudge at the final time to better match observations
        # Blend the propagated state with observation nudging
        x_final[obs_idx] = 0.7 * x_final[obs_idx] + 0.3 * observations
        
        updated[m] = x_final
    
    # Ensure finiteness and bounds
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated

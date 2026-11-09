"""Bermudan exercise-policy: High-resolution grid-based backward induction using known dynamics."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)

    N, n_total_steps = training_paths.shape
    n_exercise = 12  # monthly exercise dates
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # Determine grid range from training data to ensure coverage, 
    # but enforce a reasonable minimum/maximum to avoid excessive computation
    # or missing tails.
    S_min_train = np.min(training_paths)
    S_max_train = np.max(training_paths)
    
    # Add a buffer to the training range
    buffer = 0.1 * (S_max_train - S_min_train) + 1.0
    S_min = max(S_min_train - buffer, 0.01)
    S_max = S_max_train + buffer
    
    # Number of grid points. 
    # 2048 points is a good balance between accuracy and speed for 12 steps.
    # If we have time, we can increase this.
    n_grid = 2048
    
    # Create grid
    S_grid = np.linspace(S_min, S_max, n_grid)
    dS = S_grid[1] - S_grid[0]
    
    # Precompute coefficients for the implicit Euler scheme (Crank-Nicolson is better but slower/stable)
    # We use an explicit scheme for speed, but with small dt it might be unstable.
    # Let's check stability condition for explicit scheme:
    # dt/dS^2 * sigma^2/2 < 1/2 => dt < dS^2 / sigma^2
    # dS = (S_max - S_min) / n_grid
    # If unstable, we must use implicit or reduce dt (increase n_grid).
    
    # Actually, for robustness and speed, let's use a simple explicit upwind scheme 
    # or just standard explicit central difference if stable.
    # Given the time limit, let's try to be smart.
    
    # Stability check
    stability_limit = dS**2 / (sigma**2)
    if dt > stability_limit:
        # Need more grid points or implicit method.
        # Let's increase n_grid to ensure stability for explicit method.
        # Required dS < sqrt(dt * sigma^2)
        required_dS = np.sqrt(dt * sigma**2) * 0.9 # safety factor
        required_n_grid = int((S_max - S_min) / required_dS) + 1
        if required_n_grid < 1000:
            n_grid = required_n_grid
        else:
            # If n_grid is too large, explicit is too slow. 
            # Let's stick to a moderate n_grid and use a simple upwind or implicit if needed.
            # For simplicity in this "different strategy" attempt, let's assume 
            # 2048-4096 is fast enough in numpy.
            n_grid = max(n_grid, 4096)
        
        S_grid = np.linspace(S_min, S_max, n_grid)
        dS = S_grid[1] - S_grid[0]
        
    # Initialize value array
    # V[i, k] is value at grid point i, exercise date k
    # We only need current and next time step
    V_next = np.zeros(n_grid)
    V_curr = np.zeros(n_grid)
    
    # Terminal condition at t=12 (index 12 in 1-based, but we iterate backwards)
    # At the last exercise date (k=12), the value is the payoff.
    # Wait, the problem says "12 monthly exercise dates". 
    # Indices in training_paths are 0..12.
    # Exercise dates are 1..12.
    # At t=12, it's maturity. Payoff = max(K-S, 0).
    
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store policy thresholds for each exercise date 1..11
    # At each date, we decide to exercise if Payoff > Continuation Value.
    # Continuation Value = Discount * E[V_next | S_curr]
    
    policy_thresholds = {}
    
    # Backward induction from k=11 down to 1
    for k in range(11, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Calculate continuation value: Discount * E[V_next(S_{t+dt}) | S_t]
        # Under risk-neutral GBM:
        # S_{t+dt} = S_t * exp((r - sigma^2/2)dt + sigma*sqrt(dt)*Z)
        # We need to compute the expectation numerically on the grid.
        # We can do this by shifting the grid or using a convolution-like operation.
        # Simpler: For each S_t, approximate the integral using the grid.
        
        # Since the transition is local, we can use a finite difference approximation 
        # of the PDE:
        # dV/dt + (r-S)dV/dS + 0.5*sigma^2*S^2*d2V/dS2 - r*V = 0
        # Using explicit Euler in time (backward in time for continuation):
        # V_curr = V_next - dt * (r*S*dV/dS + 0.5*sigma^2*S^2*d2V/dS2 - r*V_next)
        # Wait, the PDE is for the continuation value C(t,S).
        # C(t,S) = E[discounted future payoff].
        # The backward induction step is:
        # C_k(S) = discount * E[C_{k+1}(S_{k+1}) | S_k]
        
        # Let's use the PDE discretization to compute C_k from C_{k+1}.
        # C_k = C_{k+1} - dt * L(C_{k+1}) where L is the generator.
        # L(C) = (r - sigma^2/2)S * dC/dS + 0.5*sigma^2*S^2 * d2C/dS2 - r*C  (Wait, discounting is handled separately?)
        
        # Let's stick to the Monte Carlo expectation on the grid to avoid PDE stability issues.
        # We can compute the expectation by assuming S_{k+1} is lognormal around S_k.
        # But doing this for every grid point is O(N_grid * N_quad).
        
        # Alternative: Use the fact that we are on a grid. 
        # We can use a simple 3-point or 5-point stencil for the expectation if we assume
        # the transition probability mass is concentrated nearby.
        # With sigma=0.2, dt=1/12, vol_per_step = 0.2 * sqrt(1/12) ~ 0.057.
        # So the distribution is quite narrow. A local approximation works well.
        
        # Let's use a simple central difference for the PDE operator, which is equivalent
        # to a specific expectation approximation.
        
        # dV/dS at i: (V[i+1] - V[i-1]) / (2*dS)
        # d2V/dS2 at i: (V[i+1] - 2*V[i] + V[i-1]) / dS^2
        
        # Continuation Value C_k[i] = discount * ( V_next[i] + dt * ( (r - sigma^2/2)*S_i * dV/dS + 0.5*sigma^2*S_i^2 * d2V/dS2 ) )
        # Note: The drift term in the PDE for the *discounted* expectation usually includes -r*V.
        # Let's derive:
        # C(t) = e^{-r dt} E[C(t+dt)]
        # C(t) = C(t+dt) - r*dt*C(t+dt) + ... (Ito terms)
        # So C_k = C_{k+1} - dt * ( r*C_{k+1} - (r - sigma^2/2)S * C' - 0.5*sigma^2*S^2 * C'' )
        # Wait, the standard PDE for the option value V is:
        # V_t + rS V_S + 0.5 sigma^2 S^2 V_SS - rV = 0
        # So V_t = rV - rS V_S - 0.5 sigma^2 S^2 V_SS
        # Backward step: V(t) = V(t+dt) - dt * V_t(t+dt)
        # V(t) = V(t+dt) - dt * ( r V(t+dt) - rS V_S(t+dt) - 0.5 sigma^2 S^2 V_SS(t+dt) )
        
        # Let's compute derivatives on V_next
        dV_dS = np.zeros(n_grid)
        d2V_dS2 = np.zeros(n_grid)
        
        # Central differences for interior
        dV_dS[1:-1] = (V_next[2:] - V_next[:-2]) / (2 * dS)
        d2V_dS2[1:-1] = (V_next[2:] - 2 * V_next[1:-1] + V_next[:-2]) / (dS * dS)
        
        # Boundary conditions: 
        # At S=0, V=K (for put), V_S=0? 
        # At S=inf, V=0, V_S=0.
        # We'll just use one-sided differences at boundaries or keep them 0.
        dV_dS[0] = (V_next[1] - V_next[0]) / dS
        dV_dS[-1] = (V_next[-1] - V_next[-2]) / dS
        d2V_dS2[0] = (V_next[1] - 2*V_next[0] + (V_next[0])) / dS**2 # Rough
        d2V_dS2[-1] = (V_next[-1] - 2*V_next[-1] + V_next[-2]) / dS**2 # Rough
        
        # Compute the PDE operator term
        # L(V) = r*V - r*S*dV/dS - 0.5*sigma^2*S^2*d2V/dS2
        # Wait, the sign in V_t = ... 
        # V_t = - (rS V_S + 0.5 sigma^2 S^2 V_SS - rV) = rV - rS V_S - 0.5 sigma^2 S^2 V_SS
        # So V_curr = V_next - dt * ( r*V_next - r*S_grid*dV_dS - 0.5*sigma^2*S_grid**2*d2V_dS2 )
        
        pde_term = r * V_next - r * S_grid * dV_dS - 0.5 * sigma**2 * S_grid**2 * d2V_dS2
        
        V_curr = V_next - dt * pde_term
        
        # The continuation value is V_curr (which is the discounted expected future value)
        # The exercise payoff is max(K - S, 0)
        payoff = np.maximum(K - S_grid, 0.0)
        
        # Optimal value at this step
        V_optimal = np.maximum(payoff, V_curr)
        
        # Determine exercise threshold
        # Exercise if payoff >= V_curr
        # Find the largest S where payoff >= V_curr
        # Since payoff is decreasing and V_curr is increasing (for put), 
        # there is a single crossing point.
        
        exercise_mask = payoff >= V_curr
        
        # Find the threshold index
        # We want the largest index i such that exercise_mask[i] is True.
        # If all are True, threshold is S_max (always exercise).
        # If none are True, threshold is S_min (never exercise).
        
        if np.any(exercise_mask):
            # Find the last True
            # Since it's monotonic (mostly), we can just find the max index
            last_true_idx = np.max(np.where(exercise_mask)[0])
            threshold = S_grid[last_true_idx]
        else:
            threshold = 0.0
            
        policy_thresholds[k] = threshold
        
        # Update for next iteration
        V_next = V_optimal

    # Build the policy object
    class Policy:
        def __init__(self, thresholds, K_val):
            self.thresholds = thresholds
            self.K = K_val

        def exercise(self, current_state, time):
            # time is the exercise date index (1-12)
            # current_state is a vector of states
            # Return boolean vector
            
            if time >= 12:
                # At maturity, always "exercise" (take payoff)
                return np.ones(len(current_state), dtype=bool)
            
            if time not in self.thresholds:
                # Fallback: simple rule
                return np.maximum(self.K - current_state, 0.0) > 0
            
            threshold = self.thresholds[time]
            # Exercise if S <= threshold
            return current_state <= threshold

    return Policy(policy_thresholds, K)
# EVOLVE-BLOCK-END

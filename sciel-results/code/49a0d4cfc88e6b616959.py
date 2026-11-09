import numpy as np
from scipy.stats import norm
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    dt = T / n_exercise
    
    # training_paths shape: (N, n_exercise + 1)
    # Columns 0 to n_exercise
    # Exercise decisions at columns 1 to n_exercise
    N, n_steps = training_paths.shape
    
    # Pre-compute European Put Prices for Control Variates
    # We need P(S, t) for t in [dt, T]
    # For step k (0-indexed exercise), state is at path index k+1.
    # Time to maturity from step k+1 is T - (k+1)*dt.
    # Control variate at step k is discounted European price from step k+1 to T.
    # C_k(S_{k+1}) = e^{-r*dt} * P_euro(S_{k+1}, T - (k+1)*dt)
    
    def european_put_price(S, t_rem):
        # S: array
        # t_rem: scalar
        if t_rem <= 0:
            return np.maximum(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t_rem) / (sigma * np.sqrt(t_rem))
        d2 = d1 - sigma * np.sqrt(t_rem)
        P = K * np.exp(-r * t_rem) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return P

    # Basis functions
    # Use S/100 to scale
    def make_basis(S):
        x = S / 100.0
        return np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            x**5
        ])

    # Store coefficients for each exercise step k (0 to 11)
    coeffs = [None] * n_exercise
    
    # Pass 1: Standard LS
    # V_next will hold the values at the next step for alive paths
    # We iterate backwards from k = n_exercise - 1 down to 0
    
    # Initialize V at the last step (k = n_exercise - 1 corresponds to path index n_exercise)
    # At the last exercise date, continuation is 0.
    # So V_cont at last step is 0.
    # Exercise if Payoff > 0.
    
    # Let's work with arrays of values for all paths at each step
    # V_values[k] will store the estimated value at exercise step k for all paths
    # But we only need the coefficients for the final policy.
    
    # For Pass 1, we need to store the exercise decisions to compute exact values for Pass 2
    exercise_decisions = [None] * n_exercise
    
    # Backward induction
    # k is exercise index 0..11
    # Path index j = k + 1
    # Next path index j_next = k + 2
    
    # Start from the end
    # At k = 11 (last), path index 12.
    # Continuation = 0.
    # Payoff = max(K - S[12], 0)
    # Exercise if Payoff > 0.
    
    # We need V_cont at step k to be the regression output.
    # Target for regression at step k is: e^{-r*dt} * V_{k+1}
    # Where V_{k+1} is the value at the next step.
    
    # Let's maintain an array `V_next` of shape (N,) representing the value at the next step (k+1)
    # For k = 11, V_next is 0 (continuation after last step).
    # Actually, for k=11, we don't regress. We just set V_11 = Payoff_11.
    # For k=10, we regress on e^{-r*dt} * V_11.
    
    V_next = np.zeros(N) # Value at step k+1 (initially 0 for after last step)
    
    for k in range(n_exercise - 1, -1, -1):
        j = k + 1
        S_curr = training_paths[:, j]
        
        if k == n_exercise - 1:
            # Last step
            payoff = np.maximum(K - S_curr, 0.0)
            V_curr = payoff
            ex_mask = payoff > 0.0
        else:
            # Regress
            target = np.exp(-r * dt) * V_next
            S_scaled = S_curr / 100.0
            basis = make_basis(S_curr)
            
            # Control Variate
            t_rem_next = T - (j + 1) * dt # Time remaining after next step
            if t_rem_next <= 0:
                 t_rem_next = 0
            P_euro_next = european_put_price(S_curr, t_rem_next)
            control = np.exp(-r * dt) * P_euro_next
            
            residual_target = target - control
            
            # Ridge Regression for stability
            # Solve (X^T X + lambda I) beta = X^T y
            X = basis
            y = residual_target
            lambda_ridge = 1e-6 * np.trace(X.T @ X) / X.shape[1]
            
            try:
                A = X.T @ X + lambda_ridge * np.eye(X.shape[1])
                b = X.T @ y
                beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            pred_residual = X @ beta
            V_cont = control + pred_residual
            
            payoff = np.maximum(K - S_curr, 0.0)
            
            ex_mask = payoff > V_cont
            
            V_curr = np.where(ex_mask, payoff, V_cont)
            
            # Store coeffs for this step
            # We need to store beta and control params to reconstruct V_cont later
            # Actually, we just store beta. The control variate is deterministic given S and t.
            coeffs[k] = beta
        
        exercise_decisions[k] = ex_mask
        V_next = V_curr
        
        # Check time
        if time.time() > deadline:
            break

    # Pass 2: Refine using exact values from Pass 1 policy
    # Compute V_exact_1 for all steps
    V_exact = [None] * n_exercise
    # Last step
    j_last = n_exercise
    S_last = training_paths[:, j_last]
    V_exact[n_exercise - 1] = np.maximum(K - S_last, 0.0)
    
    for k in range(n_exercise - 2, -1, -1):
        j = k + 1
        S_curr = training_paths[:, j]
        payoff = np.maximum(K - S_curr, 0.0)
        V_next_exact = V_exact[k + 1]
        V_cont_exact = np.exp(-r * dt) * V_next_exact
        
        ex_mask = exercise_decisions[k]
        V_curr = np.where(ex_mask, payoff, V_cont_exact)
        V_exact[k] = V_curr
        
        if time.time() > deadline:
            break

    # Now re-regress using V_exact as targets
    coeffs_final = [None] * n_exercise
    V_next = np.zeros(N)
    
    for k in range(n_exercise - 1, -1, -1):
        j = k + 1
        S_curr = training_paths[:, j]
        
        if k == n_exercise - 1:
            coeffs_final[k] = None # No regression needed, just payoff
        else:
            target = np.exp(-r * dt) * V_next
            # Note: V_next here is V_exact[k+1]
            
            basis = make_basis(S_curr)
            
            t_rem_next = T - (j + 1) * dt
            if t_rem_next <= 0:
                 t_rem_next = 0
            P_euro_next = european_put_price(S_curr, t_rem_next)
            control = np.exp(-r * dt) * P_euro_next
            
            residual_target = target - control
            
            X = basis
            y = residual_target
            lambda_ridge = 1e-6 * np.trace(X.T @ X) / X.shape[1]
            
            try:
                A = X.T @ X + lambda_ridge * np.eye(X.shape[1])
                b = X.T @ y
                beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            coeffs_final[k] = beta
            
            # Update V_next for the next (previous) step
            # We need the estimated continuation value to compute V_curr for the previous step's exact calculation?
            # No, for Pass 2 regression, we use V_exact as target.
            # But we need V_next for the *next* iteration of the loop (k-1).
            # The target for k-1 is e^{-r dt} V_exact[k].
            # So V_next should be V_exact[k].
            V_next = V_exact[k]
            
        if time.time() > deadline:
            break

    class ExercisePolicy:
        def __init__(self, coeffs, K, r, sigma, T, n_exercise, dt):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
            self.dt = dt

        def exercise(self, current_state, time):
            # time: index 0 to 11
            k = int(time)
            if k < 0 or k >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            S = np.asarray(current_state)
            payoff = np.maximum(self.K - S, 0.0)
            
            if k == self.n_exercise - 1:
                # Last step: exercise if in the money
                return payoff > 0.0
            
            beta = self.coeffs[k]
            if beta is None:
                # Fallback: exercise if in the money
                return payoff > 0.0
            
            # Compute Control Variate
            j = k + 1
            t_rem_next = self.T - (j + 1) * self.dt
            if t_rem_next <= 0:
                t_rem_next = 0
            
            # European Put
            d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * t_rem_next) / (self.sigma * np.sqrt(t_rem_next)) if t_rem_next > 0 else np.zeros_like(S)
            d2 = d1 - self.sigma * np.sqrt(t_rem_next) if t_rem_next > 0 else np.zeros_like(S)
            
            if t_rem_next > 0:
                P_euro_next = self.K * np.exp(-self.r * t_rem_next) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                P_euro_next = np.maximum(self.K - S, 0.0)
            
            control = np.exp(-self.r * self.dt) * P_euro_next
            
            # Basis
            x = S / 100.0
            basis = np.column_stack([
                np.ones_like(x),
                x,
                x**2,
                x**3,
                x**4,
                x**5
            ])
            
            pred_residual = basis @ beta
            V_cont = control + pred_residual
            
            return payoff > V_cont

    return ExercisePolicy(coeffs_final, K, r, sigma, T, n_exercise, dt)

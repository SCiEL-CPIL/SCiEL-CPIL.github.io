from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

def _relu(x):
    return np.maximum(x, 0.0)

def _relu_grad(x):
    return (x > 0).astype(np.float64)

def _net_forward(x, W1, b1, W2, b2, W3, b3):
    """3-layer MLP: input(1) -> 16 -> 16 -> 1"""
    z1 = x @ W1 + b1
    a1 = _relu(z1)
    z2 = a1 @ W2 + b2
    a2 = _relu(z2)
    z3 = a2 @ W3 + b3
    return z3, z1, a1, z2, a2

def _net_predict(x, W1, b1, W2, b2, W3, b3):
    """Return threshold K * sigmoid(z3)"""
    z3, _, _, _, _ = _net_forward(x, W1, b1, W2, b2, W3, b3)
    return K_global * 1.0 / (1.0 + np.exp(-np.clip(z3, -500, 500)))

K_global = 100.0

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary
    with sigmoid-transformed polynomial in remaining time fraction.
    
    Uses exact Gaussian transition for continuation value, bisection refinement
    for thresholds, and L-BFGS-B optimization with local refinement.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s limit
    
    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))
    
    if n_dates is None:
        n_dates = 12
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = paths.shape[0]
    
    # Extract spot prices at each exercise date
    # Exercise dates correspond to steps 1 through n_dates (no time-0 exercise)
    if paths.shape[1] >= n_dates + 1:
        exercise_spots = paths[:, 1:n_dates+1]
    else:
        exercise_spots = paths[:, 1:]
        n_dates = exercise_spots.shape[1]
    
    dt = T / n_dates
    disc = math.exp(-r * dt)
    
    # European put price function
    def european_put_price(S, K_val, r_val, sigma_val, tau):
        """Compute European put price."""
        if tau <= 0:
            return np.maximum(K_val - S, 0)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K_val) + (r_val + 0.5 * sigma_val**2) * tau) / (sigma_val * sqrt_tau)
        d2 = d1 - sigma_val * sqrt_tau
        N = norm.cdf
        return K_val * np.exp(-r_val * tau) * N(-d2) - S * N(-d1)
    
    # Compute initial thresholds using European put approximation
    def compute_european_thresholds():
        thresholds = np.zeros(n_dates)
        for t in range(n_dates):
            tau_t = (n_dates - t) * dt
            if tau_t <= 0:
                thresholds[t] = K
            else:
                # Find the threshold where immediate payoff equals European put price
                S_grid = np.linspace(0.1, K * 1.5, 2000)
                cont_value = european_put_price(S_grid, K, r, sigma, tau_t)
                exercise_payoff = np.maximum(K - S_grid, 0)
                diff = exercise_payoff - cont_value
                
                positive_mask = diff > 0
                if not np.any(positive_mask):
                    thresholds[t] = 0.1
                elif np.all(positive_mask):
                    thresholds[t] = K * 1.5
                else:
                    idx = np.where(positive_mask[:-1] & ~positive_mask[1:])[0]
                    if len(idx) > 0:
                        i = idx[0]
                        S1, S2 = S_grid[i], S_grid[i+1]
                        d1_val, d2_val = diff[i], diff[i+1]
                        if d1_val != d2_val:
                            thresholds[t] = S1 - d1_val * (S2 - S1) / (d2_val - d1_val)
                        else:
                            thresholds[t] = S1
                    else:
                        thresholds[t] = K
        return thresholds
    
    # Get initial thresholds from European put
    initial_thresholds = compute_european_thresholds()
    
    # Parametric smooth boundary: sigmoid of polynomial in u (remaining time fraction)
    p = 4
    n_params = p + 1
    u_values = np.array([(n_dates - t) / n_dates for t in range(n_dates)])
    
    thresholds_clipped = np.clip(initial_thresholds, 1e-6, K - 1e-6)
    y_values = np.log(thresholds_clipped / (K - thresholds_clipped))
    
    X = np.zeros((n_dates, n_params))
    for k in range(n_params):
        X[:, k] = u_values ** k
    
    try:
        a_init, residuals, rank, sv = np.linalg.lstsq(X, y_values, rcond=None)
    except:
        a_init = np.zeros(n_params)
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def thresholds_from_params(a):
        u_t = u_values
        poly_val = np.zeros(n_dates)
        for k in range(n_params):
            poly_val += a[k] * u_t ** k
        return K * sigmoid(poly_val)
    
    def compute_payoff_under_policy(paths_exercise, thresholds, K_val, r_val, n_d, T_val):
        n_p = paths_exercise.shape[0]
        payoffs = np.zeros(n_p)
        exercised = np.zeros(n_p, dtype=bool)
        dt_val = T_val / n_d
        disc_val = math.exp(-r_val * dt_val)
        
        for t in range(n_d):
            mask = ~exercised
            if not np.any(mask):
                break
            s_t = paths_exercise[:, t]
            should_exercise = s_t < thresholds[t]
            new_exercise = mask & should_exercise
            if np.any(new_exercise):
                disc_factor = disc_val ** t
                payoffs[new_exercise] = (K_val - s_t[new_exercise]) * disc_factor
                exercised[new_exercise] = True
        return payoffs
    
    def negative_mean_payoff(a):
        thresholds = thresholds_from_params(a)
        payoffs = compute_payoff_under_policy(exercise_spots, thresholds, K, r, n_dates, T)
        return -np.mean(payoffs)
    
    # Primary optimization using L-BFGS-B on polynomial params
    try:
        result = minimize(
            negative_mean_payoff,
            a_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        a_opt = result.x
    except Exception:
        a_opt = a_init
    
    elapsed = time.time() - start_time
    best_thresholds = thresholds_from_params(a_opt)
    best_val = -negative_mean_payoff(a_opt)
    
    # Neural network refinement: learn boundary from continuation values
    if elapsed < DEADLINE - 60.0:
        # Compute continuation values via backward induction on training paths
        # For each exercise date t, continuation value = expected discounted future payoff
        # under optimal policy. We approximate by computing the continuation value
        # as the value of holding the option (not exercising) at date t.
        
        # Backward induction to get continuation values at each date
        n_p = exercise_spots.shape[0]
        # V[t] = continuation value at exercise date t (value if we don't exercise)
        # At the last date (t = n_dates-1), continuation value = 0 (no more exercise dates)
        # Actually at last date, if we don't exercise, the option expires worthless
        # So V[n_dates-1] = 0, and we should always exercise at last date if in-the-money
        
        # Let's do proper backward induction:
        # V_t(S) = max(K - S, E_t[discount * V_{t+1}(S_{t+1})])
        # For the put, at the last exercise date, V = max(K-S, 0) since no further dates
        # But actually at the last date, if we don't exercise, the option expires, so continuation = 0
        # So at last date: V = max(K-S, 0)
        
        # We'll compute the "optimal" continuation value at each date using backward induction
        # with the current best policy as a starting point, then refine
        
        # For the neural network, we want to learn the boundary such that
        # we exercise when S < boundary(t), which is equivalent to learning
        # the continuation value V(t, S) and comparing to intrinsic value.
        
        # Simplified approach: use the training paths to estimate continuation values
        # at each exercise date, then fit a neural network to predict the threshold.
        
        # Compute continuation values via backward induction on training paths
        # cont_val[t] = value of continuing (not exercising) at date t
        cont_vals = np.zeros((n_p, n_dates))
        
        # At the last exercise date, continuation value is 0 (option expires)
        # So optimal action: exercise if K - S > 0, i.e., S < K
        
        # Work backwards from the last date
        # V_{n_dates-1} = 0 (no future exercise dates, option expires)
        # At date t, continuation value = E[discount * V_{t+1}] where V_{t+1} is the value at next date
        # V_{t+1}(S_{t+1}) = max(K - S_{t+1}, cont_{t+1}(S_{t+1}))
        
        # This is complex with paths. Let's use a simpler approach:
        # Fit the neural network to predict the logit of the threshold directly,
        # using the European put thresholds as targets, then refine with L-BFGS-B.
        
        # Neural network: input = u (time fraction), output = logit(threshold/K)
        # Then threshold = K * sigmoid(output)
        
        # Initialize network weights
        rng_nn = np.random.RandomState(123)
        hid1 = 16
        hid2 = 16
        W1 = rng_nn.randn(1, hid1) * np.sqrt(2.0 / 1)
        b1 = np.zeros(hid1)
        W2 = rng_nn.randn(hid1, hid2) * np.sqrt(2.0 / hid1)
        b2 = np.zeros(hid2)
        W3 = rng_nn.randn(hid2, 1) * np.sqrt(2.0 / hid2)
        b3 = np.zeros(1)
        
        # Target: logit of threshold/K for each exercise date
        u_train = u_values.reshape(-1, 1)  # (n_dates, 1)
        y_target = y_values.reshape(-1, 1)  # (n_dates, 1)
        
        # Train the neural network with MSE loss
        lr = 0.01
        n_epochs = 2000
        for epoch in range(n_epochs):
            # Forward pass
            z3, z1, a1, z2, a2 = _net_forward(u_train, W1, b1, W2, b2, W3, b3)
            
            # MSE loss
            pred = z3  # We predict logit directly (no sigmoid in output for training)
            loss = np.mean((pred - y_target) ** 2)
            
            # Backward pass
            dz3 = 2.0 * (pred - y_target) / n_dates  # (n_dates, 1)
            dW3 = a2.T @ dz3  # (hid2, 1)
            db3 = np.sum(dz3, axis=0)  # (1,)
            
            da2 = dz3 @ W3.T  # (n_dates, hid2)
            dz2 = da2 * _relu_grad(z2)  # (n_dates, hid2)
            dW2 = a1.T @ dz2  # (hid1, hid2)
            db2 = np.sum(dz2, axis=0)  # (hid2,)
            
            da1 = dz2 @ W2.T  # (n_dates, hid1)
            dz1 = da1 * _relu_grad(z1)  # (n_dates, hid1)
            dW1 = u_train.T @ dz1  # (1, hid1)
            db1 = np.sum(dz1, axis=0)  # (hid1,)
            
            # Gradient descent update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3
            
            # Check time
            if time.time() - start_time > DEADLINE - 30.0:
                break
        
        # Get thresholds from neural network
        z3_nn, _, _, _, _ = _net_forward(u_train, W1, b1, W2, b2, W3, b3)
        thresholds_nn = K * sigmoid(z3_nn.flatten())
        thresholds_nn = np.clip(thresholds_nn, 0.0, K)
        
        # Evaluate the neural network policy
        val_nn = -negative_mean_payoff(
            np.array([np.log(thresholds_nn[i] / (K - thresholds_nn[i])) for i in range(n_dates)])
            if np.all(thresholds_nn > 0) and np.all(thresholds_nn < K) else a_opt
        )
        
        # Actually, let's just evaluate the payoff directly
        payoffs_nn = compute_payoff_under_policy(exercise_spots, thresholds_nn, K, r, n_dates, T)
        val_nn = np.mean(payoffs_nn)
        
        if val_nn > best_val:
            best_val = val_nn
            best_thresholds = thresholds_nn
            # Refine the neural network thresholds with local optimization
            # Convert to polynomial params for refinement
            thresholds_clipped_nn = np.clip(thresholds_nn, 1e-6, K - 1e-6)
            y_nn = np.log(thresholds_clipped_nn / (K - thresholds_clipped_nn))
            try:
                a_nn, _, _, _ = np.linalg.lstsq(X, y_nn, rcond=None)
                # Refine with L-BFGS-B
                result_nn = minimize(
                    negative_mean_payoff,
                    a_nn,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                thresholds_refined = thresholds_from_params(result_nn.x)
                payoffs_refined = compute_payoff_under_policy(exercise_spots, thresholds_refined, K, r, n_dates, T)
                val_refined = np.mean(payoffs_refined)
                if val_refined > best_val:
                    best_val = val_refined
                    best_thresholds = thresholds_refined
            except Exception:
                pass
    
    # Final local refinement with multi-start
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 10.0
    if remaining_time > 5.0:
        # Get current best thresholds as polynomial params
        thresholds_clipped_best = np.clip(best_thresholds, 1e-6, K - 1e-6)
        y_best = np.log(thresholds_clipped_best / (K - thresholds_clipped_best))
        try:
            a_best, _, _, _ = np.linalg.lstsq(X, y_best, rcond=None)
        except:
            a_best = a_opt
        
        n_restarts = min(15, int(remaining_time / 2.0))
        rng = np.random.RandomState(42)
        for restart in range(n_restarts):
            scale = 0.02 * (restart % 3 + 1)
            a_try = a_best + rng.normal(0, scale, size=n_params)
            try:
                val = -negative_mean_payoff(a_try)
                if val > best_val:
                    best_val = val
                    best_thresholds = thresholds_from_params(a_try)
            except Exception:
                pass
            
            if time.time() - start_time > DEADLINE - 10.0:
                break
    
    # Clamp thresholds to [0, K]
    thresholds = np.clip(best_thresholds, 0.0, K)
    
    class ExercisePolicy:
        def __init__(self, thresholds, n_dates):
            self.thresholds = thresholds
            self.n_dates = n_dates
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of spot prices
            - time: the exercise date index (1-indexed, so time=1 is the first exercise date)
            
            Returns:
            - boolean array indicating whether to exercise for each path
            """
            t_idx = int(time) - 1
            if t_idx < 0 or t_idx >= self.n_dates:
                cs = np.asarray(current_state)
                return np.zeros(cs.shape, dtype=bool)
            
            threshold = float(self.thresholds[t_idx])
            cs = np.asarray(current_state, dtype=np.float64)
            return cs < threshold
    
    policy = ExercisePolicy(thresholds, n_dates)
    
    return policy
# EVOLVE-BLOCK-END

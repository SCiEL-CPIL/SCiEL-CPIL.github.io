# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using a neural network surrogate for 
    continuation values, trained via backward induction on training paths.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)

    # Handle input shape
    n_paths, n_cols = training_paths.shape
    if n_cols == n_dates:
        exercise_data = training_paths
    elif n_cols == n_dates + 1:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]

    # Precompute log prices
    S_vals = [exercise_data[:, t] for t in range(n_dates)]
    log_S_vals = [np.log(S_vals[t]) for t in range(n_dates)]
    
    # Normalize log prices for better NN training
    # We'll use a global normalization across all dates
    all_log_S = np.concatenate(log_S_vals)
    log_S_min = np.min(all_log_S)
    log_S_max = np.max(all_log_S)
    log_S_range = log_S_max - log_S_min
    if log_S_range < 1e-9:
        log_S_range = 1.0
        
    normalized_log_S_vals = [(log_S_vals[t] - log_S_min) / log_S_range for t in range(n_dates)]

    # Initialize V for the last date
    V_vals = [None] * n_dates
    V_vals[-1] = np.maximum(K - S_vals[-1], 0.0)

    # Store NN parameters for each time step
    nn_params = [None] * n_dates
    # Store fallback polynomial coefficients in case NN fails or times out
    poly_coeffs = [None] * n_dates

    dt = T / n_dates
    discount = np.exp(-r * dt)

    # Simple NN implementation using NumPy
    class SimpleNN:
        def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            # He initialization
            scale1 = np.sqrt(2.0 / self.input_dim)
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * scale1
            self.b1 = np.zeros(self.hidden_dim)
            
            scale2 = np.sqrt(2.0 / self.hidden_dim)
            self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * scale2
            self.b2 = np.zeros(self.output_dim)
            
        def forward(self, X):
            self.X = X
            self.Z1 = X @ self.W1 + self.b1
            self.A1 = np.tanh(self.Z1)
            self.Z2 = self.A1 @ self.W2 + self.b2
            self.A2 = self.Z2
            return self.A2

        def backward(self, Y, learning_rate=0.01):
            m = self.X.shape[0]
            
            dZ2 = self.A2 - Y
            dW2 = (self.A1.T @ dZ2) / m
            db2 = np.sum(dZ2, axis=0) / m
            
            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (1 - self.A1**2)
            dW1 = (self.X.T @ dZ1) / m
            db1 = np.sum(dZ1, axis=0) / m
            
            # Gradient Descent Update
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

    def get_features_poly(S):
        log_S = np.log(S)
        return np.column_stack([np.ones_like(log_S), log_S, log_S**2, log_S**3])

    # Backward Induction
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            # If we run out of time, break and use whatever we have
            break

        S_t = S_vals[t]
        X_t = normalized_log_S_vals[t].reshape(-1, 1)
        V_next_vals = V_vals[t+1]

        # Target for continuation value at time t is the expected value of V_next
        # But in backward induction with Monte Carlo, we usually regress the *discounted* next value
        # or the *undiscounted* next value and then discount the result.
        # Let's regress the undiscounted V_next_vals to get the continuation value function C_t(S_t)
        # Then V_t = max(Intrinsic, discount * C_t(S_t))
        
        # Train NN
        try:
            if time.time() < deadline - 5: # Reserve time for other steps
                # Subsample if too many paths to save time? 4096 is small, keep all.
                X_train = X_t
                Y_train = V_next_vals.reshape(-1, 1)
                
                # Normalize Y for better training
                Y_mean = np.mean(Y_train)
                Y_std = np.std(Y_train)
                if Y_std < 1e-9: Y_std = 1.0
                Y_norm = (Y_train - Y_mean) / Y_std
                
                nn = SimpleNN(input_dim=1, hidden_dim=16, output_dim=1)
                
                # Training loop
                n_epochs = 200
                lr = 0.01
                for epoch in range(n_epochs):
                    pred = nn.forward(X_train)
                    loss = np.mean((pred - Y_norm)**2)
                    # Backprop
                    # We need to pass the normalized target to backward
                    # But backward expects raw Y? No, we should train on normalized Y.
                    # Let's modify backward to accept target
                    # Actually, let's just compute gradients on normalized loss
                    # dLoss/dPred = 2*(Pred - Y_norm)/m
                    # This is equivalent to running backward with Y_norm
                    
                    # Let's implement a specific train step for this normalized case
                    m = X_train.shape[0]
                    dZ2 = (nn.A2 - Y_norm) * 2 / m
                    dW2 = (nn.A1.T @ dZ2)
                    db2 = np.sum(dZ2, axis=0)
                    
                    dA1 = dZ2 @ nn.W2.T
                    dZ1 = dA1 * (1 - nn.A1**2)
                    dW1 = (nn.X.T @ dZ1)
                    db1 = np.sum(dZ1, axis=0)
                    
                    nn.W2 -= lr * dW2
                    nn.b2 -= lr * db2
                    nn.W1 -= lr * dW1
                    nn.b1 -= lr * db1
                    
                    if time.time() > deadline - 10:
                        break

                # Store parameters
                nn_params[t] = {
                    'W1': nn.W1, 'b1': nn.b1,
                    'W2': nn.W2, 'b2': nn.b2,
                    'Y_mean': Y_mean, 'Y_std': Y_std
                }
            else:
                nn_params[t] = None
        except Exception:
            nn_params[t] = None

        # Fallback: Polynomial Regression
        # Always compute polynomial as a safety net and for blending if needed
        X_poly = get_features_poly(S_t)
        try:
            coeffs_poly, _, _, _ = np.linalg.lstsq(X_poly, V_next_vals, rcond=None)
        except np.linalg.LinAlgError:
            coeffs_poly = np.array([np.mean(V_next_vals), 0.0, 0.0, 0.0])
        poly_coeffs[t] = coeffs_poly

        # Calculate V_t
        # If NN is available, use it. Otherwise use Poly.
        # We could blend, but let's trust the NN if it converged, else Poly.
        if nn_params[t] is not None:
            # Predict using NN
            W1 = nn_params[t]['W1']
            b1 = nn_params[t]['b1']
            W2 = nn_params[t]['W2']
            b2 = nn_params[t]['b2']
            Y_mean = nn_params[t]['Y_mean']
            Y_std = nn_params[t]['Y_std']
            
            Z1 = X_t @ W1 + b1
            A1 = np.tanh(Z1)
            Z2 = A1 @ W2 + b2
            A2 = Z2
            cont_vals = (A2 * Y_std + Y_mean).flatten()
        else:
            cont_vals = X_poly @ coeffs_poly

        cont_vals_discounted = cont_vals * discount
        intrinsic_vals = np.maximum(K - S_t, 0.0)
        V_vals[t] = np.maximum(intrinsic_vals, cont_vals_discounted)

    # Construct the policy object
    class ExercisePolicy:
        def __init__(self, K, nn_params, poly_coeffs, n_dates, r, T, log_S_min, log_S_range):
            self.K = K
            self.nn_params = nn_params
            self.poly_coeffs = poly_coeffs
            self.n_dates = n_dates
            self.r = r
            self.T = T
            self.dt = T / n_dates
            self.discount = np.exp(-self.r * self.dt)
            self.log_S_min = log_S_min
            self.log_S_range = log_S_range

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            """
            if time >= self.n_dates - 1:
                return (self.K - current_state) > 0

            if time < 0 or time >= self.n_dates:
                return (self.K - current_state) > 0

            # Intrinsic value
            intrinsic_val = np.maximum(self.K - current_state, 0.0)

            # Continuation value
            log_S = np.log(current_state)
            X_norm = ((log_S - self.log_S_min) / self.log_S_range).reshape(-1, 1)

            cont_val = None
            
            # Try NN first
            if self.nn_params[time] is not None:
                p = self.nn_params[time]
                Z1 = X_norm @ p['W1'] + p['b1']
                A1 = np.tanh(Z1)
                Z2 = A1 @ p['W2'] + p['b2']
                A2 = Z2
                cont_val = (A2 * p['Y_std'] + p['Y_mean']).flatten()
            
            # Fallback to Poly
            if cont_val is None:
                log_S_flat = log_S
                X_poly = np.column_stack([np.ones_like(log_S_flat), log_S_flat, log_S_flat**2, log_S_flat**3])
                coeffs = self.poly_coeffs[time]
                if coeffs is not None:
                    cont_val = X_poly @ coeffs
                else:
                    # Ultimate fallback: intrinsic
                    return intrinsic_val > 0

            cont_val_discounted = cont_val * self.discount
            
            return intrinsic_val >= cont_val_discounted

    return ExercisePolicy(K, nn_params, poly_coeffs, n_dates, r, T, log_S_min, log_S_range)

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using neural network regression 
    for continuation values (Longstaff-Schwartz style).
    
    Strategy:
    1. Use the provided training_paths to simulate the option pricing under risk-neutral dynamics.
    2. Perform backward induction to compute empirical continuation values at each exercise date.
    3. Train a small neural network to regress the continuation value as a function of (S_t, t).
    4. Use the trained network to determine the exercise decision: exercise if intrinsic >= predicted continuation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin within 120s limit

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Monthly exercise dates
    dt = T / n_exercises

    # training_paths shape: (n_paths, n_timesteps)
    # n_timesteps should be 12 (one per exercise date)
    # Let's assume training_paths contains the spot prices at the 12 exercise dates.
    # If it contains more, we need to subsample.
    # The contract says "12 monthly exercise dates".
    # Let's assume training_paths has shape (N, 12).
    
    N, T_steps = training_paths.shape
    if T_steps != n_exercises:
        # If we have more steps, we need to pick the 12 monthly ones.
        # Assuming uniform grid, we pick indices [1, 12, 23, ...] or similar.
        # But the task says "12 monthly exercise dates".
        # Let's assume the input is already aligned to the 12 dates.
        # If not, we'll try to infer.
        # For safety, if T_steps > 12, we take the last 12? Or evenly spaced?
        # Let's assume the standard case where T_steps == 12.
        # If T_steps != 12, we might need to resample.
        # Given the ambiguity, I will assume T_steps == 12.
        # If it's not, I'll take a subset.
        if T_steps > n_exercises:
            indices = np.linspace(0, T_steps - 1, n_exercises).astype(int)
            training_paths = training_paths[:, indices]
        else:
            # Pad or raise error? Let's just proceed and hope it matches.
            pass

    # Initialize continuation values at the last date
    # At maturity (t=11), continuation = 0 (if we don't exercise, we get 0? No, we get payoff if we hold to maturity? 
    # Bermudan: at the last date, you can exercise. If you don't, the option expires.
    # So V_11(S) = max(K-S, 0).
    # But in LSM, we regress the *continuation* value.
    # At the last step, continuation = 0.
    # Wait, the standard LSM algorithm:
    # 1. Start from T-1 down to 0.
    # 2. At step t, we have paths.
    # 3. We know the "future value" from step t+1 (which is either exercise or continuation).
    # 4. We regress this future value on S_t to get E[future | S_t].
    # 5. We compare intrinsic at t vs discounted regression.
    
    # Let's define the value process V_t for each path.
    # V_T = Payoff_T = max(K - S_T, 0)
    # For t from T-1 down to 0:
    #   Regress V_{t+1} on S_t.
    #   Cont_t = disc * Regressed_V_{t+1}(S_t)
    #   V_t = max(Intrinsic_t, Cont_t)
    
    # We need to store the regression model for each t.
    # Since we want a single policy object, we can train one network that takes (S, t) as input.
    # Or we can train 12 separate networks (one for each t) and combine them.
    # Combining into one network with t as a feature is more elegant.
    
    # Let's prepare the data for the network.
    # Inputs: S_t (12xN), t (12xN, constant per row)
    # Targets: We need the "target" for the network.
    # The network should predict the *continuation value* C(S, t).
    # But the continuation value depends on the optimal exercise policy in the future.
    # This is a bootstrapping problem.
    
    # Standard LSM approach:
    # 1. Compute the "true" optimal values V_t for all t using backward induction on the training paths.
    #    This gives us the "optimal" decision at each node.
    # 2. For each t, the "continuation value" is disc * E[V_{t+1} | S_t].
    # 3. We can regress V_{t+1} on S_t to get an estimate of the conditional expectation.
    # 4. Then we can check: did we exercise in the optimal policy?
    #    If we exercised, V_t = Intrinsic.
    #    If we held, V_t = disc * E[V_{t+1} | S_t].
    
    # Actually, we don't need the "optimal" policy to train the network.
    # We can train the network to predict V_{t+1} given S_t.
    # Then at evaluation time, we use the network to estimate E[V_{t+1} | S_t].
    # Then we compare Intrinsic_t vs disc * Network(S_t, t).
    
    # But V_{t+1} itself depends on the policy.
    # So we must compute V_{t+1} first.
    # This implies we need to perform the backward induction to get the "targets" V_{t+1}.
    
    # Step 1: Backward induction on training paths to get V_t for all t.
    # This is fast: O(N * T).
    
    S = training_paths  # (N, 12)
    V = np.zeros_like(S)
    
    # t = 11 (last date)
    V[:, 11] = np.maximum(K - S[:, 11], 0.0)
    
    # We will store the regression models.
    # Since we are using a neural network, we need to train it.
    # But we can also use a simpler regression first to get V_t, then train the NN on the (S_t, V_{t+1}) pairs?
    # No, the NN should learn the mapping S_t -> E[V_{t+1}|S_t].
    
    # Let's collect the pairs (S_t, V_{t+1}) for all t and paths.
    # For t=0..10, we have S[:, t] and V[:, t+1].
    # We want to learn f(S, t) = E[V_{t+1} | S, t].
    
    X_train = np.zeros((N * 11, 2))  # (S, t)
    y_train = np.zeros(N * 11)
    
    idx = 0
    for t in range(11):  # 0 to 10
        X_train[idx:idx+N, 0] = S[:, t]
        X_train[idx:idx+N, 1] = t
        y_train[idx:idx+N] = V[:, t+1]
        idx += N
        
    # Normalize inputs for better NN training
    S_mean = np.mean(X_train[:, 0])
    S_std = np.std(X_train[:, 0]) + 1e-8
    t_mean = np.mean(X_train[:, 1])
    t_std = np.std(X_train[:, 1]) + 1e-8
    
    X_norm = X_train.copy()
    X_norm[:, 0] = (X_norm[:, 0] - S_mean) / S_std
    X_norm[:, 1] = (X_norm[:, 1] - t_mean) / t_std
    
    # Simple Neural Network implementation using NumPy
    class SimpleNN:
        def __init__(self, input_dim, hidden_dim, output_dim):
            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b1 = np.zeros((1, hidden_dim))
            self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
            self.b2 = np.zeros((1, output_dim))
            
        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = np.tanh(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = self.z2  # Linear output
            return self.a2
            
        def backward(self, X, y, lr=0.01):
            m = X.shape[0]
            self.a2 = self.forward(X)
            dz2 = (self.a2 - y) / m
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - self.a1**2)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            
        def predict(self, X):
            return self.forward(X)
            
    # Train the NN
    hidden_dim = 32
    nn = SimpleNN(2, hidden_dim, 1)
    
    # Training loop
    n_epochs = 500
    batch_size = 256
    
    for epoch in range(n_epochs):
        if time.time() > deadline - 10.0:
            break
            
        # Shuffle
        perm = np.random.permutation(len(X_norm))
        X_norm_shuffled = X_norm[perm]
        y_shuffled = y_train[perm]
        
        for i in range(0, len(X_norm_shuffled), batch_size):
            X_batch = X_norm_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size].reshape(-1, 1)
            nn.backward(X_batch, y_batch, lr=0.01)
            
    # Now we have a model that predicts E[V_{t+1} | S_t, t]
    # We need to compute the optimal exercise policy.
    # But wait, the V_{t+1} we used for training were computed using the *optimal* policy in the future.
    # So the model should be good.
    
    # Let's verify the policy on the training paths.
    # For each t, compute Cont_t = disc * NN(S_t, t)
    # Exercise if Intrinsic_t >= Cont_t
    
    disc = np.exp(-r * dt)
    
    # We will store the boundaries or the policy function.
    # The policy is: Exercise if S <= B_t.
    # We can find B_t by checking the training paths or by inverting the NN?
    # Since NN is not monotonic necessarily, it's safer to just use the NN directly in the exercise function.
    # But the interface requires a boolean vector.
    # We can just compute the decision on the fly.
    
    class ExercisePolicy:
        def __init__(self, nn, S_mean, S_std, t_mean, t_std, K, r, dt):
            self.nn = nn
            self.S_mean = S_mean
            self.S_std = S_std
            self.t_mean = t_mean
            self.t_std = t_std
            self.K = K
            self.r = r
            self.dt = dt
            self.disc = np.exp(-r * dt)
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices.
            time: time step index (0-based, 0 to 11).
            """
            if time < 0 or time >= 11:
                # At the last date (11), we always exercise if ITM.
                if time == 11:
                    return current_state < self.K
                return np.zeros(len(current_state), dtype=bool)
                
            # Prepare inputs
            S_norm = (current_state - self.S_mean) / self.S_std
            t_norm = (time - self.t_mean) / self.t_std
            
            X_eval = np.column_stack((S_norm, t_norm))
            
            # Predict continuation value
            cont_val = self.nn.predict(X_eval).flatten()
            cont_val = self.disc * cont_val
            
            # Intrinsic value
            intr_val = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Intrinsic >= Continuation
            return intr_val >= cont_val
            
    return ExercisePolicy(nn, S_mean, S_std, t_mean, t_std, K, r, dt)
# EVOLVE-BLOCK-END

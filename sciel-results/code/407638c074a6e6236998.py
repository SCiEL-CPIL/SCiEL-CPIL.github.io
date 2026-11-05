"""Rewritten program for Bermudan exercise-policy research."""
# EVOLVE-BLOCK-START
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction to generate
    optimal exercise labels, then train a neural network classifier to predict
    exercise decisions from (spot, time) features.
    
    The approach:
    1. Parse contract parameters.
    2. Perform backward induction on a fine price grid to compute optimal
       exercise decisions at each exercise date.
    3. Generate training samples from the training paths with labels from
       the backward induction decisions.
    4. Train a simple neural network (via gradient descent) to classify
       exercise decisions.
    5. Return an object whose exercise(current_state, time) method uses the
       trained network to make predictions.
    """
    # Extract contract parameters
    T = contract.get('T', 1.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    K = contract.get('K', 100.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Training paths shape: (n_paths, n_time_steps + 1)
    n_paths, n_steps = training_paths.shape
    dt = T / (n_steps - 1)
    
    # Exercise indices: last n_exercise time steps
    exercise_indices = list(range(n_steps - n_exercise, n_steps))
    
    # Build fine price grid for backward induction
    n_grid = 500
    min_price = max(1.0, K * 0.01)
    max_price = max(K * 3.0, np.max(training_paths[:, 1:]))
    
    # Use log-spaced grid for better resolution at low prices
    grid = np.geomspace(min_price, max_price, n_grid)
    
    # Backward induction
    discount_factor = math.exp(-r * dt)
    mu = (r - sigma**2 / 2) * dt
    sigma_dt = sigma * math.sqrt(dt)
    
    # Use Gauss-Hermite-like quadrature points
    n_quad = 200
    z_points = np.linspace(-5, 5, n_quad)
    # Use proper normal density weights
    from scipy.stats import norm
    z_weights = norm.pdf(z_points) * (z_points[1] - z_points[0])
    z_weights /= np.sum(z_weights)
    
    # Store continuation values
    cont_values = {}
    
    # Last exercise date: always exercise if ITM
    last_idx = exercise_indices[-1]
    payoff_last = np.maximum(K - grid, 0.0)
    cont_values[last_idx] = payoff_last.copy()
    
    # Work backwards
    for i in range(len(exercise_indices) - 2, -1, -1):
        current_idx = exercise_indices[i]
        next_idx = exercise_indices[i + 1]
        
        cont_next = cont_values[next_idx]
        
        # Compute expected continuation value for each grid point
        expected_cont = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = grid[j]
            log_S_next = math.log(S_j) + mu + sigma_dt * z_points
            S_next = np.exp(log_S_next)
            
            # Interpolate continuation value
            cont_at_next = np.interp(S_next, grid, cont_next)
            expected_cont[j] = np.dot(z_weights, cont_at_next)
        
        expected_cont *= discount_factor
        
        payoff_current = np.maximum(K - grid, 0.0)
        
        # Optimal value
        optimal_value = np.maximum(payoff_current, expected_cont)
        cont_values[current_idx] = optimal_value
    
    # Now generate training labels from training paths
    # For each path and each exercise date, determine if we should exercise
    # based on the backward induction decision boundary
    
    # Build decision boundaries: for each exercise date, find the threshold
    # below which we exercise (for puts, we exercise when S is low enough)
    decision_thresholds = {}
    for idx in exercise_indices:
        payoff = np.maximum(K - grid, 0.0)
        cont = np.zeros(n_grid)
        
        if idx == exercise_indices[-1]:
            # At last date, exercise if ITM
            decision_thresholds[idx] = K
        else:
            next_idx = exercise_indices[exercise_indices.index(idx) + 1]
            cont_next = cont_values[next_idx]
            
            expected_cont = np.zeros(n_grid)
            for j in range(n_grid):
                S_j = grid[j]
                log_S_next = math.log(S_j) + mu + sigma_dt * z_points
                S_next = np.exp(log_S_next)
                cont_at_next = np.interp(S_next, grid, cont_next)
                expected_cont[j] = np.dot(z_weights, cont_at_next)
            
            expected_cont *= discount_factor
            
            # Find the threshold where payoff = expected_cont
            # For puts, exercise when S < threshold
            diff = payoff - expected_cont
            # Find where diff changes from positive to negative
            exercise_mask = diff >= 0
            
            # Find the largest S where we exercise
            exercise_S = grid[exercise_mask]
            if len(exercise_S) > 0:
                decision_thresholds[idx] = np.max(exercise_S)
            else:
                decision_thresholds[idx] = 0.0
    
    # Now create training data for the neural network
    # Features: (normalized spot, normalized time)
    # Label: 1 if exercise, 0 if continue
    
    # Collect samples from training paths
    samples = []
    labels = []
    
    for path_idx in range(n_paths):
        for ex_idx in exercise_indices:
            S = training_paths[path_idx, ex_idx]
            t = ex_idx * dt
            threshold = decision_thresholds[ex_idx]
            
            # Label: exercise if S <= threshold
            label = 1.0 if S <= threshold else 0.0
            
            # Normalize features
            S_norm = S / max_price
            t_norm = t / T
            
            samples.append((S_norm, t_norm))
            labels.append(label)
    
    samples = np.array(samples)
    labels = np.array(labels)
    
    # Train a simple neural network with one hidden layer
    # Architecture: 2 inputs -> 32 hidden (tanh) -> 1 output (sigmoid)
    np.random.seed(42)
    
    n_samples = len(samples)
    n_hidden = 32
    
    # Initialize weights
    W1 = np.random.randn(2, n_hidden) * np.sqrt(2.0 / 2)
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros(1)
    
    # Training hyperparameters
    lr = 0.01
    n_epochs = 500
    batch_size = 256
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        samples_shuffled = samples[perm]
        labels_shuffled = labels[perm]
        
        # Mini-batch gradient descent
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = samples_shuffled[start:end]
            y_batch = labels_shuffled[start:end]
            
            # Forward pass
            hidden = np.tanh(X_batch @ W1 + b1)
            output = 1.0 / (1.0 + np.exp(-hidden @ W2 - b2))
            
            # Compute gradients
            m = len(X_batch)
            
            # Output layer gradient
            d_output = (output - y_batch.reshape(-1, 1)) / m
            
            dW2 = hidden.T @ d_output
            db2 = np.sum(d_output, axis=0)
            
            # Hidden layer gradient
            d_hidden = d_output @ W2.T * (1 - hidden ** 2)
            
            dW1 = X_batch.T @ d_hidden
            db1 = np.sum(d_hidden, axis=0)
            
            # Update weights
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
    
    # Create policy object
    class ExercisePolicy:
        def __init__(self, W1, b1, W2, b2, max_price, T):
            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
            self.max_price = max_price
            self.T = T
        
        def exercise(self, current_state, time):
            """
            current_state: array of current spot prices
            time: current time (float)
            
            Returns: boolean vector indicating whether to exercise each path.
            """
            n = len(current_state)
            
            # Normalize features
            S_norm = current_state / self.max_price
            t_norm = time / self.T
            
            # Create feature matrix
            X = np.column_stack([S_norm, np.full(n, t_norm)])
            
            # Forward pass
            hidden = np.tanh(X @ self.W1 + self.b1)
            output = 1.0 / (1.0 + np.exp(-hidden @ self.W2 - self.b2))
            
            # Exercise if probability > 0.5
            return (output.flatten() > 0.5)
    
    return ExercisePolicy(W1, b1, W2, b2, max_price, T)
# EVOLVE-BLOCK-END

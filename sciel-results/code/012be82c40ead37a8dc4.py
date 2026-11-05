import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    T = contract.get('T', 1.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    K = contract.get('K', 100.0)
    n_exercise = contract.get('n_exercise', 12)
    
    n_paths, n_steps = training_paths.shape
    dt = T / (n_steps - 1)
    
    exercise_indices = list(range(n_steps - n_exercise, n_steps))
    
    # 1. Backward Induction on a grid to generate labels
    n_grid = 500
    grids = {}
    for idx in exercise_indices:
        prices = training_paths[:, idx]
        qs = np.linspace(0.001, 0.999, n_grid)
        grid = np.quantile(prices, qs)
        grid = np.sort(grid)
        grids[idx] = grid
        
    cont_values = {}
    
    last_idx = exercise_indices[-1]
    grid_last = grids[last_idx]
    cont_values[last_idx] = np.zeros(n_grid)
    
    discount_factor = math.exp(-r * dt)
    mu = (r - sigma**2 / 2) * dt
    sigma_dt = sigma * math.sqrt(dt)
    
    n_quad = 100
    z_points = np.linspace(-4, 4, n_quad)
    dz = z_points[1] - z_points[0]
    weights = np.ones(n_quad) * dz
    
    for i in range(len(exercise_indices) - 2, -1, -1):
        current_idx = exercise_indices[i]
        next_idx = exercise_indices[i + 1]
        
        grid_current = grids[current_idx]
        grid_next = grids[next_idx]
        cont_next = cont_values[next_idx]
        
        expected_cont = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = grid_current[j]
            if S_j <= 0:
                expected_cont[j] = 0
                continue
            log_S_next = math.log(S_j) + mu + sigma_dt * z_points
            S_next = np.exp(log_S_next)
            cont_at_next = np.interp(S_next, grid_next, cont_next)
            expected_cont[j] = np.sum(weights * cont_at_next)
        
        expected_cont *= discount_factor
        cont_values[current_idx] = expected_cont
        
    # 2. Generate Labels
    X_data = []
    y_data = []
    
    for idx in exercise_indices:
        time_val = idx * dt
        grid = grids[idx]
        cont_vals = cont_values[idx]
        
        spots = training_paths[:, idx]
        cont_at_spots = np.interp(spots, grid, cont_vals)
        payoffs = np.maximum(K - spots, 0.0)
        
        labels = (payoffs >= cont_at_spots).astype(float)
        
        for s, l in zip(spots, labels):
            X_data.append([s, time_val])
            y_data.append(l)
            
    X = np.array(X_data)
    y = np.array(y_data)
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # 3. Train Neural Network
    class SimpleNN:
        def __init__(self, input_dim, hidden_dim, output_dim, lr=0.05, epochs=500):
            self.lr = lr
            self.epochs = epochs
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
            self.b1 = np.zeros((1, hidden_dim))
            self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
            self.b2 = np.zeros((1, output_dim))
            
        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = np.tanh(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = 1 / (1 + np.exp(-self.z2))
            return self.a2
            
        def backward(self, X, y):
            m = X.shape[0]
            dz2 = (self.a2 - y) / m
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - self.a1**2)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            
        def train(self, X, y):
            for _ in range(self.epochs):
                y_pred = self.forward(X)
                self.backward(X, y)
                
        def predict(self, X):
            return self.forward(X)
            
    nn = SimpleNN(input_dim=2, hidden_dim=32, output_dim=1, lr=0.05, epochs=500)
    y_train = y.reshape(-1, 1)
    nn.train(X_norm, y_train)
    
    class ExercisePolicy:
        def __init__(self, nn, X_mean, X_std):
            self.nn = nn
            self.X_mean = X_mean
            self.X_std = X_std
            
        def exercise(self, current_state, time):
            n_eval = len(current_state)
            X_eval = np.zeros((n_eval, 2))
            X_eval[:, 0] = current_state
            X_eval[:, 1] = time
            
            X_eval_norm = (X_eval - self.X_mean) / self.X_std
            probs = self.nn.predict(X_eval_norm)
            
            return (probs > 0.5).flatten()
            
    return ExercisePolicy(nn, X_mean, X_std)

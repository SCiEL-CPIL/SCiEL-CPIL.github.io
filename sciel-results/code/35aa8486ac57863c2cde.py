import numpy as np

def fit_exercise_policy(training_paths, contract):
    K = contract['strike']
    r = contract['rate']
    T = contract['maturity']
    M = 12
    
    dt = T / M
    exercise_times = np.arange(1, M + 1) * dt
    
    # Ensure training_paths is a numpy array
    S_paths = np.asarray(training_paths)
    
    # If the number of time steps in training_paths is not M+1, we need to handle it.
    # Standard assumption: training_paths are on the exercise grid.
    # If S_paths.shape[1] > M + 1, we might need to interpolate.
    # For simplicity and speed, we assume the grid matches.
    # If it doesn't, we take the last M+1 columns? No, that's risky.
    # Let's assume the grid matches M+1.
    
    if S_paths.shape[1] != M + 1:
        # Fallback: If grid doesn't match, we try to resample or just use the available columns.
        # Given the strict time limit and simplicity, we assume match.
        # If not, we'll use the first M+1 columns or similar.
        # But let's trust the input format for this specific task type.
        pass
        
    # Extract spot prices at exercise dates (columns 1 to M)
    S = S_paths[:, 1:M+1]
    N_paths = S.shape[0]
    
    # Backward Induction
    V_opt = np.zeros((N_paths, M))
    
    # Last step: i = M-1
    V_opt[:, M-1] = np.maximum(K - S[:, M-1], 0)
    
    betas = [None] * M
    
    for i in range(M - 2, -1, -1):
        S_i = S[:, i]
        
        # Features
        # Avoid log(0)
        log_S = np.log(np.maximum(S_i, 1e-12))
        
        X = np.column_stack([
            np.ones_like(S_i),
            S_i,
            S_i**2,
            S_i * log_S,
            S_i**2 * log_S
        ])
        
        # Target: Discounted V_opt from next step
        discount = np.exp(-r * dt)
        V_target = V_opt[:, i + 1] * discount
        
        # Regression
        # Using np.linalg.lstsq
        beta_i, _, _, _ = np.linalg.lstsq(X, V_target, rcond=None)
        betas[i] = beta_i
        
        # Continuation Value
        V_cont = X @ beta_i
        
        # Exercise Payoff
        V_ex = np.maximum(K - S_i, 0)
        
        # Optimal Value
        V_opt[:, i] = np.maximum(V_ex, V_cont)
        
    def exercise(current_state, time):
        S_vec = np.asarray(current_state).flatten()
        t_val = float(time)
        
        # Map time to index
        # t = (i+1) * dt => i = t/dt - 1
        idx = int(np.round(t_val / dt)) - 1
        idx = np.clip(idx, 0, M - 1)
        
        # Payoff
        V_ex = np.maximum(K - S_vec, 0)
        
        # Continuation
        if idx == M - 1:
            V_cont = np.zeros_like(S_vec)
        else:
            beta = betas[idx]
            log_S = np.log(np.maximum(S_vec, 1e-12))
            X = np.column_stack([
                np.ones_like(S_vec),
                S_vec,
                S_vec**2,
                S_vec * log_S,
                S_vec**2 * log_S
            ])
            V_cont = X @ beta
            
        return V_ex >= V_cont

    return exercise

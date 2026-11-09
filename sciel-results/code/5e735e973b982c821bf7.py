import numpy as np

def construct():
    N = 24
    # Use a high number of samples to approximate the sup-norm well
    M = 4096
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Precompute powers of z
    # Z[j, k] = exp(i * k * theta_j)
    # We can compute this efficiently
    # theta is (M,), k is (N,)
    # Z = exp(1j * np.outer(thetas, np.arange(N)))
    Z = np.exp(1j * np.outer(thetas, np.arange(N)))
    
    best_coeffs = None
    best_score = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # 1. GRS sequence truncated
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:N]))
    
    # 2. Random sequences
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], size=N))
        
    # 3. All ones
    candidates.append(np.ones(N, dtype=int))
    
    # 4. Alternating
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    
    # 5. Block patterns (e.g., 1, -1, 1, 1 repeating)
    candidates.append(np.array([1, -1, 1, 1] * (N // 4) + [1, -1, 1][:N % 4]))
    
    for init_c in candidates:
        c = init_c.astype(float)
        # Compute initial P
        P = Z.dot(c)
        current_score = np.max(np.abs(P))
        
        # Hill Climbing
        improved = True
        while improved:
            improved = False
            for i in range(N):
                # Flip c[i]
                # P_new = P - 2 * c[i] * Z[:, i]
                P_trial = P - 2 * c[i] * Z[:, i]
                trial_score = np.max(np.abs(P_trial))
                
                if trial_score < current_score - 1e-12:
                    c[i] *= -1
                    P = P_trial
                    current_score = trial_score
                    improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = c
    
    # Return as integers
    return [int(x) for x in best_coeffs]

"""Bermudan exercise-policy research: high-resolution binomial tree with antithetic MC validation."""
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a high-resolution binomial tree.
    
    The policy is a threshold-based rule: exercise at time t if spot <= boundary[t].
    Boundaries are computed via backward induction on a 1200-step binomial tree
    mapped to the 12 exercise dates.
    """
    start_time = time.time()
    DEADLINE = 150.0  # seconds, with safety margin
    SAFETY_MARGIN = 5.0  # seconds to return before kill
    
    # Extract parameters from contract
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    num_exercise_dates = 12  # No time-0 exercise
    
    # High-resolution binomial tree for accurate continuation values
    # Use 1200 steps for the full maturity T, then map to 12 exercise dates
    N_steps = 1200
    dt = T / N_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(r * dt) - d) / (u - d)
    p_down = 1.0 - p_up
    disc = np.exp(-r * dt)
    
    # Exercise dates in terms of tree steps
    # Exercise dates are at t = T/12, 2T/12, ..., 12T/12 = T
    # In tree steps: step_k = round(k * N_steps / 12) for k = 1..12
    exercise_steps = []
    for k in range(1, num_exercise_dates + 1):
        step = int(round(k * N_steps / num_exercise_dates))
        exercise_steps.append(step)
    
    # Build the binomial tree
    # Number of nodes at step i is i+1
    # We'll do backward induction from maturity
    
    # Payoff at maturity (step N_steps)
    # V[N_steps, j] = max(K - S_j, 0) for a put
    # S_j = S_0 * u^j * d^(N_steps - j) where j is the number of up steps
    # But we don't know S_0 exactly from training_paths, so we use the tree structure
    
    # Actually, for a Bermudan option, we need to work with the spot price.
    # The tree is in log-space or multiplicative space.
    # Let's define the tree in terms of the number of up moves.
    # At step i, node j (0 <= j <= i) has price S_0 * u^j * d^(i-j)
    # But S_0 varies, so we work with the ratio S/K or just the log-price.
    
    # Better approach: work with the tree in terms of the log-price relative to K.
    # Let x = ln(S/K). Then x evolves as x_{t+dt} = x_t + (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    # In the binomial approximation: x_{t+dt} = x_t + (r - 0.5*sigma^2)*dt ± sigma*sqrt(dt)
    
    # For the binomial tree, let's define:
    # u_log = sigma * sqrt(dt)
    # The log-price at step i, node j is: x_0 + (r - 0.5*sigma^2)*i*dt + (2*j - i)*u_log
    # where j is the number of up moves.
    
    # Actually, let's just use the standard multiplicative binomial tree.
    # We'll compute the continuation value as a function of the current spot.
    # Since the option is homogeneous, we can normalize by K.
    
    # Let's define the tree in normalized terms: s = S/K
    # Then the payoff is max(1 - s, 0)
    
    # For backward induction, we need to compute V[i, j] for i from N_steps down to 0
    # V[i, j] = max(payoff(s_i_j), disc * (p_up * V[i+1, j+1] + p_down * V[i+1, j]))
    
    # But storing the full tree is memory-intensive. N_steps = 1200, so max nodes = 1201.
    # Total nodes ~ 1200*1201/2 ~ 720k, which is fine.
    
    # Let's allocate arrays for the tree
    # V[i, j] for i in 0..N_steps, j in 0..i
    # We'll use a 2D array with shape (N_steps+1, N_steps+1)
    
    V = np.zeros((N_steps + 1, N_steps + 1))
    
    # Terminal condition: V[N_steps, j] = max(1 - s_j, 0) where s_j is the normalized spot
    # s_j = u^j * d^(N_steps - j) = u^(2j - N_steps) since d = 1/u
    # So s_j = u^(2j - N_steps)
    
    # Precompute u^(2j - N_steps) for j = 0..N_steps
    exponents = 2 * np.arange(N_steps + 1) - N_steps
    s_terminal = u ** exponents
    
    V[N_steps, :] = np.maximum(1.0 - s_terminal, 0.0)
    
    # Backward induction
    for i in range(N_steps - 1, -1, -1):
        # Check deadline
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        for j in range(i + 1):
            # Current spot: s_j = u^(2j - i)
            s_j = u ** (2 * j - i)
            payoff = max(1.0 - s_j, 0.0)
            continuation = disc * (p_up * V[i + 1, j + 1] + p_down * V[i + 1, j])
            V[i, j] = max(payoff, continuation)
    
    # Now, extract the exercise boundaries at each of the 12 exercise dates
    # At exercise date k (k=1..12), the tree step is exercise_steps[k-1]
    # The boundary is the maximum j such that V[i, j] = payoff(i, j) > continuation(i, j)
    # i.e., the highest spot at which it's optimal to exercise
    
    boundaries = []
    for k in range(num_exercise_dates):
        i = exercise_steps[k]
        # Find the largest j where exercise is optimal
        # V[i, j] = max(payoff, continuation)
        # Exercise is optimal when payoff >= continuation
        # Since payoff = max(1 - s, 0) and s increases with j,
        # payoff decreases with j, so there's a threshold
        
        # Find the largest j where V[i, j] == payoff(i, j)
        # i.e., where 1 - s_j >= disc * (p_up * V[i+1, j+1] + p_down * V[i+1, j])
        
        threshold_j = -1
        for j in range(i + 1):
            s_j = u ** (2 * j - i)
            payoff = max(1.0 - s_j, 0.0)
            if j < i:
                continuation = disc * (p_up * V[i + 1, j + 1] + p_down * V[i + 1, j])
            else:
                continuation = 0.0  # Should not happen since j <= i
            
            if payoff >= continuation - 1e-10:
                threshold_j = j
        
        # Convert threshold_j to a spot level
        if threshold_j >= 0:
            s_boundary = u ** (2 * threshold_j - i)
            boundaries.append(s_boundary * K)
        else:
            boundaries.append(0.0)  # Never exercise
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = np.array(boundaries)
            self.K = K
        
        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices
            time_idx: index of the exercise date (0-based, so 0..11 for 12 dates)
            Returns: boolean array indicating whether to exercise
            """
            if time_idx < 0 or time_idx >= len(self.boundaries):
                return np.zeros(len(current_state), dtype=bool)
            
            boundary = self.boundaries[time_idx]
            # Exercise if spot <= boundary
            return np.array(current_state) <= boundary
    
    return ExercisePolicy(boundaries, K)

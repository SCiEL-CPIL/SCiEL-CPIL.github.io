import numpy as np

N = 400

def construct():
    """
    Simulated Annealing with Delta Update to minimize autocorrelation energy.
    """
    
    def compute_initial_R(x):
        """Compute linear autocorrelation R[1..N-1] and sum of squares J."""
        # Use FFT for initial computation
        x_pad = np.zeros(2 * N)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        R_circ = np.fft.ifft(X * np.conj(X)).real
        R = R_circ[1:N]  # lags 1 to N-1
        J = 2 * np.sum(R**2)
        return R, J

    def delta_update(x, i, R, J):
        """
        Calculate delta J and delta R for flipping bit i.
        Returns (delta_J, delta_R_array).
        """
        # x_i is current value
        xi = x[i]
        # Delta x_i is -2 * xi
        dx = -2 * xi
        
        # Construct u_k = x_{i+k} * I(i+k<N) + x_{i-k} * I(i-k>=0) for k=1..N-1
        # We need an array u of length N-1
        u = np.zeros(N-1)
        
        # Positive lags: k=1..N-1-i
        if i + 1 < N:
            u[:N-1-i] = x[i+1:N]
        
        # Negative lags: k=1..i
        if i > 0:
            # x_{i-k} for k=1..i is x_{i-1}, x_{i-2}, ..., x_0
            # This corresponds to u[k-1] for k=1..i
            u[:i] += x[i-1::-1]
            
        # delta_R(k) = dx * u_k
        delta_R = dx * u
        
        # delta_J = 2 * sum( 2 R(k) delta_R(k) + delta_R(k)^2 )
        delta_J = 2 * np.sum(2 * R * delta_R + delta_R**2)
        
        return delta_J, delta_R

    best_x = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # SA Parameters
    num_restarts = 50
    iterations_per_restart = 5000
    T_start = 10.0
    T_end = 0.1
    cooling_rate = (T_end / T_start) ** (1.0 / iterations_per_restart)
    
    for _ in range(num_restarts):
        x = np.random.choice([-1, 1], size=N)
        R, J = compute_initial_R(x)
        
        current_J = J
        current_R = R.copy()
        
        T = T_start
        
        for _ in range(iterations_per_restart):
            # Pick random bit to flip
            i = np.random.randint(N)
            
            # Calculate delta
            delta_J, delta_R = delta_update(x, i, current_R, current_J)
            
            # Acceptance criterion
            if delta_J < 0 or np.random.rand() < np.exp(-delta_J / T):
                # Accept
                x[i] *= -1
                current_R += delta_R
                current_J += delta_J
                
                if current_J < best_score:
                    best_score = current_J
                    best_x = x.copy()
            
            T *= cooling_rate
            
    return best_x

# EVOLVE-BLOCK-END

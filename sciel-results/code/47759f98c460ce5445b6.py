import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak autocorrelation sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)

    def compute_peak_sidelobe(h):
        """
        Computes the maximum absolute value of the autocorrelation
        at non-zero lags using linear autocorrelation.
        """
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        # Linear autocorrelation: R[0..n-1] are positive lags, R[nfft-n+1..nfft-1] negative lags
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        return np.max(np.abs(all_sidelobes))

    def compute_gradient(h):
        """
        Computes the gradient of the peak sidelobe with respect to the continuous variables.
        This is a subgradient approximation.
        """
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        
        peak_val = np.max(np.abs(all_sidelobes))
        peak_idx = np.argmax(np.abs(all_sidelobes))
        
        # Determine which lag corresponds to this peak
        if peak_idx < n - 1:
            lag = peak_idx + 1
        else:
            lag = -(n - 1 - peak_idx)
            
        # The gradient of R[lag] with respect to h is a shifted version of h
        # R[lag] = sum h[i] h[i+lag]
        # dR[lag]/dh[i] = h[i+lag] + h[i-lag] (roughly, depending on boundary handling)
        
        # For a more robust subgradient, we can use the fact that we want to minimize max|R[k]|.
        # The subgradient of max|f(x)| is f'(x) * sign(f(x)) for the active constraint.
        
        # Let's approximate the gradient contribution from the active lag.
        # If lag > 0:
        # R[lag] = sum_{i=0}^{N-lag-1} h[i] h[i+lag]
        # dR[lag]/dh[i] = h[i+lag] (if i < N-lag) + h[i-lag] (if i >= lag)
        
        grad = np.zeros(n)
        
        if lag > 0:
            # Contribution from h[i] * h[i+lag]
            grad[:n-lag] += h[lag:]
            # Contribution from h[i-lag] * h[i]
            grad[lag:] += h[:n-lag]
        else:
            lag = -lag
            # Symmetric
            grad[:n-lag] += h[lag:]
            grad[lag:] += h[:n-lag]
            
        # Multiply by sign of the peak value
        # If R[lag] is positive, we want to decrease it, so move in -grad direction.
        # If R[lag] is negative, we want to increase it (make less negative), so move in +grad direction?
        # Wait, we are minimizing |R[lag]|.
        # If R[lag] > 0, we want to reduce R[lag]. Gradient ascent on R[lag] increases it. So we want to go against gradient.
        # If R[lag] < 0, we want to increase R[lag]. Gradient ascent on R[lag] increases it. So we want to go with gradient.
        # Actually, let's just use the sign of the peak value to determine direction.
        # peak_val is the value of |R[lag]|. The actual value is R[lag].
        # We want to move h to reduce |R[lag]|.
        # If R[lag] > 0, decrease R[lag].
        # If R[lag] < 0, increase R[lag].
        
        # The derivative of R[lag] w.r.t h is 'grad'.
        # So to change R[lag] by delta_R, we change h by delta_R * grad / ||grad||^2 ?
        # Simpler: just use grad * sign(R[lag]) as the direction to move AGAINST for minimization?
        # Let's check: if R[lag] > 0, we want to make it smaller. 
        # Gradient of R[lag] is grad. Moving h in direction -grad decreases R[lag].
        # So direction = -grad * sign(R[lag]).
        
        # Get the actual signed value of the peak
        if peak_idx < n - 1:
            actual_peak = R[peak_idx + 1]
        else:
            actual_peak = R[nfft - n + 1 + peak_idx]
            
        direction = -grad * np.sign(actual_peak)
        
        return direction

    def projected_gradient_descent(h, steps=5000, lr=0.01, deadline_check=True):
        """
        Continuous gradient descent with projection to {-1, 1}.
        """
        h_cont = h.astype(np.float64)
        current_score = compute_peak_sidelobe(h)
        best_h = h.copy()
        best_score = current_score
        
        for i in range(steps):
            if deadline_check and time.time() > deadline - 5:
                break
                
            # Compute gradient
            grad = compute_gradient(h_cont)
            
            # Normalize gradient to avoid huge steps
            norm = np.linalg.norm(grad)
            if norm > 0:
                grad = grad / norm
            
            # Update
            h_cont = h_cont - lr * grad
            
            # Project to {-1, 1}
            h_new = np.sign(h_cont)
            # Handle zeros
            h_new[h_new == 0] = 1.0
            
            new_score = compute_peak_sidelobe(h_new)
            
            if new_score < current_score:
                current_score = new_score
                h_cont = h_new.astype(np.float64)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
            else:
                # If no improvement, maybe reduce learning rate or revert
                # For simplicity, just revert h_cont to previous h_new
                h_cont = h_new.astype(np.float64)
                
        return best_h, best_score

    def simulated_annealing(h, iterations=3000, T_start=2.0, T_end=0.05):
        """
        Simulated annealing with a mix of flip and swap moves.
        """
        current_score = compute_peak_sidelobe(h)
        T = T_start

        for iteration in range(iterations):
            if time.time() > deadline - 10:
                break

            # Choose move type
            move_type = np.random.choice(['flip', 'swap', 'adj_swap'], p=[0.5, 0.3, 0.2])

            if move_type == 'flip':
                i = np.random.randint(N)
                h[i] *= -1
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i] *= -1

            elif move_type == 'swap':
                i, j = np.random.choice(N, 2, replace=False)
                h[i], h[j] = h[j], h[i]
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i], h[j] = h[j], h[i]

            else:  # adj_swap
                i = np.random.randint(N - 1)
                h[i], h[i+1] = h[i+1], h[i]
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i], h[i+1] = h[i+1], h[i]

            # Cooling
            T = T_start * (T_end / T_start) ** (iteration / iterations)

        return h, current_score

    best_h = None
    best_score = float('inf')

    # Generate diverse initial sequences
    initial_sequences = []
    
    # 1. Random
    np.random.seed(42)
    for _ in range(5):
        initial_sequences.append(np.random.choice([-1, 1], N))
        
    # 2. Thue-Morse
    seq = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        seq[i] = 1 if cnt % 2 == 0 else -1
    initial_sequences.append(seq)
    
    # 3. Sinusoidal based (continuous parameterization idea)
    # h[i] = sign(sin(2*pi*f*i + phase))
    for f in [0.1, 0.2, 0.3, 0.5]:
        for phase in [0, np.pi/2, np.pi]:
            h_sin = np.sign(np.sin(2 * np.pi * f * np.arange(N) + phase))
            h_sin[h_sin == 0] = 1
            initial_sequences.append(h_sin.astype(np.int8))

    # Main search loop
    for init_idx, init_h in enumerate(initial_sequences):
        if time.time() > deadline - 20:
            break

        h = init_h.copy().astype(np.float64)
        
        # Phase 1: Continuous Gradient Descent (Projection)
        # This helps navigate the landscape more smoothly than discrete flips initially
        h, pgd_score = projected_gradient_descent(h, steps=1000, lr=0.1)
        
        # Phase 2: Simulated Annealing to escape local minima
        h, sa_score = simulated_annealing(h, iterations=1000, T_start=1.0, T_end=0.01)
        
        # Phase 3: Final Polish with Gradient Descent
        h, final_score = projected_gradient_descent(h, steps=500, lr=0.01)
        
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

    if best_h is None:
        best_h = np.random.choice([-1, 1], N)

    return best_h.astype(np.float64)

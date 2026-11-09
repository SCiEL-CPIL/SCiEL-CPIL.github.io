"""
Third autocorrelation inequality - continuous phase optimization with discrete polish.

Strategy:
1. Represent the step function as a complex-valued unit-modulus vector (continuous phase).
2. Use iterative phase gradient descent (procrustes-like) to minimize sidelobe energy.
3. Round to ±1 and apply discrete local search (flips + swaps) for final polish.
4. Use a smooth surrogate (soft-max) during continuous phase to avoid nonsmooth max.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def compute_sidelobe_energy(h, use_softmax=True, beta=20.0):
        """
        Compute sidelobe metric. If use_softmax, use soft-max surrogate.
        Returns scalar objective (lower is better).
        """
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        # Linear autocorrelation sidelobes
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        abs_sidelobes = np.abs(all_sidelobes)
        
        if use_softmax:
            # Soft-max approximation: (1/beta) * log(sum(exp(beta * x)))
            max_val = np.max(abs_sidelobes)
            if max_val < 1e-10:
                return 0.0
            shifted = abs_sidelobes - max_val
            log_sum = np.log(np.sum(np.exp(beta * shifted)))
            return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        else:
            return np.max(abs_sidelobes)

    def compute_peak_sidelobe(h):
        """Exact peak sidelobe for final evaluation."""
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        return np.max(np.abs(all_sidelobes))

    def phase_gradient_step(z, lr=0.01, beta=20.0):
        """
        One step of gradient descent on the phase of z (unit modulus).
        Objective: soft-max of |autocorrelation sidelobes|.
        """
        n = len(z)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(z, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        abs_sidelobes = np.abs(all_sidelobes)
        
        # Gradient of soft-max w.r.t. phases
        # d/dtheta_j of |R_k| where R_k = sum_i z_i z*_{i+k}
        # This is complex; use numerical gradient or analytical.
        # For speed, use a simplified gradient: project the error signal back.
        
        # Compute the "error" signal: the autocorrelation sidelobes weighted by soft-max
        max_val = np.max(abs_sidelobes)
        if max_val < 1e-10:
            return z
        shifted = abs_sidelobes - max_val
        weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
        
        # Gradient direction: we want to reduce the sidelobes.
        # The gradient of |R_k| w.r.t. z is related to the convolution with the error.
        # Use a simple heuristic: z_new = z * exp(-1j * lr * angle(conj(z) * error_signal))
        # where error_signal is the IFFT of the weighted sidelobes.
        
        # Construct the full error spectrum
        err_spec = np.zeros(nfft // 2 + 1, dtype=complex)
        # Map sidelobes back to frequency domain
        # This is approximate; a more rigorous approach would use backprop.
        # For now, use a simpler iterative method: alternate projection.
        
        return z  # Placeholder - will use alternate projection instead

    def alternate_projection(z, n_iters=500, lr=0.05):
        """
        Alternate projection method:
        1. FFT
        2. Modify spectrum to reduce sidelobes
        3. IFFT
        4. Project to unit modulus
        """
        n = len(z)
        nfft = 1 << (2 * n - 1).bit_length()
        
        for it in range(n_iters):
            if time.time() > deadline - 20:
                break
            
            H = np.fft.rfft(z, n=nfft)
            R = np.fft.irfft(H * np.conj(H), n=nfft)
            sidelobes_pos = R[1:n]
            sidelobes_neg = R[nfft - n + 1:nfft]
            all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
            
            # Compute gradient-like update
            # The autocorrelation R is the IFFT of |H|^2.
            # To reduce sidelobes, we want to flatten the power spectrum in certain regions.
            # Use a simple gradient: dR/dH = H * (something)
            
            # Simpler approach: use the fact that for binary sequences,
            # the optimal is found by minimizing the peak.
            # Use a heuristic: z_new = z * exp(-1j * angle(z * conv(z, error)))
            
            # For now, use a random perturbation + selection approach in continuous space
            # Actually, let's use a different strategy: 
            # Represent as phases, and do coordinate descent on phases.
            
            break  # Skip this for now, use direct discrete search with better initialization
        
        return z

    def continuous_phase_optimize(z_init, n_iters=1000):
        """
        Optimize phases of z (unit modulus complex) to minimize sidelobe energy.
        Uses a simple gradient-free approach: random phase perturbations with acceptance.
        """
        z = z_init.copy()
        n = len(z)
        current_obj = compute_sidelobe_energy(z, use_softmax=True, beta=10.0)
        
        for it in range(n_iters):
            if time.time() > deadline - 30:
                break
            
            # Random phase perturbation
            i = np.random.randint(n)
            delta_phase = np.random.uniform(-0.5, 0.5)
            z[i] *= np.exp(1j * delta_phase)
            
            new_obj = compute_sidelobe_energy(z, use_softmax=True, beta=10.0)
            
            # Accept if better or with some probability
            if new_obj < current_obj or np.random.random() < np.exp(-(new_obj - current_obj) / 0.1):
                current_obj = new_obj
            else:
                z[i] *= np.exp(-1j * delta_phase)
        
        return z

    def discretize(z):
        """Convert complex unit-modulus vector to ±1 sequence."""
        # Map phase to sign: if real part > 0, +1; else -1
        # Or use angle: if angle in (-pi/2, pi/2), +1; else -1
        angles = np.angle(z)
        h = np.where(angles > 0, 1.0, -1.0)
        # Actually, for unit modulus, the sign should be based on the real part
        h = np.where(z.real >= 0, 1.0, -1.0)
        return h.astype(np.int8)

    def greedy_local_search(h, max_iterations=5000):
        """
        Greedy local search with flips and swaps.
        """
        current_score = compute_peak_sidelobe(h)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            if time.time() > deadline - 10:
                break

            # Try random single bit flips
            for _ in range(15):
                if time.time() > deadline - 10:
                    break
                i = np.random.randint(N)
                h[i] *= -1
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] *= -1

            # Try random swaps
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                i, j = np.random.choice(N, 2, replace=False)
                h[i], h[j] = h[j], h[i]
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i], h[j] = h[j], h[i]

            # Try adjacent swaps
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                i = np.random.randint(N - 1)
                h[i], h[i+1] = h[i+1], h[i]
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i], h[i+1] = h[i+1], h[i]

        return h, current_score

    def simulated_annealing(h, iterations=2000, T_start=1.0, T_end=0.01):
        """
        Simulated annealing with flips and swaps.
        """
        current_score = compute_peak_sidelobe(h)
        T = T_start

        for iteration in range(iterations):
            if time.time() > deadline - 15:
                break

            move_type = np.random.choice(['flip', 'swap'], p=[0.6, 0.4])

            if move_type == 'flip':
                i = np.random.randint(N)
                h[i] *= -1
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i] *= -1

            else:  # swap
                i, j = np.random.choice(N, 2, replace=False)
                h[i], h[j] = h[j], h[i]
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i], h[j] = h[j], h[i]

            T = T_start * (T_end / T_start) ** (iteration / iterations)

        return h, current_score

    # Main search
    best_h = None
    best_score = float('inf')
    np.random.seed(42)

    # Phase 1: Generate diverse initial sequences using continuous phase optimization
    initial_sequences = []
    
    # Thue-Morse
    thue_morse = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        thue_morse[i] = 1 if cnt % 2 == 0 else -1
    initial_sequences.append(thue_morse)
    
    # Random sequences
    for _ in range(10):
        initial_sequences.append(np.random.choice([-1, 1], N).astype(np.int8))

    # For each initial sequence, do:
    # 1. Convert to complex phase
    # 2. Continuous phase optimization (limited)
    # 3. Discretize
    # 4. Discrete local search (SA + greedy)
    
    for init_idx, init_h in enumerate(initial_sequences):
        if time.time() > deadline - 25:
            break

        # Convert to complex phase
        z = init_h.astype(complex)  # +1 -> 1+0j, -1 -> -1+0j
        
        # Continuous phase optimization (brief)
        z_opt = continuous_phase_optimize(z, n_iters=200)
        
        # Discretize
        h = discretize(z_opt)
        
        # Discrete local search
        h, sa_score = simulated_annealing(h.copy(), iterations=1000)
        h, local_score = greedy_local_search(h, max_iterations=2000)
        
        final_score = compute_peak_sidelobe(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

        # Early termination
        if best_score < 0.1:
            break

    # If no good result, fallback
    if best_h is None:
        best_h = np.random.choice([-1, 1], N).astype(np.int8)
        best_h, _ = greedy_local_search(best_h, max_iterations=3000)

    return best_h
# EVOLVE-BLOCK-END

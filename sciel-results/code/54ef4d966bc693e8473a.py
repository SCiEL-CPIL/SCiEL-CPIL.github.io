import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    rng = np.random.default_rng(12345)

    def compute_peak(h):
        """Compute the max absolute periodic autocorrelation (lags 1..N-1)."""
        n_fft = 1 << (2 * N).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        R_full = np.fft.irfft(power, n=n_fft)
        R = R_full[1:N]
        return np.max(np.abs(R))

    def greedy_improve(h, time_limit):
        """Perform greedy single-bit flips to reduce peak autocorrelation."""
        h = h.astype(np.float64).copy()
        n_fft = 1 << (2 * N).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        R_full = np.fft.irfft(power, n=n_fft)
        R = R_full[1:N].copy()
        current_peak = np.max(np.abs(R))
        best_h = h.copy()
        best_peak = current_peak

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N):
                # Compute delta for flipping bit i
                # delta_R[k] = -2 * h[i] * (h[(i+k)%N] + h[(i-k)%N]) for k=1..N-1
                h_plus = np.roll(h, -i)
                h_minus = np.roll(h, i)
                term = h_plus[1:N] + h_minus[1:N]
                delta_R = -2.0 * h[i] * term
                new_R = R + delta_R
                new_peak = np.max(np.abs(new_R))

                if new_peak < current_peak:
                    # Accept the flip
                    h[i] = -h[i]
                    R = new_R
                    current_peak = new_peak
                    improved = True

                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_h = h.copy()

                    if time.time() > time_limit:
                        break

        return best_h, best_peak

    def simulated_annealing(h, T_start, T_end, time_limit):
        """Simulated annealing to escape local minima."""
        h = h.astype(np.float64).copy()
        n_fft = 1 << (2 * N).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        R_full = np.fft.irfft(power, n=n_fft)
        R = R_full[1:N].copy()
        current_peak = np.max(np.abs(R))
        best_h = h.copy()
        best_peak = current_peak

        T = T_start
        # Estimate steps per second to set cooling
        # We'll just run until time limit
        while time.time() < time_limit:
            i = rng.integers(N)
            h_plus = np.roll(h, -i)
            h_minus = np.roll(h, i)
            term = h_plus[1:N] + h_minus[1:N]
            delta_R = -2.0 * h[i] * term
            new_R = R + delta_R
            new_peak = np.max(np.abs(new_R))

            # Acceptance criterion
            if new_peak < current_peak or (T > 0.01 and rng.random() < np.exp(-(new_peak - current_peak) / T)):
                h[i] = -h[i]
                R = new_R
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()

            # Cool down
            T = max(T_end, T * 0.9995)

        return best_h, best_peak

    # Multi-restart strategy
    best_h_overall = None
    best_score_overall = float('inf')

    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts

    for restart in range(num_restarts):
        if time.time() >= deadline - 5:
            break

        # Generate initial random sequence
        h_init = rng.choice([-1.0, 1.0], size=N)

        # Phase 1: Simulated annealing to find a good basin
        sa_time = time_per_restart * 0.5
        h_sa, peak_sa = simulated_annealing(h_init, T_start=5.0, T_end=0.1, time_limit=start_time + sa_time)

        # Phase 2: Greedy refinement
        greedy_time = time_per_restart * 0.5
        h_final, peak_final = greedy_improve(h_sa, time_limit=start_time + greedy_time)

        if peak_final < best_score_overall:
            best_score_overall = peak_final
            best_h_overall = h_final.copy()

    # Final check: if we have time, do one more greedy pass on the best result
    if best_h_overall is not None and time.time() < deadline - 2:
        h_final, peak_final = greedy_improve(best_h_overall, time_limit=deadline - 1)
        if peak_final < best_score_overall:
            best_h_overall = h_final
            best_score_overall = peak_final

    if best_h_overall is None:
        # Fallback
        best_h_overall = np.ones(N)

    return best_h_overall

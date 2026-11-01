"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(h):
    """Compute max absolute autocorrelation sidelobe (excluding lag 0) using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.fft(h, n=n_fft)
    power = np.abs(F) ** 2
    acf = np.fft.ifft(power).real
    # acf[0] is the peak (N), we want max of |acf[1:N]| and |acf[n_fft-N+1:n_fft]|
    # Due to circular autocorrelation, lags 1..N-1 appear at indices 1..N-1 and n_fft-N+1..n_fft-1
    sidelobe1 = np.abs(acf[1:N])
    sidelobe2 = np.abs(acf[n_fft - N + 1:n_fft])
    return max(np.max(sidelobe1), np.max(sidelobe2))


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill

    rng = np.random.RandomState(12345)

    # Best solution found
    best_h = None
    best_val = np.inf

    # Number of restarts we can afford
    max_restarts = 200

    for restart in range(max_restarts):
        if time.time() >= deadline - 10.0:
            break

        # Initialize with random ±1 sequence
        h = rng.choice([1, -1], N).astype(np.float64)
        current_val = _max_sidelobe(h)

        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()

        # Simulated annealing / local search
        # Start temperature high, decay
        T = 5.0
        T_min = 0.1
        cooling = 0.999

        # Try to do many single-bit flips
        max_iters = 50000
        iters_per_restart = min(max_iters, int((deadline - 10.0 - time.time()) / max_restarts / (1e-6)))
        iters_per_restart = max(iters_per_restart, 1000)

        for it in range(iters_per_restart):
            if time.time() >= deadline - 10.0:
                break

            # Pick a random position to flip
            idx = rng.randint(N)
            old_val = h[idx]
            h[idx] = -h[idx]

            new_val = _max_sidelobe(h)
            delta = new_val - current_val

            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[idx] = old_val

            T = max(T_min, T * cooling)

            # Early stop if we've reached a very good value
            if best_val <= 1.0:
                break

        if best_val <= 1.0:
            break

    if best_h is None:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END

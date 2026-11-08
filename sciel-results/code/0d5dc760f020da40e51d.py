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
        at non-zero lags.
        """
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H) ** 2).real
        peak = np.max(np.abs(R[1:]))
        return peak

    best_h = None
    best_score = float('inf')

    np.random.seed(42)

    def random_sequence():
        return np.random.choice([-1, 1], N)

    # Strategy: Combine simulated annealing with block moves and greedy local search
    # This is a structural change from pure greedy to a more sophisticated search method

    # Phase 1: Multiple SA restarts with block moves
    max_restarts = 50
    sa_iterations = 2000
    T_start = 5.0
    T_end = 0.01
    block_sizes = [1, 2, 4, 8, 16]

    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        h = random_sequence()
        current_score = compute_peak_sidelobe(h)
        T = T_start

        for iteration in range(sa_iterations):
            if time.time() > deadline - 10:
                break

            # Choose block size
            block_len = np.random.choice(block_sizes)
            if block_len >= N:
                block_len = N

            # Choose random start position
            start_pos = np.random.randint(0, N - block_len + 1)

            # Compute delta before flipping
            old_score = current_score

            # Flip the block
            h[start_pos:start_pos + block_len] *= -1

            new_score = compute_peak_sidelobe(h)
            delta = new_score - old_score

            # Accept or reject
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                h[start_pos:start_pos + block_len] *= -1

            # Cooling
            T = T_start * (T_end / T_start) ** (iteration / sa_iterations)

        # Local polish with greedy single-flip moves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_score = compute_peak_sidelobe(h)
            indices = np.random.permutation(N)

            for i in indices:
                if time.time() > deadline - 5:
                    break

                h[i] *= -1
                new_score = compute_peak_sidelobe(h)

                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] *= -1

        # Update best
        final_score = compute_peak_sidelobe(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

        # Early termination if score is very good
        if best_score < 0.5:
            break

    if best_h is None:
        best_h = random_sequence()

    return best_h

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 array."""
    # Thue-Morse: t(i) = parity of number of 1s in binary representation of i
    # Map to ±1: +1 if even parity, -1 if odd parity
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1 if count % 2 == 0 else -1
    return seq


def _autocorr_max_abs(f):
    """Compute max absolute value of autoconvolution for lags 1..N-1."""
    n = len(f)
    # Use FFT for linear convolution (autoconvolution)
    # Pad to at least 2n to avoid circular convolution
    size = 1
    while size < 2 * n:
        size *= 2
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad)
    conv = np.fft.ifft(F * F).real
    # conv[0] is zero-lag, we want max |conv[k]| for k=1..n-1
    # Also conv[k] for k>=n corresponds to negative lags
    # For real symmetric autocorrelation, we just need lags 1 to n-1
    if n > 1:
        return np.max(np.abs(conv[1:n]))
    return 0.0


def _compute_score(f):
    """The score is the upper bound on C3 (lower is better)."""
    return _autocorr_max_abs(f)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Start with Thue-Morse sequence - has structurally low autocorrelation
    best_f = _thue_morse(N)
    best_score = _compute_score(best_f)

    # Local search: try flipping individual signs
    # Use a smart order: try positions that contribute most to the peak
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 15:
        improved = False
        iteration += 1

        # Get the current autocorrelation to find the peak lag
        n = len(best_f)
        size = 1
        while size < 2 * n:
            size *= 2
        f_pad = np.zeros(size)
        f_pad[:n] = best_f
        F = np.fft.fft(f_pad)
        conv = np.fft.ifft(F * F).real

        # Find the lag with maximum absolute value
        if n > 1:
            abs_conv = np.abs(conv[1:n])
            peak_lag = np.argmax(abs_conv) + 1  # 1-indexed lag
            peak_val = abs_conv[peak_lag - 1]

            # Try flipping each position and see if it reduces the peak
            # To speed up, only try a subset of positions per iteration
            # Focus on positions that affect the peak lag most
            for i in range(n):
                if time.time() > deadline - 15:
                    break

                old_val = best_f[i]
                best_f[i] = -old_val

                new_score = _compute_score(best_f)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    improved = True
                else:
                    best_f[i] = old_val

    # Second pass: try pairs of flips (swap two positions)
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 10:
                break

            # Try random pair swaps
            i1 = np.random.randint(N)
            i2 = np.random.randint(N)
            if i1 == i2:
                continue

            old1, old2 = best_f[i1], best_f[i2]
            best_f[i1], best_f[i2] = -old1, -old2

            new_score = _compute_score(best_f)
            if new_score < best_score - 1e-12:
                best_score = new_score
            else:
                best_f[i1], best_f[i2] = old1, old2

    return best_f
# EVOLVE-BLOCK-END

"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time

N = 400
TIME_LIMIT = 115  # safety margin before 120s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n

    # Linear autoconvolution via FFT
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h

    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real

    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs

    return c3_bound


def _fast_local_search(heights, max_iter=200, time_budget=30.0):
    """
    Fast local search: try flipping individual signs to reduce the C3 bound.
    Uses FFT-based bound computation for speed.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    start_time = time.time()

    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1

        # Try flipping each position
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val

            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True

    return h, current_bound


def _pairwise_swap_search(heights, max_iter=50, time_budget=20.0):
    """
    Pairwise swap search: try swapping signs at two positions to reduce the C3 bound.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    start_time = time.time()

    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        improved = False
        iters += 1
        best_gain = 0
        best_pair = (-1, -1)

        # Try swapping pairs (limit to reduce computation)
        for i in range(0, N, 2):
            for j in range(i + 2, N, 3):
                old_i, old_j = h[i], h[j]
                h[i], h[j] = old_j, old_i
                new_bound = _compute_c3_bound(h)
                h[i], h[j] = old_i, old_j

                gain = current_bound - new_bound
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)

        if best_pair[0] >= 0:
            i, j = best_pair
            h[i], h[j] = h[j], h[i]
            current_bound -= best_gain
            improved = True

    return h, current_bound


def _gradient_descent_smooth(heights, steps=50, lr=0.05, beta=10.0, time_budget=25.0):
    """
    Gradient descent on a smooth surrogate: minimize soft-max of autoconvolution.
    The soft-max approximates the max and provides gradients.
    """
    h = np.array(heights, dtype=np.float64)
    start_time = time.time()

    for step in range(steps):
        if time.time() - start_time > time_budget:
            break

        # Compute autoconvolution
        n = len(h)
        dx = 0.5 / n
        h_pad = np.zeros(2 * n - 1)
        h_pad[:n] = h
        f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
        conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
        conv = dx * conv

        # Soft-max surrogate: (1/beta) * log(sum(exp(beta * |conv|)))
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        exp_vals = np.exp(beta * (abs_conv - max_val))
        soft_max = max_val + np.log(np.sum(exp_vals)) / beta

        # Gradient of soft-max w.r.t. h
        # d(soft_max)/dh[i] = sum_k (d|conv[k]|/dh[i]) * exp(beta*|conv[k]|) / sum(exp(beta*|conv|))
        # This is complex; use numerical gradient for simplicity
        eps = 1e-4
        grad = np.zeros_like(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] -= eps

            # Compute soft-max for perturbed
            for h_pert, sign in [(h_plus, 1), (h_minus, -1)]:
                hp_pad = np.zeros(2 * n - 1)
                hp_pad[:n] = h_pert
                fp_fft = np.fft.fft(hp_pad, n=2 * (2 * n - 1))
                conv_pert = np.fft.ifft(fp_fft * np.conj(fp_fft)).real * dx
                abs_pert = np.abs(conv_pert)
                max_pert = np.max(abs_pert)
                exp_pert = np.exp(beta * (abs_pert - max_pert))
                soft_max_pert = max_pert + np.log(np.sum(exp_pert)) / beta
                if sign == 1:
                    grad[i] = (soft_max_pert - soft_max) / eps
                else:
                    grad[i] = (soft_max - soft_max_pert) / eps

        # Update
        h -= lr * grad
        # Project to nearest signed values (keep magnitudes, update signs)
        h = np.sign(h) * np.ones_like(h)

    return h


def _logistic_map_sequence(n, r, x0, skip=1000):
    """
    Generate a sequence of signs from a logistic map in chaotic regime.
    x_{k+1} = r * x_k * (1 - x_k)
    Sign is +1 if x_k > 0.5, else -1.
    """
    x = np.zeros(n + skip)
    x[0] = x0
    for k in range(skip + n):
        if k < skip + n - 1:
            x[k+1] = r * x[k] * (1.0 - x[k])
    # Map to signs: threshold at 0.5
    signs = np.where(x[skip:skip+n] > 0.5, 1.0, -1.0)
    return signs


def _generate_logistic_initializations(num_trials, n=N):
    """
    Generate multiple initializations from logistic maps with different parameters.
    Focus on chaotic regime (r in [3.5, 4.0]) with various initial conditions.
    """
    sequences = []
    # Chaotic parameter values
    r_values = [3.5, 3.6, 3.7, 3.8, 3.9, 3.95, 3.99, 4.0]
    x0_values = [0.1, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.9]
    
    idx = 0
    for r in r_values:
        for x0 in x0_values:
            if idx >= num_trials:
                break
            seq = _logistic_map_sequence(n, r, x0, skip=1000)
            sequences.append(seq)
            idx += 1
        if idx >= num_trials:
            break
    
    return sequences


def construct():
    start_time = time.time()

    # Initialize with random signs
    np.random.seed(42)
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)

    # Try several random initializations
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()

    # Try logistic map initializations (chaotic sequences)
    if time.time() - start_time < TIME_LIMIT - 30:
        logistic_seqs = _generate_logistic_initializations(num_trials=32)
        for seq in logistic_seqs:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            bound = _compute_c3_bound(seq)
            if bound < best_bound:
                best_bound = bound
                best_h = seq.copy()

    # Local search from best initialization
    if time.time() - start_time < TIME_LIMIT - 20:
        best_h, best_bound = _fast_local_search(best_h, max_iter=300, time_budget=40.0)

    # Pairwise swap search
    if time.time() - start_time < TIME_LIMIT - 15:
        best_h, best_bound = _pairwise_swap_search(best_h, max_iter=30, time_budget=20.0)

    # Gradient-based smooth optimization from best
    if time.time() - start_time < TIME_LIMIT - 10:
        h_smooth = _gradient_descent_smooth(best_h, steps=40, lr=0.04, beta=8.0, time_budget=20.0)
        bound_smooth = _compute_c3_bound(h_smooth)
        if bound_smooth < best_bound:
            best_bound = bound_smooth
            best_h = h_smooth.copy()
            # Follow up with local search
            if time.time() - start_time < TIME_LIMIT - 5:
                best_h, best_bound = _fast_local_search(best_h, max_iter=150, time_budget=15.0)

    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            if bound < best_bound:
                h, bound = _fast_local_search(h, max_iter=100, time_budget=10.0)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()

    # Final check
    final_bound = _compute_c3_bound(best_h)
    if final_bound < best_bound:
        best_bound = final_bound

    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")

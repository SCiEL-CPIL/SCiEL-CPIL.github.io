"""Rewritten program: minimize the peak autoconvolution of a 400-step function via
gradient descent on the normalized heights, using a smooth soft-max surrogate for the
non-smooth max in the objective."""
import numpy as np
import time

N = 400
TIME_LIMIT = 120.0  # seconds; leave margin for evaluator overhead


def _compute_value(h):
    """Compute the evaluator's value for a given h."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))


def _soft_max(x, temperature=1.0):
    """Numerically stable soft-max (log-sum-exp style) for a 1D array."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp((x - m) / temperature)) * temperature)


def _soft_max_grad(x, temperature=1.0):
    """Gradient of soft-max w.r.t. x."""
    m = np.max(x)
    shifted = (x - m) / temperature
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    return weights / temperature


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5.0  # 5s safety margin

    # Start from a random initialization, normalized
    rng = np.random.default_rng(42)
    h = rng.standard_normal(N)
    h = h - np.mean(h)  # zero-mean to avoid trivial scaling issues
    if np.sum(h) == 0:
        h = np.ones(N)

    best_h = h.copy()
    best_val = _compute_value(best_h)

    # Multi-stage gradient descent with decreasing temperature
    # We minimize: soft_max(convolve(h, h)) / sum(h)^2  (up to the 2*N factor)
    # Actually, we minimize: max(conv(h,h)) / sum(h)^2, using soft-max as surrogate.

    # We'll use Adam optimizer for better convergence
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros(N)  # first moment
    v = np.zeros(N)  # second moment
    t = 0

    # Schedule of temperatures for soft-max
    temperatures = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    steps_per_temp = 2000

    for temp in temperatures:
        for step in range(steps_per_temp):
            if time.time() > deadline:
                break

            t += 1
            s = np.sum(h)
            if s == 0 or not np.all(np.isfinite(h)):
                h = best_h.copy()
                s = np.sum(h)
                continue

            conv = np.convolve(h, h)
            # Objective: soft_max(conv) / s^2
            # Gradient w.r.t. h:
            # d/dh [soft_max(conv) / s^2] = (grad_soft_max(conv) @ dconv/dh) / s^2 - 2 * soft_max(conv) / s^3 * dh/ds
            # dconv/dh is a Toeplitz-like operation. conv[i] = sum_j h[j]*h[i-j]
            # d(conv[k])/dh[j] = h[k-j] + h[j-k] (for full convolution)
            # So d(conv)/dh is a 2N-1 x N matrix. We need the gradient of soft_max w.r.t. conv,
            # then back-propagate through conv.

            # soft_max gradient w.r.t. conv:
            sm_grad = _soft_max_grad(conv, temperature=temp)  # shape (2N-1,)

            # Back-propagate: grad_h[k] = sum_i sm_grad[i] * d(conv[i])/dh[k]
            # d(conv[i])/dh[k] = h[i-k] if 0 <= i-k < N else 0, plus h[k-i] if 0 <= k-i < N else 0
            # More carefully: conv[i] = sum_j h[j] * h[i-j] where i ranges 0..2N-2
            # d(conv[i])/dh[k] = h[i-k] (if 0<=i-k<N) + h[k-i] (if 0<=k-i<N, i.e., k>=i)
            # Actually, conv[i] = sum_{j=0}^{N-1} h[j] h[i-j] where h is zero-padded outside [0,N-1]
            # So d(conv[i])/dh[k] = h[i-k] (when i-k in [0,N-1]) + h[k-i] (when k-i in [0,N-1])
            # But note: if i-k == k-i, i.e., i=2k, then it's 2*h[0]... no wait.
            # Let me be more careful. conv[i] = sum_j h[j] h[i-j].
            # d(conv[i])/dh[k] = h[i-k] * delta(j=k in first term) + h[j] * delta(i-j=k) in second term
            # = h[i-k] (if 0<=i-k<N) + h[i-k] (if 0<=i-k<N, from second term where j=i-k)
            # Wait, no. Let me redo.
            # conv[i] = sum_{j=0}^{N-1} h[j] * h[i-j]
            # d(conv[i])/dh[k] = sum_j [delta(j=k)*h[i-j] + h[j]*delta(i-j=k)]
            # = h[i-k] (if 0<=i-k<N) + h[i-k] (if 0<=i-k<N)
            # = 2*h[i-k] if 0<=i-k<N, else 0? No, that's not right either.
            #
            # Actually: d/dh[k] of h[j]*h[i-j]:
            # If j=k: h[i-k]
            # If i-j=k, i.e., j=i-k: h[i-k]
            # These are the same term when j=k=i-k, i.e., i=2k.
            # So d(conv[i])/dh[k] = h[i-k] (if 0<=i-k<N) + h[i-k] (if 0<=i-k<N) but if i=2k, it's just h[0]... 
            #
            # Let me just use the fact that the gradient of the full convolution can be computed as:
            # grad_h = np.convolve(sm_grad, h, mode='full') + np.convolve(sm_grad[::-1], h, mode='full')
            # No, that's not right either. Let me think differently.
            #
            # The convolution is: conv[i] = sum_j h[j] h[i-j]
            # This is the autocorrelation. The gradient of sum_i g[i] conv[i] w.r.t. h[k] is:
            # sum_i g[i] * (h[i-k] + h[k-i]) where the terms are zero when indices are out of bounds.
            #
            # So grad_h[k] = sum_i g[i] h[i-k] + sum_i g[i] h[k-i]
            # = (g * h)[k] + (g_rev * h)[k] where g is zero-padded and we use appropriate convolutions.
            #
            # More precisely, if we define g as a vector of length 2N-1, then:
            # sum_i g[i] h[i-k] for k in [0,N-1]: this is like a cross-correlation.
            # Let's just compute it directly for correctness, since N=400 is small.

            # Direct computation of gradient
            grad_h = np.zeros(N)
            for k in range(N):
                # sum_i sm_grad[i] * h[i-k] where 0<=i-k<N, i.e., k<=i<k+N
                # and sum_i sm_grad[i] * h[k-i] where 0<=k-i<N, i.e., 0<=i<=k
                term1 = 0.0
                term2 = 0.0
                # term1: i from k to min(k+N-1, 2N-2)
                i_start = k
                i_end = min(k + N, 2 * N - 1)
                if i_start < i_end:
                    term1 = np.dot(sm_grad[i_start:i_end], h[0:i_end - i_start])
                # term2: i from 0 to k
                if k >= 0:
                    term2 = np.dot(sm_grad[0:k + 1], h[k:-1:-1] if k > 0 else h[k:k+1])
                # Wait, h[k-i] for i=0..k is h[k], h[k-1], ..., h[0]
                # So term2 = sum_{i=0}^{k} sm_grad[i] * h[k-i] = dot(sm_grad[0:k+1], h[k::-1])
                # But h[k::-1] is h[k], h[k-1], ..., h[0]
                grad_h[k] = term1 + term2

            # Normalize by s^2 and subtract the 2/s^3 * soft_max * (1/s) * sum term
            # d/dh [f/h_sum^2] where f = soft_max(conv)
            # = (grad_f * s^2 - f * 2 * s * 1) / s^4 = grad_f/s^2 - 2*f/(s^3)
            # where 1 is the vector of ones (d(sum h)/dh = 1)
            sm_val = _soft_max(conv, temperature=temp)
            grad_obj = grad_h / (s * s) - 2.0 * sm_val / (s * s * s) * np.ones(N)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad_obj
            v = beta2 * v + (1 - beta2) * grad_obj ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Check validity
            if not np.all(np.isfinite(h_new)) or np.sum(h_new) == 0:
                continue

            h = h_new
            val = _compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()

            # Adaptive learning rate: decrease if stuck
            if step % 500 == 499:
                lr *= 0.95

        if time.time() > deadline:
            break

    # Final polish: try a few random perturbations around best_h
    if time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            perturbation = rng.standard_normal(N) * 0.01
            h_trial = best_h + perturbation
            val_trial = _compute_value(h_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_h = h_trial.copy()

    return best_h

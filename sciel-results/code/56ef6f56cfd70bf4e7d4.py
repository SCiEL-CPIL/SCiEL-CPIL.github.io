"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(h):
    """Compute circular autocorrelation A(k) = sum_i h[i]*h[(i+k)%N] for all k."""
    n = len(h)
    # Use FFT for efficiency
    H = np.fft.rfft(h, n=2 * n)
    acf = np.fft.irfft(H * np.conj(H), n=2 * n)[:n]
    return acf


def _max_overlap(h):
    """Compute max overlap O(k) = sum_i h[i]*(1-h[(i+k)%N])."""
    n = len(h)
    acf = _autocorr(h)
    # O(k) = sum h[i] - acf(k) = N/2 - acf(k)
    return np.max(N / 2.0 - acf)


def _min_autocorr(h):
    """Compute minimum autocorrelation (excluding k=0)."""
    acf = _autocorr(h)
    return np.min(acf[1:])


def _project(h):
    """Project onto {h in [0,1]^N : sum h = N/2}."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to satisfy sum constraint
    target = N / 2.0
    s = np.sum(h)
    if s != target:
        # Add/subtract uniformly from elements that can absorb the change
        diff = target - s
        if diff > 0:
            # Need to increase sum; add to elements < 1
            mask = h < 1.0
            if np.any(mask):
                room = 1.0 - h[mask]
                total_room = np.sum(room)
                if total_room > 0:
                    add = np.minimum(diff, total_room)
                    h[mask] += add * room / total_room
                # If still short, just set to 1 where possible
                if np.sum(h) < target - 1e-10:
                    h[mask] = 1.0
            # If still not enough, set some to 1
            if np.sum(h) < target - 1e-10:
                order = np.argsort(-h)
                for idx in order:
                    if np.sum(h) >= target - 1e-10:
                        break
                    h[idx] = 1.0
        else:
            # Need to decrease sum; subtract from elements > 0
            mask = h > 0.0
            if np.any(mask):
                room = h[mask]
                total_room = np.sum(room)
                if total_room > 0:
                    sub = np.minimum(-diff, total_room)
                    h[mask] -= sub * room / total_room
                if np.sum(h) > target + 1e-10:
                    h[mask] = 0.0
            if np.sum(h) > target + 1e-10:
                order = np.argsort(h)
                for idx in order:
                    if np.sum(h) <= target + 1e-10:
                        break
                    h[idx] = 0.0
    # Final clip and exact sum fix
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if abs(s - target) > 1e-12:
        # Distribute the tiny residual
        h += (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)

    # Subgradient descent to maximize min autocorrelation
    # Using smooth surrogate: -log(sum exp(-beta * A(k))) for k=1..N-1
    h = np.full(N, 0.5)
    beta = 50.0
    lr = 0.01
    n_iter = 50000

    for it in range(n_iter):
        if time.time() > deadline - 5.0:
            break

        acf = _autocorr(h)
        acf_rest = acf[1:]  # exclude k=0

        # Smooth min: -log(sum exp(-beta * acf_rest))
        # Gradient w.r.t. h: need d/dh of this objective
        # Let f = -log(sum_k exp(-beta * A(k)))
        # df/dh[i] = -sum_k w_k * dA(k)/dh[i] where w_k = exp(-beta*A(k)) / sum exp(-beta*A(k))
        # A(k) = sum_j h[j]*h[(j+k)%N]
        # dA(k)/dh[i] = h[(i-k)%N] + h[(i+k)%N] (for k != 0)

        # Compute weights
        shifted = -beta * acf_rest
        shifted -= np.max(shifted)  # numerical stability
        exp_vals = np.exp(shifted)
        w = exp_vals / np.sum(exp_vals)  # weights for k=1..N-1

        # Gradient of f w.r.t. h
        # f = -log(sum exp(-beta*A(k)))
        # df/dh[i] = sum_k w_k * beta * (h[(i-k)%N] + h[(i+k)%N])
        # We want to MAXIMIZE min A(k), so we maximize f (which approximates min A)
        # Gradient ascent on f

        grad = np.zeros(N)
        for k in range(1, N):
            wk = w[k - 1] * beta
            # dA(k)/dh[i] = h[(i-k)%N] + h[(i+k)%N]
            grad += wk * (np.roll(h, k) + np.roll(h, -k))

        # Projected gradient ascent
        h_new = h + lr * grad
        h_new = _project(h_new)

        # Check improvement
        cur_overlap = _max_overlap(h_new)
        if cur_overlap < best_score - 1e-10:
            best_score = cur_overlap
            best_h = h_new.copy()

        h = h_new

        # Adaptive learning rate
        if it % 5000 == 0 and it > 0:
            lr *= 0.8
            beta *= 1.5

    # Final polish: local optimization with tighter tolerance
    if time.time() < deadline - 10.0:
        h = best_h.copy()
        lr2 = 0.001
        for it in range(10000):
            if time.time() > deadline - 5.0:
                break
            acf = _autocorr(h)
            acf_rest = acf[1:]

            # Find the worst autocorrelation (minimum)
            min_idx = np.argmin(acf_rest) + 1  # k value

            # Subgradient of min A(k) at the active constraint
            # dA(k)/dh[i] = h[(i-k)%N] + h[(i+k)%N]
            grad = np.zeros(N)
            k = min_idx
            grad = np.roll(h, k) + np.roll(h, -k)

            # Also consider nearby k values for robustness
            for dk in [-2, -1, 1, 2]:
                k2 = (min_idx + dk) % N
                if k2 == 0:
                    continue
                # Weight by how close to minimum
                gap = acf_rest[min_idx - 1] - acf_rest[k2 - 1]
                if gap < 0.01:
                    grad += 0.5 * (np.roll(h, k2) + np.roll(h, -k2))

            h_new = h + lr2 * grad
            h_new = _project(h_new)

            cur_overlap = _max_overlap(h_new)
            if cur_overlap < best_score - 1e-12:
                best_score = cur_overlap
                best_h = h_new.copy()

            h = h_new

    # Ensure exact sum
    best_h = _project(best_h)

    return best_h
# EVOLVE-BLOCK-END

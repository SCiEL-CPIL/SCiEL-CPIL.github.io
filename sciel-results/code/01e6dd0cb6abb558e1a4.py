"""Spectral + soft-max optimizer for the Erdos minimum-overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
# Number of shift samples to approximate the continuous max overlap
M = 4 * N

_start = None

def _time_remaining(deadline):
    return max(0.0, deadline - time.time())

def _overlap_grid(h):
    """Compute the overlap O(t) = sum_j h_j * (1 - h_{j+t}) at M shift samples.
    Uses a linear-interpolated (continuous) shift for step functions.
    Returns array of length M."""
    h = np.asarray(h, dtype=float)
    N = len(h)
    # For shift s (in step units, 0 <= s < N), the shifted value at position j is
    # linear interpolation: h_{j+s} = (1-f)*h_{j+floor(s)} + f*h_{j+ceil(s)}, f = s - floor(s)
    # O(s) = (2/N) * sum_j h_j * (1 - h_{j+s})
    # We sample s = m * N / M for m in range(M).
    s_vals = np.arange(M) * (N / M)
    overlaps = np.empty(M)
    for idx, s in enumerate(s_vals):
        i0 = int(np.floor(s)) % N
        f = s - np.floor(s)
        i1 = (i0 + 1) % N
        if f < 1e-12:
            shifted = np.roll(h, -i0)
        else:
            shifted = (1 - f) * np.roll(h, -i0) + f * np.roll(h, -i1)
        overlaps[idx] = np.dot(h, 1.0 - shifted)
    return overlaps

def _max_overlap(h):
    return float(np.max(_overlap_grid(h)))

def _project(h):
    """Project h onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    s = np.sum(h)
    if s <= 0:
        h[:] = 0.5
        return h
    # Scale to hit target, then clip and fix residual
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if diff > 1e-12:
        idx = np.where(h < 1.0)[0]
        if len(idx) > 0:
            # distribute
            add = diff / len(idx)
            h[idx] = np.clip(h[idx] + add, 0.0, 1.0)
    elif diff < -1e-12:
        idx = np.where(h > 0.0)[0]
        if len(idx) > 0:
            add = diff / len(idx)
            h[idx] = np.clip(h[idx] + add, 0.0, 1.0)
    # final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.where(h < 1.0)[0]
        else:
            idx = np.where(h > 0.0)[0]
        if len(idx) > 0:
            h[idx[0]] += diff
            h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max(overlaps, beta):
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp((overlaps - m) * beta)) / len(overlaps)) / beta

def construct():
    global _start
    _start = time.time()
    deadline = _start + 140.0  # safety margin before 150s kill

    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)

    def objective(h):
        ov = _overlap_grid(h)
        return _soft_max(ov, beta=40.0)

    def objective_grad(h):
        # subgradient of the soft-max; approximate via finite differences of max is expensive.
        # Use the overlap at the argmax shift for a subgradient of the max.
        ov = _overlap_grid(h)
        k = int(np.argmax(ov))
        s = k * (N / M)
        i0 = int(np.floor(s)) % N
        f = s - np.floor(s)
        i1 = (i0 + 1) % N
        if f < 1e-12:
            shifted = np.roll(h, -i0)
            # d O / d h_j = (1 - h_{j+s}) - h_{j-s}  (for the product, careful)
            # O = sum_j h_j (1 - h_{j+s}); dO/dh_j = (1 - h_{j+s}) - h_{j-s}
            grad = (1.0 - np.roll(h, -i0)) - np.roll(h, i0)
        else:
            shifted = (1 - f) * np.roll(h, -i0) + f * np.roll(h, -i1)
            # dO/dh_j = (1 - shifted_j) - (1-f)*h_{j-i0} - f*h_{j-i1}
            grad = (1.0 - shifted) - (1 - f) * np.roll(h, i0) - f * np.roll(h, i1)
        return grad

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0,
                    'jac': lambda h: np.ones(N)}]

    # Initializations
    inits = [np.full(N, 0.5)]
    # random binary balanced
    rng = np.random.default_rng(0)
    for _ in range(3):
        b = (rng.random(N) < 0.5).astype(float)
        # balance to exactly N/2 ones
        ones = np.sum(b)
        if ones < N / 2:
            idx0 = np.where(b == 0)[0]
            b[idx0[:int(N/2 - ones)]] = 1
        elif ones > N / 2:
            idx1 = np.where(b == 1)[0]
            b[idx1[:int(ones - N/2)]] = 0
        inits.append(b)
    # smooth sinusoidal perturbation
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.clip(0.5 + 0.3*np.sin(t), 0, 1))
    inits.append(np.clip(0.5 + 0.3*np.cos(2*t), 0, 1))

    for h0 in inits:
        if _time_remaining(deadline) < 8.0:
            break
        h0 = _project(h0)
        # SLSQP
        try:
            res = minimize(objective, h0, method='SLSQP', jac=objective_grad,
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-10})
            hc = _project(res.x)
            v = _max_overlap(hc)
            if v < best_val - 1e-9:
                best_val = v
                best_h = hc.copy()
        except Exception:
            pass
        # L-BFGS-B with penalty
        if _time_remaining(deadline) < 6.0:
            break
        def obj_pen(h):
            ov = _overlap_grid(h)
            sm = _soft_max(ov, beta=40.0)
            pen = 1000.0 * (np.sum(h) - N/2.0)**2
            return sm + pen
        try:
            res = minimize(obj_pen, best_h.copy(), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            hc = _project(res.x)
            v = _max_overlap(hc)
            if v < best_val - 1e-9:
                best_val = v
                best_h = hc.copy()
        except Exception:
            pass

    # Local polish: coordinate descent on the argmax shift
    if _time_remaining(deadline) > 5.0:
        for it in range(400):
            if _time_remaining(deadline) < 3.0:
                break
            ov = _overlap_grid(best_h)
            k = int(np.argmax(ov))
            s = k * (N / M)
            i0 = int(np.floor(s)) % N
            f = s - np.floor(s)
            i1 = (i0 + 1) % N
            if f < 1e-12:
                grad = (1.0 - np.roll(best_h, -i0)) - np.roll(best_h, i0)
            else:
                shifted = (1 - f) * np.roll(best_h, -i0) + f * np.roll(best_h, -i1)
                grad = (1.0 - shifted) - (1 - f) * np.roll(best_h, i0) - f * np.roll(best_h, i1)
            for lr in (1e-3, 5e-4, 1e-4, 5e-5):
                h_new = _project(best_h - lr * grad)
                v = _max_overlap(h_new)
                if v < best_val - 1e-12:
                    best_val = v
                    best_h = h_new
                    break
    return best_h

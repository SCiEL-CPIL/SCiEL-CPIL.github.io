import numpy as np
from scipy.optimize import minimize
import time

N = 100

def objective(h, beta=100.0):
    # Soft max approximation using log-sum-exp for numerical stability
    corr = np.correlate(h, 1.0 - h, mode='full')
    # Shift to prevent overflow in exp
    c_max = np.max(corr)
    if c_max == 0:
        return 0.0
    # log(sum(exp(beta * corr))) / beta approximates max(corr)
    # To keep numbers manageable, we can scale
    scaled = beta * (corr - c_max)
    # Avoid overflow in exp
    exp_scaled = np.exp(scaled)
    log_sum = np.log(np.sum(exp_scaled))
    return (c_max + log_sum / beta)

def project_to_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin under 150s kill

    best_h = np.full(N, 0.5)
    # Use the true max for evaluation of the best candidate
    best_val = np.max(np.correlate(best_h, 1.0 - best_h, mode='full'))

    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]

    rng = np.random.RandomState(12345)
    starts = [np.full(N, 0.5)]

    # (A) Direct spreading patterns
    # Alternating 0.25 / 0.75
    starts.append(np.where(np.arange(N) % 2 == 0, 0.25, 0.75).copy())

    # Two-block spread
    for m in [10, 20, 25, 30, 40]:
        h = np.zeros(N)
        idx = np.linspace(0, N - 1, m).astype(int)
        h[idx] = 1.0
        s = h.sum()
        if s > 0:
            h = h * (N / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
        h = project_to_feasible(h)
        starts.append(h.copy())

    # Sine-spreading (low-frequency so mass is distributed, not clumped)
    # Add more frequencies and amplitudes
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.3, 0.45, 0.49]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            starts.append(project_to_feasible(h))
            
            # Also try cosine
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * np.arange(N) / N)
            starts.append(project_to_feasible(h_cos))

    # Random smooth seeds (using low-pass filtered noise)
    for _ in range(8):
        noise = rng.randn(N)
        # Simple moving average smoothing
        kernel = np.ones(5) / 5.0
        smooth_noise = np.convolve(noise, kernel, mode='same')
        h_pert = 0.5 + 0.4 * smooth_noise
        starts.append(project_to_feasible(h_pert))

    # Two-level: first 50 ones (block baseline)
    h_bin = np.zeros(N)
    h_bin[:50] = 1.0
    starts.append(h_bin.copy())

    # Linear ramp
    starts.append(project_to_feasible(np.linspace(0.0, 1.0, N)))

    # (B) Refine each seed with the optimizer using soft-max objective
    # We use a lambda to capture beta
    beta = 50.0 # Moderate sharpness
    obj_func = lambda x: objective(x, beta=beta)
    
    for h0 in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                obj_func,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False}
            )
            h_res = project_to_feasible(res.x)
            # Evaluate with TRUE max
            val = np.max(np.correlate(h_res, 1.0 - h_res, mode='full'))
            if val < best_val - 1e-9:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass

    # Final polish on the best found
    # Try a few more iterations on the best_h with a higher beta (sharper)
    if time.time() < deadline - 5:
        try:
            beta_sharp = 200.0
            obj_sharp = lambda x: objective(x, beta=beta_sharp)
            res = minimize(
                obj_sharp,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
            )
            h_res = project_to_feasible(res.x)
            val = np.max(np.correlate(h_res, 1.0 - h_res, mode='full'))
            if val < best_val - 1e-12:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass

    # Final projection to guarantee validity.
    best_h = project_to_feasible(best_h)
    # Enforce the sum tolerance strictly.
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 0:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    best_h = project_to_feasible(best_h)

    return best_h

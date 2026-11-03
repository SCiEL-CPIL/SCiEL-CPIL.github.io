"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Build a diverse pool of 'extreme' nonnegative basis vectors (cosine bumps,
   rectangles, Gaussians, plateaus, and random smooth shapes).
2. Represent candidate step functions as convex combinations of these bases
   (weights >= 0, sum = 1).
3. Optimize the weights with a fast, bounded stochastic search (differential
   evolution / Nelder-Mead hybrids) using a smooth log-sum-exp surrogate for
   the max of the autoconvolution.
4. Convert the best convex combination back to a 600-bin step function.
5. Apply sensitivity-based local refinement and a high-precision polish pass
   directly on the 600-bin vector.
6. Deadline-guarded: always return the best result found before the safety
   margin.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill
X_GRID = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N


def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)


def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max + log_sum


def build_basis_pool(rng, n_bases=24):
    """Build a diverse pool of nonnegative basis vectors of length N."""
    bases = []
    x = X_GRID

    # 1) Centered cosine bumps with varying widths
    for w in [0.08, 0.12, 0.16, 0.20, 0.25, 0.30]:
        d = np.abs(x - 0.0)
        mask = d <= w / 2.0
        b = np.zeros(N)
        b[mask] = np.cos(np.pi * x[mask] / w) ** 2
        if np.sum(b) > 1e-12:
            bases.append(b / np.sum(b))

    # 2) Rectangular (box) functions
    for hw in [0.05, 0.08, 0.10, 0.125, 0.15, 0.20]:
        b = np.zeros(N)
        b[np.abs(x) <= hw] = 1.0
        if np.sum(b) > 1e-12:
            bases.append(b / np.sum(b))

    # 3) Gaussian bumps
    for sigma in [0.02, 0.04, 0.06, 0.08, 0.10]:
        b = np.exp(-0.5 * (x / sigma) ** 2)
        if np.sum(b) > 1e-12:
            bases.append(b / np.sum(b))

    # 4) Plateau with smooth edges
    for inner in [0.04, 0.08, 0.12]:
        for edge in [0.02, 0.04]:
            d = np.abs(x)
            b = np.zeros(N)
            b[d <= inner] = 1.0
            mask = (d > inner) & (d <= inner + edge)
            b[mask] = 0.5 * (1 + np.cos(np.pi * (d[mask] - inner) / edge))
            if np.sum(b) > 1e-12:
                bases.append(b / np.sum(b))

    # 5) Off-center cosine bumps
    for c in [-0.1, -0.05, 0.05, 0.1]:
        for w in [0.12, 0.18]:
            d = np.abs(x - c)
            mask = d <= w / 2.0
            b = np.zeros(N)
            b[mask] = np.cos(np.pi * x[mask] / w) ** 2
            if np.sum(b) > 1e-12:
                bases.append(b / np.sum(b))

    # 6) Random smooth shapes (low-frequency random Fourier series)
    for _ in range(8):
        coeffs = rng.normal(size=5)
        b = np.zeros(N)
        for k, c in enumerate(coeffs, start=1):
            b += c * np.cos(2 * np.pi * k * (x + 0.25) / 0.5)
        b = np.abs(b) + 0.1
        if np.sum(b) > 1e-12:
            bases.append(b / np.sum(b))

    # Trim to requested size, prefer diversity by keeping first n_bases
    bases = bases[:n_bases]

    # Ensure all are nonnegative and normalized to sum 1
    processed = []
    for b in bases:
        b = np.maximum(b, 0.0)
        s = np.sum(b)
        if s > 1e-12:
            processed.append(b / s)
    return np.array(processed)


def weights_to_step(weights, basis_pool, scale=1.0):
    """Convert convex weights to a step function of length N."""
    w = np.maximum(weights, 0.0)
    s = np.sum(w)
    if s < 1e-15:
        return np.zeros(N)
    w = w / s
    f = w @ basis_pool  # shape (N,)
    f = np.maximum(f, 0.0)
    # Scale to a reasonable amplitude (score is scale-invariant, but keep
    # values in a stable range for optimization)
    return f * scale


def objective_weights(weights, basis_pool, beta=20.0):
    """Objective: smooth max of autoconvolution normalized by (sum)^2."""
    f = weights_to_step(weights, basis_pool)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)


def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init

    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)

        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)

        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]

        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]

        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std

        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)

        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)

        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)

        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
            if alpha < 1e-8:
                break

    return best_h, best_val


def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02

    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)

        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)

        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

        threshold = max_c * 0.99
        peak_indices = np.where(conv >= threshold)[0]

        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]

        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std

        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)

        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)

        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)

        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break

    return best_h, best_val


def differential_evolution_weights(basis_pool, n_bases, start_time, max_time,
                                   pop_size=30, beta=20.0, seed=42):
    """Bounded differential evolution over simplex weights."""
    rng = np.random.RandomState(seed)
    deadline = start_time + max_time

    # Initialize population on the simplex
    def random_simplex(rng, n):
        w = rng.dirichlet(np.ones(n))
        return w

    pop = np.array([random_simplex(rng, n_bases) for _ in range(pop_size)])
    best_w = pop[0].copy()
    best_val = objective_weights(best_w, basis_pool, beta)

    # Sort pop by objective
    vals = np.array([objective_weights(w, basis_pool, beta) for w in pop])
    order = np.argsort(vals)
    pop = pop[order]
    vals = vals[order]
    best_w = pop[0].copy()
    best_val = vals[0]

    F = 0.8
    CR = 0.9
    gen = 0
    while time.time() < deadline:
        gen += 1
        # Update F and CR adaptively
        F = max(0.3, 0.8 * (1.0 - gen / 50.0))
        CR = min(0.95, 0.7 + 0.3 * (gen / 50.0))

        new_pop = pop.copy()
        new_vals = vals.copy()

        for i in range(pop_size):
            if time.time() > deadline:
                break
            # Select three distinct individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(pop_size, size=3, replace=False)

            v = pop[a] + F * (pop[b] - pop[c])
            # Project to simplex: clip negatives, renormalize
            v = np.maximum(v, 0.0)
            sv = np.sum(v)
            if sv < 1e-15:
                v = random_simplex(rng, n_bases)
            else:
                v = v / sv

            # Crossover
            trial = pop[i].copy()
            for j in range(n_bases):
                if rng.random() < CR or j == rng.randint(n_bases):
                    trial[j] = v[j]
            # Ensure simplex
            trial = np.maximum(trial, 0.0)
            st = np.sum(trial)
            if st < 1e-15:
                trial = random_simplex(rng, n_bases)
            else:
                trial = trial / st

            trial_val = objective_weights(trial, basis_pool, beta)
            if trial_val < new_vals[i]:
                new_pop[i] = trial
                new_vals[i] = trial_val

        pop = new_pop
        vals = new_vals
        order = np.argsort(vals)
        pop = pop[order]
        vals = vals[order]
        if vals[0] < best_val:
            best_val = vals[0]
            best_w = pop[0].copy()

        # Early stopping if no improvement in last 10 generations
        if gen > 10 and vals[0] >= best_val * (1.0 - 1e-9):
            if gen % 10 == 0 and gen > 20:
                break

    return best_w, best_val


def construct():
    start_time = time.time()

    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)

    # Build basis pool
    rng = np.random.RandomState(42)
    basis_pool = build_basis_pool(rng, n_bases=24)
    n_bases = basis_pool.shape[0]

    # Phase 1: Differential evolution over convex weights
    if time.time() - start_time < TIME_LIMIT - 50:
        de_w, de_val = differential_evolution_weights(
            basis_pool, n_bases, start_time, TIME_LIMIT - 50,
            pop_size=30, beta=20.0, seed=42
        )
        f_de = weights_to_step(de_w, basis_pool)
        actual_score = compute_score(f_de)
        if actual_score < best_score:
            best_f = f_de
            best_score = actual_score

    # Phase 2: Nelder-Mead refinement on weights starting from DE result
    if time.time() - start_time < TIME_LIMIT - 30:
        # Convert best_f back to approximate weights via least squares
        A = basis_pool.T  # shape (n_bases, N)
        # Solve min ||A w - best_f||^2 s.t. w >= 0, sum w = 1
        # Use simple projection: start from DE weights
        w0 = de_w.copy()

        def obj_w(w):
            f = weights_to_step(w, basis_pool)
            return compute_score(f)

        try:
            res = minimize(
                obj_w, w0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            w_nm = np.maximum(res.x, 0.0)
            sw = np.sum(w_nm)
            if sw > 1e-15:
                w_nm = w_nm / sw
            f_nm = weights_to_step(w_nm, basis_pool)
            score_nm = compute_score(f_nm)
            if score_nm < best_score:
                best_f = f_nm
                best_score = score_nm
        except Exception:
            pass

    # Phase 3: Local refinement on the 600-bin vector
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 8
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    # Phase 4: Final polish pass
    if time.time() - start_time < TIME_LIMIT - 3:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 1
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score

    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]


if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")

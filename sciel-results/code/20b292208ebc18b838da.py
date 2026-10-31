import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _smooth_max(x, beta=100.0):
    m = np.max(x)
    if m < 0:
        m = 0
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _negative_ratio_smooth(params, beta=100.0):
    h = np.exp(params)
    h = h / np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = _smooth_max(conv, beta)
    if max_conv < 1e-15:
        return 0.0
    return -l2_sq / max_conv

def _exact_ratio(h):
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

# EVOLVE-BLOCK-START
def _sidon_indices(n, max_size=15):
    """Find a Sidon set in Z_n (no repeated pairwise sums)."""
    # Greedy: try to add indices that don't create duplicate sums
    selected = [0]
    sums = set()
    for i in range(1, n):
        # Check if adding i creates any duplicate sum
        valid = True
        for j in selected:
            s = (i + j) % n
            if s in sums:
                valid = False
                break
            sums.add(s)
        if valid and len(selected) < max_size:
            selected.append(i)
        if len(selected) >= max_size:
            break
    return selected

def _generate_sidon_init(n, max_size=12, seed=0):
    """Generate a sparse Sidon-based initialization."""
    h = np.zeros(n)
    indices = _sidon_indices(n, max_size)
    for i in indices:
        h[i] = 1.0
    if np.sum(h) > 0:
        h = h / np.sum(h)
    else:
        h[0] = 1.0
    return h

def _random_sparse_init(n, num_nonzero, seed=0):
    """Generate a random sparse initialization."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_nonzero, replace=False)
    h = np.zeros(n)
    for i in indices:
        h[i] = 1.0
    h = h / np.sum(h)
    return h

def _optimize_from_init(h_init, deadline, max_iters=200):
    """Run smooth optimization from a given initialization."""
    params_init = np.log(h_init + 1e-10)
    best_score = _exact_ratio(h_init)
    best_h = h_init.copy()

    # Progressive beta refinement
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 10:
            break

        result = minimize(
            _negative_ratio_smooth,
            params_init,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
        )

        h_result = np.exp(result.x)
        h_result = h_result / np.sum(h_result)

        score = _exact_ratio(h_result)
        if score > best_score:
            best_score = score
            best_h = h_result.copy()
            params_init = result.x

    return best_score, best_h

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin

    best_score = 0.0
    best_h = None

    # Multiple initialization strategies
    inits = []

    # 1. Sidon set (sparse, structurally optimal for low autoconv peak)
    h_sidon = _generate_sidon_init(N, max_size=12)
    inits.append(h_sidon)

    # 2. Larger Sidon set
    h_sidon_large = _generate_sidon_init(N, max_size=15)
    inits.append(h_sidon_large)

    # 3. Random sparse inits with different seeds
    for seed in [1, 2, 3]:
        h_rand = _random_sparse_init(N, num_nonzero=10, seed=seed)
        inits.append(h_rand)

    # 4. Uniform init
    h_uniform = np.ones(N) / N
    inits.append(h_uniform)

    # Try each initialization
    for i, h_init in enumerate(inits):
        if time.time() > deadline - 5:
            break

        score, h_opt = _optimize_from_init(h_init, deadline)

        if score > best_score:
            best_score = score
            best_h = h_opt.copy()

    # If no good result yet, fall back to basic Sidon
    if best_h is None:
        best_h = _generate_sidon_init(N, max_size=12)
        best_score = _exact_ratio(best_h)

    # Final polish: high-precision local optimization from best found
    if time.time() < deadline - 5:
        params_final = np.log(best_h + 1e-10)
        result = minimize(
            _negative_ratio_smooth,
            params_final,
            args=(300.0,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        h_polish = np.exp(result.x)
        h_polish = h_polish / np.sum(h_polish)
        score_polish = _exact_ratio(h_polish)
        if score_polish > best_score:
            best_score = score_polish
            best_h = h_polish.copy()

    # Ensure nonnegativity and return
    best_h = np.maximum(best_h, 0)

    return best_h.tolist()
# EVOLVE-BLOCK-END

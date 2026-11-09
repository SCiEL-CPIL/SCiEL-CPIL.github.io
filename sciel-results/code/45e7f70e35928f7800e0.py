# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS


def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    s = counts.sum()
    if s <= 1e-30:
        return 0.0
    q = counts / s
    q = q[q > 0.0]
    return float(-(q * np.log2(q)).sum())


def _compute_value(P):
    """Compute D/M ratio from a 10x10 probability matrix P."""
    p = P.ravel()
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    p = p / s
    
    # Distribution of x - y (values range from -9 to 9)
    d_counts = np.zeros(_N)
    np.add.at(d_counts, _XM1Y, p)
    hd = _entropy_from_counts(d_counts)
    
    # Distributions for M: H_0 (x), H_1 (x+y), H_2 (x+2y)
    # x ranges 0-9
    m0_counts = np.zeros(_N)
    np.add.at(m0_counts, _XS, p)
    hm0 = _entropy_from_counts(m0_counts)
    
    # x+y ranges 0-18
    m1_counts = np.zeros(2 * _N - 1)
    np.add.at(m1_counts, _XY, p)
    hm1 = _entropy_from_counts(m1_counts)
    
    # x+2y ranges 0-27
    m2_counts = np.zeros(2 * _N + _N - 1)
    np.add.at(m2_counts, _X2Y, p)
    hm2 = _entropy_from_counts(m2_counts)
    
    mmax = max(hm0, hm1, hm2)
    if mmax <= 1e-12:
        return 0.0
    return hd / mmax


def _build_from_theta(theta):
    """Build probability matrix from log-weights theta (10x10)."""
    w = np.exp(theta)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 1e-12:
        return np.full((_N, _N), 1.0 / (_N * _N))
    return w / s


def _gradient_ascent(theta0, lr=0.1, n_steps=200, tol=1e-8):
    """Perform gradient ascent on the D/M ratio using numerical gradients."""
    theta = theta0.copy()
    best_val = _compute_value(_build_from_theta(theta))
    best_theta = theta.copy()
    
    for step in range(n_steps):
        # Compute numerical gradient
        grad = np.zeros_like(theta)
        eps = 1e-5
        for i in range(_N):
            for j in range(_N):
                theta[i, j] += eps
                val_plus = _compute_value(_build_from_theta(theta))
                theta[i, j] -= 2 * eps
                val_minus = _compute_value(_build_from_theta(theta))
                theta[i, j] += eps
                grad[i, j] = (val_plus - val_minus) / (2 * eps)
        
        # Update
        theta = theta + lr * grad
        
        # Check convergence
        val = _compute_value(_build_from_theta(theta))
        if val > best_val:
            best_val = val
            best_theta = theta.copy()
        
        if abs(val - best_val) < tol:
            break
    
    return best_theta, best_val


def _gaussian_mixture(theta_centers, sigmas, weights):
    """Build a probability matrix from a Gaussian mixture model."""
    P = np.zeros((_N, _N))
    for center, sigma, weight in zip(theta_centers, sigmas, weights):
        y, x = center
        # 2D Gaussian
        dy = _YS.reshape(_N, _N) - y
        dx = _XS.reshape(_N, _N) - x
        r2 = dy**2 + dx**2
        P += weight * np.exp(-r2 / (2 * sigma**2))
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-12:
        return np.full((_N, _N), 1.0 / (_N * _N))
    return P / s


def _optimize_single_gaussian(center, sigma):
    """Optimize a single Gaussian using local search."""
    best_val = _compute_value(_gaussian_mixture([center], [sigma], [1.0]))
    best_params = (center, sigma)
    
    # Local search in center and sigma
    for _ in range(50):
        improved = False
        for dy in [-0.5, 0.5]:
            for dx in [-0.5, 0.5]:
                for ds in [-0.5, 0.5]:
                    new_center = (center[0] + dy, center[1] + dx)
                    new_sigma = max(0.5, sigma + ds)
                    val = _compute_value(_gaussian_mixture([new_center], [new_sigma], [1.0]))
                    if val > best_val:
                        best_val = val
                        best_params = (new_center, new_sigma)
                        improved = True
        if not improved:
            break
        center, sigma = best_params
    
    return best_params, best_val


def _optimize_mixture(n_components=3, max_iter=100):
    """Optimize a Gaussian mixture model."""
    rng = np.random.RandomState(42)
    
    # Initialize with random centers
    centers = [(rng.randint(0, _N), rng.randint(0, _N)) for _ in range(n_components)]
    sigmas = [rng.uniform(1.0, 3.0) for _ in range(n_components)]
    weights = np.ones(n_components) / n_components
    
    best_val = _compute_value(_gaussian_mixture(centers, sigmas, weights))
    best_state = (centers.copy(), sigmas.copy(), weights.copy())
    
    for _ in range(max_iter):
        # Perturb each component
        new_centers = []
        new_sigmas = []
        for i in range(n_components):
            c = centers[i]
            s = sigmas[i]
            new_c = (c[0] + rng.uniform(-1, 1), c[1] + rng.uniform(-1, 1))
            new_s = max(0.5, s + rng.uniform(-0.5, 0.5))
            new_centers.append(new_c)
            new_sigmas.append(new_s)
        
        # Perturb weights
        new_weights = weights + rng.uniform(-0.1, 0.1, n_components)
        new_weights = np.clip(new_weights, 0.01, None)
        new_weights /= new_weights.sum()
        
        val = _compute_value(_gaussian_mixture(new_centers, new_sigmas, new_weights))
        if val > best_val:
            best_val = val
            best_state = (new_centers.copy(), new_sigmas.copy(), new_weights.copy())
            centers, sigmas, weights = new_centers, new_sigmas, new_weights
    
    return best_state, best_val


def _local_search(P, n_iter=200):
    """Perform local search on the probability matrix."""
    best_P = P.copy()
    best_val = _compute_value(P)
    
    for _ in range(n_iter):
        improved = False
        for i in range(_N):
            for j in range(_N):
                # Try increasing
                if best_P[i, j] < 0.9:
                    new_P = best_P.copy()
                    new_P[i, j] *= 1.1
                    new_P /= new_P.sum()
                    val = _compute_value(new_P)
                    if val > best_val:
                        best_P = new_P
                        best_val = val
                        improved = True
                
                # Try decreasing
                if best_P[i, j] > 0.01:
                    new_P = best_P.copy()
                    new_P[i, j] *= 0.9
                    new_P /= new_P.sum()
                    val = _compute_value(new_P)
                    if val > best_val:
                        best_P = new_P
                        best_val = val
                        improved = True
        
        if not improved:
            break
    
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    t0 = time.time()
    DEADLINE = 595.0
    
    def guard():
        return time.time() - t0 < DEADLINE
    
    best_P = None
    best_val = 0.0
    
    # Strategy 1: Linear model with L-BFGS-B
    theta0 = np.zeros((_N, _N))
    for a in [0.2, 0.4, 0.6]:
        for b in [0.0, 0.2, 0.4]:
            if not guard():
                break
            th = a * _YS.reshape(_N, _N) + b * _XS.reshape(_N, _N)
            P = _build_from_theta(th)
            val = _compute_value(P)
            if val > best_val:
                best_val = val
                best_P = P.copy()
    
    # Strategy 2: Gradient ascent from best linear model
    if best_P is not None and guard():
        theta_best = np.log(best_P + 1e-10)
        theta_opt, val_opt = _gradient_ascent(theta_best, lr=0.05, n_steps=100)
        P_opt = _build_from_theta(theta_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
    
    # Strategy 3: Gaussian mixture
    if guard():
        state, val_gm = _optimize_mixture(n_components=2, max_iter=50)
        if val_gm > best_val:
            best_val = val_gm
            best_P = _gaussian_mixture(state[0], state[1], state[2])
    
    # Strategy 4: Local search refinement
    if best_P is not None and guard():
        P_refined, val_refined = _local_search(best_P, n_iter=100)
        if val_refined > best_val:
            best_val = val_refined
            best_P = P_refined
    
    # Final safety check
    if best_P is None:
        best_P = np.full((_N, _N), 1.0 / (_N * _N))
    
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s <= 1e-12:
        best_P = np.full((_N, _N), 1.0 / (_N * _N))
    else:
        best_P = best_P / s
    
    return best_P


# EVOLVE-BLOCK-END

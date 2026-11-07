import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s wall-clock

# Global state for adaptive beta
_N_EVALS = 0

# ---------- Cosine8 8d (f15 in the DRL/BOB suite) ----------
def _cosine8(x):
    """x shape (n, 8) in [-1,1]^8. Returns (n,) values; global min -0.8."""
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(-1, 8)
    out = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(8):
        out += (1.0 + np.sin(np.pi * x[:, i])) / (1.0 + i)
    base = sum(1.0 / (1.0 + i) for i in range(8))
    out = out - base
    out = out - 0.8
    return out

# ---------- Instance k: minimizer moved to uniform point in inner 80% ----------
def _instance_offset(k, dim=8):
    rng = np.random.RandomState(1000 + k)
    s = rng.uniform(-0.8, 0.8, size=dim)
    return s

def _f_instance(x, k):
    off = _instance_offset(k)
    return _cosine8(x - off)

# ---------- Cost model ----------
def _cost_of(x):
    scaled = (np.asarray(x, dtype=np.float64) + 1.0) / 2.0
    return 1.0 + np.mean(scaled, axis=-1)

# ---------- GP prediction utilities ----------
def _kernel_matern52(A, B, ls=0.2, noise=1e-6):
    A = (A + 1.0) / 2.0
    B = (B + 1.0) / 2.0
    d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    r = np.sqrt(np.maximum(d, 0.0)) / ls
    K = np.sqrt(5.0 * r) * np.exp(-r)
    if A.shape[0] == B.shape[0] and A.shape[0] == 1:
        pass
    elif A.shape[0] == B.shape[0]:
        K[np.arange(len(A)), np.arange(len(A))] += noise
    return K

def _gp_predict(X, Y, C):
    """
    X: (n, d) observed points
    Y: (n,) observed values
    C: (m, d) candidate points
    Returns mu: (m,), sigma: (m,) in objective units
    """
    mu_y = Y.mean()
    sd_y = Y.std() + 1e-9
    Ys = (Y - mu_y) / sd_y
    
    K = _kernel_matern52(X, C)
    Kss = np.diag(_kernel_matern52(C, C))
    
    Kxx = _kernel_matern52(X, X)
    try:
        L = np.linalg.cholesky(Kxx)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ys))
    except np.linalg.LinAlgError:
        try:
            Kinv = np.linalg.inv(Kxx + 1e-8 * np.eye(len(X)))
            alpha = Kinv @ Ys
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(Kxx + 1e-8 * np.eye(len(X)))
            alpha = Kinv @ Ys
    
    mu_std = K @ alpha
    v = np.linalg.solve(L, K.T) if 'L' in dir() else None
    if v is None:
        v = np.linalg.cholesky(Kxx + 1e-8 * np.eye(len(X)))
        v = np.linalg.solve(v.T, np.linalg.solve(v, K.T))
    var_std = Kss - np.sum(v * v, axis=0)
    var_std = np.maximum(var_std, 1e-9)
    sigma_std = np.sqrt(var_std)
    
    mu_obj = mu_std * sd_y + mu_y
    sigma_obj = sigma_std * sd_y
    return mu_obj, sigma_obj

# ---------- LCB acquisition ----------
def _lcb_score_candidates(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition function.
    Score = (f_best - (mu - beta * sigma)) / cost
    Higher score is better.
    """
    global _N_EVALS
    
    # Adaptive beta: increases with number of evaluations to encourage exploration
    # Start with beta=1.0 and increase slowly
    # A common choice is beta_t = sqrt(2 * log(t)) or similar
    # Here we use a simple heuristic: beta = 1.0 + 0.5 * sqrt(_N_EVALS)
    # But since we don't know the exact t (it varies per instance), we use a fixed
    # reasonable beta or estimate based on sigma magnitude.
    # Let's use a beta that scales with the typical sigma.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate beta based on the current uncertainty
    # If we are early on, sigma is high. We want to explore.
    # If we are late, sigma is low. We want to exploit.
    # A standard LCB for minimization is: mu - beta * sigma
    # We want to maximize: -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    # Or equivalently, maximize: (f_best - (mu - beta * sigma)) / cost
    
    # Let's use a dynamic beta.
    # Based on the observation that LCB with adaptive exploration works well.
    # We'll use beta = 1.0 + 0.2 * _N_EVALS (capped at some value)
    # Actually, since we don't track _N_EVALS reliably across calls in the same instance
    # (the state is reset? No, _N_EVALS is global, but the instances are separate loops?
    # The problem says "Each function runs on five seeded instances".
    # The acquisition function is called repeatedly.
    # We can estimate the number of evaluations by the magnitude of f_best or just use a fixed beta.
    # Let's try a fixed beta first, or one that depends on the range of sigma.
    
    # Let's use beta = 1.0 for a start, or slightly higher.
    # The inspiration mentioned "adaptive exploration (beta*sigma)".
    # Let's try beta = 1.5.
    
    beta = 1.5
    
    # LCB value to minimize: mu - beta * sigma
    # We want to maximize the "improvement" over f_best, penalized by cost.
    # Improvement = f_best - (mu - beta * sigma)
    # Score = Improvement / cost
    
    lcb = mu - beta * sigma_safe
    improvement = f_best - lcb
    score = improvement / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e9)
    
    # Update global counter (approximate)
    # This is a heuristic, not perfect, but helps adapt beta if we want to.
    _N_EVALS += 1
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function.
    Returns scores of shape (2000,), higher = evaluate next.
    """
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, -1e9)
        return score
    
    # LCB
    scores = _lcb_score_candidates(mu, sigma, f_best, cost)
    return scores

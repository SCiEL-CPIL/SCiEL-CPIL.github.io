import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 2.0  # safety margin in seconds before the 150s kill

# ---- Precomputed localization kernel (circular, N=20) ----
_N = 20
_LOCSIG = 3.2
_dist = np.arange(_N)
_circ = np.minimum(_dist, _N - _dist)
_LOC = np.exp(-0.5 * (_circ / _LOCSIG) ** 2)
_LOC = _LOC / _LOC[0]  # normalize so observed variable has weight 1

# ---- Inflation state (persists across analyze calls) ----
_INFL = [1.0]

def _lorenz96(x, F):
    n = x.shape[0]
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4(x, dt, F):
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5 * dt * k1, F)
    k3 = _lorenz96(x + 0.5 * dt * k2, F)
    k4 = _lorenz96(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    ensemble = np.asarray(ensemble, dtype=float)
    observations = np.asarray(observations, dtype=float)
    M = ensemble.shape[0]
    N = ensemble.shape[1]
    obs_idx = np.asarray(observed_indices, dtype=int)
    R = float(observation_variance)

    # Infer forcing from the ensemble dynamics (consistent across members)
    x0 = ensemble[0]
    dx = _lorenz96(x0, 0.0)
    # Estimate F: dx = (x_{i+1}-x_{i-2})x_{i-1} - x_i + F  =>  F = dx - (x_{i+1}-x_{i-2})x_{i-1} + x_i
    F_est = np.mean(dx - (np.roll(x0, -1) - np.roll(x0, 2)) * np.roll(x0, 1) + x0)

    # Short propagation of each member to the observation time (5 steps of dt=0.01)
    dt = 0.01
    result = ensemble.copy()
    for m in range(M):
        for _ in range(5):
            result[m] = _rk4(result[m], dt, F_est)

    # ---- Adaptive (SP) inflation from observed-space fit ----
    innov = observations - result[:, obs_idx]
    innov_mean = np.mean(innov, axis=0)
    resid = innov - innov_mean
    s2 = np.mean(resid * resid) / max(M - 1, 1)  # average sample variance in obs space
    # target: match R; inflate to close the gap
    gap = max(0.0, (R - s2) / (R + 1e-12))
    infl = _INFL[0] * (1.0 + 0.5 * gap)
    infl = min(infl, 2.0)
    infl = max(infl, 1.0)
    _INFL[0] = 0.9 * infl + 0.1 * 1.0  # relax toward 1 to avoid runaway

    # Inflate about the ensemble mean
    mean = np.mean(result, axis=0)
    result = mean + infl * (result - mean)

    # ---- Ensemble Kalman update (analysis) ----
    # Ensemble anomalies
    anom = result - mean  # (M, N)
    # Sample covariance in observed space
    B = (anom[:, obs_idx] - np.mean(anom[:, obs_idx], axis=0))
    # Localize B: scale each (obs, state) pair by distance kernel
    # B is (M, no); covariance C = B^T B/(M-1); localize columns of C
    no = obs_idx.size
    C = (B.T @ B) / max(M - 1, 1)  # (no, no)
    # Localize each column by distance to that observed variable
    for j in range(no):
        d = np.minimum(np.abs(obs_idx - obs_idx[j]), N - np.abs(obs_idx - obs_idx[j]))
        w = np.exp(-0.5 * (d / _LOCSIG) ** 2)
        C[:, j] *= w
    # Innovation covariance: S = C + R*I
    S = C + R * np.eye(no)
    # Kalman gain: K = B_loc^T S^{-1}, where B_loc = localized cross-cov (state, obs)
    B_loc = np.zeros((N, no))
    for j in range(no):
        d = np.minimum(np.abs(obs_idx - obs_idx[j]), N - np.abs(obs_idx - obs_idx[j]))
        w = np.exp(-0.5 * (d / _LOCSIG) ** 2)
        B_loc[:, j] = (anom.T @ (B[:, j] * 1.0)) / max(M - 1, 1) * w
    # Solve S^{-1} innov
    try:
        gain = B_loc @ np.linalg.solve(S, np.eye(no))  # (N, no)
    except np.linalg.LinAlgError:
        gain = B_loc @ np.linalg.pinv(S)
    K = gain  # (N, no)
    # Analysis: x_a = x_f + K (y - H x_f)
    Hx = result[:, obs_idx]
    upd = innov  # (M, no)
    result = result + upd @ K.T

    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)

    # Time guard
    if time.time() > _DEADLINE - _MARGIN:
        return result

    return result
# EVOLVE-BLOCK-END

"""Lorenz-96 assimilation using Ensemble Kalman Filter with adaptive localization and inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None


def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE


def _lorenz96_rhs(x, F):
    """Compute the Lorenz-96 right-hand side."""
    n = len(x)
    dxdt = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    return dxdt


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Kalman Filter analysis update with adaptive localization and inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - updated ensemble with the same (16, 20) shape
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)

    ens = ensemble.astype(np.float64).copy()

    # Estimate F from the ensemble (it should be roughly constant across members and time)
    # For Lorenz-96, F is the forcing parameter. We can estimate it from the equilibrium.
    # Actually, we don't need F for the analysis update - it's only for propagation.
    # But we need to know F to properly tune localization. Let's use a reasonable default.
    # The task says F=6,8,10 for different runs. We'll use a moderate localization.

    obs_var = observation_variance

    # Adaptive localization: use a width that balances bias and variance
    # For 16 members, we need enough localization to avoid spurious correlations
    # but not so much that we lose information
    local_decay = 1.8

    # Precompute localization weights for all pairs
    # loc_weights[i, j] = exp(-d(i,j)^2 / (2 * local_decay^2))
    loc_weights = np.zeros((n_state, n_state))
    for i in range(n_state):
        for j in range(n_state):
            d = abs(i - j)
            d = min(d, n_state - d)
            loc_weights[i, j] = np.exp(-d * d / (2.0 * local_decay * local_decay))

    # Ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean  # (n_members, n_state)

    # Compute ensemble covariance (with localization applied to off-diagonal)
    # C = (1/(n-1)) * A^T * A
    C = (anomalies.T @ anomalies) / (n_members - 1)  # (n_state, n_state)

    # Apply localization to the covariance matrix
    C_loc = C * loc_weights

    # Observation operator: H selects observed indices
    # H is (n_obs, n_state)
    H = np.zeros((n_obs, n_state))
    for j in range(n_obs):
        H[j, observed_indices[j]] = 1.0

    # Innovation: y - H * x_mean
    innov = observations - H @ ens_mean  # (n_obs,)

    # Innovation covariance: S = H * C_loc * H^T + R
    # R is diagonal with obs_var
    HC = H @ C_loc  # (n_obs, n_state)
    S = HC @ H.T + obs_var * np.eye(n_obs)  # (n_obs, n_obs)

    # Invert S (it's small, n_obs x n_obs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    # Kalman gain: K = C_loc * H^T * S_inv
    # K is (n_state, n_obs)
    K = C_loc @ H.T @ S_inv

    # Update ensemble mean
    ens_mean_new = ens_mean + K @ innov

    # Update anomalies: A_new = A - K * H * A
    # H @ anomalies gives (n_obs, n_members)
    H_anom = H @ anomalies  # (n_obs, n_members)
    anomalies_new = anomalies - K @ H_anom  # (n_state, n_members)

    ens_new = ens_mean_new[:, np.newaxis] + anomalies_new.T  # (n_members, n_state)

    # Adaptive inflation: adjust ensemble spread
    # Compute the expected innovation variance from the ensemble
    # and compare to the actual innovation variance
    # If the ensemble is too tight, inflate; if too wide, deflate

    # For each observed coordinate, check if the ensemble spread is appropriate
    inflation_factors = []
    for j in range(n_obs):
        idx = observed_indices[j]
        ens_spread = np.std(ens_new[:, idx])
        # The expected spread of the innovation is sqrt(C_loc[idx,idx] + obs_var)
        expected_innov_var = C_loc[idx, idx] + obs_var
        expected_spread = np.sqrt(expected_innov_var)
        
        if ens_spread > 1e-10 and expected_spread > 1e-10:
            # Target: ensemble spread should be roughly consistent with the model
            # Use a soft target
            ratio = expected_spread / ens_spread
            # Clip to avoid extreme inflation/deflation
            inflation_factors.append(np.clip(ratio, 0.8, 1.3))

    if inflation_factors:
        # Use geometric mean of inflation factors
        geo_mean = np.exp(np.mean(np.log(inflation_factors)))
        inflation = np.clip(geo_mean, 0.9, 1.2)
    else:
        inflation = 1.0

    # Apply inflation around the mean
    ens_new = ens_mean_new + inflation * (ens_new - ens_mean_new)

    # Clip and ensure finiteness
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.nan_to_num(ens_new, nan=0.0, posinf=100.0, neginf=-100.0)

    return ens_new

# EVOLVE-BLOCK-END

"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
import time

N = 150

def _spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _log_energy_and_grad(params):
    theta = params[:N]
    phi = params[N:]
    pts = _spherical_to_cartesian(theta, phi)
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    # Clamp to avoid log(0)
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Gradient with respect to theta and phi
    # d/idx log(1/d_ij) = -(1/d_ij) * (xi - xj)·(dxi/dtheta_i) / d_ij ... 
    # Let's use: grad_theta_i = sum_{j != i} (1/d_ij^2) * (xi - xj) · dxi/dtheta_i
    # dxi/dtheta_i = (cos(theta_i)*cos(phi_i), cos(theta_i)*sin(phi_i), -sin(theta_i))
    # dxi/dphi_i = (-sin(phi_i), cos(phi_i), 0)
    
    grad_theta = np.zeros(N)
    grad_phi = np.zeros(N)
    
    for i in range(N):
        dxi_dtheta = np.array([np.cos(theta[i]) * np.cos(phi[i]),
                                np.cos(theta[i]) * np.sin(phi[i]),
                                -np.sin(theta[i])])
        dxi_dphi = np.array([-np.sin(phi[i]), np.cos(phi[i]), 0.0])
        
        for j in range(N):
            if i == j:
                continue
            dij = dist[i, j]
            if dij < 1e-15:
                continue
            # grad = -1/d * (xi-xj)·(dxi/dparam) / d = -(xi-xj)·(dxi/dparam)/d^2
            dot_t = np.dot(pts[i] - pts[j], dxi_dtheta)
            dot_p = np.dot(pts[i] - pts[j], dxi_dphi)
            # d energy / d param = -sum_j (1/d) * d/dparam log(d) = -sum_j (1/d) * (d/dparam d)/d
            # = -sum_j (1/d^2) * (d/dparam d)
            # d/dparam_i d_ij = (xi - xj)·(dxi/dparam_i) / d_ij
            # So d energy/dparam = -sum_j (1/d_ij^2) * (xi-xj)·(dxi/dparam_i)/d_ij
            # Wait, let me redo:
            # E = -sum_{i<j} log(d_ij)
            # dE/dtheta_i = -sum_{j != i} (1/d_ij) * d(d_ij)/dtheta_i
            # d(d_ij)/dtheta_i = (xi - xj)·(dxi/dtheta_i) / d_ij
            # So dE/dtheta_i = -sum_{j != i} (1/d_ij) * (xi-xj)·(dxi/dtheta_i)/d_ij
            # = -sum_{j != i} (xi-xj)·(dxi/dtheta_i) / d_ij^2
            grad_theta[i] += -dot_t / (dij * dij)
            grad_phi[i] += -dot_p / (dij * dij)
    
    grad = np.concatenate([grad_theta, grad_phi])
    return energy, grad

def _log_energy_only(params):
    theta = params[:N]
    phi = params[N:]
    pts = _spherical_to_cartesian(theta, phi)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = np.inf
    best_params = None
    
    # Initial point: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    theta_init = np.arccos(np.clip(z, -1, 1))
    phi_init = np.mod(phi_init, 2 * np.pi)
    
    params0 = np.concatenate([theta_init, phi_init])
    
    # Try multiple restarts with different initializations
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            p0 = params0.copy()
        else:
            # Perturbed starting point
            noise = rng.normal(0, 0.1, size=2*N)
            p0 = params0.copy() + noise
            p0[:N] = np.clip(p0[:N], 1e-6, np.pi - 1e-6)
            p0[N:] = np.mod(p0[N:], 2 * np.pi)
        
        try:
            result = minimize(
                _log_energy_and_grad,
                p0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < best_energy:
                e = _log_energy_only(result.x)
                if e < best_energy:
                    best_energy = e
                    best_params = result.x.copy()
        except Exception:
            continue
        
        # Check if we've run out of time
        if time.time() > deadline - 60:
            break
    
    # Final polish with tighter tolerances if we have time
    if best_params is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                _log_energy_and_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80}
            )
            e = _log_energy_only(result.x)
            if e < best_energy:
                best_energy = e
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to initial
        best_params = params0
    
    theta = best_params[:N]
    phi = best_params[N:]
    theta = np.clip(theta, 1e-8, np.pi - 1e-8)
    phi = np.mod(phi, 2 * np.pi)
    
    pts = _spherical_to_cartesian(theta, phi)
    
    # Normalize to unit sphere (safety)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    return pts

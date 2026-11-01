"""Direct analytic approach with sphere reparameterization and multi-method polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195


def _hermite_coeffs():
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound_from_coeffs(a0, a4, a8):
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0: continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
            
    q = p * _SGN
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0: continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
            
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18
    return np.sqrt(vf * vF)


def _bound(params):
    return _bound_from_coeffs(params[0], params[1], params[2])


def _bound_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    m0 = 0.0; m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0: continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
            
    q = p * _SGN
    n0 = 0.0; n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0: continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
            
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        dm0 = 0.0; dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0: continue
            s0 = 0.0; s2 = 0.0
            for j in range(7):
                s0 += p[j] * _I[i+j]
                s2 += p[j] * _I[1+i+j]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
            
        dq = dp * _SGN
        dn0 = 0.0; dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0: continue
            s0 = 0.0; s2 = 0.0
            for j in range(7):
                s0 += q[j] * _I[i+j]
                s2 += q[j] * _I[1+i+j]
            dn0 += 4.0 * np.pi * dqi * s0
            dn2 += 4.0 * np.pi * dqi * s2
            
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    return bound, grad


def _objective(params):
    return _bound(params)

def _objective_grad(params):
    return _bound_grad(params)


def construct():
    from scipy.optimize import minimize
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Sphere reparameterization
    # a0 = cos(phi1) * cos(phi2)
    # a4 = cos(phi1) * sin(phi2)
    # a8 = sin(phi1)
    
    def sphere_obj(theta):
        phi1, phi2 = theta
        a0 = np.cos(phi1) * np.cos(phi2)
        a4 = np.cos(phi1) * np.sin(phi2)
        a8 = np.sin(phi1)
        return _bound_from_coeffs(a0, a4, a8)
    
    # Initial points on sphere
    # Seed normalized
    seed = np.array([-1.13, 0.04, 0.000308642])
    norm_seed = np.linalg.norm(seed)
    seed_norm = seed / norm_seed
    # Convert to angles
    a0, a4, a8 = seed_norm
    phi1_seed = np.arcsin(np.clip(a8, -1, 1))
    phi2_seed = np.arctan2(a4, a0)
    
    starts_sphere = [
        np.array([phi1_seed, phi2_seed]),
        np.array([0.1, 0.1]),
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        np.array([3.0, 3.0]),
        np.array([4.0, 4.0]),
        np.array([5.0, 5.0]),
        np.array([0.0, 0.0]),
        np.array([np.pi/2, 0.0]),
        np.array([0.0, np.pi/2]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(50):
        phi1 = rng.uniform(0, np.pi)
        phi2 = rng.uniform(0, 2*np.pi)
        starts_sphere.append(np.array([phi1, phi2]))
        
    # Optimize on sphere
    for start in starts_sphere:
        if time.time() > _DEADLINE - 20: break
        try:
            res = minimize(sphere_obj, start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                phi1, phi2 = res.x
                a0 = np.cos(phi1) * np.cos(phi2)
                a4 = np.cos(phi1) * np.sin(phi2)
                a8 = np.sin(phi1)
                best_params = np.array([a0, a4, a8])
        except:
            pass
            
    # Also run standard 3D L-BFGS-B from best sphere point and seed
    starts_3d = [best_params.copy(), seed]
    for _ in range(20):
        starts_3d.append(rng.normal(best_params, [0.1, 0.1, 0.1]))
        
    for start in starts_3d:
        if time.time() > _DEADLINE - 10: break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
            
    # Final polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Powell',
                          options={'maxiter': 5000, 'ftol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except:
            pass
            
    return best_params
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1

def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W

_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)

def _objective(c):
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)

def _gradient(c):
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad

def _complex_to_real(z):
    """Convert complex amplitude vector to real coefficient vector.
    z[0] = c_0 (real)
    z[m] for m=1..K: z[m] = c_{-m} + i*c_m, so c_{-m} = Re(z[m]), c_m = Im(z[m])
    """
    c = np.zeros(_NC)
    c[_K] = float(np.real(z[0]))
    for m in range(1, _K + 1):
        z_m = z[m]
        c[_K - m] = float(np.real(z_m))
        c[_K + m] = float(np.imag(z_m))
    return c

def _real_to_complex(c):
    """Convert real coefficient vector to complex amplitude vector."""
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = complex(c[_K], 0.0)
    for m in range(1, _K + 1):
        z[m] = complex(c[_K - m], c[_K + m])
    return z

def _objective_complex(z):
    c = _complex_to_real(z)
    return _objective(c)

def _gradient_complex(z):
    c = _complex_to_real(z)
    grad_c = _gradient(c)
    grad_z = np.zeros(_K + 1, dtype=np.complex128)
    grad_z[0] = complex(grad_c[_K], 0.0)
    for m in range(1, _K + 1):
        grad_z[m] = complex(grad_c[_K - m], grad_c[_K + m])
    return grad_z

def _polish_complex(z0, deadline):
    """Polish in complex amplitude space using SLSQP with normalization constraint."""
    z = np.asarray(z0, dtype=np.complex128).copy()
    # Normalize
    nrm = np.sqrt(np.sum(np.abs(z)**2))
    if nrm < 1e-14:
        return None
    z = z / nrm

    # Convert to real vector for scipy: [Re(z0), Im(z0), Re(z1), Im(z1), ..., Re(zK), Im(zK)]
    # But z0 is real-valued (c_0), so we treat it specially
    def z_to_real(x):
        r = np.zeros(2 * _K + 1)
        r[0] = x[0]  # c_0
        for m in range(1, _K + 1):
            r[2*m - 1] = x[1 + 2*(m-1)]  # Re(z_m)
            r[2*m] = x[2 + 2*(m-1)]      # Im(z_m)
        return r

    def real_to_z(r):
        z = np.zeros(_K + 1, dtype=np.complex128)
        z[0] = complex(r[0], 0.0)
        for m in range(1, _K + 1):
            z[m] = complex(r[2*m - 1], r[2*m])
        return z

    def fun(x):
        z = real_to_z(x)
        c = _complex_to_real(z)
        return _objective(c)

    def jac(x):
        z = real_to_z(x)
        c = _complex_to_real(z)
        grad_c = _gradient(c)
        grad_x = np.zeros(2 * _K + 1)
        grad_x[0] = grad_c[_K]
        for m in range(1, _K + 1):
            grad_x[2*m - 1] = grad_c[_K - m]
            grad_x[2*m] = grad_c[_K + m]
        return grad_x

    # Constraint: ||z||^2 = 1, i.e., x[0]^2 + sum(|z_m|^2) = 1
    def constraint_fun(x):
        return x[0]**2 + np.sum(np.abs(x[1:])**2) - 1.0

    def constraint_jac(x):
        j = np.zeros(2 * _K + 1)
        j[0] = 2.0 * x[0]
        j[1:] = 2.0 * x[1:]
        return j

    x0 = z_to_real(z)
    best_x = x0.copy()
    best_val = fun(x0)

    try:
        res = minimize(fun, x0, jac=jac, method='SLSQP',
                       constraints=[{'type': 'eq', 'fun': constraint_fun, 'jac': constraint_jac}],
                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Second pass with tighter tolerance
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res2 = minimize(fun, best_x, jac=jac, method='SLSQP',
                           constraints=[{'type': 'eq', 'fun': constraint_fun, 'jac': constraint_jac}],
                           options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass

    z_best = real_to_z(best_x)
    c_best = _complex_to_real(z_best)
    nrm = np.linalg.norm(c_best)
    if nrm < 1e-14:
        return None
    return c_best / nrm, float(best_val)

def _polish(c0, deadline):
    """Polish in real coefficient space using SLSQP with sphere constraint."""
    c = np.asarray(c0, dtype=np.float64).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    best_c = c.copy()
    best_val = fun(c)

    try:
        constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
        res = minimize(fun, c, jac=jac, method='SLSQP',
                       constraints=[constraint],
                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            constraint2 = {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1.0, 'jac': lambda x: 2.0 * x}
            res2 = minimize(fun, best_c, jac=jac, method='SLSQP',
                           constraints=[constraint2],
                           options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    nrm = np.linalg.norm(best_c)
    if nrm < 1e-14:
        return None
    return best_c / nrm, float(best_val)

def _grid_search_complex(deadline):
    """Deterministic grid search over complex amplitudes exploiting rotational symmetry.
    We search over |z_m| for m=1..K and the phase of z_m relative to z_1 (setting z_1 real).
    This reduces the 21D real space to (K-1) phases + K magnitudes + c_0 = 1 + 10 + 10 = 21,
    but with the constraint that ||z||^2 = 1, we have 10 free magnitude/phase params + c_0.
    
    Actually, let's use a simpler approach: grid over a few key parameters.
    """
    best_c = None
    best_val = np.inf

    # Strategy: grid over the relative magnitudes and phases of the complex amplitudes
    # z_m for m=1..10, with z_1 set to be real and positive (gauge fixing)
    # Then we have 9 complex amplitudes (z_2..z_10) + c_0 + |z_1| = 1 + 18 + 1 = 20 free params
    # But with ||z||^2 = 1 constraint, we have 19 free params.
    
    # For a practical grid, let's focus on the most important modes:
    # m = 10, 9, 8, ... and see which combinations give the flattest harmonic.
    
    # Grid over relative weights of top few modes
    # Let's try a structured search over the dominant mode combinations
    
    # First, try pure modes and simple combinations
    rng = np.random.default_rng(42)
    
    # Grid over |z_10|, |z_9|, |z_8|, |z_7| with the rest zero
    # and phases of z_9, z_8, z_7 relative to z_10
    n_grid = 5  # points per dimension for magnitudes
    n_phase = 4  # points per dimension for phases
    
    # Magnitudes: 0, 0.25, 0.5, 0.75, 1.0 (normalized)
    mag_vals = np.linspace(0.0, 1.0, n_grid)
    # Phases: 0, pi/2, pi, 3pi/2
    phase_vals = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    
    count = 0
    for m10 in mag_vals:
        for m9 in mag_vals:
            for m8 in mag_vals:
                for m7 in mag_vals:
                    # Skip if all zero
                    if m10 + m9 + m8 + m7 < 1e-6:
                        continue
                    # Normalize magnitudes
                    mag_sq = m10**2 + m9**2 + m8**2 + m7**2
                    if mag_sq < 1e-6:
                        continue
                    # c_0 can be 0 for now
                    # Phases for m9, m8, m7 (m10 phase fixed to 0)
                    for p9 in phase_vals:
                        for p8 in phase_vals:
                            for p7 in phase_vals:
                                if time.time() > deadline - 30.0:
                                    return best_c, best_val
                                
                                z = np.zeros(_K + 1, dtype=np.complex128)
                                z[0] = 0.0  # c_0 = 0
                                z[10] = m10 / np.sqrt(mag_sq)  # real
                                z[9] = (m9 / np.sqrt(mag_sq)) * np.exp(1j * p9)
                                z[8] = (m8 / np.sqrt(mag_sq)) * np.exp(1j * p8)
                                z[7] = (m7 / np.sqrt(mag_sq)) * np.exp(1j * p7)
                                
                                c = _complex_to_real(z)
                                val = _objective(c)
                                if val < best_val:
                                    best_val = val
                                    best_c = c.copy()
                                count += 1
                                if count % 1000 == 0 and time.time() > deadline - 60.0:
                                    return best_c, best_val
    
    return best_c, best_val

def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 620.0  # 10s safety margin
    
    best_c = None
    best_val = np.inf
    
    # 1. Grid search in complex amplitude space
    if time.time() < deadline - 120.0:
        grid_c, grid_val = _grid_search_complex(deadline)
        if grid_c is not None:
            best_c = grid_c
            best_val = grid_val
    
    # 2. Polish the grid result in complex space
    if best_c is not None and time.time() < deadline - 30.0:
        z_best = _real_to_complex(best_c)
        polished = _polish_complex(z_best, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 3. Differential Evolution in real space (broader search)
    bounds = [(-10.0, 10.0) for _ in range(_NC)]
    
    try:
        if time.time() < deadline - 60.0:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
    except Exception:
        pass
    
    # 4. Polish DE result
    if best_c is not None and time.time() < deadline - 20.0:
        polished = _polish(best_c, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 5. Additional restarts from structured starts
    rng = np.random.default_rng(123)
    
    structured_starts = []
    # Pure modes
    for m in range(-_K, _K + 1):
        c0 = np.zeros(_NC)
        c0[m + _K] = 1.0
        structured_starts.append(c0)
    
    # Pairs of modes with same |m|
    for m in range(1, _K + 1):
        idx_pos = m + _K
        idx_neg = -m + _K
        for angle in [0.0, np.pi/4, np.pi/2, 3.0*np.pi/4, np.pi]:
            c0 = np.zeros(_NC)
            c0[idx_pos] = np.cos(angle)
            c0[idx_neg] = np.sin(angle)
            structured_starts.append(c0)
    
    # Random starts
    for _ in range(20):
        c0 = rng.standard_normal(_NC)
        c0 /= np.linalg.norm(c0)
        structured_starts.append(c0)
    
    rng.shuffle(structured_starts)
    
    for c0 in structured_starts:
        if time.time() >= deadline - 15.0:
            break
        polished = _polish(c0, deadline)
        if polished:
            c, val = polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 6. Final high-precision polish
    if best_c is not None and time.time() < deadline - 5.0:
        final_polished = _polish(best_c, deadline)
        if final_polished:
            c, val = final_polished
            if val < best_val:
                best_val = val
                best_c = c
    
    # 7. Fallback
    if best_c is None:
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
    
    best_c = np.asarray(best_c, dtype=np.float64)
    
    # Validation
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(_NC, dtype=np.float64)
        best_c[_K] = 1.0
        
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
        
    return best_c

# EVOLVE-BLOCK-END

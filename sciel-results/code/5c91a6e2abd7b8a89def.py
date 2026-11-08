import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

# --- Precompute moment integrals ---
# I1[n] = ∫ x^(2n) exp(-x²) dx, n=0..24
# I2[n] = ∫ x^(2n) exp(-2x²) dx, n=0..48
I1 = np.zeros(25)
I2 = np.zeros(49)
for n in range(25):
    if n == 0:
        I1[n] = np.sqrt(np.pi)
    else:
        result = 1.0
        for k in range(n):
            result *= (2*k + 1)
        I1[n] = result * np.sqrt(np.pi) / (2.0 ** n)

for n in range(49):
    if n == 0:
        I2[n] = np.sqrt(np.pi / 2.0)
    else:
        result = 1.0
        for k in range(n):
            result *= (2*k + 1)
        I2[n] = result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))

# Precompute matrix forms
# I1_mat[i,j] = I1[i+j] for i,j in 0..6
I1_mat = I1[np.arange(7)[:, None] + np.arange(7)[None, :]]

# For L4: S[u] = sum_{a=0}^{min(u,6)} P[a]*P[u-a], u=0..12
# L4_4th = sum_u sum_v S[u]*S[v]*I2[u+v]
# I2_sub[u,v] = I2[u+v] for u,v in 0..12
I2_sub = I2[np.arange(13)[:, None] + np.arange(13)[None, :]]

# Hermite polynomial coefficients (even powers only: x^0, x^2, ..., x^degree)
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Pad to length 7
H0_7 = np.zeros(7); H0_7[0] = 1.0
H4_7 = np.zeros(7); H4_7[0] = 12.0; H4_7[1] = -48.0; H4_7[2] = 16.0
H8_7 = np.zeros(7); H8_7[:5] = H8_coeffs
H12_7 = np.zeros(7); H12_7[:] = H12_coeffs

# c3 = -(c0 + 12*c1 + 3360*c2) / 1039500
# dP/dc0 = H0_7 + (-1/1039500)*H12_7
# dP/dc1 = H4_7 + (-12/1039500)*H12_7
# dP/dc2 = H8_7 + (-3360/1039500)*H12_7
dP_dc0 = H0_7 + (-1.0/1039500.0) * H12_7
dP_dc1 = H4_7 + (-12.0/1039500.0) * H12_7
dP_dc2 = H8_7 + (-3360.0/1039500.0) * H12_7

def _compute_P(c0, c1, c2):
    """Compute monomial coefficients of P(x) given free coefficients c0, c1, c2."""
    c3 = -(c0 + 12.0*c1 + 3360.0*c2) / 1039500.0
    P = c0 * H0_7 + c1 * H4_7 + c2 * H8_7 + c3 * H12_7
    return P

def _compute_S(P):
    """Compute S[u] = sum_{a=0}^{min(u,6)} P[a]*P[u-a] for u=0..12."""
    S = np.zeros(13)
    for u in range(13):
        total = 0.0
        for a in range(7):
            b = u - a
            if 0 <= b <= 6:
                total += P[a] * P[b]
        S[u] = total
    return S

def _compute_dS(dP, P):
    """Compute dS/du where S[u] = sum P[a]*P[u-a]."""
    dS = np.zeros(13)
    for u in range(13):
        total = 0.0
        for a in range(7):
            b = u - a
            if 0 <= b <= 6:
                total += dP[a] * P[b] + P[a] * dP[b]
        dS[u] = total
    return dS

def _objective_and_grad(c0, c1, c2):
    """Compute C4 upper bound and its gradient w.r.t. (c0, c1, c2)."""
    P = _compute_P(c0, c1, c2)
    
    # L2_sq = P^T I1_mat P
    L2_sq = float(P @ I1_mat @ P)
    
    # L4_4th = S^T I2_sub S
    S = _compute_S(P)
    L4_4th = float(S @ I2_sub @ S)
    
    if L2_sq <= 0 or L4_4th <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    # Gradients
    dL2_0 = 2.0 * float(P @ I1_mat @ dP_dc0)
    dL2_1 = 2.0 * float(P @ I1_mat @ dP_dc1)
    dL2_2 = 2.0 * float(P @ I1_mat @ dP_dc2)
    
    dS_0 = _compute_dS(dP_dc0, P)
    dS_1 = _compute_dS(dP_dc1, P)
    dS_2 = _compute_dS(dP_dc2, P)
    
    dL4_0 = 2.0 * float(dS_0 @ I2_sub @ S)
    dL4_1 = 2.0 * float(dS_1 @ I2_sub @ S)
    dL4_2 = 2.0 * float(dS_2 @ I2_sub @ S)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_0 = (0.5 * dL4_0 / sqrt_L4 - dL2_0 / L2_sq) * C4
    dC4_1 = (0.5 * dL4_1 / sqrt_L4 - dL2_1 / L2_sq) * C4
    dC4_2 = (0.5 * dL4_2 / sqrt_L4 - dL2_2 / L2_sq) * C4
    
    return C4, np.array([dC4_0, dC4_1, dC4_2])

def _objective_only(c):
    return _objective_and_grad(c[0], c[1], c[2])[0]

def _grad_only(c):
    return _objective_and_grad(c[0], c[1], c[2])[1]

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s/210s limit
    
    # Known good starting point (seed)
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_C4 = _objective_and_grad(best_c[0], best_c[1], best_c[2])[0]
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [best_c.copy()]
    
    # Perturbed starts around known good point at various scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        for _ in range(4):
            perturbation = rng.normal(0, scale, 3)
            starts.append(best_c + perturbation)
    
    # Random starts in various ranges
    for _ in range(30):
        scale = 10.0 ** rng.uniform(-2, 3)
        starts.append(rng.uniform(-scale, scale, 3))
    
    # Structured grid starts
    for a in [-5, -1, -0.1, 0.1, 1, 5]:
        for b in [-5, -1, -0.1, 0.1, 1, 5]:
            for c in [-5, -1, -0.1, 0.1, 1, 5]:
                starts.append(np.array([a, b, c]))
    
    # Evaluate and sort starting points
    start_objs = []
    for s in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            obj, _ = _objective_and_grad(s[0], s[1], s[2])
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    # Multi-start BFGS optimization
    for obj, s in start_objs:
        if time.time() > deadline - 10.0:
            break
        if obj >= best_C4:
            continue
        
        try:
            result = minimize(
                _objective_only,
                s,
                jac=_grad_only,
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-12}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # High-precision polish: tighter tolerance, more iterations
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective_only,
                best_c,
                jac=_grad_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxls': 50}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Additional polish with Nelder-Mead for potential escape from numerical stagnation
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                _objective_only,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final verification
    try:
        final_obj, _ = _objective_and_grad(best_c[0], best_c[1], best_c[2])
        if final_obj < best_C4:
            best_C4 = final_obj
    except:
        pass
    
    return np.array(best_c)

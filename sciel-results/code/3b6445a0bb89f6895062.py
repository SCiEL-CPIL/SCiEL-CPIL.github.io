import numpy as np
from scipy.optimize import minimize
import time

# Precompute moment integrals for all needed values
# _integral_x2n_exp_minus_x2(n) for n = 0..24 (max i+j for 7-element P)
# _integral_x2n_exp_minus_2x2(n) for n = 0..48 (max i+j+k+l for 7-element P)

def _precompute_moments():
    """Precompute all needed moment integrals."""
    # For L2: ∫ x^(2n) exp(-x²) dx, n = 0..24
    I1 = np.zeros(25)
    for n in range(25):
        if n == 0:
            I1[n] = np.sqrt(np.pi)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            I1[n] = result * np.sqrt(np.pi) / (2.0 ** n)
    
    # For L4: ∫ x^(2n) exp(-2x²) dx, n = 0..48
    I2 = np.zeros(49)
    for n in range(49):
        if n == 0:
            I2[n] = np.sqrt(np.pi / 2.0)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            I2[n] = result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))
    
    return I1, I2

I1, I2 = _precompute_moments()

# Precompute Hermite polynomial coefficients
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Precompute dP/dc_i vectors (these don't change)
# c3 = -(c0 + 12*c1 + 3360*c2) / 1039500
# dP/dc0 = H0 + (-1/1039500)*H12
# dP/dc1 = H4 + (-12/1039500)*H12
# dP/dc2 = H8 + (-3360/1039500)*H12

def _compute_P_coeffs(c0, c1, c2, c3):
    """Compute the monomial coefficients of P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    P = np.zeros(7)
    P[0] += c0 * H0_coeffs[0]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    return P

# Precompute dP/dc vectors
dP_dc0_base = np.zeros(7)
dP_dc0_base[0] = H0_coeffs[0]
dP_dc1_base = np.zeros(7)
dP_dc1_base[0] = H4_coeffs[0]
dP_dc1_base[1] = H4_coeffs[1]
dP_dc1_base[2] = H4_coeffs[2]
dP_dc2_base = np.zeros(7)
for i in range(5):
    dP_dc2_base[i] = H8_coeffs[i]

H12_full = np.zeros(7)
for i in range(7):
    H12_full[i] = H12_coeffs[i]

dP_dc0 = dP_dc0_base + (-1.0/1039500.0) * H12_full
dP_dc1 = dP_dc1_base + (-12.0/1039500.0) * H12_full
dP_dc2 = dP_dc2_base + (-3360.0/1039500.0) * H12_full

def _objective_and_grad(c0, c1, c2):
    """Compute the objective (upper bound on C4) and its gradient."""
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    # Vectorized L2_sq = sum_i sum_j P[i]*P[j]*I1[i+j]
    # Using matrix form: P @ (I1_matrix) @ P
    I1_mat = I1[np.arange(7)[:, None] + np.arange(7)[None, :]]
    L2_sq = P @ I1_mat @ P
    
    # Vectorized L4_4th = sum_{i,j,k,l} P[i]*P[j]*P[k]*P[l]*I2[i+j+k+l]
    # = (P ⊗ P) @ I2_mat @ (P ⊗ P) where I2_mat[a,b] = I2[a+b] for a,b in 0..48
    P_outer = np.outer(P, P)  # 49 elements, index = i*7+j
    # Need to map (i,j,k,l) -> i+j+k+l
    # P_outer has index i*7+j, so we need I2[i+j+k+l]
    # Let's build a 49x49 matrix M where M[(i*7+j), (k*7+l)] = I2[i+j+k+l]
    # Actually, L4_4th = sum_{i,j,k,l} P[i]P[j]P[k]P[l] I2[i+j+k+l]
    # = sum_{a=0}^{6} sum_{b=0}^{6} sum_{c=0}^{6} sum_{d=0}^{6} P[a]P[b]P[c]P[d] I2[a+b+c+d]
    # Let u = a+b (0..12), v = c+d (0..12)
    # = sum_u sum_v (sum_{a+b=u} P[a]P[b]) (sum_{c+d=v} P[c]P[d]) I2[u+v]
    
    # Compute S[u] = sum_{a=0}^{min(u,6)} P[a]*P[u-a] for u=0..12
    S = np.zeros(13)
    for u in range(13):
        for a in range(min(u, 6) + 1):
            if u - a <= 6:
                S[u] += P[a] * P[u-a]
    
    # L4_4th = sum_u sum_v S[u]*S[v]*I2[u+v]
    I2_sub = I2[np.arange(13)[:, None] + np.arange(13)[None, :]]
    L4_4th = S @ I2_sub @ S
    
    if L2_sq <= 0 or L4_4th <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    # Gradient: dC4/dc_i = (0.5 * dL4/dc_i / sqrt(L4) - dL2/dc_i / L2) * C4
    # dL2/dc_i = 2 * P @ I1_mat @ dP_dc_i
    dL2_0 = 2.0 * P @ I1_mat @ dP_dc0
    dL2_1 = 2.0 * P @ I1_mat @ dP_dc1
    dL2_2 = 2.0 * P @ I1_mat @ dP_dc2
    
    # dL4/dc_i: L4 = S @ I2_sub @ S where S depends on P
    # dS/du = sum_a dP_dc_i[a]*P[u-a] + sum_a P[a]*dP_dc_i[u-a]
    # dL4/dc_i = (dS/dP_dc_i) @ I2_sub @ S + S @ I2_sub @ (dS/dP_dc_i)
    # = 2 * (dS_vec) @ I2_sub @ S (by symmetry)
    
    def _dS_vec(dP):
        dS = np.zeros(13)
        for u in range(13):
            for a in range(min(u, 6) + 1):
                if u - a <= 6:
                    dS[u] += dP[a] * P[u-a] + P[a] * dP[u-a]
            return dS if u == 12 else dS
        return dS
    
    # More carefully:
    def _compute_dS(dP, P):
        dS = np.zeros(13)
        for u in range(13):
            total = 0.0
            for a in range(7):
                b = u - a
                if 0 <= b <= 6:
                    total += dP[a] * P[b] + P[a] * dP[b]
            dS[u] = total
        return dS
    
    dS_0 = _compute_dS(dP_dc0, P)
    dS_1 = _compute_dS(dP_dc1, P)
    dS_2 = _compute_dS(dP_dc2, P)
    
    dL4_0 = 2.0 * dS_0 @ I2_sub @ S
    dL4_1 = 2.0 * dS_1 @ I2_sub @ S
    dL4_2 = 2.0 * dS_2 @ I2_sub @ S
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_0 = (0.5 * dL4_0 / sqrt_L4 - dL2_0 / L2_sq) * C4
    dC4_1 = (0.5 * dL4_1 / sqrt_L4 - dL2_1 / L2_sq) * C4
    dC4_2 = (0.5 * dL4_2 / sqrt_L4 - dL2_2 / L2_sq) * C4
    
    return C4, np.array([dC4_0, dC4_1, dC4_2])

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    # Known good starting point
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_C4 = _objective_and_grad(best_c[0], best_c[1], best_c[2])[0]
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [best_c.copy()]
    
    # Perturbed starts around known good point
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        for _ in range(4):
            perturbation = rng.normal(0, scale, 3)
            starts.append(best_c + perturbation)
    
    # Random starts in various ranges
    for _ in range(30):
        scale = 10.0 ** rng.uniform(-2, 3)
        starts.append(rng.uniform(-scale, scale, 3))
    
    # Some structured starts
    for a in [-5, -1, -0.1, 0.1, 1, 5]:
        for b in [-5, -1, -0.1, 0.1, 1, 5]:
            for c in [-5, -1, -0.1, 0.1, 1, 5]:
                starts.append(np.array([a, b, c]))
    
    # Sort by initial objective
    start_objs = []
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            obj, _ = _objective_and_grad(s[0], s[1], s[2])
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    # Multi-start BFGS
    for obj, s in start_objs:
        if time.time() > deadline - 10.0:
            break
        if obj >= best_C4:
            continue
        
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                s,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-12}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # High-precision polish with L-BFGS-B and tighter tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 50}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final polish with even tighter settings
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 1000, 'gtol': 1e-15, 'maxfun': 5000}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)

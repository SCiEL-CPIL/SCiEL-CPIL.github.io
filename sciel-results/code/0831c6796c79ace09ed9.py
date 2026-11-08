import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients as arrays indexed by power/2
# H0: [1]
# H4: [12, -48, 16]
# H8: [3360, -13440, 11520, -3072, 256]
# H12: [1039500, -6048000, 8064000, -3932160, 846720, -98304, 4096]

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Precompute integral values
# I(n) = ∫ x^(2n) exp(-x²) dx for n = 0..12
# J(n) = ∫ x^(2n) exp(-2x²) dx for n = 0..24 (since max sum of 4 indices from 0..6 is 24)

def _precompute_integrals():
    """Precompute all needed integral values."""
    # I(n) = (2n-1)!! * sqrt(pi) / 2^n
    I = np.zeros(13)  # n = 0 to 12
    for n in range(13):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            I[n] = result * np.sqrt(np.pi) / (2.0 ** n)
    
    # J(n) = ∫ x^(2n) exp(-2x²) dx = (2n-1)!! * sqrt(pi) / 2^(2n+0.5)
    J = np.zeros(25)  # n = 0 to 24
    for n in range(25):
        if n == 0:
            J[n] = np.sqrt(np.pi / 2.0)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            J[n] = result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))
    
    return I, J

I_int, J_int = _precompute_integrals()

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

def _poly_square_moment(P):
    """Compute ∫ P(x)² * exp(-x²) dx using precomputed integrals."""
    n = len(P)
    # Use matrix form: P^T @ M @ P where M[i,j] = I(i+j)
    M = np.array([[I_int[i+j] for j in range(n)] for i in range(n)])
    return float(P @ M @ P)

def _poly_fourth_moment(P):
    """Compute ∫ P(x)⁴ * exp(-2x²) dx using precomputed integrals."""
    n = len(P)
    # P(x)^4 = (sum_i P[i] x^(2i))^4
    # = sum_{i,j,k,l} P[i]P[j]P[k]P[l] x^(2(i+j+k+l))
    # Use: P^4 integral = sum over all tuples
    # More efficiently: compute P(x)^2 first, then square it
    # P(x)^2 = sum_i sum_j P[i]P[j] x^(2(i+j))
    # Let Q[m] = sum_{i+j=m} P[i]P[j] for m=0..12
    Q = np.zeros(13)
    for i in range(n):
        for j in range(n):
            Q[i+j] += P[i] * P[j]
    # Now (P^2)^2 = sum_m sum_n Q[m]Q[n] x^(2(m+n))
    # ∫ (P^2)^2 exp(-2x²) = sum_m sum_n Q[m]Q[n] J(m+n)
    result = 0.0
    for m in range(13):
        for nn in range(13):
            result += Q[m] * Q[nn] * J_int[m+nn]
    return result

def _objective_and_grad(c0, c1, c2):
    """Compute the objective (upper bound on C4) and its gradient."""
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    L2_sq = _poly_square_moment(P)
    L4_4th = _poly_fourth_moment(P)
    
    if L2_sq <= 0 or L4_4th <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    # Compute gradients analytically
    # dP/dc0 = H0 + (dc3/dc0)*H12
    # dc3/dc0 = -1/1039500
    # dc3/dc1 = -12/1039500  
    # dc3/dc2 = -3360/1039500
    
    dP_dc0 = np.zeros(7)
    dP_dc0[0] += 1.0  # from H0
    for i in range(7):
        dP_dc0[i] += (-1/1039500.0) * H12_coeffs[i]
    
    dP_dc1 = np.zeros(7)
    for i in range(3):
        dP_dc1[i] += H4_coeffs[i]
    for i in range(7):
        dP_dc1[i] += (-12/1039500.0) * H12_coeffs[i]
    
    dP_dc2 = np.zeros(7)
    for i in range(5):
        dP_dc2[i] += H8_coeffs[i]
    for i in range(7):
        dP_dc2[i] += (-3360/1039500.0) * H12_coeffs[i]
    
    # dL2_sq/dc = 2 * P^T M dP where M[i,j] = I(i+j)
    n = len(P)
    M = np.array([[I_int[i+j] for j in range(n)] for i in range(n)])
    
    dL2_sq_dc0 = 2.0 * float(P @ M @ dP_dc0)
    dL2_sq_dc1 = 2.0 * float(P @ M @ dP_dc1)
    dL2_sq_dc2 = 2.0 * float(P @ M @ dP_dc2)
    
    # For L4_4th: 
    # L4_4th = sum_{m,n} Q[m]Q[n] J(m+n) where Q[m] = sum_{i+j=m} P[i]P[j]
    # dL4_4th/dc = 2 * sum_{m,n} (dQ[m]Q[n] + Q[m]dQ[n]) J(m+n)
    # = 4 * sum_{m,n} dQ[m] Q[n] J(m+n) (by symmetry)
    # dQ[m] = sum_{i+j=m} (dP[i]P[j] + P[i]dP[j]) = 2 * sum_{i+j=m} dP[i]P[j]
    
    def _compute_dQ(dP, P):
        dQ = np.zeros(13)
        for i in range(7):
            for j in range(7):
                dQ[i+j] += 2.0 * dP[i] * P[j]
        return dQ
    
    dQ_dc0 = _compute_dQ(dP_dc0, P)
    dQ_dc1 = _compute_dQ(dP_dc1, P)
    dQ_dc2 = _compute_dQ(dP_dc2, P)
    
    # Compute Q
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    # dL4_4th/dc = 4 * sum_{m,n} dQ[m] Q[n] J(m+n)
    def _dL4_4th(dQ, Q):
        result = 0.0
        for m in range(13):
            for nn in range(13):
                result += 4.0 * dQ[m] * Q[nn] * J_int[m+nn]
        return result
    
    dL4_4th_dc0 = _dL4_4th(dQ_dc0, Q)
    dL4_4th_dc1 = _dL4_4th(dQ_dc1, Q)
    dL4_4th_dc2 = _dL4_4th(dQ_dc2, Q)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_dc0 = (0.5 * dL4_4th_dc0 / sqrt_L4 - dL2_sq_dc0 / L2_sq) * C4
    dC4_dc1 = (0.5 * dL4_4th_dc1 / sqrt_L4 - dL2_sq_dc1 / L2_sq) * C4
    dC4_dc2 = (0.5 * dL4_4th_dc2 / sqrt_L4 - dL2_sq_dc2 / L2_sq) * C4
    
    return C4, np.array([dC4_dc0, dC4_dc1, dC4_dc2])

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin (210s limit, leave buffer)
    
    # Known good starting point
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_C4, _ = _objective_and_grad(best_c[0], best_c[1], best_c[2])
    
    # Generate starting points: focused grid + random
    rng = np.random.RandomState(42)
    starts = [best_c.copy()]
    
    # Grid around the known good point
    for dc0 in np.linspace(-0.5, 0.5, 15):
        for dc1 in np.linspace(-0.2, 0.2, 10):
            for dc2 in np.linspace(-0.1, 0.1, 8):
                if time.time() > deadline - 30.0:
                    break
                s = best_c + np.array([dc0, dc1, dc2])
                starts.append(s)
    
    # Add some wider random starts
    for _ in range(50):
        s = best_c + rng.normal(0, 5.0, 3)
        starts.append(s)
    
    # Evaluate and sort starting points
    start_objs = []
    for s in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            obj, _ = _objective_and_grad(s[0], s[1], s[2])
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    # Run BFGS from best starting points
    for obj, s in start_objs:
        if time.time() > deadline - 5.0:
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
                best_c = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-14}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
    
    return np.array(best_c)

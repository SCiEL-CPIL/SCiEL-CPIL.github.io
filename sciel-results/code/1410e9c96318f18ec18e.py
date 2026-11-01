import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 200.0 # Safety margin before 210s kill

def _check_time():
    return time.time() > _DEADLINE

def _fact_ratio(n):
    """
    Computes I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    """
    if n < 0:
        return 0.0
    # Use log-space for stability for larger n, though n is small here (max 12)
    log_val = 0.0
    for m in range(1, 2*n + 1):
        log_val += np.log(m)
    log_val -= n * np.log(4.0)
    for m in range(1, n + 1):
        log_val -= np.log(m)
    log_val += 0.5 * np.log(np.pi)
    return np.exp(log_val)

# Precompute integrals I_n for n=0..12
# Max power in P(x) is 12. P(x)^2 has max power 24.
# We need integrals of x^{2k} e^{-x^2} for k up to 12.
_I = np.array([_fact_ratio(n) for n in range(13)])

def _bound_and_grad(params):
    """
    Computes the uncertainty product bound and its analytic gradient.
    params: [c0, c4, c8]
    """
    c0, c4, c8 = params[0], params[1], params[2]
    
    # Constraint: g(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    
    # Polynomial P(x) = sum c_k H_k(x)
    # We need the coefficients of P(x) in the monomial basis x^{2j} to compute moments easily.
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    # p_j is the coefficient of x^{2j} in P(x)
    # j=0 (x^0): c0*1 + c4*3 + c8*105 + c12*945
    # j=1 (x^2): c4*(-6) + c8*(-420) + c12*(-7560)
    # j=2 (x^4): c4*1 + c8*210 + c12*13860
    # j=3 (x^6): c8*(-28) + c12*(-6930)
    # j=4 (x^8): c8*1 + c12*1155
    # j=5 (x^10): c12*(-66)
    # j=6 (x^12): c12*1
    
    p = np.zeros(7)
    p[0] = c0 + 3.0 * c4 + 105.0 * c8 + 945.0 * c12
    p[1] = -6.0 * c4 - 420.0 * c8 - 7560.0 * c12
    p[2] = c4 + 210.0 * c8 + 13860.0 * c12
    p[3] = -28.0 * c8 - 6930.0 * c12
    p[4] = c8 + 1155.0 * c12
    p[5] = -66.0 * c12
    p[6] = c12
    
    # Moments of |g|^2 = P(x)^2 e^{-x^2}
    # M_k = integral x^{2k} |g|^2 dx = sum_{i,j} p_i p_j integral x^{2(k+i+j)} e^{-x^2} dx
    # M_k = sum_{i,j} p_i p_j I_{k+i+j}
    
    # We need M_0, M_1, M_2 for variance <x^2> = M_2/M_0 - (M_1/M_0)^2 ?
    # Wait, g is even? 
    # H_n(x) for even n is even. P(x) is even. g(x) is even.
    # So <x> = M_1/M_0 = 0.
    # Variance = M_2/M_0.
    
    M0 = 0.0
    M1 = 0.0 # Should be 0
    M2 = 0.0
    
    for i in range(7):
        pi_val = p[i]
        if pi_val == 0: continue
        for j in range(7):
            pij = pi_val * p[j]
            if pij == 0: continue
            # Indices for I
            idx0 = i + j
            idx1 = 1 + i + j
            idx2 = 2 + i + j
            if idx0 < 13:
                M0 += pij * _I[idx0]
            if idx1 < 13:
                M1 += pij * _I[idx1]
            if idx2 < 13:
                M2 += pij * _I[idx2]
                
    if M0 <= 1e-15 or not np.isfinite(M0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    # Fourier side
    # \hat{g}(\xi) = \sqrt{2\pi} P(-i\xi) e^{-\xi^2/2}
    # P(-i\xi) = sum p_j (-i\xi)^{2j} = sum p_j (-1)^j \xi^{2j}
    # Let Q(\xi) = sum q_j \xi^{2j} where q_j = p_j (-1)^j
    # |\hat{g}|^2 = 2\pi Q(\xi)^2 e^{-\xi^2}
    
    q = p * np.array([1, -1, 1, -1, 1, -1, 1]) # (-1)^j for j=0..6
    
    N0 = 0.0
    N1 = 0.0
    N2 = 0.0
    
    for i in range(7):
        qi_val = q[i]
        if qi_val == 0: continue
        for j in range(7):
            qij = qi_val * q[j]
            if qij == 0: continue
            idx0 = i + j
            idx1 = 1 + i + j
            idx2 = 2 + i + j
            if idx0 < 13:
                N0 += qij * _I[idx0]
            if idx1 < 13:
                N1 += qij * _I[idx1]
            if idx2 < 13:
                N2 += qij * _I[idx2]
                
    # Normalize by 2*pi factor which cancels in ratio
    # N0, N1, N2 are proportional to the integrals of |\hat{g}|^2
    # Variance in freq domain: <\xi^2> = N2/N0 (since <\xi>=0)
    
    if N0 <= 1e-15 or not np.isfinite(N0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    var_x = M2 / M0
    var_k = N2 / N0
    
    if var_x <= 0 or var_k <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(var_x * var_k)
    
    # Gradient Calculation
    # J = sqrt(Vx Vk)
    # dJ/dc = (1/(2J)) * (Vk * dVx/dc + Vx * dVk/dc)
    # Vx = M2/M0 -> dVx = (M2' M0 - M2 M0') / M0^2
    # Vk = N2/N0 -> dVk = (N2' N0 - N2 N0') / N0^2
    
    # Derivatives of p_j w.r.t c0, c4, c8
    # c12 depends on c0, c4, c8
    # dc12/dc0 = -1/945, dc12/dc4 = -3/945, dc12/dc8 = -105/945
    
    dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_j/dc_k = (direct terms) + (terms from c12) * dc12/dc_k
    # Direct terms for c0: only p0 has 1
    # Direct terms for c4: p0 has 3, p1 has -6, p2 has 1
    # Direct terms for c8: p0 has 105, p1 has -420, p2 has 210, p3 has -28, p4 has 1
    
    direct_dp = np.zeros((3, 7))
    direct_dp[0, 0] = 1.0
    direct_dp[1, 0] = 3.0
    direct_dp[1, 1] = -6.0
    direct_dp[1, 2] = 1.0
    direct_dp[2, 0] = 105.0
    direct_dp[2, 1] = -420.0
    direct_dp[2, 2] = 210.0
    direct_dp[2, 3] = -28.0
    direct_dp[2, 4] = 1.0
    
    # dp_j/dc_k = direct_dp[k, j] + dp_j/dc12 * dc12[k]
    # dp_j/dc12 is the column of p_j corresponding to c12
    dp_dc12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    grad = np.zeros(3)
    
    for k in range(3):
        dp = direct_dp[k, :] + dc12[k] * dp_dc12
        
        # Calculate dM0, dM1, dM2
        # M_k = sum_{i,j} p_i p_j I_{k+i+j}
        # dM_k = sum_{i,j} (dp_i p_j + p_i dp_j) I_{k+i+j} = 2 sum_i dp_i sum_j p_j I_{k+i+j}
        
        dM = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0: continue
            for m in range(3): # m=0,1,2
                s = 0.0
                for j in range(7):
                    idx = m + i + j
                    if idx < 13:
                        s += p[j] * _I[idx]
                dM[m] += 2.0 * dpi * s
        
        # Calculate dN0, dN1, dN2
        # q_j = p_j (-1)^j -> dq_j = dp_j (-1)^j
        dq = dp * np.array([1, -1, 1, -1, 1, -1, 1])
        
        dN = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0: continue
            for m in range(3):
                s = 0.0
                for j in range(7):
                    idx = m + i + j
                    if idx < 13:
                        s += q[j] * _I[idx]
                dN[m] += 2.0 * dqi * s
        
        dVx = (dM[2] * M0 - M2 * dM[0]) / (M0 * M0)
        dVk = (dN[2] * N0 - N2 * dN[0]) / (N0 * N0)
        
        grad[k] = (var_k * dVx + var_x * dVk) / (2.0 * bound)
        
    return bound, grad

def construct():
    """
    Returns np.array([c0, c4, c8])
    """
    if _check_time():
        return np.array([0.0, 0.0, 0.0])
        
    # Define objective wrapper for scipy
    def objective(x):
        val, _ = _bound_and_grad(x)
        return val
        
    def objective_grad(x):
        val, grad = _bound_and_grad(x)
        return val, grad

    # Starting points
    # Based on previous bests and random exploration
    starts = [
        np.array([-1.13, 0.04, 0.000308642]), # Previous seed/best
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.normal(0, 0.5, 3))
        
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if _check_time():
            break
            
        try:
            # L-BFGS-B with analytic gradient
            res = minimize(objective_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final polish with high precision
    if best_params is not None and not _check_time():
        try:
            res = minimize(objective_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        best_params = np.array([0.0, 0.0, 0.0])
        
    return best_params

if __name__ == "__main__":
    coeffs = construct()
    val, _ = _bound_and_grad(coeffs)
    print(f"Optimal coefficients: {coeffs}")
    print(f"Upper bound on C4: {val}")

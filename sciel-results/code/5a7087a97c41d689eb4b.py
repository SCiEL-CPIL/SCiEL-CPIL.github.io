import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 190.0  # Safety margin for 210s limit

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Precompute integrals I_n = int_{-inf}^{inf} x^{2n} e^{-x^2} dx
    # I_n = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # We need moments up to x^{24} for P(x)^2 where deg P = 12.
    # So we need I_n for n up to 12.
    sp = np.sqrt(np.pi)
    I = np.zeros(13) # indices 0 to 12
    I[0] = sp
    for n in range(1, 13):
        # I_n = (2n-1)/n * I_{n-1} ? No.
        # I_n / I_{n-1} = (2n-1)/2 * ... let's just use factorial ratio or recurrence
        # I_n = (2n-1)!! sqrt(pi) / 2^n
        # I_{n-1} = (2n-3)!! sqrt(pi) / 2^{n-1}
        # I_n = I_{n-1} * (2n-1)/2
        I[n] = I[n-1] * (2*n - 1) / 2.0

    def get_poly_coeffs(c0, c4, c8):
        """
        Returns coefficients p_k for P(x) = sum_{k=0}^6 p_k x^{2k}
        H0 = 1
        H4 = x^4 - 6x^2 + 3
        H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
        
        Constraint: g(0) = 0 => P(0) = 0
        P(0) = c0*1 + c4*3 + c8*105 + c12*945 = 0
        c12 = -(c0 + 3c4 + 105c8) / 945
        """
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
        
        p = np.zeros(7)
        # p0 (x^0)
        p[0] = c0 + 3.0*c4 + 105.0*c8 + 945.0*c12
        # p1 (x^2)
        p[1] = -6.0*c4 - 420.0*c8 - 7560.0*c12
        # p2 (x^4)
        p[2] = c4 + 210.0*c8 + 13860.0*c12
        # p3 (x^6)
        p[3] = -28.0*c8 - 6930.0*c12
        # p4 (x^8)
        p[4] = c8 + 1155.0*c12
        # p5 (x^10)
        p[5] = -66.0*c12
        # p6 (x^12)
        p[6] = c12
        
        return p

    def objective_and_grad(params):
        c0, c4, c8 = params
        p = get_poly_coeffs(c0, c4, c8)
        
        # Calculate moments m_k = int x^{2k} |P(x)|^2 e^{-x^2} dx
        # |P|^2 = (sum p_i x^{2i}) (sum p_j x^{2j}) = sum_{i,j} p_i p_j x^{2(i+j)}
        # m_k = sum_{i,j} p_i p_j int x^{2(k+i+j)} e^{-x^2} dx = sum_{i,j} p_i p_j I[k+i+j]
        
        m0 = 0.0
        m2 = 0.0
        # We only need m0 and m2 for variance.
        # m0 = sum_{i,j} p_i p_j I[i+j]
        # m2 = sum_{i,j} p_i p_j I[1+i+j] (since x^2 * x^{2(i+j)} = x^{2(1+i+j)})
        
        for i in range(7):
            if p[i] == 0.0: continue
            for j in range(7):
                if p[j] == 0.0: continue
                val = p[i] * p[j]
                m0 += val * I[i+j]
                m2 += val * I[1+i+j]
        
        # Calculate moments n_k for Fourier side
        # Q(x) = P(-ix) => q_k = p_k (-1)^k
        # n0 = 2pi sum_{i,j} q_i q_j I[i+j]
        # n2 = 2pi sum_{i,j} q_i q_j I[1+i+j]
        
        q = p * np.array([1, -1, 1, -1, 1, -1, 1])
        
        n0 = 0.0
        n2 = 0.0
        for i in range(7):
            if q[i] == 0.0: continue
            for j in range(7):
                if q[j] == 0.0: continue
                val = q[i] * q[j]
                n0 += val * I[i+j]
                n2 += val * I[1+i+j]
                
        n0 *= 2.0 * np.pi
        n2 *= 2.0 * np.pi
        
        if m0 <= 1e-30 or n0 <= 1e-30:
            return 1e18, np.array([1e18, 1e18, 1e18])
            
        vx = m2 / m0
        vxi = n2 / n0
        
        if vx <= 0 or vxi <= 0:
            return 1e18, np.array([1e18, 1e18, 1e18])
            
        bound = np.sqrt(vx * vxi)
        
        # Gradient
        # d(bound)/dc = 1/(2*bound) * (vxi * d(vx)/dc + vx * d(vxi)/dc)
        # d(vx)/dc = (dm2/dc * m0 - m2 * dm0/dc) / m0^2
        # d(vxi)/dc = (dn2/dc * n0 - n2 * dn0/dc) / n0^2
        
        grad = np.zeros(3)
        
        # Derivatives of p with respect to c0, c4, c8
        # p0 = c0 + 3c4 + 105c8 + 945c12
        # p1 = -6c4 - 420c8 - 7560c12
        # p2 = c4 + 210c8 + 13860c12
        # p3 = -28c8 - 6930c12
        # p4 = c8 + 1155c12
        # p5 = -66c12
        # p6 = c12
        # c12 = -(c0 + 3c4 + 105c8)/945
        # dc12/dc0 = -1/945, dc12/dc4 = -3/945, dc12/dc8 = -105/945
        
        dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
        
        # Matrix of derivatives dp_k/dc_l
        # dp0/dc = [1, 3, 105] + 945*dc12
        # dp1/dc = [0, -6, -420] - 7560*dc12
        # ...
        
        base_dp = np.zeros((7, 3))
        base_dp[0] = [1.0, 3.0, 105.0]
        base_dp[1] = [0.0, -6.0, -420.0]
        base_dp[2] = [0.0, 1.0, 210.0]
        base_dp[3] = [0.0, 0.0, -28.0]
        base_dp[4] = [0.0, 0.0, 1.0]
        base_dp[5] = [0.0, 0.0, 0.0]
        base_dp[6] = [0.0, 0.0, 0.0]
        
        coeff_dp = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
        
        dp = base_dp + np.outer(coeff_dp, dc12)
        
        # Calculate dm0/dc, dm2/dc, dn0/dc, dn2/dc
        # dm0/dc_l = 2 * sum_i dp_il * sum_j p_j I[i+j]
        # dm2/dc_l = 2 * sum_i dp_il * sum_j p_j I[1+i+j]
        
        # Precompute sums for speed
        # S0_i = sum_j p_j I[i+j]
        # S1_i = sum_j p_j I[1+i+j]
        
        S0 = np.zeros(7)
        S1 = np.zeros(7)
        for i in range(7):
            s0 = 0.0
            s1 = 0.0
            for j in range(7):
                s0 += p[j] * I[i+j]
                s1 += p[j] * I[1+i+j]
            S0[i] = s0
            S1[i] = s1
            
        dm0 = 2.0 * (dp * S0).sum(axis=0) # shape (3,)
        dm2 = 2.0 * (dp * S1).sum(axis=0)
        
        # For Fourier side
        dq = dp * np.array([1, -1, 1, -1, 1, -1, 1])[:, np.newaxis]
        
        T0 = np.zeros(7)
        T1 = np.zeros(7)
        for i in range(7):
            t0 = 0.0
            t1 = 0.0
            for j in range(7):
                t0 += q[j] * I[i+j]
                t1 += q[j] * I[1+i+j]
            T0[i] = t0
            T1[i] = t1
            
        dn0 = 2.0 * (2.0 * np.pi) * (dq * T0).sum(axis=0)
        dn2 = 2.0 * (2.0 * np.pi) * (dq * T1).sum(axis=0)
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad = (vxi * dvm2 + vx * dvn2) / (2.0 * bound)
        
        return bound, grad

    def objective(params):
        val, _ = objective_and_grad(params)
        return val

    def objective_grad(params):
        return objective_and_grad(params)

    # Seed solution
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score, _ = objective_and_grad(best_coeffs)
    
    np.random.seed(42)
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    num_starts = 50
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 10:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # Polish
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(objective, best_coeffs, jac=objective_grad, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return best_coeffs

if __name__ == "__main__":
    coeffs = construct()
    print(coeffs)

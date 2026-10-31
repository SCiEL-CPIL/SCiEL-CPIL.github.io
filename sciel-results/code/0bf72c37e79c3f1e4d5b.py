import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Indices of Hermite polynomials
    # We need H0, H4, H8, H12
    # Coefficients c0, c4, c8 are free. c12 is fixed by g(0)=0.
    
    # Precompute H_n(0) values
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Constraint: c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    # We need to compute the matrices M_norm, M_pos, M_mom for indices [0, 4, 8, 12]
    # M_norm[m,n] = int H_m H_n e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_mn
    # M_pos[m,n] = int x^2 H_m H_n e^{-x^2} dx
    # M_mom[m,n] = int (H_m' - x H_m) (H_n' - x H_n) e^{-x^2} dx
    
    indices = [0, 4, 8, 12]
    N = len(indices)
    
    M_norm = np.zeros((N, N))
    M_pos = np.zeros((N, N))
    M_mom = np.zeros((N, N))
    
    sqrt_pi = np.sqrt(np.pi)
    
    def fact(n):
        return float(np.math.factorial(n)) if hasattr(np.math, 'factorial') else float(np.prod(np.arange(1, n+1)))
        
    # Helper to compute factorial for large n if needed, but 12! is small
    # np.math.factorial is available in newer numpy, or use scipy.special.factorial
    from scipy.special import factorial
    
    for i, m in enumerate(indices):
        for j, n in enumerate(indices):
            # Norm
            if m == n:
                M_norm[i, j] = sqrt_pi * (2 ** n) * factorial(n, exact=True)
            
            # Position: int x^2 H_m H_n e^{-x^2}
            # x^2 H_n = 0.5 H_{n+2} + n H_n + 0.5 n(n-1) H_{n-2}
            # Integral is non-zero only if m matches one of n+2, n, n-2
            # And the integral of H_a H_b is sqrt(pi) 2^b b! delta_ab
            # So we need to sum over the components of x^2 H_n that match H_m
            
            # Term 1: 0.5 H_{n+2}
            if m == n + 2:
                M_pos[i, j] += 0.5 * sqrt_pi * (2 ** m) * factorial(m, exact=True)
            # Term 2: n H_n
            if m == n:
                M_pos[i, j] += n * sqrt_pi * (2 ** n) * factorial(n, exact=True)
            # Term 3: 0.5 n(n-1) H_{n-2}
            if m == n - 2:
                M_pos[i, j] += 0.5 * n * (n - 1) * sqrt_pi * (2 ** m) * factorial(m, exact=True)
                
            # Momentum: int Q_m Q_n e^{-x^2}
            # Q_k = 1.5 k H_{k-1} - 0.5 H_{k+1}
            # Q_m Q_n = (1.5 m H_{m-1} - 0.5 H_{m+1}) (1.5 n H_{n-1} - 0.5 H_{n+1})
            # = 2.25 mn H_{m-1}H_{n-1} - 0.75 m H_{m-1}H_{n+1} - 0.75 n H_{m+1}H_{n-1} + 0.25 H_{m+1}H_{n+1}
            
            def int_h(a, b):
                if a == b and a >= 0:
                    return sqrt_pi * (2 ** b) * factorial(b, exact=True)
                return 0.0
            
            M_mom[i, j] = (2.25 * m * n * int_h(m-1, n-1) 
                           - 0.75 * m * int_h(m-1, n+1) 
                           - 0.75 * n * int_h(m+1, n-1) 
                           + 0.25 * int_h(m+1, n+1))

    # Construct A matrix to map v=[c0,c4,c8] to C=[c0,c4,c8,c12]
    # c12 = -(1*c0 + 12*c4 + 1680*c8) / 665280
    K = H12_0
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/K, -12.0/K, -1680.0/K]
    ])
    
    M_norm_p = A.T @ M_norm @ A
    M_pos_p = A.T @ M_pos @ A
    M_mom_p = A.T @ M_mom @ A
    
    # Objective function
    def objective(v):
        norm_val = v @ M_norm_p @ v
        if norm_val <= 1e-12:
            return 1e20
        pos_val = v @ M_pos_p @ v
        mom_val = v @ M_mom_p @ v
        return (pos_val * mom_val) / (norm_val * norm_val)
        
    # Gradient
    def gradient(v):
        norm_val = v @ M_norm_p @ v
        pos_val = v @ M_pos_p @ v
        mom_val = v @ M_mom_p @ v
        
        # f = (P * M) / N^2
        # df/dv = ( (dP * M + P * dM) * N^2 - P * M * 2 N * dN ) / N^4
        #       = ( (dP * M + P * dM) * N - 2 P M dN ) / N^3
        
        dP = 2 * M_pos_p @ v
        dM = 2 * M_mom_p @ v
        dN = 2 * M_norm_p @ v
        
        num = (dP * mom_val + pos_val * dM) * norm_val - 2 * pos_val * mom_val * dN
        den = norm_val ** 3
        
        return num / den

    # Initial guess from seed
    v0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_v = v0.copy()
    best_val = objective(v0)
    
    # Multi-start local search
    # Since the problem is small, we can try a few random starts and the seed
    starts = [v0]
    
    # Add some perturbed starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        v_rand = v0 * rng.uniform(0.5, 1.5)
        starts.append(v_rand)
        
    # Also try a start far away
    starts.append(np.array([1.0, 0.0, 0.0]))
    starts.append(np.array([0.0, 1.0, 0.0]))
    starts.append(np.array([0.0, 0.0, 1.0]))

    for v_start in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(objective, v_start, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_v = res.x
        except Exception:
            pass

    return np.array(best_v)

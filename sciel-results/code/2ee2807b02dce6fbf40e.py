import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def _hermite_phys(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_nm1 = np.ones_like(x)
    H_n = 2 * x
    for k in range(2, n + 1):
        H_np1 = 2 * x * H_n - 2 * (k - 1) * H_nm1
        H_nm1 = H_n
        H_n = H_np1
    return H_n

def _compute_integrals(c0, c4, c8, c12):
    # Coefficients for P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) is even.
    # We need to compute N = int P^2 exp(-x^2/2)
    # M2 = int x^2 P^2 exp(-x^2/2)
    # K = int Q^2 exp(-x^2/2) where Q = P' - x/2 P
    
    # We will represent P(x) and Q(x) as polynomials in x.
    # H_n(x) can be converted to monomial form.
    # Or we can just evaluate the integrals using the property:
    # int x^k H_i(x) H_j(x) exp(-x^2/2) dx
    # This requires the moments of the weight e^{-x^2/2} against Hermite products.
    # Since degrees are low, let's expand H_n in monomials.
    
    # Monomial coefficients for H_n
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # ...
    # We can generate these up to n=13.
    
    max_n = 13
    # coeffs[n][k] is coefficient of x^k in H_n(x)
    coeffs = [[0.0] * (n + 1) for n in range(max_n + 1)]
    coeffs[0][0] = 1.0
    coeffs[1][1] = 2.0
    for n in range(2, max_n + 1):
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        for k in range(n + 1):
            val = 0.0
            if k - 1 >= 0:
                val += 2.0 * coeffs[n-1][k-1]
            if n - 2 >= 0 and k <= n - 2:
                val -= 2.0 * (n - 1) * coeffs[n-2][k]
            coeffs[n][k] = val
            
    # P(x) coefficients
    # P = c0 H0 + c4 H4 + c8 H8 + c12 H12
    p_coeffs = [0.0] * 13
    for k in range(13):
        p_coeffs[k] = c0 * coeffs[0][k] + c4 * coeffs[4][k] + c8 * coeffs[8][k] + c12 * coeffs[12][k]
        
    # Q(x) coefficients
    # Q = P' - x/2 P
    # P' = sum (k+1) p_{k+1} x^k
    # x/2 P = sum 0.5 p_k x^{k+1}
    q_coeffs = [0.0] * 14 # degree of Q is 13 (from H12' -> H11)
    for k in range(13):
        if k + 1 < 13:
            q_coeffs[k] += (k + 1) * p_coeffs[k + 1]
        if k < 14:
            q_coeffs[k + 1] -= 0.5 * p_coeffs[k]
            
    # Function to compute integral of x^m * (poly1 * poly2) * exp(-x^2/2)
    # poly1, poly2 are coefficient lists
    # Result = sum_{i,j} c1_i c2_j int x^{i+j+m} exp(-x^2/2) dx
    # int x^n exp(-x^2/2) dx = sqrt(pi/2) 2^(n/2) Gamma((n+1)/2)
    # Precompute moments M_n = int x^n exp(-x^2/2) dx for n up to 26
    
    M_n = [0.0] * 27
    for n in range(27):
        if n % 2 == 1:
            M_n[n] = 0.0
        else:
            # int x^{2k} exp(-x^2/2) dx = 2^k (2k-1)!! sqrt(pi/2)
            # (2k-1)!! = (2k)! / (2^k k!)
            # So M_{2k} = 2^k * (2k)! / (2^k k!) * sqrt(pi/2) = (2k)! / k! * sqrt(pi/2)
            k = n // 2
            M_n[n] = factorial(2*k) / factorial(k) * np.sqrt(np.pi / 2.0)
            
    def calc_int(p1, p2, m):
        total = 0.0
        max_deg1 = len(p1) - 1
        max_deg2 = len(p2) - 1
        for i in range(max_deg1 + 1):
            if p1[i] == 0: continue
            for j in range(max_deg2 + 1):
                if p2[j] == 0: continue
                n = i + j + m
                if n < 27:
                    total += p1[i] * p2[j] * M_n[n]
        return total
        
    N = calc_int(p_coeffs, p_coeffs, 0)
    M2 = calc_int(p_coeffs, p_coeffs, 2)
    K = calc_int(q_coeffs, q_coeffs, 0)
    
    if N <= 0:
        return float('inf')
        
    return (M2 * K) / (N * N)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin
    
    # Seed
    c0_seed, c4_seed, c8_seed = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    
    # Constraint: g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # c0 + 12 c4 + 1680 c8 + 665280 c12 = 0
    # c12 = -(c0 + 12 c4 + 1680 c8) / 665280
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
        return _compute_integrals(c0, c4, c8, c12)
        
    x0 = np.array([c0_seed, c4_seed, c8_seed])
    
    # Check if we are within time
    if time.time() > deadline - 10:
        return np.array([c0_seed, c4_seed, c8_seed])
        
    try:
        # Use Nelder-Mead or L-BFGS-B
        # The function is smooth.
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
        
        if time.time() > deadline - 5:
             return np.array([c0_seed, c4_seed, c8_seed])
             
        if res.success:
            best_c = res.x
        else:
            # Compare with seed
            seed_val = objective(x0)
            res_val = res.fun
            if res_val < seed_val:
                best_c = res.x
            else:
                best_c = x0
                
        # Optional: Polish with BFGS
        if time.time() < deadline - 20:
             res2 = minimize(objective, best_c, method='BFGS', 
                            options={'gtol': 1e-12, 'disp': False, 'maxiter': 100})
             if res2.fun < objective(best_c):
                 best_c = res2.x
                 
        return np.array(best_c)
        
    except Exception:
        return np.array([c0_seed, c4_seed, c8_seed])

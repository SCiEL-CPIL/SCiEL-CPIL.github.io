"""Bootstrap spawn: SEARCH approach – generate many varied candidates, keep the best."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import comb as scipy_comb

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0."""
    if k == 1:
        return 2.4048255576957728
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _build_laguerre_coeffs_fast():
    """
    Pre-compute coefficients of L_{2k}^{(-1/2)}(x) in monomial basis.
    Returns a matrix C of shape (26, 51) where C[j, k] is coeff of x^k in L_{2j}^{(-1/2)}.
    """
    C = np.zeros((26, 51))
    for j in range(26):
        n = 2 * j
        for k in range(n + 1):
            # binom(n + alpha, n - k) where alpha = -1/2
            # scipy.special.comb(n + alpha, n - k, exact=False)
            coeff = scipy_comb(n - 0.5, n - k, exact=False) * ((-1.0)**k) / math.factorial(k)
            C[j, k] = coeff
    return C

def _compute_score_fast(z_list, C):
    """
    Fast numerical approximation of the score.
    z_list: list of 12 floats.
    C: pre-computed Laguerre coefficient matrix.
    """
    try:
        z = np.array(z_list, dtype=float)
        if np.any(z <= 0) or np.any(np.diff(np.sort(z)) < 1e-8):
            return 1e100
        
        # 1. Solve for c
        # A matrix construction
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0 => sum c_j L_{2j}(0) = 0
        # L_n^{(alpha)}(0) = binom(n+alpha, n)
        for j in range(26):
            n = 2*j
            A[0, j] = scipy_comb(n - 0.5, n, exact=False)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1 => sum c_j L'_{2j}(0) = 1
        # L_n^{(alpha)}(x) = sum_{k=0}^n a_k x^k. L'(0) = a_1.
        # a_1 = binom(n+alpha, n-1) * (-1)^1 / 1! = -binom(n+alpha, n-1)
        for j in range(26):
            n = 2*j
            if n >= 1:
                A[1, j] = -scipy_comb(n - 0.5, n - 1, exact=False)
            else:
                A[1, j] = 0.0
        b[1] = 1.0
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        # We need L_{2j}(z) and L'_{2j}(z)
        # L_{2j}(z) = sum_{k=0}^{2j} C[j,k] z^k
        # L'_{2j}(z) = sum_{k=1}^{2j} k C[j,k] z^{k-1}
        
        for i in range(12):
            zi = z[i]
            row_g = 2 + 2*i
            row_dp = 3 + 2*i
            
            # Precompute powers of zi
            powers = np.zeros(51)
            powers[0] = 1.0
            for k in range(1, 51):
                powers[k] = powers[k-1] * zi
            
            for j in range(26):
                n = 2*j
                # g(zi)
                A[row_g, j] = np.dot(C[j, :n+1], powers[:n+1])
                # g'(zi)
                # sum k C[j,k] zi^{k-1}
                # = sum_{k=1}^n k C[j,k] zi^{k-1}
                # Let m = k-1. sum_{m=0}^{n-1} (m+1) C[j, m+1] zi^m
                # = dot( (1..n) * C[j, 1..n], powers[0..n-1] )
                coeffs_deriv = np.arange(1, n+1) * C[j, 1:n+1]
                A[row_dp, j] = np.dot(coeffs_deriv, powers[:n])
            
            b[row_g] = 0.0
            b[row_dp] = 0.0
        
        try:
            c = np.linalg.solve(A, b)
        except:
            return 1e100
            
        # 2. Construct g(x) coefficients
        # g_k = sum_j c_j C[j, k]
        g_coeffs = np.dot(c, C) # shape (51,)
        
        # 3. Construct D(x) = x * prod (x - zi)^2
        # Start with [1] for x^0? No, D has degree 25.
        # D(x) = x * P(x)^2 where P(x) = prod (x - zi)
        # P(x) = x^12 + p_11 x^11 + ...
        # Let's compute P(x) coefficients
        p_coeffs = np.ones(13) # p_12=1
        for zi in z:
            # multiply by (x - zi)
            new_p = np.zeros(14)
            for k in range(13):
                new_p[k+1] += p_coeffs[k]
                new_p[k] -= p_coeffs[k] * zi
            p_coeffs = new_p[:13] # keep degree 12
            
        # P(x)^2
        p2_coeffs = np.convolve(p_coeffs, p_coeffs) # degree 24
        
        # D(x) = x * P(x)^2 => shift by 1
        d_coeffs = np.zeros(26) # degree 25
        d_coeffs[1:] = p2_coeffs
        
        # 4. Polynomial division g / D
        # g has degree 50, D has degree 25. q has degree 25.
        # We want q such that g = q * D + R (R=0 ideally)
        # Long division
        g_work = g_coeffs.copy()
        q_coeffs = np.zeros(25)
        
        # Leading term of D is d_coeffs[25] = 1 (since p_coeffs[0]=1, p2_coeffs[0]=1, d_coeffs[25]=1)
        # Actually, let's check.
        # p_coeffs[0] is coeff of x^12. p_coeffs = [p_0, ..., p_12].
        # My convolve assumes standard order?
        # np.convolve([1, a], [1, b]) = [1, a+b, ab].
        # If p_coeffs[0] is x^12 coeff, then this is reversed.
        # Let's use standard math: p_coeffs[k] is coeff of x^k.
        # p_coeffs[12] = 1.
        # p2_coeffs[24] = 1.
        # d_coeffs[25] = 1.
        
        # Division:
        for k in range(50, 24, -1):
            if g_work[k] != 0:
                q_term = g_work[k] / d_coeffs[25]
                q_coeffs[k-25] = q_term
                # Subtract q_term * D(x) * x^(k-25)
                for m in range(26):
                    g_work[k-25+m] -= q_term * d_coeffs[m]
                    
        # Now q_coeffs are in standard order (q_coeffs[0] is x^0 coeff? No.)
        # In the loop, k goes 50 down to 25.
        # q_coeffs[k-25] corresponds to x^(k-25).
        # So q_coeffs[0] is x^0, q_coeffs[25] is x^25.
        
        # 5. Find roots of q(x)
        # np.roots expects coefficients in descending order.
        q_roots = np.roots(q_coeffs[::-1])
        
        # 6. Filter real roots
        real_roots = [r.real for r in q_roots if abs(r.imag) < 1e-9]
        
        if not real_roots:
            return 1e100
            
        # 7. Check sign change
        eps = 1e-9
        max_r = 0
        for r in real_roots:
            q_minus = np.polyval(q_coeffs[::-1], r - eps)
            q_plus = np.polyval(q_coeffs[::-1], r + eps)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        return max_r / (2 * math.pi)
        
    except Exception as e:
        return 1e100

def construct():
    """Return 12 double-root locations using SEARCH: many varied candidates, keep best."""
    start_time = time.time()
    # Set a hard deadline well before the 630s limit to allow for safe return
    deadline = start_time + 580.0
    
    # Pre-compute Laguerre coefficients for fast score
    C = _build_laguerre_coeffs_fast()
    
    best_score = float('inf')
    best_result = None
    
    # Initial candidate: Bessel zeros
    baseline = [_bessel_zero_J0(i+1)**2 for i in range(12)]
    best_result = list(baseline)
    best_score = _compute_score_fast(best_result, C)
    
    # Optimization using fast score
    # Parameterize z_i = exp(y_i) to ensure positivity
    def objective(y):
        z = np.exp(y)
        # Ensure distinctness in the fast score check? 
        # _compute_score_fast handles it.
        return _compute_score_fast(z, C)
        
    # Initial y
    y0 = np.log(baseline)
    
    # 1. Powell optimization from baseline
    if time.time() < deadline - 100:
        try:
            res = minimize(objective, y0, method='Powell', options={'maxiter': 100, 'ftol': 1e-6, 'xtol': 1e-6})
            if res.fun < best_score:
                z_opt = np.exp(res.x)
                if np.all(z_opt > 0) and len(set(np.round(z_opt, 10))) == 12:
                    best_result = list(z_opt)
                    best_score = res.fun
        except Exception:
            pass
        
    # 2. Random restarts with Powell
    # We have ~400s left. Each Powell run takes maybe 5-20s?
    # Let's try 5 restarts.
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 100:
            break
        # Generate a random candidate near the baseline or uniformly
        # Uniform in log-space between 0.1 and 100
        y_rand = np.random.uniform(np.log(0.1), np.log(100), 12)
        # Ensure distinctness in y-space to avoid singularities in exp? 
        # _compute_score_fast checks distinctness in z-space.
        try:
            res_r = minimize(objective, y_rand, method='Powell', options={'maxiter': 50, 'ftol': 1e-5, 'xtol': 1e-5})
            if res_r.fun < best_score:
                z_r = np.exp(res_r.x)
                if np.all(z_r > 0) and len(set(np.round(z_r, 10))) == 12:
                    best_result = list(z_r)
                    best_score = res_r.fun
        except Exception:
            pass

    # 3. Local refinement around best_result with Nelder-Mead
    if best_result is not None and time.time() < deadline - 50:
        try:
            res_nm = minimize(objective, np.log(best_result), method='Nelder-Mead', options={'maxiter': 200, 'fatol': 1e-8, 'xtol': 1e-8})
            if res_nm.fun < best_score:
                z_nm = np.exp(res_nm.x)
                if np.all(z_nm > 0) and len(set(np.round(z_nm, 10))) == 12:
                    best_result = list(z_nm)
                    best_score = res_nm.fun
        except Exception:
            pass

    final_candidate = best_result
    if final_candidate is None:
        final_candidate = baseline
        
    # Ensure distinctness and positivity
    final_candidate = sorted([max(0.01, z) for z in final_candidate])
    for i in range(1, 12):
        if final_candidate[i] - final_candidate[i-1] < 1e-6:
            final_candidate[i] = final_candidate[i-1] + 0.01
            
    return final_candidate

# EVOLVE-BLOCK-END

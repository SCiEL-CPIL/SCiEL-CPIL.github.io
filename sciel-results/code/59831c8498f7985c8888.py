import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

def get_hermite_coeffs(P_coeffs, degrees):
    """
    Converts coefficients of P(x) = sum c_k H_{d_k}(x) 
    to a dictionary or array of coefficients in the full Hermite basis H_0..H_max.
    """
    max_deg = max(degrees)
    coeffs = np.zeros(max_deg + 1)
    for c, d in zip(P_coeffs, degrees):
        if d >= 0:
            coeffs[d] += c
    return coeffs

def compute_uncertainty(c0, c4, c8):
    # 1. Determine c12
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    p_coeffs = np.array([c0, c4, c8, c12])
    degrees = np.array([0, 4, 8, 12])
    
    # Map to full basis 0..12
    p_full = np.zeros(13)
    p_full[0] = c0
    p_full[4] = c4
    p_full[8] = c8
    p_full[12] = c12
    
    # 2. Compute N = <P, P>
    # <H_n, H_n> = sqrt(pi) 2^n n!
    norms = np.sqrt(np.pi) * (2 ** np.arange(13)) * factorial(np.arange(13))
    N = np.sum(p_full * p_full * norms)
    
    if N < 1e-10:
        return 1e10 # Avoid division by zero
        
    # 3. Compute mu_x = <x P, P>
    # x H_n = 0.5 H_{n+1} + n H_{n-1}
    # <x H_n, H_m> = 0.5 <H_{n+1}, H_m> + n <H_{n-1}, H_m>
    # Non-zero if m = n+1 or m = n-1.
    mu_x = 0.0
    for n in range(13):
        if p_full[n] == 0: continue
        # Term from 0.5 H_{n+1}
        if n + 1 < 13:
            mu_x += p_full[n] * p_full[n+1] * 0.5 * norms[n+1]
        # Term from n H_{n-1}
        if n - 1 >= 0:
            mu_x += p_full[n] * p_full[n-1] * n * norms[n-1]
            
    # Note: The sum above counts each pair (n, n+1) twice?
    # Let's check.
    # Sum over n of p_n p_{n+1} 0.5 <H_{n+1}, H_{n+1}> ? No.
    # <x P, P> = sum_{n,m} p_n p_m <x H_n, H_m>
    # = sum_{n,m} p_n p_m [ 0.5 delta_{m, n+1} <H_{n+1}, H_{n+1}> + n delta_{m, n-1} <H_{n-1}, H_{n-1}> ]
    # = sum_n p_n p_{n+1} 0.5 norms[n+1] + sum_n p_n p_{n-1} n norms[n-1]
    # Let j = n+1 in first sum: sum_j p_{j-1} p_j 0.5 norms[j]
    # Let j = n-1 in second sum: sum_j p_{j+1} p_j (j+1) norms[j]
    # So total for pair (j, j+1) is: p_j p_{j+1} [ 0.5 norms[j+1] + (j+1) norms[j] ]?
    # Wait, the indices in the second sum: n = j+1. Term is p_{j+1} p_j (j+1) norms[j].
    # The first sum term for pair (j, j+1) comes from n=j. Term is p_j p_{j+1} 0.5 norms[j+1].
    # So yes, we sum over j.
    
    # My loop above:
    # For n, adds p_n p_{n+1} 0.5 norms[n+1].
    # For n, adds p_n p_{n-1} n norms[n-1].
    # Let k = n-1. Adds p_{k+1} p_k (k+1) norms[k].
    # So for a fixed pair of indices (k, k+1), the contribution is:
    # From first part (n=k): p_k p_{k+1} 0.5 norms[k+1]
    # From second part (n=k+1): p_{k+1} p_k (k+1) norms[k]
    # This is correct.
    
    # 4. Compute mu_x2 = <x^2 P, P>
    # x^2 H_n = 0.25 H_{n+2} + n H_n + 0.5 n(n-1) H_{n-2}
    mu_x2 = 0.0
    for n in range(13):
        if p_full[n] == 0: continue
        # Term 0.25 H_{n+2}
        if n + 2 < 13:
            mu_x2 += p_full[n] * p_full[n+2] * 0.25 * norms[n+2]
        # Term n H_n
        mu_x2 += p_full[n] * p_full[n] * n * norms[n]
        # Term 0.5 n(n-1) H_{n-2}
        if n - 2 >= 0:
            mu_x2 += p_full[n] * p_full[n-2] * 0.5 * n * (n-1) * norms[n-2]
            
    # Symmetry check:
    # Pair (k, k+2):
    # From n=k: p_k p_{k+2} 0.25 norms[k+2]
    # From n=k+2: p_{k+2} p_k 0.5 (k+2)(k+1) norms[k]
    # This is correct.
    
    delta_x2 = mu_x2 / N - (mu_x / N) ** 2
    
    # 5. Compute Q(x) coefficients
    # Q = P'' - 2x P' + (x^2 - 1) P
    # P = sum p_n H_n
    # P' = sum 2n p_n H_{n-1}
    # P'' = sum 4n(n-1) p_n H_{n-2}
    
    q_full = np.zeros(13)
    
    # P'' contribution
    for n in range(2, 13):
        q_full[n-2] += 4 * n * (n-1) * p_full[n]
        
    # -2x P' contribution
    # P' = sum_{k} p'_k H_k where p'_k = 2(k+1) p_{k+1}
    # x P' = sum_k p'_k x H_k = sum_k p'_k (0.5 H_{k+1} + k H_{k-1})
    # -2x P' = sum_k p'_k (-H_{k+1} - 2k H_{k-1})
    for k in range(13):
        p_k_prime = 0
        if k + 1 < 13:
            p_k_prime = 2 * (k+1) * p_full[k+1]
        else:
            p_k_prime = 0
        if p_k_prime == 0: continue
        # -H_{k+1}
        if k + 1 < 13:
            q_full[k+1] -= p_k_prime
        # -2k H_{k-1}
        if k - 1 >= 0:
            q_full[k-1] -= 2 * k * p_k_prime
            
    # (x^2 - 1) P contribution
    # x^2 P = sum_n p_n x^2 H_n = sum_n p_n (0.25 H_{n+2} + n H_n + 0.5 n(n-1) H_{n-2})
    # -P = - sum_n p_n H_n
    # Total (x^2-1)P = sum_n p_n (0.25 H_{n+2} + (n-1) H_n + 0.5 n(n-1) H_{n-2})
    for n in range(13):
        if p_full[n] == 0: continue
        # 0.25 H_{n+2}
        if n + 2 < 13:
            q_full[n+2] += 0.25 * p_full[n]
        # (n-1) H_n
        q_full[n] += (n - 1) * p_full[n]
        # 0.5 n(n-1) H_{n-2}
        if n - 2 >= 0:
            q_full[n-2] += 0.5 * n * (n-1) * p_full[n]
            
    # 6. Compute integral of Q^2
    # <Q, Q> = sum q_j^2 norms[j]
    integral_Q2 = np.sum(q_full * q_full * norms)
    
    # Delta_xi^2 = <xi^2> - <xi>^2
    # <xi^2> = integral_Q2 / N
    # <xi> = <x> in Fourier space? 
    # For real f, <xi> = 0 if f is even/odd? 
    # f(x) = P(x) e^{-x^2/2}. P has only even degrees. So P is even. f is even.
    # Fourier transform of even function is even.
    # <xi> = \int xi |hat{f}|^2 / \int |hat{f}|^2. Since |hat{f}|^2 is even, <xi> = 0.
    # So Delta_xi^2 = integral_Q2 / N.
    
    delta_xi2 = integral_Q2 / N
    
    return np.sqrt(delta_x2 * delta_xi2)

def construct():
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.000308])
    
    # Optimize
    result = minimize(compute_uncertainty, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    # Return the optimal coefficients
    return result.x


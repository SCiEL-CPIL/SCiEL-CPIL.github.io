import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 200.0  # 180s limit + safety margin, but evaluator kills at 210s. Use 190s to be safe.
    
    # Indices for the Hermite polynomials
    K = [0, 4, 8, 12]
    # Values of H_k(0)
    H0_vals = np.array([1, -12, 1008, -133056])
    
    # Precompute S_k = sqrt(2^k k! sqrt(pi))
    S = np.array([np.sqrt(2**k * np.math.factorial(k) * np.sqrt(np.pi)) for k in K])
    
    # Constraint: sum c_k H_k(0) = 0
    # c_k = d_k / S_k
    # sum (d_k / S_k) H_k(0) = 0
    # d_12 = - sum_{k in 0,4,8} (d_k H_k(0) / S_k) * (S_12 / H_12(0))
    coeff_constraint = np.array([
        -(H0_vals[0] / S[0]) * (S[3] / H0_vals[3]),
        -(H0_vals[1] / S[1]) * (S[3] / H0_vals[3]),
        -(H0_vals[2] / S[2]) * (S[3] / H0_vals[3]),
        1.0
    ])
    # d_12 = coeff_constraint[0]*d_0 + coeff_constraint[1]*d_4 + coeff_constraint[2]*d_8
    
    def get_full_d(d_free):
        d_full = np.zeros(4)
        d_full[0] = d_free[0]
        d_full[1] = d_free[1]
        d_full[2] = d_free[2]
        d_full[3] = np.dot(coeff_constraint[:3], d_free)
        return d_full

    def objective(d_free):
        d_full = get_full_d(d_free)
        
        # M0 = sum d_k^2
        M0 = np.dot(d_full, d_full)
        if M0 < 1e-10:
            return 1e10
        
        # M2 = d^T X2 d
        # X2 matrix elements: <m|x^2|n> = 0.5 * (sqrt(n(n-1)) delta_{m,n-2} + (2n+1) delta_{mn} + sqrt((n+1)(n+2)) delta_{m,n+2})
        # Indices 0..3 correspond to k=0,4,8,12
        # Note: The formula above is for adjacent indices m, n in the full harmonic oscillator basis.
        # Here our basis is sparse: 0, 4, 8, 12.
        # We need <k_i | x^2 | k_j> where k_i, k_j in K.
        # Since x^2 connects n to n-2, n, n+2, and our indices differ by 4, 
        # <k_i | x^2 | k_j> is 0 if |k_i - k_j| > 2.
        # Here min difference is 4. So X2 is diagonal?
        # Let's check.
        # <0|x^2|0> = 0.5 * (0 + 1 + 0) = 0.5
        # <4|x^2|4> = 0.5 * (4*3 + 9 + 5*6) = 0.5 * (12 + 9 + 30) = 25.5
        # Wait, formula: 0.5 * (sqrt(n(n-1)) + (2n+1) + sqrt((n+1)(n+2)))? No.
        # <n|x^2|n> = 0.5 * (0 + (2n+1) + 0) = n + 0.5.
        # So <4|x^2|4> = 4.5.
        # Let's re-verify.
        # x^2 = 0.5(a^2 + aa^dag + a^dag a + (a^dag)^2).
        # <n|aa^dag|n> = n+1. <n|a^dag a|n> = n. Sum = 2n+1.
        # <n|a^2|n> = 0. <n|(a^dag)^2|n> = 0.
        # So <n|x^2|n> = 0.5(2n+1) = n + 0.5.
        # <0|x^2|0> = 0.5
        # <4|x^2|4> = 4.5
        # <8|x^2|8> = 8.5
        # <12|x^2|12> = 12.5
        # Off-diagonals:
        # <0|x^2|4> = 0 (diff 4 > 2)
        # So X2 is diagonal with entries [0.5, 4.5, 8.5, 12.5].
        
        X2_diag = np.array([0.5, 4.5, 8.5, 12.5])
        M2 = np.dot(d_full**2, X2_diag)
        
        # E = ||D d||^2
        # D matrix:
        # phi_k' = sqrt(k/2) phi_{k-1} - sqrt((k+1)/2) phi_{k+1}
        # We need to express this in the basis K={0,4,8,12}.
        # k=0: phi_0' = -sqrt(1/2) phi_1. phi_1 is NOT in K.
        # This is a problem. The derivative of phi_0 introduces phi_1, which is not in our truncated basis.
        # However, E = integral |g'|^2.
        # g = sum d_k phi_k.
        # g' = sum d_k phi_k'.
        # E = sum_{i,j} d_i d_j <phi_i'|phi_j'>.
        # <phi_i'|phi_j'> = <phi_i | d^2/dx^2 | phi_j> (integration by parts, boundary terms vanish).
        # So E = d^T H d, where H_{ij} = <phi_i | -d^2/dx^2 | phi_j> = <phi_i | p^2 | phi_j>.
        # p = -i d/dx. p^2 = -d^2/dx^2.
        # Harmonic oscillator Hamiltonian H_O = p^2 + x^2 = 2a^dag a + 1.
        # So p^2 = H_O - x^2 = (2a^dag a + 1) - x^2.
        # <n|p^2|n> = (2n+1) - (n+0.5) = n + 0.5.
        # <n|p^2|m> = 0 if |n-m|>2 (since x^2 and a^dag a connect neighbors).
        # So H (kinetic energy matrix) is also diagonal in this sparse basis?
        # <0|p^2|0> = 0.5
        # <4|p^2|4> = 4.5
        # <8|p^2|8> = 8.5
        # <12|p^2|12> = 12.5
        # Off-diagonals are 0.
        
        H_diag = np.array([0.5, 4.5, 8.5, 12.5])
        E = np.dot(d_full**2, H_diag)
        
        # Objective: minimize P = sqrt(M2 * E / (4 * pi^2 * M0^2))
        # Equivalent to minimizing M2 * E / M0^2
        val = (M2 * E) / (M0 * M0)
        return val

    best_score = np.inf
    best_d = None
    
    # Random restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize d_free (d_0, d_4, d_8)
        # Scale doesn't matter for the ratio, but helps optimizer.
        # Try random directions
        d_init = np.random.randn(3)
        
        res = minimize(objective, d_init, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success or res.fun < best_score:
            if res.fun < best_score:
                best_score = res.fun
                best_d = res.x

    if best_d is None:
        # Fallback to seed
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

    # Convert best_d (d_0, d_4, d_8) back to c_0, c_4, c_8
    # d_k = c_k S_k => c_k = d_k / S_k
    c_best = np.array([
        best_d[0] / S[0],
        best_d[1] / S[1],
        best_d[2] / S[2]
    ])
    
    return c_best


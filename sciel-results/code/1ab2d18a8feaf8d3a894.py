import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

# Precompute Hermite polynomials coefficients for H_0, H_4, H_8, H_12
# H_n(x) = sum_{k=0}^{floor(n/2)} (-1)^k * n! / (k! * (n-2k)!) * (2x)^(n-2k)
# Actually, standard physicists' Hermite polynomials.
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 113400x^2 + 10395

# Coefficients for powers of x: x^0, x^2, x^4, ..., x^12
# Index 0: x^0, Index 1: x^2, ..., Index 6: x^12
# H_0: [1, 0, 0, 0, 0, 0, 0]
# H_4: [3, -6, 1, 0, 0, 0, 0]
# H_8: [105, -420, 210, -28, 1, 0, 0]
# H_12: [10395, -113400, 56700, -12600, 1260, -60, 1]

H_coeffs = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [3, -6, 1, 0, 0, 0, 0],
    [105, -420, 210, -28, 1, 0, 0],
    [10395, -113400, 56700, -12600, 1260, -60, 1]
], dtype=float)

# Weights for the linear combination
# g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) exp(-x^2/2)
# c12 is determined by g(0)=0.
# g(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
# c12 = -(c0 + 3c4 + 105c8) / 10395

def compute_moments(c0, c4, c8):
    """
    Computes E[x^2], E[x^4], E[x^6], E[x^8], E[x^10], E[x^12] for the distribution
    proportional to |g(x)|^2.
    Returns (norm, m2, m4, m6, m8, m10, m12)
    """
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    c = np.array([c0, c4, c8, c12])
    
    # Polynomial P(x) = sum c_i H_i(x)
    # P(x) is a polynomial of degree 12.
    # Coefficients of P(x) in basis 1, x^2, ..., x^12
    # Let P(x) = sum_{k=0}^6 p_k (x^2)^k
    p = c @ H_coeffs
    
    # We need integrals of (x^2)^i (x^2)^j exp(-x^2) dx from -inf to inf
    # Integral I_{ij} = ∫ (x^2)^{i+j} e^{-x^2} dx = ∫ x^{2(i+j)} e^{-x^2} dx
    # Let m = i+j. ∫ x^{2m} e^{-x^2} dx = Γ(m+0.5) = 0.5 * 2^{-m-0.5} * Γ(m+0.5) * 2?
    # Formula: ∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = \frac{(2m)!}{4^m m!} \sqrt{\pi}
    
    max_power = 12 # max degree of P is 12, so P(x)^2 is degree 24, x^2 is degree 2, total 26?
    # Wait, |g(x)|^2 = P(x)^2 e^{-x^2}.
    # P(x) is degree 12. P(x)^2 is degree 24.
    # We need moments up to x^12 for sigma_x^2 and sigma_xi^2?
    # sigma_x^2 = E[x^2] - E[x]^2. By symmetry E[x]=0. So sigma_x^2 = E[x^2].
    # sigma_xi^2 = E[xi^2]. xi is Fourier dual.
    # For g(x) = P(x) e^{-x^2/2}, the Fourier transform is also a polynomial times Gaussian.
    # Specifically, if P(x) = sum p_k x^{2k}, then \hat{P}(\xi) involves coefficients.
    # Alternatively, use the fact that for any function, sigma_x^2 sigma_xi^2 >= 1/4.
    # We can compute sigma_x^2 directly from moments of |g|^2.
    # For sigma_xi^2, we can use the property that E[xi^2] = E[ - d^2/dx^2 ]? No.
    # E[xi^2] = (1/||g||^2) ∫ |g''(x)|^2 dx ? No, that's for specific cases.
    # Actually, E[xi^2] = (1/||g||^2) ∫ | \hat{g}'(\xi) |^2 d\xi ? No.
    # Uncertainty product: sigma_x^2 sigma_xi^2 = (E[x^2] - E[x]^2) (E[xi^2] - E[xi]^2).
    # E[xi^2] = (1/||g||^2) ∫ |g''(x)|^2 dx is NOT correct generally.
    # Correct relation: E[xi^2] = (1/||g||^2) ∫ | -i \hat{g}'(\xi) |^2 d\xi = (1/||g||^2) ∫ |g''(x)|^2 dx ?
    # Yes, Parseval: ∫ |g''(x)|^2 dx = ∫ | (i\xi)^2 \hat{g}(\xi) |^2 d\xi = ∫ \xi^4 |\hat{g}|^2 d\xi. That's E[xi^4].
    # We need E[xi^2].
    # ∫ |g'(x)|^2 dx = ∫ |i\xi \hat{g}(\xi)|^2 d\xi = ∫ \xi^2 |\hat{g}|^2 d\xi = ||g||^2 E[xi^2].
    # So sigma_xi^2 = (1/||g||^2) ∫ |g'(x)|^2 dx.
    
    # So we need:
    # 1. Norm squared: N = ∫ P(x)^2 e^{-x^2} dx
    # 2. E[x^2] = (1/N) ∫ x^2 P(x)^2 e^{-x^2} dx
    # 3. ∫ |g'(x)|^2 dx = ∫ (P'(x) e^{-x^2/2} - x P(x) e^{-x^2/2})^2 dx
    #    = ∫ (P'(x) - x P(x))^2 e^{-x^2} dx
    
    # Let's compute these integrals using the polynomial coefficients.
    # P(x) = sum_{k=0}^6 p_k x^{2k}
    # P'(x) = sum_{k=1}^6 2k p_k x^{2k-1}
    # Q(x) = P'(x) - x P(x)
    # Q(x) is a polynomial.
    
    # Integral of x^{2m} e^{-x^2} is J_m = \frac{(2m)!}{4^m m!} \sqrt{\pi}
    
    # We need to compute:
    # N = ∫ P(x)^2 e^{-x^2} dx
    # M2 = ∫ x^2 P(x)^2 e^{-x^2} dx
    # D = ∫ Q(x)^2 e^{-x^2} dx
    
    # P(x)^2 = (sum p_i x^{2i}) (sum p_j x^{2j}) = sum_{i,j} p_i p_j x^{2(i+j)}
    # N = sum_{i,j} p_i p_j J_{i+j}
    # M2 = sum_{i,j} p_i p_j J_{i+j+1}
    
    # Q(x) = P'(x) - x P(x)
    # P'(x) = sum_{k=1}^6 2k p_k x^{2k-1}
    # x P(x) = sum_{k=0}^6 p_k x^{2k+1}
    # Let's express Q(x) in terms of odd powers x^{2k+1}.
    # Coeff of x^{2k+1} in P'(x): if 2k+1 = 2m-1 => m=k+1. Coeff is 2(k+1)p_{k+1}.
    # Coeff of x^{2k+1} in x P(x): p_k.
    # So Q(x) = sum_{k=0}^5 (2(k+1)p_{k+1} - p_k) x^{2k+1}  (Note: p_6 term in P' is 12 p_6 x^{11}, in xP is p_5 x^{11} + p_6 x^{13}?)
    # Let's be careful with indices.
    # P(x) = p_0 + p_1 x^2 + ... + p_6 x^{12}
    # P'(x) = 2 p_1 x + 4 p_2 x^3 + ... + 12 p_6 x^{11}
    # x P(x) = p_0 x + p_1 x^3 + ... + p_5 x^{11} + p_6 x^{13}
    # Q(x) = (2 p_1 - p_0) x + (4 p_2 - p_1) x^3 + ... + (12 p_6 - p_5) x^{11} - p_6 x^{13}
    # Q(x) = sum_{k=0}^6 q_k x^{2k+1}
    # q_0 = 2 p_1 - p_0
    # q_k = 2(k+1) p_{k+1} - p_k for k=1..5
    # q_6 = -p_6
    
    q = np.zeros(7)
    q[0] = 2*p[1] - p[0]
    for k in range(1, 6):
        q[k] = 2*(k+1)*p[k+1] - p[k]
    q[6] = -p[6]
    
    # D = ∫ Q(x)^2 e^{-x^2} dx
    # Q(x)^2 = sum_{i,j} q_i q_j x^{2i+1} x^{2j+1} = sum_{i,j} q_i q_j x^{2(i+j+1)}
    # D = sum_{i,j} q_i q_j J_{i+j+1}
    
    # Precompute J_m for m=0..13 (max i+j+1 is 6+6+1=13)
    # J_m = (2m)! / (4^m m!) * sqrt(pi)
    J = np.zeros(14)
    for m in range(14):
        # Use log gamma for stability? m is small, factorial is fine.
        J[m] = (factorial(2*m) / (4**m * factorial(m))) * np.sqrt(np.pi)
        
    N = 0.0
    M2 = 0.0
    for i in range(7):
        for j in range(7):
            val = p[i] * p[j]
            N += val * J[i+j]
            M2 += val * J[i+j+1]
            
    D = 0.0
    for i in range(7):
        for j in range(7):
            val = q[i] * q[j]
            D += val * J[i+j+1]
            
    return N, M2, D

def objective_and_grad(c):
    c0, c4, c8 = c
    N, M2, D = compute_moments(c0, c4, c8)
    if N <= 1e-12:
        return 1e10, np.zeros(3)
    
    sigma_x2 = M2 / N
    sigma_xi2 = D / N
    
    # Objective is sigma_x2 * sigma_xi2
    obj = sigma_x2 * sigma_xi2
    
    # Gradient
    # We need dN/dc, dM2/dc, dD/dc
    # This requires differentiating the sums.
    # p_k are linear in c. q_k are linear in c.
    # N = sum p_i p_j J_{i+j}
    # dN/dc_m = sum_{i,j} (dp_i/dc_m p_j + p_i dp_j/dc_m) J_{i+j}
    # dp_k/dc_m:
    # c = [c0, c4, c8]
    # p = c @ H_coeffs
    # dp_k/dc0 = H_coeffs[0, k]
    # dp_k/dc4 = H_coeffs[1, k]
    # dp_k/dc8 = H_coeffs[2, k]
    # dp_k/dc12 is not needed directly as c12 is a function of c.
    # Wait, c12 depends on c0, c4, c8.
    # c12 = -(c0 + 3c4 + 105c8) / 10395
    # dc12/dc0 = -1/10395
    # dc12/dc4 = -3/10395
    # dc12/dc8 = -105/10395
    
    # Let's compute gradients of p and q with respect to c0, c4, c8.
    dp_dc = np.zeros((7, 3))
    for m in range(3):
        # c_m is c0, c4, c8
        # c12 changes by dc12_dc[m]
        dc12 = -np.array([1, 3, 105])[m] / 10395.0
        # dp_k = sum_{i in 0..3} c_i H_i_k
        # d p_k / d c_m = H_m_k + H_12_k * dc12_dc_m
        for k in range(7):
            dp_dc[k, m] = H_coeffs[m, k] + H_coeffs[3, k] * dc12
    
    dq_dc = np.zeros((7, 3))
    # q_k are linear in p.
    # q_0 = 2 p_1 - p_0
    # q_k = 2(k+1) p_{k+1} - p_k
    # q_6 = -p_6
    for m in range(3):
        dq_dc[0, m] = 2*dp_dc[1, m] - dp_dc[0, m]
        for k in range(1, 6):
            dq_dc[k, m] = 2*(k+1)*dp_dc[k+1, m] - dp_dc[k, m]
        dq_dc[6, m] = -dp_dc[6, m]
        
    grad_N = np.zeros(3)
    grad_M2 = np.zeros(3)
    grad_D = np.zeros(3)
    
    for m in range(3):
        dN_m = 0.0
        dM2_m = 0.0
        for i in range(7):
            for j in range(7):
                # d(p_i p_j)/dc_m = dp_i/dc_m p_j + p_i dp_j/dc_m
                term = dp_dc[i, m] * p[j] + p[i] * dp_dc[j, m]
                dN_m += term * J[i+j]
                dM2_m += term * J[i+j+1]
        grad_N[m] = dN_m
        grad_M2[m] = dM2_m
        
        dD_m = 0.0
        for i in range(7):
            for j in range(7):
                term = dq_dc[i, m] * q[j] + q[i] * dq_dc[j, m]
                dD_m += term * J[i+j+1]
        grad_D[m] = dD_m
        
    # obj = (M2/N) * (D/N) = M2 * D / N^2
    # d obj / dc = (dM2 * D * N^2 - M2 * D * 2 N dN + M2 * dD * N^2 - M2 * D * 2 N dN) / N^4
    # = (D/N^2) dM2 + (M2/N^2) dD - 2 (M2 D / N^3) dN
    
    grad_obj = (D / (N**2)) * grad_M2 + (M2 / (N**2)) * grad_D - 2 * (M2 * D / (N**3)) * grad_N
    
    return obj, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, keep margin
    
    # Initial guesses
    # Seed: [-1.13, 0.04, 1/3240]
    initial_guesses = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([0.5, -0.5, 0.1]),
        np.array([-0.5, 0.5, -0.1])
    ]
    
    best_score = float('inf')
    best_c = initial_guesses[0]
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                guess, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # If we have time, do a few more random starts
    np.random.seed(42)
    while time.time() < deadline - 20:
        guess = np.random.uniform(-2, 2, size=3)
        try:
            res = minimize(
                objective_and_grad, 
                guess, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    return best_c

# Keep the function name exactly as requested
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # Initial guesses
    initial_guesses = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([0.5, -0.5, 0.1]),
        np.array([-0.5, 0.5, -0.1])
    ]
    
    best_score = float('inf')
    best_c = initial_guesses[0]
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                guess, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # Random restarts
    np.random.seed(42)
    while time.time() < deadline - 20:
        guess = np.random.uniform(-2, 2, size=3)
        try:
            res = minimize(
                objective_and_grad, 
                guess, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    return best_c

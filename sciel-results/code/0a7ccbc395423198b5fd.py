import numpy as np
import scipy.optimize as opt
import time

# Define Hermite Polynomials H_n(x) (Physicists')
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H_12(x) = 4096x^12 - 73728x^10 + 524280x^8 - 1720320x^6 + 2612736x^4 - 1720320x^2 + 103216

def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16*x**4 - 48*x**2 + 12

def H8(x):
    return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680

def H12(x):
    return 4096*x**12 - 73728*x**10 + 524280*x**8 - 1720320*x**6 + 2612736*x**4 - 1720320*x**2 + 103216

# Values at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 103216.0

def compute_variances(c0, c4, c8):
    # Calculate c12 from constraint g(0)=0
    c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    # We need to integrate |P(x)|^2 e^{-x^2} and x^2 |P(x)|^2 e^{-x^2}
    # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # Instead of numerical integration which might be slow or inaccurate, 
    # we can use the orthogonality of Hermite polynomials.
    # \int H_n(x) H_m(x) e^{-x^2} dx = 2^n n! \sqrt{\pi} \delta_{nm}
    # \int x^2 H_n(x) H_m(x) e^{-x^2} dx can be derived.
    # x^2 H_n = 0.5 (H_{n+2} + 2n H_n)  ... wait, relation is x H_n = 0.5 H_{n+1} + n H_{n-1}
    # So x^2 H_n = x(0.5 H_{n+1} + n H_{n-1}) = 0.5(0.5 H_{n+2} + (n+1)H_n) + n(0.5 H_n + (n-1)H_{n-2})
    # x^2 H_n = 0.25 H_{n+2} + 0.5(n+1)H_n + 0.5n H_n + n(n-1)H_{n-2}
    # x^2 H_n = 0.25 H_{n+2} + (n+0.5)H_n + n(n-1)H_{n-2}
    
    # Let's define a function to get coefficients of x^2 H_k in terms of H_j
    # For k=0: x^2 H0 = 0.25 H2 + 0.5 H0
    # For k=4: x^2 H4 = 0.25 H6 + 4.5 H4 + 12 H2
    # For k=8: x^2 H8 = 0.25 H10 + 8.5 H8 + 56 H6
    # For k=12: x^2 H12 = 0.25 H14 + 12.5 H12 + 132 H10
    
    # Norms: N_n = 2^n n! sqrt(pi)
    # We only need the diagonal terms for the variance if the basis is orthogonal?
    # No, P(x) is a linear combination. The cross terms vanish in the norm integral due to orthogonality.
    # But in the x^2 integral, cross terms might appear if the expansion of x^2 H_k contains H_m with m != k.
    # However, the integral \int (sum c_i H_i) (sum c_j x^2 H_j) e^{-x^2}
    # = sum_i sum_j c_i c_j \int H_i x^2 H_j e^{-x^2}
    # Since x^2 H_j is a linear combo of H_{j-2}, H_j, H_{j+2}, the integral is non-zero only if i is in {j-2, j, j+2}.
    
    # Let's precompute the matrix M where M[i,j] = \int H_i x^2 H_j e^{-x^2} / sqrt(N_i N_j) ?
    # Actually, let's just compute the integrals directly using the expansion.
    
    # Coefficients of P(x)
    coeffs = {0: c0, 4: c4, 8: c8, 12: c12}
    
    # Norm integral I0 = sum_{n in coeffs} c_n^2 N_n
    # N_n = 2^n * n! * sqrt(pi)
    N = lambda n: (2**n) * (np.math.factorial(n)) * np.sqrt(np.pi)
    
    I0 = 0.0
    I2 = 0.0
    
    # Expansion of x^2 H_j:
    # x^2 H_j = A_j H_{j+2} + B_j H_j + C_j H_{j-2}
    # A_j = 0.25
    # B_j = j + 0.5
    # C_j = j(j-1)
    
    # We need \int H_i (A_j H_{j+2} + B_j H_j + C_j H_{j-2}) e^{-x^2}
    # = A_j N_i \delta_{i, j+2} + B_j N_i \delta_{i, j} + C_j N_i \delta_{i, j-2}
    # Note: The integral is symmetric in i,j for the x^2 term? 
    # \int H_i x^2 H_j = \int H_j x^2 H_i.
    # So we can sum over j (the index from the right side of x^2) and match i from the left.
    
    # Let's iterate over j (indices present in P)
    for j in coeffs:
        cj = coeffs[j]
        Nj = N(j)
        
        # Terms in x^2 H_j
        terms = []
        if j+2 in coeffs: # i = j+2
             # Contribution to I2 from c_i (which is c_{j+2}) * c_j * (A_j * N_{j+2})
             # Wait, the integral is \sum_i \sum_j c_i c_j \int H_i x^2 H_j
             # Let's fix j and sum over possible i.
             pass
             
    # Simpler: Just compute the polynomial P(x) and integrate numerically? 
    # No, analytical is faster and exact.
    
    # Let's build the vector of coefficients c_n for n in [0, 2, ..., 14]
    # Max degree in P is 12. x^2 P has max degree 14.
    # We need \int P(x) (x^2 P(x)) e^{-x^2} dx.
    # Let P(x) = \sum c_n H_n(x).
    # x^2 P(x) = \sum c_n (0.25 H_{n+2} + (n+0.5) H_n + n(n-1) H_{n-2})
    # Let Q(x) = x^2 P(x) = \sum q_m H_m(x).
    # q_m = 0.25 c_{m-2} + (m+0.5) c_m + m(m-1) c_{m+2}
    # (with c_k = 0 if k not in {0,4,8,12})
    
    # Then I2 = \int P Q e^{-x^2} = \sum_m c_m q_m N_m.
    
    c_map = {0: c0, 4: c4, 8: c8, 12: c12}
    
    # Get all indices involved
    indices = [0, 2, 4, 6, 8, 10, 12, 14]
    
    # Calculate q_m for each m
    q = {}
    for m in indices:
        val = 0.0
        # Term from c_{m-2} * 0.25
        if m-2 in c_map:
            val += 0.25 * c_map[m-2]
        # Term from c_m * (m+0.5)
        if m in c_map:
            val += (m + 0.5) * c_map[m]
        # Term from c_{m+2} * m(m-1)
        if m+2 in c_map:
            val += m * (m - 1) * c_map[m+2]
        q[m] = val
        
    # Calculate I0 and I2
    for m in indices:
        cm = c_map.get(m, 0.0)
        qm = q[m]
        Nm = N(m)
        I0 += cm * cm * Nm
        I2 += cm * qm * Nm
        
    sigma_x2 = I2 / I0
    
    # Now for sigma_xi2
    # \hat{g}(\xi) \propto \sum d_n H_n(\xi) e^{-\xi^2/2}
    # d_0 = c0, d_4 = -c4, d_8 = c8, d_12 = -c12
    d_map = {0: c0, 4: -c4, 8: c8, 12: -c12}
    
    qd = {}
    for m in indices:
        val = 0.0
        if m-2 in d_map:
            val += 0.25 * d_map[m-2]
        if m in d_map:
            val += (m + 0.5) * d_map[m]
        if m+2 in d_map:
            val += m * (m - 1) * d_map[m+2]
        qd[m] = val
        
    I0_d = 0.0
    I2_d = 0.0
    for m in indices:
        dm = d_map.get(m, 0.0)
        qdm = qd[m]
        Nm = N(m)
        I0_d += dm * dm * Nm
        I2_d += dm * qdm * Nm
        
    sigma_xi2 = I2_d / I0_d
    
    return sigma_x2 * sigma_xi2

def objective(vars):
    c0, c4, c8 = vars
    return compute_variances(c0, c4, c8)

def construct():
    start_time = time.time()
    deadline = 180.0 # Seconds
    
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    best_score = objective(x0)
    best_x = x0
    
    # Perform optimization
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 5.0:
             # Maybe try a different starting point or more iterations?
             # Nelder-Mead is usually fast.
             pass
             
    except Exception as e:
        pass
        
    # Ensure we return an array of 3 floats
    return np.array(best_x)

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16*x**4 - 48*x**2 + 12

def H8(x):
    return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680

def H12(x):
    return 4096*x**12 - 73728*x**10 + 524280*x**8 - 1720320*x**6 + 2612736*x**4 - 1720320*x**2 + 103216

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 103216.0

def compute_variances(c0, c4, c8):
    c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    N = lambda n: (2**n) * (np.math.factorial(n)) * np.sqrt(np.pi)
    
    c_map = {0: c0, 4: c4, 8: c8, 12: c12}
    indices = [0, 2, 4, 6, 8, 10, 12, 14]
    
    q = {}
    for m in indices:
        val = 0.0
        if m-2 in c_map:
            val += 0.25 * c_map[m-2]
        if m in c_map:
            val += (m + 0.5) * c_map[m]
        if m+2 in c_map:
            val += m * (m - 1) * c_map[m+2]
        q[m] = val
        
    I0 = 0.0
    I2 = 0.0
    for m in indices:
        cm = c_map.get(m, 0.0)
        qm = q[m]
        Nm = N(m)
        I0 += cm * cm * Nm
        I2 += cm * qm * Nm
        
    sigma_x2 = I2 / I0
    
    d_map = {0: c0, 4: -c4, 8: c8, 12: -c12}
    
    qd = {}
    for m in indices:
        val = 0.0
        if m-2 in d_map:
            val += 0.25 * d_map[m-2]
        if m in d_map:
            val += (m + 0.5) * d_map[m]
        if m+2 in d_map:
            val += m * (m - 1) * d_map[m+2]
        qd[m] = val
        
    I0_d = 0.0
    I2_d = 0.0
    for m in indices:
        dm = d_map.get(m, 0.0)
        qdm = qd[m]
        Nm = N(m)
        I0_d += dm * dm * Nm
        I2_d += dm * qdm * Nm
        
    sigma_xi2 = I2_d / I0_d
    
    return sigma_x2 * sigma_xi2

def objective(vars):
    c0, c4, c8 = vars
    return compute_variances(c0, c4, c8)

def construct():
    start_time = time.time()
    deadline = 180.0
    
    x0 = np.array([-1.13, 0.04, 0.0003])
    best_score = objective(x0)
    best_x = x0
    
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
             
    except Exception:
        pass
        
    return np.array(best_x)
# EVOLVE-BLOCK-END

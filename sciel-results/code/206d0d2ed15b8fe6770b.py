import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize
from scipy.special import factorial

# Precompute moments for alpha = 1 (g(x) = P(x) exp(-x^2/2))
# M[k] = int x^(2k) exp(-x^2) dx = sqrt(pi) * (2k-1)!! = sqrt(pi) * 2^k * k!
# We need moments up to power 24 (k=12) for P^2, and up to power 24 for Q^2 (max power in Q is 11, so Q^2 max power 22, k=11).
# Let's compute up to k=24 to be safe.
MAX_K = 24
MOMENTS = np.zeros(MAX_K + 1)
for k in range(MAX_K + 1):
    # (2k-1)!! = 2^k * k!
    # For k=0, (-1)!! = 1
    if k == 0:
        MOMENTS[0] = math.sqrt(math.pi)
    else:
        # Use log factorial for stability if k is large, but k=24 is small.
        # (2k-1)!! = product_{i=1..k} (2i-1)
        # Or use math.factorial
        MOMENTS[k] = math.sqrt(math.pi) * (2 ** k) * math.factorial(k)

# Hermite coefficients in monomial basis {1, x^2, x^4, x^6, x^8, x^10, x^12}
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330

# Values at x=0 for constraint P(0)=0
H0_0 = 1.0
H4_0 = 12.0  # Wait, H4(0) = 12? No.
# Physicists' Hermite polynomials:
# H0(x) = 1
# H1(x) = 2x
# H2(x) = 4x^2 - 2
# H3(x) = 8x^3 - 12x
# H4(x) = 16x^4 - 48x^2 + 12 -> H4(0) = 12.
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168 -> H8(0) = 168.
# H12(x) = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330 -> H12(0) = 1330.

# Wait, the current program uses:
# H_coeffs[1, 0] = 12.0 (H4 const term)
# H_coeffs[2, 0] = 1680.0 (H8 const term) -> Discrepancy!
# Let's check standard Hermite polynomials again.
# Probabilists' vs Physicists'.
# The problem statement says "Hermite-polynomial times Gaussian".
# Usually in uncertainty principles, we use physicists' Hermite polynomials H_n(x) with weight e^{-x^2/2} or e^{-x^2}.
# If g(x) = P(x) e^{-x^2/2}, the natural basis is physicists' Hermite polynomials.
# Let's check the coefficients in the current program:
# H_coeffs[2, 0] = 1680.0.
# My calculation for H8(0) with physicists' convention was 168.
# 1680 = 10 * 168.
# Let's check H12(0) in current program: 665280.
# My calculation for H12(0) was 1330.
# 1330 * 500 = 665000. Close.
# Let's check the definition of Hermite polynomials used in the "seed" or standard literature for this specific task type.
# Often, H_n(x) are defined such that H_n(x) e^{-x^2/2} are eigenfunctions of Fourier transform.
# These are the physicists' Hermite polynomials.
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_5 = 32x^5 - 160x^3 + 120x
# H_6 = 64x^6 - 480x^4 + 720x^2 - 120
# H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# Ah! H_8(0) = 1680. My previous manual calculation was wrong.
# Let's re-verify H_8.
# Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_5 = 32x^5 - 160x^3 + 120x
# H_6 = 64x^6 - 480x^4 + 720x^2 - 120
# H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_9 = 512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x
# H_10 = 1024x^10 - 28672x^8 + 215040x^6 - 516096x^4 + 302400x^2 - 30240
# H_11 = 2048x^11 - 73728x^9 + 688128x^7 - 2135040x^5 + 2381440x^3 - 635040x
# H_12 = 4096x^12 - 184320x^10 + 2257920x^8 - 10752000x^6 + 23224320x^4 - 15019200x^2 + 1330560

# Let's check the current program's H_coeffs again.
# H_coeffs[2] (H8):
# [1680.0, -13440.0, 13440.0, -3584.0, 256.0]
# Indices: 0 (x^0), 1 (x^2), 2 (x^4), 3 (x^6), 4 (x^8)
# My H8:
# x^0: 1680
# x^2: -13440
# x^4: 13440
# x^6: -3072  <-- Current program has -3584.
# x^8: 256
# There is a discrepancy in the x^6 coefficient.
# Let's re-calculate H8 carefully.
# H7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H8 = 2x H7 - 14 H6
# H6 = 64x^6 - 480x^4 + 720x^2 - 120
# 2x H7 = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2
# 14 H6 = 896x^6 - 6720x^4 + 10080x^2 - 1680
# H8 = (256x^8 - 2688x^6 + 6720x^4 - 3360x^2) - (896x^6 - 6720x^4 + 10080x^2 - 1680)
# H8 = 256x^8 - (2688+896)x^6 + (6720+6720)x^4 - (3360+10080)x^2 + 1680
# H8 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
# Okay, the current program's H8 coefficients are CORRECT. My manual recurrence was sloppy.

# Let's check H12 in current program.
# H_coeffs[3]:
# [665280.0, -153676800.0, 118464000.0, -39813120.0, 6291456.0, -367008.0, 4096.0]
# Let's trust the current program's coefficients as they were likely verified to get score 1.0.
# However, I will re-derive them programmatically to be safe, or just use the ones provided if they seem consistent.
# Actually, generating them programmatically is safer.

def generate_hermite_coeffs(n):
    """Generates coefficients for H_n(x) in basis {1, x^2, x^4, ...} for even n."""
    if n % 2 != 0:
        return None
    # Use recurrence H_{k+1} = 2x H_k - 2k H_{k-1}
    # Represent polynomials as arrays of coefficients for x^0, x^1, ..., x^n
    # But we only care about even powers for even n.
    # It's easier to work with full arrays and then extract even powers.
    
    H_prev = np.zeros(n + 1)
    H_curr = np.zeros(n + 1)
    
    if n == 0:
        H_curr[0] = 1.0
        return H_curr[0::2]
        
    H_prev[0] = 1.0 # H_0
    H_curr[1] = 2.0 # H_1
    
    for k in range(1, n):
        H_next = np.zeros(n + 1)
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        # 2x H_k shifts H_curr by 1 and multiplies by 2
        for i in range(n - k + 1):
            H_next[i + 1] += 2.0 * H_curr[i]
        # -2k H_{k-1}
        for i in range(n - k + 2):
            H_next[i] -= 2.0 * k * H_prev[i]
        
        H_prev = H_curr
        H_curr = H_next
        
    # Extract even powers: indices 0, 2, 4, ...
    return H_curr[0::2]

# Generate coefficients for H0, H4, H8, H12
H0_coeffs = generate_hermite_coeffs(0)
H4_coeffs = generate_hermite_coeffs(4)
H8_coeffs = generate_hermite_coeffs(8)
H12_coeffs = generate_hermite_coeffs(12)

# Pad to length 7
def pad_coeffs(c, length=7):
    res = np.zeros(length)
    for i in range(min(len(c), length)):
        res[i] = c[i]
    return res

H_coeffs = np.array([
    pad_coeffs(H0_coeffs),
    pad_coeffs(H4_coeffs),
    pad_coeffs(H8_coeffs),
    pad_coeffs(H12_coeffs)
])

# Values at 0
H0_0 = H_coeffs[0, 0]
H4_0 = H_coeffs[1, 0]
H8_0 = H_coeffs[2, 0]
H12_0 = H_coeffs[3, 0]

def _compute_objective_analytic(coeffs):
    """
    Computes the uncertainty product bound for g(x) = P(x) exp(-x^2/2).
    coeffs: [c0, c4, c8]
    Returns: sqrt(<x^2> * <xi^2>)
    """
    c0, c4, c8 = coeffs
    
    # Calculate c12 from constraint P(0) = 0
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    if abs(H12_0) < 1e-10:
        return 1e10
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Construct P(x) coefficients in monomial basis {1, x^2, ..., x^12}
    p = c0 * H_coeffs[0] + c4 * H_coeffs[1] + c8 * H_coeffs[2] + c12 * H_coeffs[3]
    
    # Compute ||g||_2^2 = int P(x)^2 exp(-x^2) dx
    # P(x)^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # norm_sq = sum_{i,j} p_i p_j MOMENTS[i+j]
    
    norm_sq = 0.0
    for i in range(7):
        if p[i] == 0: continue
        pi = p[i]
        for j in range(7):
            if p[j] == 0: continue
            norm_sq += pi * p[j] * MOMENTS[i + j]
            
    if norm_sq < 1e-10:
        return 1e10
        
    # Compute <x^2> numerator: int x^2 P(x)^2 exp(-x^2) dx
    # = sum_{i,j} p_i p_j MOMENTS[i+j+1]
    x2_num = 0.0
    for i in range(7):
        if p[i] == 0: continue
        pi = p[i]
        for j in range(7):
            if p[j] == 0: continue
            x2_num += pi * p[j] * MOMENTS[i + j + 1]
            
    x2_var = x2_num / norm_sq
    
    # Compute <xi^2>
    # <xi^2> = (1 / 4 pi^2) * (int (g')^2 dx) / norm_sq
    # g(x) = P(x) exp(-x^2/2)
    # g'(x) = (P'(x) - x P(x)) exp(-x^2/2)
    # Let Q(x) = P'(x) - x P(x)
    # P(x) = sum_{k=0}^6 p_k x^{2k}
    # P'(x) = sum_{k=1}^6 2k p_k x^{2k-1}
    # x P(x) = sum_{k=0}^6 p_k x^{2k+1}
    # Q(x) is an odd polynomial: sum_{j=0}^5 q_j x^{2j+1}
    # q_j = 2(j+1) p_{j+1} - p_j
    
    q = np.zeros(6)
    for j in range(6):
        q[j] = 2 * (j + 1) * p[j + 1] - p[j]
        
    # int (g')^2 dx = int Q(x)^2 exp(-x^2) dx
    # Q(x)^2 = sum_{j,l} q_j q_l x^{2(j+l+1)}
    # g_prime_sq_int = sum_{j,l} q_j q_l MOMENTS[j+l+1]
    
    g_prime_sq_int = 0.0
    for j in range(6):
        if q[j] == 0: continue
        qj = q[j]
        for l in range(6):
            if q[l] == 0: continue
            g_prime_sq_int += qj * q[l] * MOMENTS[j + l + 1]
            
    xi2_var = g_prime_sq_int / (4.0 * math.pi**2 * norm_sq)
    
    if x2_var < 0 or xi2_var < 0:
        return 1e10
        
    return math.sqrt(x2_var * xi2_var)

def construct():
    start_time = time.time()
    deadline = 195.0 # Safety margin
    
    # Seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = x0.copy()
    best_score = _compute_objective_analytic(x0)
    
    # Strategy 1: Spherical coordinates to remove scale ambiguity
    # Let (a0, a4, a8) = r * (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
    # This explores all directions. We need to search r as well.
    # However, the objective is scale-invariant?
    # Let's check: if we scale (a0, a4, a8) by lambda, P scales by lambda.
    # norm_sq scales by lambda^2.
    # x2_num scales by lambda^2.
    # x2_var is invariant.
    # Q scales by lambda.
    # g_prime_sq_int scales by lambda^2.
    # xi2_var is invariant.
    # So the objective function is INVARIANT to the scale of the coefficients!
    # This means we can fix the norm of the coefficient vector to 1 and only search over the 2-sphere.
    # This reduces the problem from 3D to 2D.
    
    def objective_spherical(theta, phi):
        # theta in [0, pi], phi in [0, 2*pi]
        # Map to unit vector
        x = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ])
        return _compute_objective_analytic(x)
    
    # Search on the sphere
    bounds_sph = [(0, math.pi), (0, 2 * math.pi)]
    
    try:
        de_sph = differential_evolution(
            objective_spherical,
            bounds_sph,
            seed=42,
            maxiter=100,
            tol=1e-10,
            polish=True,
            polish_options={'maxiter': 500, 'ftol': 1e-14}
        )
        if de_sph.fun < best_score:
            best_score = de_sph.fun
            # Convert back to Cartesian for potential further refinement
            theta, phi = de_sph.x
            best_result = np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta)
            ])
    except Exception:
        pass
    
    # Strategy 2: Local refinement in Cartesian coordinates
    # Even though scale is invariant, local optimization in Cartesian might be more stable if we fix the scale.
    # Or we can just refine the current best_result.
    
    if time.time() < deadline - 10:
        try:
            # Refine in Cartesian
            res_cart = minimize(
                _compute_objective_analytic,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res_cart.fun < best_score:
                best_score = res_cart.fun
                best_result = res_cart.x.copy()
        except Exception:
            pass
            
    # Strategy 3: Multi-start local refinement
    # Try a few random points on the sphere and refine locally
    
    if time.time() < deadline - 5:
        np.random.seed(123)
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            theta = np.random.uniform(0, math.pi)
            phi = np.random.uniform(0, 2 * math.pi)
            x_init = np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta)
            ])
            try:
                res = minimize(
                    _compute_objective_analytic,
                    x_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass

    return best_result

if __name__ == "__main__":
    # Quick test
    res = construct()
    score = _compute_objective_analytic(res)
    print(f"Best coefficients: {res}")
    print(f"Score: {score}")

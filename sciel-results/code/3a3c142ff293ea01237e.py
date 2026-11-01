import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Hermite polynomial values at 0 (physicists' convention)
    # H_0(0) = 1, H_4(0) = 12, H_8(0) = 1680, H_12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Normalization constants: N_n = pi^(1/2) * 2^n * n!
    # N_0 = sqrt(pi) * 1 = sqrt(pi)
    # N_4 = sqrt(pi) * 16 * 24 = 384 * sqrt(pi)
    # N_8 = sqrt(pi) * 256 * 40320 = 10321920 * sqrt(pi)
    # N_12 = sqrt(pi) * 4096 * 479001600 = 1962007275520 * sqrt(pi)
    # We can factor out sqrt(pi) since it cancels in the ratio
    N0 = 1.0
    N4 = 384.0
    N8 = 10321920.0
    N12 = 1962007275520.0
    
    # For the second moment, we need:
    # <x^2> = (1/4) * [sum over m,n c_m c_n * I_2(m,n) + 2 * <1>]
    # where I_2(m,n) = integral H_2 * H_m * H_n * exp(-x^2) dx
    # 
    # Using H_2 * H_m expansion:
    # For the product H_2 * H_m, we use the recurrence relation
    # x^2 = (H_2 + 2)/4
    # 
    # Actually, let's compute <x^2> directly using the fact that:
    # For g = sum c_n H_n exp(-x^2/2), we have:
    # <x^2> / <1> can be computed from the coefficients
    #
    # Using the identity: x^2 H_n = (1/4) H_{n+2} + (n+1/2) H_n + n(n-1)/4 H_{n-2}
    # So <x^2> = sum_{m,n} c_m c_n <H_m x^2 H_n>
    # = sum_{m,n} c_m c_n [ (1/4)<H_m H_{n+2}> + (n+1/2)<H_m H_n> + n(n-1)/4 <H_m H_{n-2}> ]
    # By orthogonality, <H_m H_k> = N_k delta_{m,k}
    # So <H_m H_{n+2}> = N_{n+2} delta_{m, n+2}
    # <H_m H_n> = N_n delta_{m,n}
    # <H_m H_{n-2}> = N_{n-2} delta_{m, n-2} (for n >= 2)
    
    # Let's define the contribution to <x^2> from each pair (m,n):
    # Term1: (1/4) * c_m * c_n * N_{n+2} * delta_{m, n+2}
    # Term2: (n+1/2) * c_m * c_n * N_n * delta_{m, n}
    # Term3: n(n-1)/4 * c_m * c_n * N_{n-2} * delta_{m, n-2} (n >= 2)
    
    # We only have n in {0, 4, 8, 12}
    # N_0 = 1, N_4 = 384, N_8 = 10321920, N_12 = 1962007275520
    # N_2 = sqrt(pi)*4*2 = 8 (but we don't have H_2, so N_2 not directly needed)
    # N_6 = sqrt(pi)*64*720 = 46080
    # N_10 = sqrt(pi)*1024*3628800 = 3715891200
    # N_14 = sqrt(pi)*16384*87178291200 = 1430717439431680000 (huge)
    
    # Actually, for delta_{m, n+2}, we need n+2 to be in {0,4,8,12}
    # n=0: n+2=2 (not in our set)
    # n=4: n+2=6 (not in our set)
    # n=8: n+2=10 (not in our set)
    # n=12: n+2=14 (not in our set)
    # So Term1 is always 0!
    
    # For delta_{m, n-2}, we need n-2 in {0,4,8,12}
    # n=4: n-2=2 (not in set)
    # n=8: n-2=6 (not in set)
    # n=12: n-2=10 (not in set)
    # n=0: n-2=-2 (not in set)
    # So Term3 is also always 0!
    
    # Wait, that means <x^2> = sum_n (n+1/2) * c_n^2 * N_n?
    # That seems too simple. Let me double-check.
    
    # Actually, I think I made an error. The formula x^2 H_n = (1/4)H_{n+2} + (n+1/2)H_n + (n(n-1)/4)H_{n-2}
    # is correct for physicists' Hermite polynomials.
    
    # So <x^2> = sum_m c_m * [sum_n c_n * ((1/4)N_{n+2} delta_{m,n+2} + (n+1/2)N_n delta_{m,n} + (n(n-1)/4)N_{n-2} delta_{m,n-2})]
    
    # For our set {0,4,8,12}:
    # The cross terms vanish because n+2 and n-2 are never in our set for n in {0,4,8,12}
    # (2,6,10,14 and -2,2,6,10 are not in {0,4,8,12})
    
    # So <x^2> = sum_n (n + 1/2) * c_n^2 * N_n
    
    # Let's verify: for pure H_0, <x^2> = (0+1/2)*1*1 = 1/2
    # For g = exp(-x^2/2), <x^2> = integral x^2 exp(-x^2) dx / integral exp(-x^2) dx = (sqrt(pi)/2) / sqrt(pi) = 1/2. Correct!
    
    # For pure H_4: <x^2> = (4+1/2)*1*384 = 4.5*384 = 1728
    # Let me verify numerically... this should be correct by the derivation.
    
    # So the objective is:
    # C4 = A(f) * A(F) = <x^2>/<1> (since A(f) = A(F) for this symmetric case)
    # Wait, is A(f)*A(F) = <x^2>/<1> or is it sqrt(<x^2>/<1>) * sqrt(<xi^2>/<1>)?
    # A(f) = sqrt(<x^2>/<1>), A(F) = sqrt(<xi^2>/<1>)
    # For our case <xi^2>/<1> = <x^2>/<1> by symmetry
    # So A(f)*A(F) = <x^2>/<1>
    
    # Therefore: C4 = [sum_n (n+0.5) c_n^2 N_n] / [sum_n c_n^2 N_n]
    
    # This is a Rayleigh quotient! The minimum is the smallest eigenvalue of the generalized
    # eigenvalue problem, but since it's diagonal, the minimum is just the minimum of (n+0.5)
    # weighted by the constraint g(0)=0.
    
    # Wait, but we have the constraint g(0)=0 which couples the coefficients.
    # g(0) = c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    # So we minimize over c0, c4, c8:
    # f(c0,c4,c8) = [0.5*c0^2*N0 + 4.5*c4^2*N4 + 8.5*c8^2*N8 + 12.5*c12^2*N12] / [c0^2*N0 + c4^2*N4 + c8^2*N8 + c12^2*N12]
    
    # where c12 = -(c0 + 12*c4 + 1680*c8)/665280
    
    # Let's define:
    # num = 0.5*c0^2 + 4.5*384*c4^2 + 8.5*10321920*c8^2 + 12.5*1962007275520*c12^2
    # den = c0^2 + 384*c4^2 + 10321920*c8^2 + 1962007275520*c12^2
    
    C0_coeff = 0.5
    C4_coeff = 4.5 * 384.0
    C8_coeff = 8.5 * 10321920.0
    C12_coeff = 12.5 * 1962007275520.0
    
    N0_val = 1.0
    N4_val = 384.0
    N8_val = 10321920.0
    N12_val = 1962007275520.0
    
    def objective(c):
        c0, c4, c8 = c
        c12 = -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
        
        num = C0_coeff * c0**2 + C4_coeff * c4**2 + C8_coeff * c8**2 + C12_coeff * c12**2
        den = N0_val * c0**2 + N4_val * c4**2 + N8_val * c8**2 + N12_val * c12**2
        
        if den <= 0:
            return np.inf
        return num / den
    
    def gradient(c):
        c0, c4, c8 = c
        c12 = -(c0 + H4_0 * c4 + H8_0 * c8) / H12_0
        dc12_dc0 = -1.0 / H12_0
        dc12_dc4 = -H4_0 / H12_0
        dc12_dc8 = -H8_0 / H12_0
        
        num = C0_coeff * c0**2 + C4_coeff * c4**2 + C8_coeff * c8**2 + C12_coeff * c12**2
        den = N0_val * c0**2 + N4_val * c4**2 + N8_val * c8**2 + N12_val * c12**2
        
        if den <= 0:
            return np.array([0.0, 0.0, 0.0])
        
        # d(num)/dc0 = 2*C0_coeff*c0 + 2*C12_coeff*c12*dc12_dc0
        # d(num)/dc4 = 2*C4_coeff*c4 + 2*C12_coeff*c12*dc12_dc4
        # d(num)/dc8 = 2*C8_coeff*c8 + 2*C12_coeff*c12*dc12_dc8
        dnum_dc0 = 2 * C0_coeff * c0 + 2 * C12_coeff * c12 * dc12_dc0
        dnum_dc4 = 2 * C4_coeff * c4 + 2 * C12_coeff * c12 * dc12_dc4
        dnum_dc8 = 2 * C8_coeff * c8 + 2 * C12_coeff * c12 * dc12_dc8
        
        # d(den)/dc0 = 2*N0_val*c0 + 2*N12_val*c12*dc12_dc0
        # etc.
        dden_dc0 = 2 * N0_val * c0 + 2 * N12_val * c12 * dc12_dc0
        dden_dc4 = 2 * N4_val * c4 + 2 * N12_val * c12 * dc12_dc4
        dden_dc8 = 2 * N8_val * c8 + 2 * N12_val * c12 * dc12_dc8
        
        # d(num/den) = (dnum*den - num*dden) / den^2
        grad0 = (dnum_dc0 * den - num * dden_dc0) / (den**2)
        grad4 = (dnum_dc4 * den - num * dden_dc4) / (den**2)
        grad8 = (dnum_dc8 * den - num * dden_dc8) / (den**2)
        
        return np.array([grad0, grad4, grad8])
    
    # Multi-start optimization
    deadline = time.time() + 190  # Safety margin from 210s limit
    
    # Initial guesses
    starts = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([1.0, -1.0/12.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, -1.0/1680.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, -0.1, 0.001]),
        np.array([-1.0, 0.1, -0.001]),
        np.array([0.5, -0.05, 0.0005]),
        np.array([2.0, -0.2, 0.002]),
    ]
    
    best_val = np.inf
    best_x = None
    
    for x0 in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Also try a few random starts if time permits
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() > deadline:
                break
            x0 = rng.normal(0, 1, 3)
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                val = objective(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
            except Exception:
                pass
    
    if best_x is None:
        return np.array([-1.13, 0.04, 1.0/3240.0])
    
    return best_x

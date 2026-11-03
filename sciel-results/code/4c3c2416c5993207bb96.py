"""Optimized search for the uncertainty inequality constant via analytic objective."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Returns three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 coefficient is fixed to ensure g(0) = 0.
    """
    deadline = time.time() + 150  # Safety margin before 180s limit
    
    # 1. Define the analytic objective function
    # We minimize U = sqrt(A(f) * A(F))
    # The matrices M (moment) and N (Laplacian) are tridiagonal in the Hermite basis.
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        # Determine a12 such that g(0) = 0
        # g(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        # H_n(0) is 0 for odd n.
        # H0(0) = 1
        # H4(0) = 12
        # H8(0) = 1344
        # H12(0) = 169344
        # g(0) = a0 + 12*a4 + 1344*a8 + 169344*a12 = 0
        a12 = -(a0 + 12*a4 + 1344*a8) / 169344.0
        
        # Vector of coefficients for n=0, 4, 8, 12
        a = np.array([a0, a4, a8, a12])
        n = np.array([0, 4, 8, 12])
        
        # Diagonal terms for M (variance)
        # M_ii = (n_i + 1) * K_{n_i}
        # K_n = (2pi)^0.5 * (n!)^0.5
        K = np.zeros(4)
        for i in range(4):
            K[i] = np.sqrt(2 * np.pi) * np.sqrt(np.math.factorial(n[i]))
            
        M_diag = (n + 1) * K
        # M_ii contribution to numerator
        num_M = sum((a[i] * a[i] * M_diag[i]) for i in range(4))
        
        # M_ij (i!=j) terms:
        # M_{n, n+2} = sqrt(2(n+1)) * K_n  (since K_{n+2} = K_n * sqrt((n+1)(n+2))/sqrt(2) ... wait)
        # K_{n+2} / K_n = sqrt((n+1)(n+2))
        # M_{n, n+2} = sqrt(2(n+1)) * K_n
        # Let's verify: <psi_n, x^2 psi_{n+2}>.
        # x psi_n = 0.5 psi_{n+1} + n psi_{n-1}
        # x^2 psi_n = 0.5 x psi_{n+1} + n x psi_{n-1}
        #           = 0.5 (0.5 psi_{n+2} + (n+1) psi_n) + n (0.5 psi_n + (n-1) psi_{n-2})
        #           = 0.25 psi_{n+2} + (0.5(n+1) + 0.5n) psi_n + ...
        #           = 0.25 psi_{n+2} + (n+0.5) psi_n + ...
        # So <psi_n, x^2 psi_{n+2}> = 0.25 * <psi_n, psi_{n+2}>? No, we want <psi_n, x^2 psi_{n+2}>.
        # By symmetry <psi_n, x^2 psi_{n+2}> = <x psi_n, x psi_{n+2}>.
        # x psi_n = 0.5 psi_{n+1} + n psi_{n-1}
        # x psi_{n+2} = 0.5 psi_{n+3} + (n+2) psi_{n+1}
        # Product: 0.25 <psi_{n+1}, psi_{n+3}> + 0.5(n+2) <psi_{n+1}, psi_{n+1}> + ...
        # = 0 + 0.5(n+2) K_{n+1} + 0 + ...
        # Wait, <psi_n, x^2 psi_{n+2}> involves overlap of n and n+2.
        # Let's use the known formula for Hermite moments.
        # <H_n, x^2 H_m> = 0 if |n-m| is odd.
        # If n = m: (2n+1) <H_n, H_n>
        # If n = m+2: sqrt(2(m+1)) <H_m, H_m> ? 
        # Let's check n=0, m=2. <H_0, x^2 H_2>.
        # H_0 = 1, H_2 = 4x^2-2.
        # x^2 H_2 = 4x^4 - 2x^2.
        # <1, 4x^4 - 2x^2> = 4*6 - 2*2 = 20? (Integrals of x^4 and x^2 against gaussian)
        # K_0 = sqrt(2pi). K_2 = sqrt(2pi)*2 = 2 sqrt(2pi).
        # Formula sqrt(2(m+1)) K_m for m=0: sqrt(2) K_0.
        # Let's stick to the matrix construction via vectorization for small N to avoid algebra errors.
        
        # Since we only have 4 basis functions, we can just compute the integrals directly using quadrature.
        # This is robust and fast enough.
        def compute_integral(f):
            # Gauss-Hermite quadrature
            # Integrate f(x) e^{-x^2} dx? No, our basis is psi_n = H_n e^{-x^2/2}.
            # Inner product <u, v> = int u(x) v(x) dx.
            # Let u = sum a_i H_i e^{-x^2/2}.
            # We can just sample x and compute.
            # Use Gauss-Hermite which integrates P(x) e^{-x^2}.
            # Let y = x. x = y.
            # e^{-x^2/2} e^{-x^2/2} = e^{-x^2}.
            # So <psi_n, psi_m> = int H_n(x) H_m(x) e^{-x^2} dx.
            # This is exactly what Gauss-Hermite handles.
            # Points and weights for degree 12+2=14? We have polynomials up to degree 12.
            # Product degree 24. Need at least 13 points. 20 is safe.
            x, w = np.polynomial.hermite.hermgauss(20)
            H = np.polynomial.hermite.hermval(x, np.array([1, 0, 0, 0])) # H0
            # We need H_n for n in {0, 4, 8, 12}
            Hs = []
            for n_val in n:
                Hs.append(np.polynomial.hermite.hermval(x, np.zeros(n_val+1)))
                Hs[-1][n_val] = 1.0
            
            Hs = np.array(Hs) # Shape (4, 20)
            
            # M_ii = int H_i^2 e^{-x^2} dx
            # M_ij = int H_i H_j e^{-x^2} dx
            M = Hs.T @ (w * Hs)
            
            # N matrix: int H_i'' H_j e^{-x^2} dx ? No.
            # N_ij = <psi_i'', psi_j> = int (H_i'' e^{-x^2/2} - ... ) ?
            # Recall: psi_i' = (H_i' - x H_i) e^{-x^2/2}
            # psi_i'' = (H_i'' - 2 H_i' + x^2 H_i - H_i) e^{-x^2/2} ?
            # Actually, psi_i'' = (H_i'' - x H_i' - H_i' - x H_i + x^2 H_i) e^{-x^2/2} ... 
            # Let's use the recurrence: psi_n'' = n(n-1) psi_{n-2} - (2n+1) psi_n + 0.25 psi_{n+2}
            # This is derived from psi_n' = n psi_{n-1} - 0.5 psi_{n+1}.
            # So N_ij = <psi_i'', psi_j> = n_i(n_i-1) <psi_{n_i-2}, psi_j> - (2n_i+1) <psi_{n_i}, psi_j> + 0.25 <psi_{n_i+2}, psi_j>
            # This requires K for n-2 and n+2.
            # To be safe and simple, let's compute N numerically as well.
            # N_ij = int psi_i''(x) psi_j(x) dx
            # psi_i''(x) = (H_i''(x) - 2 H_i'(x) ... wait)
            # Let's just evaluate psi_i'' directly.
            # psi_n(x) = H_n(x) e^{-x^2/2}
            # psi_n'(x) = (H_n'(x) - x H_n(x)) e^{-x^2/2}
            # psi_n''(x) = (H_n''(x) - H_n'(x) - H_n'(x) - x H_n'(x) - H_n(x) + x^2 H_n(x)) e^{-x^2/2}
            #            = (H_n''(x) - 2 H_n'(x) - x H_n'(x) - H_n(x) + x^2 H_n(x)) e^{-x^2/2}
            
            # We need H_n' and H_n''.
            # H_n'(x) = 2n H_{n-1}(x)
            # H_n''(x) = 2n H_{n-1}'(x) = 4n(n-1) H_{n-2}(x)
            
            N = np.zeros((4, 4))
            for i in range(4):
                n_i = n[i]
                # Construct psi_i'' on grid
                # We need H_{n_i}, H_{n_i}', H_{n_i}''
                H_ni = Hs[i]
                if n_i > 0:
                    H_ni_m1 = np.polynomial.hermite.hermval(x, np.zeros(n_i+1))
                    H_ni_m1[n_i-1] = 1.0
                    H_ni_prime = 2 * n_i * H_ni_m1
                else:
                    H_ni_prime = 0.0
                    
                if n_i > 1:
                    H_ni_m2 = np.polynomial.hermite.hermval(x, np.zeros(n_i+1))
                    H_ni_m2[n_i-2] = 1.0
                    H_ni_double_prime = 4 * n_i * (n_i - 1) * H_ni_m2
                elif n_i == 1:
                    H_ni_double_prime = 4 * 1 * 0 * 0 # 0
                else:
                    H_ni_double_prime = 0.0
                
                psi_dd = (H_ni_double_prime - 2 * H_ni_prime - x * H_ni_prime - H_ni + x**2 * H_ni) * np.exp(-x**2 / 2.0)
                
                for j in range(4):
                    psi_j = Hs[j] * np.exp(-x**2 / 2.0)
                    N[i, j] = np.sum(w * psi_dd * psi_j * np.exp(-x**2/2.0) * np.exp(x**2/2.0)) # Wait, psi_j already has exp.
                    # N_ij = int psi_dd * psi_j dx
                    # psi_dd has exp(-x^2/2), psi_j has exp(-x^2/2).
                    # Product has exp(-x^2).
                    # So int (poly) exp(-x^2) dx.
                    # We used hermgauss which integrates P(x) exp(-x^2).
                    # So N[i,j] = np.sum(w * (psi_dd * np.exp(x**2/2.0)) * (psi_j * np.exp(x**2/2.0)))
                    # No, psi_dd = P_dd * exp(-x^2/2). psi_j = P_j * exp(-x^2/2).
                    # Product = P_dd P_j exp(-x^2).
                    # So we just need P_dd and P_j.
                    P_dd = H_ni_double_prime - 2 * H_ni_prime - x * H_ni_prime - H_ni + x**2 * H_ni
                    N[i, j] = np.sum(w * P_dd * Hs[j])
            
            # Calculate A(f) and A(F)
            # A(f) = sum a_i a_j M_ij / (sum a_i a_j K_i)? No.
            # A(f) = <f, x^2 f> / <f, f>
            num_A = a @ M @ a
            den_A = a @ (K) # <f,f> = sum a_i a_j <psi_i, psi_j>. <psi_i, psi_j> = K_i delta_ij.
            # Wait, M is <psi_i, x^2 psi_j>.
            # <f, f> = a @ K_vec @ a where K_vec is diagonal of M?
            # M_ii = (n_i+1) K_i.
            # <psi_i, psi_i> = K_i.
            # So den_A = sum a_i^2 K_i.
            den_A = np.sum(a**2 * K)
            
            A_f = num_A / den_A
            
            # A(F) = <F, (-d^2/dx^2) F> / <F, F>
            # F is Fourier transform. For Hermite, F[psi_n] = i^n psi_n (up to sign/phase, irrelevant for energy).
            # Actually, the operator is same form.
            # num_F = a @ N @ a ?
            # N_ij = <psi_i'', psi_j>.
            # <F, -F''> = <F, -F''>.
            # If F = sum b_i psi_i, then <F, -F''> = sum b_i b_j <psi_i, -psi_j''> = - a @ N @ a.
            num_F = - (a @ N @ a)
            den_F = den_A
            
            A_F = num_F / den_F
            
            return np.sqrt(A_f * A_F)
            
    # Initial guess
    x0 = np.array([-113 / 100.0, 1 / 25.0, 1 / 3240.0])
    
    best_x = x0
    best_val = objective(x0)
    
    # Try local optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Try a few random restarts if time permits
    start_time = time.time()
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        x_rand = x0 + np.random.normal(0, 0.5, 3)
        try:
            res_r = minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_r.success and res_r.fun < best_val:
                best_x = res_r.x
                best_val = res_r.fun
        except Exception:
            continue
            
    return best_x

# EVOLVE-BLOCK-END

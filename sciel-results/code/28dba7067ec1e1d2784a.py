import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 170  # 10s safety margin
    
    # Hermite Polynomials (Probabilists' He_n)
    # He_0(x) = 1
    # He_4(x) = x^4 - 6x^2 + 3
    # He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He_12(x) = x^12 - 42x^10 + 756x^8 - 6048x^6 + 15120x^4 - 15120x^2 + 10395
    
    def He0(x): return np.ones_like(x)
    def He4(x): return x**4 - 6*x**2 + 3
    def He8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def He12(x): return x**12 - 42*x**10 + 756*x**8 - 6048*x**6 + 15120*x**4 - 15120*x**2 + 10395
    
    polys = [He0, He4, He8, He12]
    degs = [0, 4, 8, 12]
    
    # Compute the matrix M where M[i,j] = int exp(-2*pi*x^2) He_i(x) He_j(x) dx
    # We use quad for high precision
    M = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(i, 4):
            if (i + j) % 2 != 0:
                M[i,j] = 0.0
                M[j,i] = 0.0
            else:
                # Integrand
                def integrand(x):
                    return np.exp(-2 * np.pi * x**2) * polys[i](x) * polys[j](x)
                
                # Symmetric integral, integrate 0 to inf and multiply by 2
                val, err = quad(integrand, 0, np.inf, epsabs=1e-15, epsrel=1e-15, limit=1000)
                M[i,j] = 2 * val
                M[j,i] = M[i,j]
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if time.time() > deadline:
        # Fallback to initial guess if integration takes too long
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

    # Constraint: g(0) = 0
    # c0*1 + c4*3 + c8*105 + c12*10395 = 0
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        c = np.array([c0, c4, c8, c12])
        
        # Check if vector is zero (avoid division by zero in normalization if we used it, but product is scale invariant)
        # We compute A_g^2 and A_hat_g^2
        Ag2 = c @ M @ c
        
        # A_hat_g^2
        # N[i,j] = (-i)^(deg_i + deg_j) * M[i,j]
        # Since degs are 0, 4, 8, 12 (all multiples of 4), (-i)^(sum) is always 1?
        # deg_i + deg_j is multiple of 4. (-i)^4 = 1.
        # So N = M?
        # Let's verify.
        # F[He_n] = (-i)^n He_n.
        # Product: (-i)^n (-i)^m = (-i)^(n+m).
        # n,m in {0,4,8,12}. n+m is multiple of 4.
        # (-i)^4 = 1.
        # So yes, N = M.
        # This implies A(g) = A(hat(g)) for this specific subspace?
        # If so, we minimize c @ M @ c.
        
        Ahat_g2 = c @ M @ c # Since N=M
        
        # We want to minimize sqrt(Ag2 * Ahat_g2) = Ag2 (since Ag2=Ahag2)
        # However, the scale is arbitrary. We can fix the scale to avoid trivial solution c=0.
        # The uncertainty product is scale invariant.
        # We can minimize (c @ M @ c) / (c @ c) or just fix one coefficient.
        # Let's fix c0 = 1 to remove scale invariance.
        # But the optimizer might drift.
        # Better: minimize log(Ag2 * Ahat_g2) subject to c0=1?
        # Or just minimize Ag2 with c0 fixed to 1.
        
        # Actually, the problem asks for coefficients. The score is the upper bound.
        # The upper bound is the value of the product for the function.
        # Since the product is scale invariant, any scalar multiple gives the same score.
        # We can return the coefficients found by the optimizer.
        
        return Ag2 * Ahat_g2 # Minimize this, but it's 0 at c=0.
        # We need to constrain the scale.
        # Let's add a penalty if the norm is too small.
        # Or simply fix c0 = 1 in the optimization variables?
        # The interface requires returning 3 coefficients.
        # Let's optimize c4, c8 with c0=1?
        # The initial guess has c0 = -1.13.
        # Let's optimize c0, c4, c8 and add a constraint c0^2 + c4^2 + c8^2 + c12^2 = 1.
        
        # For simplicity and robustness, let's just minimize Ag2 * Ahat_g2 / (c @ c)
        # This is the Rayleigh quotient, which is scale invariant.
        norm_sq = c @ c
        if norm_sq < 1e-12:
            return 1e100
        return (Ag2 * Ahat_g2) / norm_sq

    # Initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        
        if time.time() > deadline:
            return x0
            
        if res.success:
            best_c = res.x
        else:
            # Try L-BFGS-B if NM fails
            res2 = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 10000, 'ftol': 1e-15})
            if res2.fun < res.fun:
                best_c = res2.x
            else:
                best_c = res.x
        
        # Normalize? The problem doesn't require normalization, but the "upper bound" is the value.
        # The value is scale invariant.
        # We return the coefficients.
        # To be safe, we can scale them such that the first coefficient is close to the initial magnitude?
        # Or just return them as is.
        
        return np.array(best_c)
        
    except Exception as e:
        return x0

# EVOLVE-BLOCK-END

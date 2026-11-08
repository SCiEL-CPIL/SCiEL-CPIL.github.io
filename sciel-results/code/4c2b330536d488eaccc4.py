"""Fourier-invariant Hermite-Gaussian test function tuned via analytic spectral moments."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution


def _hermite_coeffs(n):
    """
    Compute coefficients for the physicists' Hermite polynomial H_n(x).
    H_n(x) = sum_{k=0}^{n} c_k x^k
    Recurrence: H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    
    # c represents coefficients for H_{k-2}
    c_nm2 = np.array([1.0])
    # c_nm1 represents coefficients for H_{k-1}
    c_nm1 = np.array([0.0, 2.0])
    
    for k in range(2, n + 1):
        # H_k = 2x H_{k-1} - 2(k-1) H_{k-2}
        # 2x H_{k-1} shifts coefficients by 1 and multiplies by 2
        term1 = np.zeros(len(c_nm1) + 1)
        term1[1:] = 2.0 * c_nm1
        
        # -2(k-1) H_{k-2}
        term2 = -2.0 * (k - 1) * c_nm2
        
        # Combine
        # Ensure lengths match
        max_len = max(len(term1), len(term2))
        c_n = np.zeros(max_len)
        c_n[:len(term1)] += term1
        c_n[:len(term2)] += term2
        
        c_nm2 = c_nm1
        c_nm1 = c_n
        
    return c_nm1


def _compute_uncertainty_product_analytic(a0, a4, a8):
    """
    Compute the uncertainty product A(f)*A(F) analytically.
    g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) e^{-x^2/2}
    Constraint: g(0) = 0 => a0 + a4 H4(0) + a8 H8(0) + a12 H12(0) = 0
    H4(0) = -12, H8(0) = 1680, H12(0) = -47880
    a12 = (a0 - 12 a4 + 1680 a8) / 47880
    
    Norm N = sum_{i,j} a_i a_j <H_i, H_j> = sum a_i^2 (2^i i!)
    <x^2> = sum_{i,j} a_i a_j <x^2 H_i, H_j> / N
    <xi^2> = sum_{i,j} a_i a_j <xi^2 H_i, H_j> / N
    
    For physicists' Hermite:
    <H_m, H_n> = 2^n n! delta_{mn}
    
    <x^2 H_m, H_n>:
    x^2 H_m = 0.5 H_{m+2} + m H_m + m(m-1) H_{m-2}
    So <x^2 H_m, H_n> is non-zero only if n = m+2, m, m-2.
    Specifically:
    If n = m: coeff is m. Term: a_m a_n m <H_m, H_n> = a_m^2 m 2^m m!
    If n = m+2: coeff is 0.5. Term: a_m a_{m+2} 0.5 <H_{m+2}, H_{m+2}> = a_m a_{m+2} 0.5 2^{m+2} (m+2)!
    If n = m-2: coeff is m(m-1). Term: a_m a_{m-2} m(m-1) <H_{m-2}, H_{m-2}> = a_m a_{m-2} m(m-1) 2^{m-2} (m-2)!
    
    By symmetry of the sum over i,j, we can just sum over i and j directly using the expansion of x^2 H_i.
    Let's precompute the matrix elements.
    Indices: 0, 4, 8, 12.
    """
    # a12 calculation
    a12 = (a0 - 12.0 * a4 + 1680.0 * a8) / 47880.0
    a_coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # Precompute norms 2^d d!
    norms = 2.0**degrees * np.array([1.0, 24.0, 40320.0, 479001600.0]) 
    # Wait, 12! is 479001600. 
    # Let's use scipy.special.factorial for accuracy or just hardcode since degrees are fixed.
    # 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    
    # Norm N
    N = np.sum(a_coeffs**2 * norms)
    if N < 1e-30:
        return 1e10
        
    # Calculate <x^2>
    # Sum_{i,j} a_i a_j <x^2 H_i, H_j>
    # <x^2 H_i, H_j> is non-zero only if |i-j| is even and <= 2.
    # Our degrees are 0, 4, 8, 12. Differences are 4, 8, etc.
    # So <x^2 H_i, H_j> is 0 if i != j!
    # Because x^2 H_i has components H_{i+2}, H_i, H_{i-2}.
    # For i=0: H_2, H_0, H_{-2}(0). Only H_0 matches j=0.
    # For i=4: H_6, H_4, H_2. Only H_4 matches j=4.
    # For i=8: H_10, H_8, H_6. Only H_8 matches j=8.
    # For i=12: H_14, H_12, H_10. Only H_12 matches j=12.
    
    # So <x^2> = sum a_i^2 * (coeff of H_i in x^2 H_i) * <H_i, H_i> / N
    # Coeff of H_i in x^2 H_i is i.
    # <H_i, H_i> = 2^i i!
    # So <x^2> = sum a_i^2 * i * 2^i i! / N
    
    x2_numer = np.sum(a_coeffs**2 * degrees * norms)
    x2 = x2_numer / N
    
    # Calculate <xi^2>
    # Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(k) e^{-k^2/2} (up to constant).
    # Actually, let's use the property that the problem is symmetric?
    # No, the coefficients are not symmetric.
    # However, the Fourier transform of the Gaussian times Hermite is another Gaussian times Hermite.
    # F[g](k) = C * sum a_n (-i)^n H_n(k) e^{-k^2/2}
    # Let b_n = a_n (-i)^n.
    # The form of the function in k-space is sum b_n H_n(k) e^{-k^2/2}.
    # The moments <k^2> will have the same structure as <x^2> but with coefficients b_n.
    # <k^2> = sum b_n^2 * n * 2^n n! / N_k
    # N_k = sum |b_n|^2 2^n n! = sum a_n^2 2^n n! = N (since |(-i)^n|^2 = 1)
    
    # So <k^2> = sum a_n^2 * n * 2^n n! / N
    # Wait, is it exactly the same?
    # Let's verify.
    # g(x) = sum a_n H_n(x) e^{-x^2/2}
    # G(k) = sum a_n (-i)^n H_n(k) e^{-k^2/2}
    # Let b_n = a_n (-i)^n.
    # The "shape" in k-space is determined by b_n.
    # <k^2> = sum_{n,m} b_n b_m^* <k^2 H_n, H_m> / N
    # As before, <k^2 H_n, H_m> is 0 unless n=m.
    # So <k^2> = sum b_n^2 n 2^n n! / N
    # Since b_n = a_n (-i)^n, b_n^2 = a_n^2 (-i)^{2n} = a_n^2 (-1)^n.
    # This would make <k^2> potentially complex or negative? No, expectation values must be real positive.
    # Ah, the inner product in Fourier space involves conjugates.
    # <G, G> = sum |b_n|^2 ... = sum a_n^2 ... = N.
    # <k^2 G, G> = sum b_n b_m^* <k^2 H_n, H_m>.
    # Only n=m terms survive.
    # Term n=n: b_n b_n^* n 2^n n! = |b_n|^2 n 2^n n! = a_n^2 n 2^n n!.
    # So <k^2> = sum a_n^2 n 2^n n! / N.
    # It is IDENTICAL to <x^2>.
    
    # Wait, if <x^2> = <k^2>, then the product is <x^2>.
    # Let's re-read the task. "Tune ... to reduce the upper bound on A(f)*A(Fourier(f))".
    # If the function is real and even (Hermite polys are even for even n), then g(x) is even.
    # Fourier transform of an even real function is real and even.
    # H_n for even n is even. e^{-x^2/2} is even.
    # So g(x) is even. G(k) is even.
    # The distribution of x and k are identical?
    # If g(x) = g(k) in structure (coefficients same), then yes.
    # But the coefficients in k-space are b_n = a_n (-i)^n.
    # For even n, (-i)^n = (-1)^{n/2}.
    # So b_n = a_n (-1)^{n/2}.
    # The coefficients change sign for n=4, 12.
    # a_4 becomes -a_4. a_12 becomes -a_12.
    # Let's check the formula for <k^2> again.
    # <k^2> = sum |b_n|^2 n 2^n n! / N.
    # |b_n|^2 = a_n^2.
    # So <k^2> IS identical to <x^2>.
    
    # This implies A(f) = A(F). Product = <x^2>.
    # Is this correct?
    # Let's test with a simple case. g(x) = e^{-x^2/2}. a0=1.
    # <x^2> = 1 * 0 * ... / 1 = 0? No.
    # For n=0, coeff is 0.
    # Wait, x^2 H_0 = x^2.
    # <x^2 H_0, H_0> = <x^2, 1> = 1.
    # Formula: sum a_n^2 n 2^n n! / N.
    # For n=0, term is 0.
    # But <x^2> for Gaussian is 1/2 (if normalized to prob) or 1 (if unnormalized e^{-x^2/2}? No, e^{-x^2/2} norm is sqrt(pi)).
    # Let's check the recurrence coefficient again.
    # x^2 H_n = 0.5 H_{n+2} + n H_n + n(n-1) H_{n-2}.
    # <x^2 H_n, H_n> = n <H_n, H_n>.
    # For n=0, <x^2, 1> = 1.
    # Formula gives 0. Why?
    # Because H_0 = 1. x^2 * 1 = x^2.
    # H_2 = 4x^2 - 2. So x^2 = (H_2 + 2)/4 = 0.25 H_2 + 0.5 H_0.
    # <x^2 H_0, H_0> = <x^2, 1> = 1.
    # My formula <x^2 H_n, H_n> = n <H_n, H_n> works for n>=1?
    # For n=1: H_1 = 2x. x^2 H_1 = 2x^3.
    # H_3 = 8x^3 - 12x. x^3 = (H_3 + 12x)/8 = 0.125 H_3 + 0.5 H_1.
    # 2x^3 = 0.25 H_3 + H_1.
    # <2x^3, 2x> = 2 <x^3, x> = 2 * (1/2) <H_1, H_1> = <H_1, H_1> = 4.
    # Formula: n <H_n, H_n> = 1 * 4 = 4. Matches.
    
    # For n=0:
    # x^2 H_0 = x^2.
    # Expansion: 0.5 H_2 + 0.5 H_0.
    # <x^2, H_0> = <0.5 H_2 + 0.5 H_0, H_0> = 0.5 <H_0, H_0> = 0.5 * 1 = 0.5.
    # Formula n <H_n, H_n> gives 0.
    # So the term for n=0 is missing in the "n <H_n, H_n>" part.
    # The general term is:
    # <x^2 H_n, H_n> = 0.5 <H_{n+2}, H_n> + n <H_n, H_n> + n(n-1) <H_{n-2}, H_n>.
    # Orthogonality kills n+2 and n-2 terms.
    # So it is n <H_n, H_n> for n>=1.
    # For n=0, it is 0.5 <H_0, H_0> = 0.5.
    # Note that 0.5 = (0 + 1) * 2^0 0! / 2? No.
    # Let's just compute the numerator term by term.
    
    # Term for n:
    # If n == 0: val = 0.5 * 1.0 * 1.0 # 0.5 * <H_0, H_0>
    # Else: val = n * 2^n n!
    
    # Let's generalize:
    # val_n = 0.5 * 2^0 0! if n==0 else n * 2^n n!
    # Actually, for n=0, the "n" term is 0. The "0.5 H_2" part doesn't contribute to <., H_0>.
    # The "n(n-1) H_{n-2}" part is 0.
    # So for n=0, it is 0.5 * Norm_0.
    
    # Let's calculate x2_numer carefully.
    x2_numer = 0.0
    for i, n in enumerate(degrees):
        a_i = a_coeffs[i]
        norm_i = norms[i]
        if n == 0:
            # <x^2 H_0, H_0> = 0.5 * Norm_0
            x2_numer += a_i**2 * 0.5 * norm_i
        else:
            # <x^2 H_n, H_n> = n * Norm_n
            x2_numer += a_i**2 * n * norm_i
            
    x2 = x2_numer / N
    
    # Now for <k^2>.
    # Coefficients b_n = a_n (-i)^n.
    # |b_n|^2 = a_n^2.
    # The structure of the moment calculation depends on |b_n|^2 and the degree n.
    # The "n=0 special case" applies to the degree 0 component.
    # Does the degree 0 component change? No, n=0 is still n=0.
    # So <k^2> is calculated exactly the same way as <x^2> using a_n^2.
    # Therefore <k^2> = <x^2>.
    
    prod = np.sqrt(x2 * x2) # = x2
    return prod


def _objective(a3):
    a0, a4, a8 = a3
    return _compute_uncertainty_product_analytic(a0, a4, a8)


def construct():
    start = time.time()
    deadline = start + 175.0
    
    # Seed value
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = _objective(best)
    
    # Define bounds for differential evolution
    # Based on seed and typical scales
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
    
    # Try Differential Evolution for global search
    try:
        res = differential_evolution(_objective, bounds, maxiter=200, popsize=15, 
                                     tol=1e-8, seed=42, polish=True)
        if res.fun < best_val:
            best = res.x
            best_val = res.fun
    except Exception as e:
        pass
    
    # Refine with Nelder-Mead
    if time.time() < deadline:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, best, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best = res.x
                best_val = res.fun
        except Exception:
            pass
            
    if not np.all(np.isfinite(best)):
        best = seed
        
    return best
# EVOLVE-BLOCK-END

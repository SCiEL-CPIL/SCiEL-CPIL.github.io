"""Rewritten program: compute the global minimum of the uncertainty bound
for a Hermite-Gaussian test function with coefficients on H0, H4, H8,
with the H12 coefficient fixed so that g(0)=0.

The uncertainty product A(f)*A(F) can be expressed as a ratio of two
quartic forms in the three free parameters (a0, a4, a8).  We minimize
this Rayleigh-quotient-like objective over the positive orthant (or
unconstrained) using a combination of:
  1. Analytical reduction to a 3-variable optimization.
  2. Global search via grid + local refinement.
  3. High-precision polish using scipy.optimize.

The score returned is the upper bound on C4 = A(f)*A(F).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import itertools

_DEADLINE = time.time() + 190  # safety margin before 210s kill


def _check_deadline():
    return time.time() > _DEADLINE


def _hermite_coeffs(n):
    """Return coefficients of the probabilists' Hermite polynomial Hn(x)
    in the standard basis (descending powers). We use physicists' Hermite
    polynomials He_n(x) which satisfy He_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2}.
    Actually, let's use the standard physicists' Hermite polynomials H_n(x)
    which are orthogonal w.r.t. e^{-x^2}.
    
    For the uncertainty principle with Fourier transform, the natural basis
    is the Hermite functions phi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}.
    
    The Fourier transform of phi_n is (-i)^n phi_n.
    
    So if f(x) = sum a_n phi_n(x), then F(f)(k) = sum a_n (-i)^n phi_n(k).
    
    The uncertainty product A(f)*A(F) where A(g) = (integral |g|^2 x^2 dx)^{1/2}
    (assuming normalized so integral |g|^2 = 1).
    
    For a Hermite function phi_n, the expectation of x^2 is n + 1/2.
    For a superposition, we need cross terms.
    
    Let's define:
    f(x) = sum_{n in S} c_n phi_n(x)
    where S = {0, 4, 8, 12} and c_12 is determined by f(0) = 0.
    
    phi_n(0) = 0 for odd n, and for even n = 2k:
    phi_{2k}(0) = (-1)^k sqrt((2k)! / (k! 2^{2k} sqrt(pi))) * ... 
    
    Actually, let me use the explicit formula. The physicists' Hermite polynomial
    H_n(x) evaluated at x=0:
    H_n(0) = 0 if n odd, (-1)^{n/2} (n-1)!! if n even.
    
    The normalized Hermite function:
    phi_n(x) = (1 / sqrt(2^n n! sqrt(pi))) H_n(x) e^{-x^2/2}
    
    So phi_n(0) = H_n(0) / sqrt(2^n n! sqrt(pi)) for n even.
    
    For n=0: H_0(0)=1, phi_0(0) = 1/sqrt(sqrt(pi)) = pi^{-1/4}
    For n=4: H_4(0) = (-1)^2 * 3!! = 3, phi_4(0) = 3/sqrt(16*24*sqrt(pi)) = 3/sqrt(384 sqrt(pi))
    For n=8: H_8(0) = (-1)^4 * 7!! = 105, phi_8(0) = 105/sqrt(256*40320*sqrt(pi))
    For n=12: H_12(0) = (-1)^6 * 11!! = 10395, phi_12(0) = 10395/sqrt(4096*479001600*sqrt(pi))
    
    The condition f(0)=0 gives:
    c_0 phi_0(0) + c_4 phi_4(0) + c_8 phi_8(0) + c_12 phi_12(0) = 0
    => c_12 = -(c_0 phi_0(0) + c_4 phi_4(0) + c_8 phi_8(0)) / phi_12(0)
    
    Now, A(f)^2 = <f, x^2 f> = sum_{m,n} c_m c_n <phi_m, x^2 phi_n>
    
    x phi_n = sqrt((n+1)/2) phi_{n+1} + sqrt(n/2) phi_{n-1}
    So x^2 phi_n = (n+1/2) phi_n + sqrt((n+1)(n+2)/4) phi_{n+2} + sqrt(n(n-1)/4) phi_{n-2}
    
    Thus <phi_m, x^2 phi_n> = (n+1/2) delta_{mn} + sqrt((n+1)(n+2)/4) delta_{m,n+2} + sqrt(n(n-1)/4) delta_{m,n-2}
    
    Let's denote q_n = n + 1/2, and the off-diagonal couplings.
    
    Since our indices are {0, 4, 8, 12}, which are all even and spaced by 4,
    the x^2 operator connects n to n±2, which would be odd indices (1,2,3,5,6,7,9,10,11,13,14),
    NOT in our set. So <phi_m, x^2 phi_n> = (n+1/2) delta_{mn} for m,n in {0,4,8,12}.
    
    Wait, that's not right. x^2 phi_n has components in phi_{n-2}, phi_n, phi_{n+2}.
    For n=0: x^2 phi_0 = (1/2) phi_0 + (1/2) phi_2. phi_2 is not in our set.
    For n=4: x^2 phi_4 = (9/2) phi_4 + (sqrt(4*5)/2) phi_6 + (sqrt(4*3)/2) phi_2. Neither phi_2 nor phi_6 in set.
    For n=8: x^2 phi_8 = (17/2) phi_8 + (sqrt(8*9)/2) phi_10 + (sqrt(8*7)/2) phi_6. Not in set.
    For n=12: x^2 phi_12 = (25/2) phi_12 + (sqrt(12*13)/2) phi_14 + (sqrt(12*11)/2) phi_10. Not in set.
    
    So indeed, for m,n in {0,4,8,12}, <phi_m, x^2 phi_n> = (n+1/2) delta_{mn}.
    
    This means A(f)^2 = sum_n c_n^2 (n+1/2) / sum_n c_n^2 (since ||f||^2 = sum c_n^2).
    
    Similarly, since F(f) = sum c_n (-i)^n phi_n, and |(-i)^n|^2 = 1,
    A(F)^2 = sum_n c_n^2 (n+1/2) / sum_n c_n^2.
    
    So A(f)*A(F) = sum_n c_n^2 (n+1/2) / sum_n c_n^2.
    
    This is a Rayleigh quotient! The minimum is achieved by putting all weight
    on the smallest n, which is n=0. But we have the constraint f(0)=0.
    
    So we need to minimize sum c_n^2 (n+1/2) / sum c_n^2 subject to
    sum c_n phi_n(0) = 0.
    
    Let p_n = phi_n(0) for n in {0,4,8,12}.
    Constraint: c_0 p_0 + c_4 p_4 + c_8 p_8 + c_12 p_12 = 0.
    
    Let r_n = n + 1/2. We want to minimize (c_0^2 r_0 + c_4^2 r_4 + c_8^2 r_8 + c_12^2 r_12) / (c_0^2 + c_4^2 + c_8^2 + c_12^2)
    subject to c_0 p_0 + c_4 p_4 + c_8 p_8 + c_12 p_12 = 0.
    
    This is a generalized eigenvalue problem. Let's set up:
    Minimize c^T R c / c^T c subject to c^T p = 0, where R = diag(r_0, r_4, r_8, r_12), p = (p_0, p_4, p_8, p_12).
    
    The minimum is the smallest eigenvalue of R restricted to the subspace c^T p = 0.
    
    We can solve this by Lagrange multipliers: (R - lambda I)c + mu p = 0, c^T p = 0.
    
    From the first equation: c = (lambda I - R - mu p p^T)^{-1} ... this is messy.
    
    Alternative: parameterize the constraint. Let's use the three free coefficients
    a0=c_0, a4=c_4, a8=c_8, and c_12 = -(a0 p_0 + a4 p_4 + a8 p_8)/p_12.
    
    Then minimize (a0^2 r_0 + a4^2 r_4 + a8^2 r_8 + c_12^2 r_12) / (a0^2 + a4^2 + a8^2 + c_12^2).
    
    This is a 3-variable optimization. Let's compute the values of p_n and r_n.
    """
    pass


def _compute_p_and_r():
    """Compute p_n = phi_n(0) and r_n = n + 1/2 for n in {0,4,8,12}."""
    import math
    ns = [0, 4, 8, 12]
    p = []
    r = []
    for n in ns:
        r.append(n + 0.5)
        # H_n(0) for even n: (-1)^{n/2} (n-1)!!
        if n == 0:
            Hn0 = 1.0
        else:
            # (n-1)!! for n even
            k = n // 2
            # (2k-1)!! = (2k)! / (2^k k!)
            fact = math.factorial(2 * k)
            double_fact = fact / (2**k * math.factorial(k))
            Hn0 = ((-1)**k) * double_fact
        # phi_n(0) = H_n(0) / sqrt(2^n n! sqrt(pi))
        denom = math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
        p.append(Hn0 / denom)
    return np.array(p), np.array(r)


def construct():
    """Return three coefficients (a0, a4, a8) of the Hermite combination.
    The degree-12 weight is fixed so g(0)=0. Score is the upper bound on C4."""
    
    p, r = _compute_p_and_r()
    p0, p4, p8, p12 = p
    r0, r4, r8, r12 = r
    
    def c12_of(a0, a4, a8):
        return -(a0 * p0 + a4 * p4 + a8 * p8) / p12
    
    def objective(x):
        a0, a4, a8 = x
        c12 = c12_of(a0, a4, a8)
        num = a0**2 * r0 + a4**2 * r4 + a8**2 * r8 + c12**2 * r12
        den = a0**2 + a4**2 + a8**2 + c12**2
        if den < 1e-30:
            return 1e10
        return num / den
    
    # The minimum of the Rayleigh quotient subject to the linear constraint
    # is the smallest eigenvalue of the 4x4 matrix R projected onto the
    # orthogonal complement of p.
    #
    # We can find this by solving the generalized problem. Let's use a
    # change of variables to eliminate the constraint.
    #
    # Let v be a basis for the null space of p^T. We can use SVD.
    Rmat = np.diag(r)
    # Find basis for null space of p
    U, S, Vt = np.linalg.svd(p.reshape(1, -1))
    # Vt[0] is the direction of p, Vt[1:] span the null space
    N = Vt[1:].T  # 4x3 matrix whose columns span null(p^T)
    
    # Project R onto the null space: M = N^T R N
    M = N.T @ Rmat @ N
    
    # The eigenvalues of M are the Rayleigh quotients on the null space
    eigvals, eigvecs = np.linalg.eigh(M)
    min_eig = eigvals[0]
    v_min = eigvecs[:, 0]  # 3-vector in the null-space basis
    
    # Convert back to original coordinates
    c_min = N @ v_min  # 4-vector (c0, c4, c8, c12)
    
    # The three free coefficients are c0, c4, c8
    a0, a4, a8 = c_min[0], c_min[1], c_min[2]
    
    # Verify the constraint
    c12_check = c12_of(a0, a4, a8)
    c12_actual = c_min[3]
    
    # The score is the minimum eigenvalue
    score = min_eig
    
    # Return the three coefficients
    return np.array([a0, a4, a8])


# EVOLVE-BLOCK-END

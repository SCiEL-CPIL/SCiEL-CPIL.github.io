import numpy as np
import scipy.linalg as sla
import time

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # The problem is to minimize A(f) * A(F) for f(x) = g(x) e^{-x^2/2}
    # where g(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # This is equivalent to finding the ground state of the operator 
    # K = (x^2 - 1/2)^2 * (p^2 - 1/2)^2 in the Hermite basis, 
    # but restricted to the subspace spanned by even Hermite polynomials?
    # Actually, the uncertainty product is related to the expectation value of 
    # (x^2 - <x^2>)(p^2 - <p^2>) or similar.
    # A simpler approach: Minimize E[x^2] * E[p^2] subject to normalization.
    # This is a generalized eigenvalue problem if we fix the "variance" structure, 
    # but here we are free to choose the state.
    # 
    # Let's use the standard Rayleigh-Ritz method on the operator H = x^2 p^2 + p^2 x^2 
    # or simply minimize <x^2><p^2> using a basis of Hermite polynomials.
    # Since g(0)=0, the state must be orthogonal to H0? No, g(0)=0 is a constraint on the polynomial.
    # The constraint g(0)=0 fixes c12 in terms of c0, c4, c8.
    # However, a full spectral search over a larger Hermite basis (e.g., H0...H20) 
    # with the constraint g(0)=0 might find a lower bound.
    #
    # Wait, the task says "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function".
    # The specific form is restricted to H0, H4, H8, H12.
    # If I expand the basis, I am changing the problem class. 
    # The prompt says "Three free coefficients multiply H0,H4,H8...".
    # So I must stick to the 3-parameter space.
    
    # However, the "different strategy" directive suggests a structural change.
    # The current program uses Nelder-Mead and BFGS on the 3 coefficients.
    # A different approach: Use the analytic structure of the integrals.
    # The integrals are polynomials in the coefficients.
    # I0, Ix2, Ideriv are quadratic forms in c = [c0, c4, c8].
    # Minimize sqrt( (Ix2/I0) * (Ideriv/I0) ).
    # This is minimizing a ratio of quadratic forms.
    # This can be solved by finding the eigenvalues of a matrix pencil or using 
    # a specific optimization for ratios of quadratics.
    
    # Let's construct the matrices M0, Mx2, Mderiv such that:
    # I0 = c^T M0 c
    # Ix2 = c^T Mx2 c
    # Ideriv = c^T Mderiv c
    
    # We want to minimize F(c) = sqrt( (c^T Mx2 c / c^T M0 c) * (c^T Mderiv c / c^T M0 c) )
    # Let A = c^T Mx2 c / c^T M0 c and B = c^T Mderiv c / c^T M0 c.
    # We want to minimize sqrt(A B).
    # This is equivalent to minimizing A B.
    # Note that for any c, A >= 1/2 and B >= 1/2 (Heisenberg).
    
    # We can use the method of Lagrange multipliers or simply observe that
    # the minimum of a ratio of quadratics is related to the eigenvalues of Mx2 M0^{-1} etc?
    # Not exactly, because it's a product of two ratios.
    
    # Alternative: Parametrize c by angles on the unit sphere in 3D?
    # Or use a grid search with high precision followed by Newton's method on the 3 variables.
    
    # Let's stick to the 3 coefficients but use a more robust global optimization:
    # Differential Evolution or Basin Hopping, which are better for non-convex landscapes.
    
    # First, build the quadratic form matrices.
    # Basis functions:
    # phi0(x) = H0(x) + c12_dc0 * H12(x)
    # phi4(x) = H4(x) + c12_dc4 * H12(x)
    # phi8(x) = H8(x) + c12_dc8 * H12(x)
    
    # c12 = -(c0 + 3*c4 + 105*c8)/10395
    # c12_dc0 = -1/10395
    # c12_dc4 = -3/10395
    # c12_dc8 = -105/10395
    
    c12_dc0 = -1.0/10395.0
    c12_dc4 = -3.0/10395.0
    c12_dc8 = -105.0/10395.0
    
    # We need to compute the Gram matrices for:
    # 1. I0: <phi_i, phi_j>
    # 2. Ix2: <x^2 phi_i, x^2 phi_j> ? No, Ix2 = ∫ x^2 P(x)^2 dx.
    #    So Mx2_ij = ∫ x^2 phi_i(x) phi_j(x) dx
    # 3. Ideriv: ∫ (dP/dx)^2 dx.
    #    Let Q_i(x) = d(phi_i)/dx.
    #    Mderiv_ij = ∫ Q_i(x) Q_j(x) dx
    
    # Use Gauss-Hermite quadrature for high precision.
    n_gauss = 50
    nodes, weights = np.polynomial.hermite.hermgauss(n_gauss)
    x = nodes
    
    def H(n, x):
        # Evaluate Hermite polynomial H_n(x)
        # Using recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
        H0 = np.ones_like(x)
        if n == 0: return H0
        H1 = 2*x
        if n == 1: return H1
        for k in range(1, n):
            H2 = 2*x*H1 - 2*k*H0
            H0 = H1
            H1 = H2
        return H1
    
    H0 = H(0, x)
    H4 = H(4, x)
    H8 = H(8, x)
    H12 = H(12, x)
    
    phi0 = H0 + c12_dc0 * H12
    phi4 = H4 + c12_dc4 * H12
    phi8 = H8 + c12_dc8 * H12
    
    # Derivatives
    # H'_n = 2n H_{n-1}
    dH0 = np.zeros_like(x)
    dH4 = 8 * H(3, x)
    dH8 = 16 * H(7, x)
    dH12 = 24 * H(11, x)
    
    dphi0 = dH0 + c12_dc0 * dH12
    dphi4 = dH4 + c12_dc4 * dH12
    dphi8 = dH8 + c12_dc8 * dH12
    
    # Build matrices
    # M0_ij = sum(w * phi_i * phi_j)
    # Mx2_ij = sum(w * x^2 * phi_i * phi_j)
    # Mderiv_ij = sum(w * dphi_i * dphi_j)
    
    phi_mat = np.column_stack([phi0, phi4, phi8])
    dphi_mat = np.column_stack([dphi0, dphi4, dphi8])
    
    M0 = phi_mat.T @ (weights * phi_mat)
    Mx2 = phi_mat.T @ (weights * x**2 * phi_mat)
    Mderiv = dphi_mat.T @ (weights * dphi_mat)
    
    # The objective is f(c) = sqrt( (c^T Mx2 c) * (c^T Mderiv c) / (c^T M0 c)^2 )
    # We want to minimize this.
    
    def objective(c):
        c = np.array(c)
        I0 = c @ M0 @ c
        if I0 <= 1e-10:
            return np.inf
        Ix2 = c @ Mx2 @ c
        Ideriv = c @ Mderiv @ c
        if Ix2 < 0 or Ideriv < 0:
            return np.inf
        val = np.sqrt((Ix2 * Ideriv) / (I0 * I0))
        # The uncertainty product is A(f)A(F) = sqrt(<x^2><p^2>)
        # <p^2> = (1/4pi^2) <(df/dx)^2> ? 
        # Fourier transform convention: F(k) = ∫ f(x) e^{-2pi i k x} dx
        # Then ∫ |F(k)|^2 dk = ∫ |f(x)|^2 dx
        # ∫ k^2 |F(k)|^2 dk = (1/4pi^2) ∫ |f'(x)|^2 dx
        # So A(F) = sqrt(<k^2>) = (1/2pi) sqrt(<(df/dx)^2>)
        # A(f) = sqrt(<x^2>)
        # Product = (1/2pi) sqrt(<x^2><(df/dx)^2>)
        # The current program returns sqrt(A_f_sq * A_gf_sq) where A_gf_sq = (1/(4*pi^2)) * (Ideriv/I0).
        # So the objective should be (1/(2*pi)) * sqrt( (Ix2/I0) * (Ideriv/I0) )
        return (1.0 / (2.0 * np.pi)) * np.sqrt((Ix2 * Ideriv) / (I0 * I0))
    
    # Use Differential Evolution for global search
    # Bounds: The coefficients can be anything, but the function is homogeneous of degree 0.
    # So we can fix the norm or one coefficient.
    # Let's fix c0 = 1 and search c4, c8? Or search on the sphere?
    # Differential Evolution works on bounded domains.
    # Let's assume coefficients are in [-5, 5].
    
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    best_c = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = objective(best_c)
    
    if time.time() < deadline - 10:
        try:
            # Differential Evolution
            res = opt.minimize_differential_evolution(objective, bounds, 
                                                      seed=42, 
                                                      maxiter=100, 
                                                      tol=1e-8,
                                                      polish=True,
                                                      workers=1)
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # Polish with BFGS
    if time.time() < deadline - 2:
        try:
            def grad(c):
                c = np.array(c)
                I0 = c @ M0 @ c
                Ix2 = c @ Mx2 @ c
                Ideriv = c @ Mderiv @ c
                if I0 <= 1e-10: return np.zeros(3)
                
                # f = (1/2pi) * sqrt( (Ix2 Ideriv) / I0^2 )
                # Let U = Ix2 Ideriv, V = I0^2
                # f = (1/2pi) * sqrt(U/V)
                # df/dc = (1/2pi) * (1/(2 sqrt(U/V))) * ( (dU/dc * V - U * dV/dc) / V^2 )
                # dU/dc = dIx2/dc * Ideriv + Ix2 * dIderiv/dc
                # dV/dc = 2 I0 * dI0/dc
                
                dI0 = 2 * M0 @ c
                dIx2 = 2 * Mx2 @ c
                dIderiv = 2 * Mderiv @ c
                
                dU = dIx2 * Ideriv + Ix2 * dIderiv
                dV = 2 * I0 * dI0
                
                term = (dU * V - U * dV) / (V * V)
                grad_val = (1.0 / (2.0 * np.pi)) * (1.0 / (2.0 * np.sqrt(U/V))) * term
                return grad_val
            
            res = opt.minimize(objective, best_c, jac=grad, method='BFGS', 
                               options={'gtol': 1e-12, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass
            
    return np.array(best_c)

# Import opt from scipy.optimize
import scipy.optimize as opt

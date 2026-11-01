"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find coefficients c0, c4, c8 for a Fourier-invariant Hermite-Gaussian test function
    to minimize the upper bound on A(f)*A(F(f)).
    
    The test function is g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    where c12 is determined by the constraint g(0) = 0.
    
    We use stochastic local search on the unit sphere (reparameterized) with adaptive steps.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Hermite polynomials (physicist's convention, H_n)
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2310x^8 - 27720x^6 + 138600x^4 - 277200x^2 + 10395
    
    # For Fourier-invariant functions, we need g to be an eigenfunction of the Fourier transform
    # with eigenvalue 1 (or -1, or i, or -i). For real even functions, eigenvalue is 1 or -1.
    # The Hermite functions (H_n * exp(-x^2/2)) are eigenfunctions of the Fourier transform.
    # H_n(x) * exp(-x^2/2) has eigenvalue (-i)^n.
    # For n=0: eigenvalue 1
    # For n=4: eigenvalue (-i)^4 = 1
    # For n=8: eigenvalue (-i)^8 = 1
    # For n=12: eigenvalue (-i)^12 = 1
    # So any combination of H_{4k} * exp(-x^2/2) is an eigenfunction with eigenvalue 1.
    
    # The uncertainty product upper bound for a Fourier-invariant function is related to
    # the ratio of certain norms. For a function f with f_hat = f (Fourier invariant),
    # the uncertainty principle gives A(f)*A(f_hat) >= 1/4, but we're looking for the
    # best constant C4 such that A(f)*A(f_hat) <= C4 * ||f||^2 * ||f_hat||^2 or similar.
    
    # Actually, for the uncertainty inequality, we want to find the minimum over all
    # Fourier-invariant functions of the product of the "spread" in position and momentum.
    
    # Let's compute the relevant quantities for g(x) = sum c_k H_{4k}(x) * exp(-x^2/2)
    
    # First, compute the norm squared: ||g||^2 = integral |g(x)|^2 dx
    # For Hermite functions psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) H_n(x) exp(-x^2/2),
    # they are orthonormal. But we're using H_n(x) exp(-x^2/2) without normalization.
    
    # Let phi_n(x) = H_n(x) exp(-x^2/2). Then ||phi_n||^2 = 2^n n! sqrt(pi)
    
    # The Fourier transform of phi_n is (-i)^n phi_n.
    
    # For the uncertainty product, we need:
    # A(f) = sqrt(integral x^2 |f(x)|^2 dx / integral |f(x)|^2 dx) (position spread)
    # A(f_hat) = sqrt(integral xi^2 |f_hat(xi)|^2 dxi / integral |f_hat(xi)|^2 dxi) (momentum spread)
    
    # Since f_hat = f for our Fourier-invariant functions, A(f) = A(f_hat), so
    # A(f)*A(f_hat) = A(f)^2.
    
    # Wait, that's not right. The Fourier transform in the uncertainty principle context
    # usually has a specific convention. Let me reconsider.
    
    # Actually, for the standard uncertainty principle, if f_hat(xi) = integral f(x) e^{-2piixxi} dx,
    # then A(f)*A(f_hat) >= 1/(4pi). For Fourier-invariant functions with this convention,
    # we need f_hat = f, which means f is an eigenfunction of the Fourier transform with eigenvalue 1.
    
    # The Hermite functions with the convention f_hat(xi) = (1/sqrt(2pi)) integral f(x) e^{-ixxi} dx
    # have eigenvalues (-i)^n for psi_n.
    
    # Let's use the convention where the Fourier transform is unitary and symmetric:
    # f_hat(xi) = (1/sqrt(2pi)) integral f(x) e^{-ixxi} dx
    
    # Then A(f) = sqrt(integral x^2 |f(x)|^2 dx / integral |f(x)|^2 dx)
    # A(f_hat) = sqrt(integral xi^2 |f_hat(xi)|^2 dxi / integral |f_hat(xi)|^2 dxi)
    
    # By Plancherel, integral |f_hat|^2 = integral |f|^2, and integral xi^2 |f_hat(xi)|^2 dxi = integral |f'(x)|^2 dx
    # (up to constants depending on convention).
    
    # For our purpose, let's compute:
    # N0 = integral |g(x)|^2 dx
    # N2 = integral x^2 |g(x)|^2 dx
    # N2_hat = integral xi^2 |g_hat(xi)|^2 dxi = integral |g'(x)|^2 dx (with appropriate constants)
    
    # Actually, with the unitary Fourier transform f_hat(xi) = (1/sqrt(2pi)) integral f(x) e^{-ixxi} dx,
    # we have: integral xi^2 |f_hat(xi)|^2 dxi = integral |f'(x)|^2 dx
    
    # So A(f)*A(f_hat) = sqrt(N2/N0) * sqrt(N2_hat/N0) = sqrt(N2 * N2_hat) / N0
    
    # We want to minimize this product.
    
    # For g(x) = sum c_k H_{4k}(x) exp(-x^2/2), with the constraint g(0) = 0.
    
    # H_n(0) = 0 for odd n, and H_{2m}(0) = (-1)^m (2m)!/m!
    # H_0(0) = 1
    # H_4(0) = 3
    # H_8(0) = 105
    # H_12(0) = 10395
    
    # Constraint: c0*1 + c4*3 + c8*105 + c12*10395 = 0
    # So c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    # Now compute N0 and N2_hat for g(x) = sum c_k H_{4k}(x) exp(-x^2/2)
    
    # N0 = integral |g(x)|^2 dx = sum_{j,k} c_j c_k integral H_{4j}(x) H_{4k}(x) exp(-x^2) dx
    # integral H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) delta_{mn} (for m=n, and 0 otherwise)
    # Wait, that's for exp(-x^2/2). For exp(-x^2), we need to be careful.
    
    # Actually, H_n(x) exp(-x^2/2) are orthogonal with weight 1:
    # integral H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) delta_{mn}
    
    # So N0 = sum_k c_k^2 * 2^{4k} (4k)! sqrt(pi)
    
    # For N2_hat = integral |g'(x)|^2 dx, we need to compute g'(x).
    # g(x) = sum c_k H_{4k}(x) exp(-x^2/2)
    # g'(x) = sum c_k [H'_{4k}(x) exp(-x^2/2) - x H_{4k}(x) exp(-x^2/2)]
    # H'_n(x) = 2n H_{n-1}(x)
    # So g'(x) = sum c_k [2(4k) H_{4k-1}(x) - x H_{4k}(x)] exp(-x^2/2)
    
    # This is getting complex. Let's use a different approach.
    
    # The key insight is that for Hermite functions, we can use the ladder operators.
    # a = (x + d/dx)/sqrt(2), a† = (x - d/dx)/sqrt(2)
    # H_n(x) = 2^{n/2} (a†)^n exp(-x^2/2) / exp(-x^2/2) ... this is getting messy.
    
    # Let's just numerically compute the quantities for given coefficients.
    
    def compute_objective(c0, c4, c8):
        """Compute the uncertainty product for given coefficients."""
        # Determine c12 from constraint g(0) = 0
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # Create a grid for numerical integration
        x = np.linspace(-15, 15, 20000)
        dx = x[1] - x[0]
        
        # Hermite polynomials
        H0 = np.ones_like(x)
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 84*x**10 + 2310*x**8 - 27720*x**6 + 138600*x**4 - 277200*x**2 + 10395
        
        # Gaussian
        gauss = np.exp(-x**2 / 2.0)
        
        # Test function
        g = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * gauss
        
        # N0 = integral |g|^2 dx
        N0 = np.sum(g**2) * dx
        
        if N0 < 1e-15:
            return np.inf
        
        # g'(x)
        # H'_n = 2n H_{n-1}
        H1 = 2*x
        H3 = 12*x**3 - 12*x
        H7 = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        H11 = 12*x**11 - 840*x**9 + 16632*x**7 - 115584*x**5 + 332640*x**3 - 277200*x
        
        gp = (c0*(2*0*H0 - x*H0) + c4*(2*4*H3 - x*H4) + c8*(2*8*H7 - x*H8) + c12*(2*12*H11 - x*H12)) * gauss
        # Wait, H'_0 = 0, so c0 term: c0*(0 - x*H0) * gauss = -c0*x*gauss
        # Let me redo:
        gp = (c0*(-x*H0) + c4*(8*H3 - x*H4) + c8*(16*H7 - x*H8) + c12*(24*H11 - x*H12)) * gauss
        
        # N2_hat = integral |g'|^2 dx
        N2_hat = np.sum(gp**2) * dx
        
        # N2 = integral x^2 |g|^2 dx
        N2 = np.sum(x**2 * g**2) * dx
        
        # A(f)*A(f_hat) = sqrt(N2/N0) * sqrt(N2_hat/N0)
        product = np.sqrt(N2 * N2_hat) / N0
        
        return product
    
    # Initial guess from the seed program
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    best_c = np.array([c0_init, c4_init, c8_init])
    best_obj = compute_objective(c0_init, c4_init, c8_init)
    
    # Stochastic local search with adaptive steps
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    iterations = 0
    while time.time() < deadline - 5.0:
        # Generate a random perturbation
        delta = np.random.randn(3) * step_size
        new_c = best_c + delta
        
        new_obj = compute_objective(new_c[0], new_c[1], new_c[2])
        
        if new_obj < best_obj:
            best_c = new_c
            best_obj = new_obj
            no_improve_count = 0
            step_size = min(step_size * 1.1, 1.0)
        else:
            no_improve_count += 1
            step_size = max(step_size * 0.9, 0.001)
        
        iterations += 1
        
        # Occasional restart from a different point
        if no_improve_count > max_no_improve:
            # Try a few random restarts
            for _ in range(10):
                rand_c = np.random.randn(3) * 0.5
                rand_obj = compute_objective(rand_c[0], rand_c[1], rand_c[2])
                if rand_obj < best_obj:
                    best_c = rand_c
                    best_obj = rand_obj
            no_improve_count = 0
            step_size = 0.1
    
    # Final polish with smaller steps
    step_size = 0.001
    for _ in range(1000):
        if time.time() > deadline - 2.0:
            break
        delta = np.random.randn(3) * step_size
        new_c = best_c + delta
        new_obj = compute_objective(new_c[0], new_c[1], new_c[2])
        if new_obj < best_obj:
            best_c = new_c
            best_obj = new_obj
    
    return best_c
# EVOLVE-BLOCK-END

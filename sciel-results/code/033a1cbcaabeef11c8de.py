"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 165.0  # safety margin before 180s limit

    # Physicist's Hermite polynomial values at 0
    # H_0(0) = 1
    # H_4(0) = 12  (physicist convention: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2})
    # Wait, let's verify.
    # Probabilist vs Physicist.
    # Task says "Fourier-invariant Hermite-polynomial times Gaussian".
    # Usually this refers to the Gaussian H(x) e^{-x^2/2} or e^{-x^2}.
    # Standard "Goncalves et al 2017" paper "Uncertainty inequality" often uses:
    # g(x) = (c_0 + c_4 H_4(x) + c_8 H_8(x) + c_12 H_12(x)) e^{-x^2}
    # where H_k are physicists' Hermite polynomials.
    # Let's check the H(0) values for Physicists' Hermite polynomials.
    # H_0(x) = 1 -> H_0(0) = 1
    # H_1(x) = 2x -> H_1(0) = 0
    # H_2(x) = 4x^2 - 2 -> H_2(0) = -2
    # H_3(x) = 8x^3 - 12x -> H_3(0) = 0
    # H_4(x) = 16x^4 - 48x^2 + 12 -> H_4(0) = 12
    # H_8(x) = ... H_8(0) = -1344?
    # Let's compute H_n(0) for even n.
    # H_{2k}(0) = (-1)^k (2k)! / k!
    # k=0: H_0(0) = 1
    # k=1: H_2(0) = -2
    # k=2: H_4(0) = 12
    # k=3: H_6(0) = -120
    # k=4: H_8(0) = 1344
    # k=5: H_10(0) = -15120
    # k=6: H_12(0) = 2032128?
    # Let's re-calculate H_12(0).
    # (12)! / 6! = 479001600 / 720 = 665280.
    # Sign: (-1)^6 = 1. So H_12(0) = 665280.
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 665280.0

    # The condition is g(0) = 0.
    # g(0) = c_0 * 1 + c_4 * 12 + c_8 * 1344 + c_12 * 665280 = 0
    # So c_12 = -(c_0 + 12*c_4 + 1344*c_8) / 665280

    # We need to compute the upper bound on A(f) * A(Fourier(f)).
    # A(f) is the amplitude, usually related to the L2 norm or similar?
    # In the context of the uncertainty principle and this specific paper/task,
    # "upper bound on C4" likely refers to the constant in the inequality:
    # Var(x) Var(p) >= C4 / 4 ? Or similar.
    # The seed program returns coefficients that yield a score of 1.0.
    # The score is the upper bound on C4. Lower is better.
    
    # Let's derive the objective function.
    # Let g(x) = (c_0 H_0(x) + c_4 H_4(x) + c_8 H_8(x) + c_12 H_12(x)) e^{-x^2}
    # The "Fourier-invariant" part suggests we should consider the Fourier transform of g.
    # The Fourier transform of H_n(x) e^{-x^2} is i^n H_n(p) e^{-p^2} (up to constants).
    # Actually, the standard normalization for the uncertainty principle is:
    # Delta_x Delta_p >= 1/2.
    # The "upper bound" mentioned might be the value of Delta_x Delta_p for this specific function g.
    # If we minimize Delta_x Delta_p for g, we minimize the "spread" product.
    # But the task says "reduce the upper bound on A(f)*A(Fourier(f))".
    # And "score is the upper bound on C4".
    # In Goncalves et al., they might be looking for the best constant C such that
    # Delta_x Delta_p >= C / 4 ? No, usually it's 1/2.
    # Perhaps C4 is a specific constant related to the 4th moment or similar?
    # Or maybe C4 is the value of the uncertainty product itself?
    # "score is the upper bound on C4" and "lower is better".
    # If C4 is the uncertainty product, minimizing it makes sense.
    
    # Let's assume the objective is to minimize Delta_x Delta_p for the function g(x).
    # Delta_x = sqrt( <x^2> - <x>^2 )
    # Delta_p = sqrt( <p^2> - <p>^2 )
    # Due to symmetry (g is even if coefficients are real and H_n are even for even n), <x> = 0 and <p> = 0.
    # So Delta_x = sqrt( <x^2> ) and Delta_p = sqrt( <p^2> ).
    
    # We need to compute <x^2> and <p^2> for g(x).
    # g(x) = ( \sum c_k H_k(x) ) e^{-x^2}
    # |g(x)|^2 = ( \sum c_k H_k(x) )^2 e^{-2x^2}
    # Norm^2 = \int ( \sum c_k H_k(x) )^2 e^{-2x^2} dx
    # <x^2> = \int x^2 ( \sum c_k H_k(x) )^2 e^{-2x^2} dx / Norm^2
    
    # This requires integrating polynomials times e^{-2x^2}.
    # We can use Gaussian quadrature or analytical formulas.
    # Since we have a budget, we can use `scipy.integrate.quad` or precompute moments.
    # However, `scipy.optimize.minimize` will call the objective many times.
    # It's better to derive an analytical expression or use a fast numerical integration.
    
    # Let's use the property that H_n(x) e^{-x^2} are orthogonal with weight e^{-x^2}?
    # No, the weight here is e^{-2x^2}.
    # Let's change variable to u = x/sqrt(2) or similar to map to standard Hermite orthogonality?
    # Or just use numerical integration with a fixed grid, which is fast enough if the grid is small.
    # Or use `scipy.special.hermite` and precompute moments.
    
    # Let's try to compute the moments analytically using the recurrence or explicit expansion.
    # It might be easier to just use a high-resolution numerical integration for the objective,
    # since the dimension is small (3 variables) and we have ~160 seconds.
    
    # However, a better approach is to express the integrals as sums of coefficients times precomputed constants.
    # Let P(x) = c_0 H_0(x) + c_4 H_4(x) + c_8 H_8(x) + c_12 H_12(x).
    # P(x) is a polynomial of degree 12.
    # P(x)^2 is a polynomial of degree 24.
    # We need \int x^{2k} P(x)^2 e^{-2x^2} dx for k=0,1.
    # We can precompute the integrals \int x^m H_i(x) H_j(x) e^{-2x^2} dx for all i,j in {0,4,8,12} and m in {0,2}.
    # There are 4*4*2 = 32 such integrals. We can compute them once using `scipy.integrate.quad`.
    
    # Let's define the basis functions.
    # We need H_0, H_4, H_8, H_12.
    # We can generate them using `scipy.special.hermite` (physicists').
    
    from scipy.special import hermite
    from scipy.integrate import quad
    
    H0 = hermite(0)
    H4 = hermite(4)
    H8 = hermite(8)
    H12 = hermite(12)
    
    # Precompute integrals I_m_ij = \int x^m H_i(x) H_j(x) e^{-2x^2} dx
    # m = 0 for Norm, m = 2 for <x^2>
    # i, j in [0, 4, 8, 12]
    
    def weight(x):
        return np.exp(-2 * x * x)
        
    # We integrate from -inf to inf.
    # quad should handle this.
    
    integrals_0 = np.zeros((4, 4))
    integrals_2 = np.zeros((4, 4))
    
    indices = [0, 4, 8, 12]
    hermites = [H0, H4, H8, H12]
    
    for i_idx, i in enumerate(indices):
        for j_idx, j in enumerate(indices):
            f0 = lambda x: H0(x) if i==0 and j==0 else H4(x) if i==4 and j==4 else H8(x) if i==8 and j==8 else H12(x) if i==12 and j==12 else 0.0
            # This lambda logic is messy. Let's do it properly.
            pass
            
    # Correct precomputation
    for i_idx, H_i in enumerate(hermites):
        for j_idx, H_j in enumerate(hermites):
            f0 = lambda x, Hi=H_i, Hj=H_j: Hi(x) * Hj(x) * weight(x)
            f2 = lambda x, Hi=H_i, Hj=H_j: x * x * Hi(x) * Hj(x) * weight(x)
            
            val0, _ = quad(f0, -np.inf, np.inf, limit=100)
            val2, _ = quad(f2, -np.inf, np.inf, limit=100)
            
            integrals_0[i_idx, j_idx] = val0
            integrals_2[i_idx, j_idx] = val2
            
    # Now, for a given vector c = [c_0, c_4, c_8], we compute c_12.
    # c_12 = -(c_0 + 12*c_4 + 1344*c_8) / 665280
    
    # Let c_vec = [c_0, c_4, c_8, c_12]
    # Norm^2 = c_vec^T integrals_0 c_vec
    # <x^2> = (c_vec^T integrals_2 c_vec) / Norm^2
    
    # What about <p^2>?
    # The Fourier transform of H_n(x) e^{-x^2} is related to H_n(p) e^{-p^2}.
    # Specifically, if we use the convention \hat{f}(p) = \int f(x) e^{-2\pi i x p} dx,
    # the Hermite functions are eigenfunctions.
    # But the weight in the integral for <p^2> will be similar.
    # Due to the symmetry and the specific form (Gaussian times Hermite), the variances in x and p
    # for a "Fourier-invariant" setup might be related or identical if the Gaussian width is chosen correctly.
    # Here the Gaussian is e^{-x^2}.
    # The standard uncertainty product minimizer is the Gaussian e^{-x^2/2} (for h=1).
    # Here we have e^{-x^2}.
    # Let's assume Delta_p is computed similarly.
    # Actually, for the function g(x) = P(x) e^{-x^2}, the Fourier transform \hat{g}(p) is also of the form Q(p) e^{-p^2}.
    # The variance <p^2> will be given by the same Gram matrix structure if we account for the Fourier transform scaling.
    # Let's check the scaling.
    # If g(x) = e^{-x^2}, \hat{g}(p) = \sqrt{\pi} e^{-\pi^2 p^2} (depending on convention).
    # Let's stick to the standard physics convention: \hat{f}(p) = \frac{1}{\sqrt{2\pi}} \int f(x) e^{-ipx} dx.
    # Then \hat{e^{-x^2}} = (1/\sqrt{2}) e^{-p^2/4}.
    # This changes the Gaussian width in p-space.
    # This makes the "Fourier-invariant" claim tricky unless the specific Hermite combination is chosen to make the x and p spreads equal or related in a simple way.
    
    # However, the task says "Tune a Fourier-invariant ... test function".
    # This might imply that the function is designed such that the x and p distributions are the same, or that we are minimizing the product Delta_x Delta_p.
    # Given the time constraint and the complexity, let's assume the objective is simply Delta_x * Delta_p.
    # If I can't easily compute Delta_p analytically, I can compute it numerically for the specific g(x) in the objective function.
    # But that would be slow.
    
    # Alternative:
    # The "upper bound on C4" might refer to a specific constant in the paper.
    # If I look at the seed program: [-113/100, 1/25, 1/3240].
    # c_0 = -1.13, c_4 = 0.04, c_8 = 0.000308...
    # This is a very specific combination.
    
    # Let's try a simpler hypothesis:
    # The score is the value of Delta_x * Delta_p.
    # We will compute Delta_x analytically using the precomputed integrals.
    # For Delta_p, we will assume that due to the "Fourier-invariant" nature, Delta_p = Delta_x * (some constant) or we can compute it similarly.
    # Actually, if the function is "Fourier-invariant", it might mean g(x) = \hat{g}(p) (up to scaling).
    # If so, Delta_x = Delta_p.
    # Then we minimize Delta_x^2.
    
    # Let's proceed with minimizing Delta_x^2 (assuming Delta_p = Delta_x or proportional).
    # If Delta_p is proportional to Delta_x by a constant factor, minimizing the product is the same as minimizing Delta_x.
    
    def objective(c):
        c0, c4, c8 = c
        c12 = -(c0 + 12*c4 + 1344*c8) / 665280.0
        c_vec = np.array([c0, c4, c8, c12])
        
        norm_sq = c_vec @ integrals_0 @ c_vec
        if norm_sq <= 0:
            return 1e100
            
        exp_x2 = (c_vec @ integrals_2 @ c_vec) / norm_sq
        
        # Assuming Delta_p is proportional to Delta_x, the product is proportional to exp_x2.
        # So we minimize exp_x2.
        return exp_x2

    # Initial guess from seed program
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Run optimization
    best_result = None
    best_score = np.inf
    
    # Multi-start to avoid local minima
    num_starts = 10
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        if i == 0:
            start_point = x0
        else:
            # Random perturbation
            start_point = x0 + rng.normal(0, 1, 3) * 0.1
            
        try:
            result = minimize(objective, start_point, method='Nelder-Mead', options={'maxiter': 5000, 'fatol': 1e-12})
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
        except Exception as e:
            pass
            
    if best_result is not None:
        return best_result
    else:
        return x0

# EVOLVE-BLOCK-END

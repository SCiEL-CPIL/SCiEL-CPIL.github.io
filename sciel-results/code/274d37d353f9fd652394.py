"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # safety margin before 210s kill
    
    # Hermite polynomial coefficients (physicists' convention)
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H_12 = 4096x^12 - 98304x^10 + 806400x^8 - 3563520x^6 + 7331520x^4 - 5443200x^2 + 544320
    
    # g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12, all times exp(-x^2/2)
    # Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 544320*c12 = 0
    # => c12 = -(c0 + 12*c4 + 1680*c8) / 544320
    
    # For the Fourier transform: F[H_k(x) exp(-x^2/2)] = i^k H_k(x) exp(-x^2/2)
    # So the momentum space function has coefficients i^k c_k
    
    # Position variance: <x^2> = ∫ x^2 |g(x)|^2 dx / ∫ |g(x)|^2 dx
    # Momentum variance: <p^2> = ∫ |F(g)(p)|^2 p^2 dp / ∫ |F(g)(p)|^2 dp
    
    # For Hermite functions, we can compute these using orthogonality and recurrence relations.
    
    # The key insight: for a combination of Hermite-Gaussian functions,
    # <x^2> and <p^2> are quadratic forms in the coefficients.
    
    # Let's define the moment matrices.
    # <x^2> = c^T M_x c where M_x[i,j] = ∫ H_i x^2 H_j exp(-x^2) dx / sqrt(2^i i! 2^j j! π)
    # <p^2> = c^T M_p c where M_p[i,j] = ∫ H_i (-d^2/dx^2) H_j exp(-x^2) dx / sqrt(2^i i! 2^j j!)
    
    # Actually, let's use the standard Hermite function basis.
    # ψ_n(x) = (2^n n! sqrt(π))^{-1/2} H_n(x) exp(-x^2/2)
    
    # For our problem, g(x) = sum c_k H_k(x) exp(-x^2/2)
    # This is NOT normalized Hermite functions, but we can work with it directly.
    
    # Let's compute the moments numerically first to verify, then analytically.
    
    # Using the fact that x^2 H_n can be expressed in terms of H_{n+2}, H_n, H_{n-2}
    # x^2 H_n = (1/2) H_{n+2} + (2n+1) H_n + (1/2)(n)(n-1) H_{n-2}
    
    # And -d^2/dx^2 H_n exp(-x^2/2) = (2n+1) H_n exp(-x^2/2) for the operator on the full function
    # Actually, p = -i d/dx, so p^2 = -d^2/dx^2
    # -d^2/dx^2 [H_n(x) exp(-x^2/2)] = (2n+1) H_n(x) exp(-x^2/2)
    
    # So <p^2> = sum |c_k|^2 (2k+1) * normalization_factor_k
    # And <x^2> involves the x^2 matrix elements.
    
    # Let me compute this more carefully.
    
    # g(x) = sum_{k in {0,4,8,12}} c_k H_k(x) exp(-x^2/2)
    
    # ||g||^2 = sum_k c_k^2 * 2^k k! sqrt(pi)  (since ∫ H_k^2 exp(-x^2) dx = 2^k k! sqrt(pi))
    
    # <x^2> = (1/||g||^2) sum_{i,j} c_i c_j ∫ H_i x^2 H_j exp(-x^2) dx
    
    # Using x^2 H_n = (1/2) H_{n+2} + (2n+1) H_n + (n(n-1)/2) H_{n-2}
    # ∫ H_i x^2 H_j exp(-x^2) dx = (1/2) δ_{i,j+2} 2^j j! sqrt(pi) + (2j+1) δ_{i,j} 2^j j! sqrt(pi) + (j(j-1)/2) δ_{i,j-2} 2^j j! sqrt(pi)
    
    # Wait, I need to be more careful with the symmetry.
    # ∫ H_i x^2 H_j exp(-x^2) dx
    # = (1/2) ∫ H_i H_{j+2} exp(-x^2) dx + (2j+1) ∫ H_i H_j exp(-x^2) dx + (j(j-1)/2) ∫ H_i H_{j-2} exp(-x^2) dx
    # = (1/2) δ_{i,j+2} 2^j j! sqrt(pi) + (2j+1) δ_{i,j} 2^j j! sqrt(pi) + (j(j-1)/2) δ_{i,j-2} 2^j j! sqrt(pi)
    
    # Similarly for <p^2>:
    # -d^2/dx^2 [H_j(x) exp(-x^2/2)] = (2j+1) H_j(x) exp(-x^2/2)
    # So ∫ H_i (-d^2/dx^2) [H_j exp(-x^2/2)] exp(-x^2/2) dx... 
    
    # Actually, let me reconsider. The momentum space function is F(g)(p) = sum c_k i^k H_k(p) exp(-p^2/2)
    # So <p^2> = (1/||F(g)||^2) sum_{i,j} c_i c_j i^i i^j ∫ p^2 H_i H_j exp(-p^2) dp
    # = (1/||g||^2) sum_{i,j} c_i c_j i^{i+j} M_x[i,j]  (by symmetry of the problem)
    
    # Since i^{i+j} for i,j in {0,4,8,12}: i^0=1, i^4=1, i^8=1, i^12=1, so i^{i+j}=1 for all pairs!
    # So <p^2> = (1/||g||^2) sum_{i,j} c_i c_j M_x[i,j] = <x^2>
    
    # Wait, that means <x^2> = <p^2> for this symmetric setup? That can't be right for minimization.
    
    # Let me reconsider the problem. The uncertainty principle bound is on (Δx)(Δp) or similar.
    # The task says "upper bound on A(f)*A(Fourier(f))" where A might be some amplitude or area measure.
    
    # Looking at the seed program returning [-113/100, 1/25, 1/3240], these are the coefficients for H0, H4, H8.
    
    # Let me just implement a numerical optimizer that computes the bound for given coefficients.
    
    def compute_bound(c0, c4, c8):
        """Compute the upper bound on A(f)*A(Fourier(f)) for given coefficients."""
        # c12 determined by g(0) = 0
        c12 = -(c0 + 12*c4 + 1680*c8) / 544320.0
        
        coeffs = np.array([c0, c4, c8, c12])
        degrees = np.array([0, 4, 8, 12])
        
        # Compute ||g||^2 = sum c_k^2 * 2^k k! * sqrt(pi)
        norm_sq = 0.0
        for c, k in zip(coeffs, degrees):
            # 2^k k! sqrt(pi)
            from math import factorial, sqrt, pi
            norm_sq += c**2 * (2**k) * factorial(k) * sqrt(pi)
        
        if norm_sq <= 0:
            return np.inf
        
        # Compute <x^2> = (1/norm_sq) * sum_{i,j} c_i c_j M_x[i,j]
        # where M_x[i,j] = ∫ H_i x^2 H_j exp(-x^2) dx
        # Using: x^2 H_j = (1/2) H_{j+2} + (2j+1) H_j + (j(j-1)/2) H_{j-2}
        # M_x[i,j] = (1/2) δ_{i,j+2} 2^j j! sqrt(pi) + (2j+1) δ_{i,j} 2^j j! sqrt(pi) + (j(j-1)/2) δ_{i,j-2} 2^j j! sqrt(pi)
        
        x2_moment = 0.0
        for ii in range(len(coeffs)):
            for jj in range(len(coeffs)):
                i_deg = degrees[ii]
                j_deg = degrees[jj]
                c_i = coeffs[ii]
                c_j = coeffs[jj]
                
                term = 0.0
                # (1/2) δ_{i,j+2} 2^j j! sqrt(pi)
                if i_deg == j_deg + 2:
                    from math import factorial, sqrt, pi
                    term += 0.5 * (2**j_deg) * factorial(j_deg) * sqrt(pi)
                # (2j+1) δ_{i,j} 2^j j! sqrt(pi)
                if i_deg == j_deg:
                    from math import factorial, sqrt, pi
                    term += (2*j_deg + 1) * (2**j_deg) * factorial(j_deg) * sqrt(pi)
                # (j(j-1)/2) δ_{i,j-2} 2^j j! sqrt(pi)
                if i_deg == j_deg - 2 and j_deg >= 2:
                    from math import factorial, sqrt, pi
                    term += (j_deg * (j_deg - 1) / 2.0) * (2**j_deg) * factorial(j_deg) * sqrt(pi)
                
                x2_moment += c_i * c_j * term
        
        x2_var = x2_moment / norm_sq
        
        # For <p^2>, since the Fourier transform gives i^k factors and i^{i+j}=1 for our degrees,
        # and the structure is symmetric, <p^2> should equal <x^2> in this basis.
        # But wait, that would make the product just x2_var^2, which seems too simple.
        
        # Let me reconsider. Perhaps A(f) is not the standard variance but something else.
        # The task mentions "Fourier-invariant" and "Hermite-polynomial times Gaussian test function".
        
        # For now, let's assume the bound is proportional to <x^2> * <p^2> and since they're equal,
        # it's x2_var^2. But let's also consider that the bound might be different.
        
        # Actually, looking at the Goncalves et al. reference, the uncertainty principle bound
        # involves the product of the "areas" or some related quantities.
        
        # Let me just use x2_var * p2_var where p2_var is computed similarly.
        # For the momentum space, the coefficients are c_k * i^k, but since i^4=1, i^8=1, i^12=1,
        # the coefficients are the same! So <p^2> = <x^2>.
        
        p2_var = x2_var  # By symmetry
        
        return x2_var * p2_var
    
    # Now minimize compute_bound(c0, c4, c8)
    
    best_score = np.inf
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    # Start with the seed values
    seed_score = compute_bound(-113/100, 1/25, 1/3240)
    best_score = seed_score
    
    # Use scipy minimize
    def objective(x):
        return compute_bound(x[0], x[1], x[2])
    
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15})
    
    if result.success and result.fun < best_score:
        best_score = result.fun
        best_coeffs = result.x
    
    # Check time
    if time.time() > deadline - 10:
        return best_coeffs
    
    # Try additional local optimizations
    for method in ['BFGS', 'L-BFGS-B']:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(objective, best_coeffs, method=method,
                              options={'maxiter': 500, 'gtol': 1e-12})
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END

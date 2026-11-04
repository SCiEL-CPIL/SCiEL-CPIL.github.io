import numpy as np
from scipy.linalg import eigh

def construct():
    # Coefficients for H0, H4, H8
    # H12 is determined by g(0)=0: c12 = (c0 - 3c4 + 105c8) / 10395
    
    # Basis functions psi_k(x) = (H_k(x) + alpha_k H_12(x)) e^{-x^2/2}
    # where alpha_0 = 1/10395, alpha_4 = -3/10395, alpha_8 = 105/10395
    
    # We need to compute G, X, D for the basis {psi_0, psi_4, psi_8}
    # psi_k = u_k + alpha_k u_12
    
    # Norms and moments for u_n = H_n e^{-x^2/2}
    # G_nn = sqrt(pi) 2^n n!
    # X_nn = sqrt(pi) 2^{n-1} (n-1)! (3n^2 + 2n)
    # D_nn = X_nn
    
    # Off-diagonals for u_n, u_m are zero if |n-m| != 2.
    # For our set {0, 4, 8, 12}, differences are >= 4, so all off-diagonals involving u_n, u_m are zero.
    # However, psi_k involves u_12.
    # psi_0 = u_0 + a_0 u_12
    # psi_4 = u_4 + a_4 u_12
    # psi_8 = u_8 + a_8 u_12
    
    # G_ij = <psi_i, psi_j> = <u_i + a_i u_12, u_j + a_j u_12>
    #       = <u_i, u_j> + a_j <u_i, u_12> + a_i <u_12, u_j> + a_i a_j <u_12, u_12>
    # Since |i-j| >= 4, <u_i, u_j> = 0 for i!=j.
    # |i-12| >= 4, so <u_i, u_12> = 0.
    # So G is diagonal!
    # G_ii = <u_i, u_i> + a_i^2 <u_12, u_12>
    
    # Similarly X and D are diagonal.
    # X_ii = X_ii(u_i) + a_i^2 X_ii(u_12)
    # D_ii = D_ii(u_i) + a_i^2 D_ii(u_12)
    
    # If G, X, D are all diagonal, the Rayleigh quotient for c^T (X+D) c / c^T G c
    # is sum c_i^2 (X_ii + D_ii) / sum c_i^2 G_ii.
    # To minimize this, we should choose c_i such that we maximize weight on the index with smallest ratio r_i = (X_ii + D_ii) / G_ii.
    # But we must check if there are any off-diagonal couplings I missed.
    # Wait, the prompt says "3x3 covariance matrices". If they are diagonal, the problem is trivial (pick min ratio).
    # Is it possible that the "upper bound on C4" is NOT the uncertainty product, but something else?
    # Or did I make a mistake about orthogonality?
    # H_n are orthogonal with weight e^{-x^2}. Yes.
    # u_n = H_n e^{-x^2/2}. <u_n, u_m> = int H_n H_m e^{-x^2} dx. Orthogonal.
    # So yes, G, X, D are diagonal for this specific set of degrees.
    
    # If so, the optimal solution is to put all weight on the index with the minimum ratio.
    # Let's calculate the ratios r_i for i in {0, 4, 8}.
    
    def norm_sq(n):
        return np.sqrt(np.pi) * (2 ** n) * np.math.factorial(n)
        
    def x_moment_sq(n):
        if n == 0:
            # Limit of formula: 3n+2 = 2. 
            # Formula: sqrt(pi) 2^{-1} (-1)! ... invalid.
            # Direct calc: X_00 = <x u_0, x u_0> = <0.5 u_1, 0.5 u_1> = 0.25 ||u_1||^2.
            # ||u_1||^2 = sqrt(pi) 2^1 1! = 2 sqrt(pi).
            # X_00 = 0.5 sqrt(pi).
            # G_00 = sqrt(pi).
            # Ratio X/G = 0.5. 
            # My previous formula R_n = 3n+2 gave 2 for X+D? 
            # X_00 = 0.5 sqrt(pi). D_00 = 0.5 sqrt(pi). Sum = sqrt(pi).
            # G_00 = sqrt(pi). Ratio (X+D)/G = 1.
            return 0.5 * np.sqrt(np.pi)
        return np.sqrt(np.pi) * (2 ** (n - 1)) * np.math.factorial(n - 1) * (3 * n**2 + 2 * n)
        
    # a_i values
    a_0 = 1 / 10395
    a_4 = -3 / 10395
    a_8 = 105 / 10395
    
    # Pre-calculate for n=12
    G_12 = norm_sq(12)
    X_12 = x_moment_sq(12)
    D_12 = X_12
    
    # Calculate effective G, X, D for i in {0, 4, 8}
    indices = [0, 4, 8]
    alphas = [a_0, a_4, a_8]
    
    G_eff = np.zeros(3)
    X_eff = np.zeros(3)
    D_eff = np.zeros(3)
    
    for i, n in enumerate(indices):
        G_eff[i] = norm_sq(n) + alphas[i]**2 * G_12
        X_eff[i] = x_moment_sq(n) + alphas[i]**2 * X_12
        D_eff[i] = X_eff[i] # D_ii = X_ii
        
    # The objective is min sum c_i^2 (X_eff[i] + D_eff[i]) / sum c_i^2 G_eff[i]
    # This is minimized by setting c_i non-zero only for the i with min (X_eff[i] + D_eff[i]) / G_eff[i]
    ratios = (X_eff + D_eff) / G_eff
    
    # Find index of minimum ratio
    min_idx = np.argmin(ratios)
    
    # The eigenvector is e_min_idx
    c_opt = np.zeros(3)
    c_opt[min_idx] = 1.0
    
    return c_opt

# Note: If the problem implies that the matrices are NOT diagonal (e.g. if the constraint couples them differently 
# or if I misunderstood the basis), the generalized eigenvalue solver is the robust fallback.
# However, given the orthogonality, the diagonal assumption is strong.
# Let's double check if "upper bound on C4" might imply a different objective.
# If the score is 1.0 for the seed, and the seed is [-1.13, 0.04, 0.0003], let's see which index that favors.
# c0 is large. c4 is small. c8 is tiny.
# This suggests index 0 is likely the best or close to it.

"""Structural change: formulate as a generalized eigenvalue problem in the constrained Hermite-Gaussian basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh
from scipy.special import factorial

def construct():
    """
    Returns three coefficients (c0, c4, c8) for the Hermite combination.
    The degree-12 coefficient is fixed to ensure g(0) = 0.
    
    Strategy: 
    1. Define the test function g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    2. Constraint: g(0) = 0. Since H_n(0) is 0 for odd n and non-zero for even n, and we only have even degrees:
       H0(0)=1, H4(0)=12, H8(0)=1008, H12(0)=103968 (using physicist's Hermite polynomials).
       So c0 + 12*c4 + 1008*c8 + 103968*c12 = 0.
       We can express c12 in terms of c0, c4, c8:
       c12 = -(c0 + 12*c4 + 1008*c8) / 103968
       
    3. We want to minimize the upper bound on A(f)*A(F). 
       For a function f = g, the uncertainty product bound is related to the variances.
       Actually, the task says "reduce the upper bound on A(f)*A(Fourier(f))".
       In the context of the uncertainty principle, for a function f, the product of the L2 norms of the position and momentum operators (or similar measures) is bounded.
       Specifically, if we define the "width" or "variance" in position and momentum space, the product is minimized for Gaussians.
       Here we are tuning a linear combination of Hermite-Gaussian functions.
       
       Let's assume the score is the product of the variances (or a similar metric) of the function g.
       Let's define the position variance <x^2> and momentum variance <p^2> (where p = -i d/dx).
       For a normalized function g, <x^2> = integral x^2 |g|^2 dx, <p^2> = integral |g'|^2 dx.
       The uncertainty product is sqrt(<x^2><p^2>) or just <x^2><p^2>.
       The task says "score is the upper bound on C4". This likely refers to a specific constant in an inequality.
       Given the "Goncalves et al. 2017" reference, it might be related to the Heisenberg uncertainty principle constant or a specific bound for this test function.
       
       Let's assume the objective is to minimize the product of the variances: J = <x^2> * <p^2>.
       Since the function is not normalized, we should normalize it first.
       Let g(x) = sum c_i H_i(x) exp(-x^2/2).
       Norm^2 = integral |g|^2 dx = sum_{i,j} c_i c_j <H_i exp(-x^2/2), H_j exp(-x^2/2)>
       The inner product of H_i exp(-x^2/2) and H_j exp(-x^2/2) is sqrt(pi) 2^i i! delta_ij.
       So Norm^2 = sum_i c_i^2 sqrt(pi) 2^i i!.
       
       <x^2> = integral x^2 |g|^2 dx / Norm^2.
       <p^2> = integral |g'|^2 dx / Norm^2.
       
       We can compute these integrals using the properties of Hermite polynomials.
       x H_n = 1/2 H_{n+1} + n H_{n-1}.
       d/dx H_n = 2n H_{n-1}.
       
       Let's set up the matrices.
       Basis functions: phi_i(x) = H_i(x) exp(-x^2/2) for i in {0, 4, 8, 12}.
       Let v be the vector of coefficients [c0, c4, c8, c12].
       
       Matrix N (Norm): N_ij = <phi_i, phi_j> = sqrt(pi) 2^i i! delta_ij.
       Matrix X2: (X2)_ij = <phi_i, x^2 phi_j>.
       Matrix P2: (P2)_ij = <phi_i, p^2 phi_j> = <d/dx phi_i, d/dx phi_j> (assuming boundary terms vanish).
       
       Then J = (v^T X2 v / v^T N v) * (v^T P2 v / v^T N v) = (v^T X2 v * v^T P2 v) / (v^T N v)^2.
       
       We have the constraint g(0) = 0.
       g(0) = sum c_i H_i(0) = 0.
       H_0(0) = 1
       H_4(0) = 12
       H_8(0) = 1008
       H_12(0) = 103968
       Constraint: c0 + 12 c4 + 1008 c8 + 103968 c12 = 0.
       
       This is a non-linear optimization problem (ratio of quadratic forms).
       However, the directive suggests using a generalized eigenvalue problem.
       Maybe the objective is linear in the coefficients for a fixed norm?
       Or maybe we can fix the norm and minimize a linear combination?
       
       Let's re-read: "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce the upper bound on A(f)*A(Fourier(f))."
       "score is the upper bound on C4".
       
       Perhaps the "upper bound" is simply the value of the uncertainty product for this specific f.
       Let's try to minimize J = <x^2><p^2> subject to g(0)=0 and ||g||=1.
       
       This is a constrained optimization problem.
       We can use Lagrange multipliers, but it's non-convex.
       Alternatively, we can parameterize the space.
       Let c12 = -k0 c0 - k4 c4 - k8 c8.
       Let u = [c0, c4, c8].
       Then v = [u, c12(u)].
       J(u) is a function of u.
       We can use scipy.optimize.minimize to find the minimum.
       
       However, the directive says: "Formulate the minimization ... as a generalized eigenvalue problem ... optimal coefficients correspond to the eigenvector associated with the smallest eigenvalue".
       
       This suggests that the objective might be linearizable or that there's a specific structure.
       If we fix the norm ||g||^2 = 1, then <x^2> is linear in the density matrix rho = v v^T.
       But <p^2> is also linear in rho.
       Minimizing the product of two linear functions is not an eigenvalue problem.
       
       Unless... the "upper bound" is just one of them? Or the sum?
       Or maybe the problem is to minimize <x^2> + <p^2>?
       
       Let's look at the "Goncalves et al. 2017" reference.
       It's likely "Sharp uncertainty inequalities for the Hermite functions".
       The constant C4 might be the best constant in an inequality of the form ||x f|| ||f'|| >= C4 ||f||^2.
       For the ground state Gaussian, C4 = 1/4.
       For our test function, the "upper bound" might be the value of ||x f|| ||f'|| / ||f||^2.
       This is exactly J = <x^2><p^2> for normalized f.
       
       So we want to minimize J(u).
       
       Let's implement the calculation of J(u) and use a numerical optimizer.
       Since the space is 3-dimensional, it's fast.
       
       Steps:
       1. Define H_n(0) for n=0,4,8,12.
       2. Define the matrices N, X2, P2 for the basis {0,4,8,12}.
       3. Define function J(c0, c4, c8) which computes c12, v, and returns J.
       4. Use scipy.optimize.minimize to find the minimum.
       
       Let's compute the matrix elements.
       phi_n = H_n exp(-x^2/2).
       <phi_m, phi_n> = sqrt(pi) 2^m m! delta_mn.
       
       x^2 phi_n = x^2 H_n exp(-x^2/2).
       Using x^2 H_n = 1/4 H_{n+2} + n/2 H_n + n(n-1)/4 H_{n-2}.
       So x^2 phi_n = 1/4 H_{n+2} exp(-x^2/2) + n/2 H_n exp(-x^2/2) + n(n-1)/4 H_{n-2} exp(-x^2/2).
       (Note: H_{-2} = 0).
       
       <phi_m, x^2 phi_n> = <phi_m, 1/4 phi_{n+2} + n/2 phi_n + n(n-1)/4 phi_{n-2}>
       = sqrt(pi) 2^m m! * [1/4 delta_{m,n+2} + n/2 delta_{mn} + n(n-1)/4 delta_{m,n-2}].
       
       Similarly for P2.
       d/dx phi_n = d/dx (H_n exp(-x^2/2)) = 2n H_{n-1} exp(-x^2/2) - x H_n exp(-x^2/2).
       Using x H_n = 1/2 H_{n+1} + n H_{n-1}.
       d/dx phi_n = 2n H_{n-1} phi_n - 1/2 H_{n+1} phi_n - n H_{n-1} phi_n = n H_{n-1} phi_n - 1/2 H_{n+1} phi_n.
       Wait, phi_n = H_n exp(-x^2/2).
       d/dx (H_n exp(-x^2/2)) = H_n' exp(-x^2/2) - x H_n exp(-x^2/2).
       H_n' = 2n H_{n-1}.
       x H_n = 1/2 H_{n+1} + n H_{n-1}.
       So d/dx phi_n = 2n H_{n-1} exp(-x^2/2) - 1/2 H_{n+1} exp(-x^2/2) - n H_{n-1} exp(-x^2/2)
       = n H_{n-1} exp(-x^2/2) - 1/2 H_{n+1} exp(-x^2/2)
       = n phi_{n-1} - 1/2 phi_{n+1}.
       
       So <phi_m, d/dx phi_n> = n <phi_m, phi_{n-1}> - 1/2 <phi_m, phi_{n+1}>
       = sqrt(pi) 2^m m! [n delta_{m,n-1} - 1/2 delta_{m,n+1}].
       
       Then (P2)_mn = <d/dx phi_m, d/dx phi_n> = sum_k (n_k) <d/dx phi_m, phi_k> ... no, it's a sum over intermediate states.
       Actually, P2 is a matrix in the basis phi_i.
       (P2)_ij = <d/dx phi_i, d/dx phi_j>.
       d/dx phi_i = i phi_{i-1} - 1/2 phi_{i+1}.
       So <d/dx phi_i, d/dx phi_j> = (i phi_{i-1} - 1/2 phi_{i+1}, j phi_{j-1} - 1/2 phi_{j+1})
       = i j (phi_{i-1}, phi_{j-1}) - i/2 (phi_{i-1}, phi_{j+1}) - j/2 (phi_{i+1}, phi_{j-1}) + 1/4 (phi_{i+1}, phi_{j+1}).
       Using orthonormality (up to factor):
       (phi_a, phi_b) = sqrt(pi) 2^a a! delta_ab.
       So non-zero terms when a=b.
       1. i j sqrt(pi) 2^{i-1} (i-1)! delta_{i-1, j-1} = i j sqrt(pi) 2^{i-1} (i-1)! delta_{ij} (for i,j >= 1).
       2. -i/2 sqrt(pi) 2^{i-1} (i-1)! delta_{i-1, j+1} = -i/2 sqrt(pi) 2^{i-1} (i-1)! delta_{i, j+2} (so j = i-2).
       3. -j/2 sqrt(pi) 2^{i+1} (i+1)! delta_{i+1, j-1} = -j/2 sqrt(pi) 2^{i+1} (i+1)! delta_{i+2, j} (so j = i+2).
       4. 1/4 sqrt(pi) 2^{i+1} (i+1)! delta_{i+1, j+1} = 1/4 sqrt(pi) 2^{i+1} (i+1)! delta_{ij} (for i,j >= 0).
       
       Let's simplify the diagonal term (i=j):
       (P2)_ii = i^2 sqrt(pi) 2^{i-1} (i-1)! + 1/4 sqrt(pi) 2^{i+1} (i+1)!
       = sqrt(pi) 2^{i-1} (i-1)! [i^2 + 1/4 * 2^2 * (i+1)i]
       = sqrt(pi) 2^{i-1} (i-1)! [i^2 + i(i+1)]
       = sqrt(pi) 2^{i-1} (i-1)! [2i^2 + i]
       = sqrt(pi) 2^{i-1} (i-1)! i (2i+1).
       
       Let's check for i=0.
       d/dx phi_0 = -1/2 phi_1.
       (P2)_00 = 1/4 <phi_1, phi_1> = 1/4 sqrt(pi) 2^1 1! = 1/2 sqrt(pi).
       Formula: i=0. 0 * (1) * sqrt(pi) ... = 0. Mismatch.
       The term i^2 (i-1)! is 0 for i=0.
       The second term: 1/4 sqrt(pi) 2^{1} 1! = 1/2 sqrt(pi).
       So the formula needs to handle i=0 separately or be written more carefully.
       For i=0: (P2)_00 = 1/4 <phi_1, phi_1> = 1/2 sqrt(pi).
       For i>0: (P2)_ii = sqrt(pi) 2^{i-1} (i-1)! (2i^2 + i).
       
       Off-diagonal terms:
       (P2)_i,i-2: from term 2. j = i-2.
       (P2)_i,i-2 = -i/2 sqrt(pi) 2^{i-1} (i-1)!.
       (P2)_i,i+2: from term 3. j = i+2.
       (P2)_i,i+2 = -i/2 sqrt(pi) 2^{i+1} (i+1)!.
       Note: (P2)_i,i+2 should be equal to (P2)_{i+2,i} by symmetry.
       (P2)_{i+2,i} = -(i+2)/2 sqrt(pi) 2^{i+1} (i+1)!.
       These are not equal unless i=0?
       Let's re-evaluate term 3.
       Term 3: -j/2 <phi_{i+1}, phi_{j-1}>.
       If j = i+2, then j-1 = i+1. So <phi_{i+1}, phi_{i+1}>.
       Value: -(i+2)/2 sqrt(pi) 2^{i+1} (i+1)!.
       
       Term 2 for (P2)_{i+2, i}:
       - (i+2)/2 <phi_{i+1}, phi_{i+1}>.
       Wait, in term 2, the indices are <phi_{i'-1}, phi_{j'+1}>.
       Here i' = i+2, j' = i.
       So <phi_{i+1}, phi_{i+1}>.
       Value: -(i+2)/2 sqrt(pi) 2^{i+1} (i+1)!.
       So they are equal. Good.
       
       So the matrix P2 is tridiagonal (in steps of 2).
       
       Now, let's implement this.
    """
    start_time = time.time()
    
    # Hermite polynomial values at 0 for even n
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1008
    # H_12(0) = 103968
    H0_0 = 1
    H4_0 = 12
    H8_0 = 1008
    H12_0 = 103968
    
    # Basis indices
    idx = [0, 4, 8, 12]
    n_basis = 4
    
    # Precompute factorials and powers of 2
    # We need 2^i i! for i in idx
    # And 2^{i-1} (i-1)! etc.
    
    # Matrix N (Norm)
    N = np.zeros((n_basis, n_basis))
    for i, n in enumerate(idx):
        N[i, i] = np.sqrt(np.pi) * (2 ** n) * factorial(n, exact=False)
    
    # Matrix X2
    X2 = np.zeros((n_basis, n_basis))
    for i, n in enumerate(idx):
        # Diagonal: n/2
        X2[i, i] += (n / 2.0) * N[i, i]
        # n+2 term
        if n + 2 in idx:
            j = idx.index(n + 2)
            X2[i, j] += 0.25 * N[i, i] # <phi_i, phi_{n+2}> is 0 if i != n+2, but here we are computing <phi_i, x^2 phi_n>
            # Wait, x^2 phi_n = 1/4 phi_{n+2} + n/2 phi_n + n(n-1)/4 phi_{n-2}
            # <phi_i, x^2 phi_n> = 1/4 <phi_i, phi_{n+2}> + n/2 <phi_i, phi_n> + n(n-1)/4 <phi_i, phi_{n-2}>
            # <phi_i, phi_k> is non-zero only if i=k.
            # So:
            # If i = n+2: 1/4 <phi_{n+2}, phi_{n+2}> = 1/4 N[n+2, n+2]
            # If i = n: n/2 N[n, n]
            # If i = n-2: n(n-1)/4 N[n-2, n-2]
            
            # Let's redo this loop correctly.
            pass
            
    # Let's fill X2 properly
    X2 = np.zeros((n_basis, n_basis))
    for i, n_i in enumerate(idx):
        for j, n_j in enumerate(idx):
            val = 0.0
            # Term 1: 1/4 delta_{n_i, n_j+2}
            if n_i == n_j + 2:
                val += 0.25 * N[j, j] # <phi_{n_j+2}, phi_{n_j+2}> = N[j,j]
            # Term 2: n_j/2 delta_{n_i, n_j}
            if n_i == n_j:
                val += (n_j / 2.0) * N[j, j]
            # Term 3: n_j(n_j-1)/4 delta_{n_i, n_j-2}
            if n_i == n_j - 2:
                val += (n_j * (n_j - 1) / 4.0) * N[j, j]
            X2[i, j] = val
            
    # Matrix P2
    P2 = np.zeros((n_basis, n_basis))
    for i, n_i in enumerate(idx):
        for j, n_j in enumerate(idx):
            val = 0.0
            # d/dx phi_i = n_i phi_{i-1} - 1/2 phi_{i+1}
            # d/dx phi_j = n_j phi_{j-1} - 1/2 phi_{j+1}
            # Product:
            # 1. n_i n_j <phi_{i-1}, phi_{j-1}>
            if n_i > 0 and n_j > 0 and n_i - 1 == n_j - 1: # i.e. n_i == n_j
                val += n_i * n_j * N[i, i] / (2 * n_i) # N[i,i] is for n_i, N for n_i-1 is N[i,i]/(2*n_i)
                # Actually <phi_{n-1}, phi_{n-1}> = sqrt(pi) 2^{n-1} (n-1)! = N[n,n] / (2n)
                val -= n_i * n_j * N[i, i] # Undo, let's calculate directly
                val = 0.0
                if n_i == n_j and n_i > 0:
                    norm_nm1 = np.sqrt(np.pi) * (2 ** (n_i - 1)) * factorial(n_i - 1, exact=False)
                    val += n_i * n_j * norm_nm1
            # 2. -n_i/2 <phi_{i-1}, phi_{j+1}>
            if n_i > 0 and n_i - 1 == n_j + 1: # i.e. n_i == n_j + 2
                norm_nj1 = np.sqrt(np.pi) * (2 ** (n_j + 1)) * factorial(n_j + 1, exact=False)
                val += -n_i / 2.0 * norm_nj1
            # 3. -n_j/2 <phi_{i+1}, phi_{j-1}>
            if n_j > 0 and n_i + 1 == n_j - 1: # i.e. n_j == n_i + 2
                norm_ni1 = np.sqrt(np.pi) * (2 ** (n_i + 1)) * factorial(n_i + 1, exact=False)
                val += -n_j / 2.0 * norm_ni1
            # 4. 1/4 <phi_{i+1}, phi_{j+1}>
            if n_i + 1 == n_j + 1: # i.e. n_i == n_j
                norm_ni1 = np.sqrt(np.pi) * (2 ** (n_i + 1)) * factorial(n_i + 1, exact=False)
                val += 0.25 * norm_ni1
            P2[i, j] = val
            
    # Objective function J(c0, c4, c8)
    def objective(u):
        c0, c4, c8 = u
        c12 = -(c0 + 12*c4 + 1008*c8) / 103968.0
        v = np.array([c0, c4, c8, c12])
        
        norm_sq = v.T @ N @ v
        if norm_sq <= 0:
            return 1e10
        x2 = v.T @ X2 @ v
        p2 = v.T @ P2 @ v
        
        # J = (x2 / norm_sq) * (p2 / norm_sq)
        return (x2 * p2) / (norm_sq * norm_sq)
        
    # Initial guess
    u0 = np.array([-113/100, 1/25, 1/3240])
    
    # Optimize
    from scipy.optimize import minimize
    
    try:
        result = minimize(objective, u0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        u_opt = result.x
    except Exception as e:
        # Fallback to seed if optimization fails
        u_opt = u0
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 180:
        return u0
        
    return np.array(u_opt)

# EVOLVE-BLOCK-END

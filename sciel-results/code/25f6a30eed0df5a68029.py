"""Optimized program: eigenvalue-based direct solution for uncertainty bound minimization."""
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    Returns the product sqrt(<x^2> * <xi^2>).
    """
    # Hermite values at 0: H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=1330
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    # a0 = c0 + 12c4 + 168c8 + 1330c12 = 0
    a0 = 0.0
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    # q[k] is coeff of x^{2k} in P(x)^2
    q = np.convolve(p, p)
    # q has length 13, indices 0 to 12 corresponding to x^0 to x^24
    
    # Integrate x^{2k} e^{-x^2}
    # \int_{-inf}^{inf} x^{2k} e^{-x^2} dx = \sqrt{\pi} (2k-1)!! = \sqrt{\pi} 2^k k!
    factors = np.zeros(13)
    for k in range(13):
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    # I[k] = \int x^{2k} P(x)^2 e^{-x^2} dx
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
    
    # Variance <x^2> = I[1] / I[0]
    x2 = I[1] / I[0]
    
    # Calculate g''(x) to get <xi^2>
    # g(x) = P(x) e^{-x^2/2}
    # g''(x) = (P'' - 2xP' + (x^2 - 1)P) e^{-x^2/2}
    # Let R(x) = P'' - 2xP' + (x^2 - 1)P
    # R(x) is even. We need \int R(x)^2 e^{-x^2} dx.
    
    # Construct R(x) coefficients r_j for x^{2j}
    # P(x) = \sum_{i=0}^6 p_i x^{2i}
    # P'(x) = \sum_{i=1}^6 2i p_i x^{2i-1}
    # P''(x) = \sum_{i=1}^6 2i(2i-1) p_i x^{2i-2}
    # -2xP'(x) = \sum_{i=1}^6 -4i p_i x^{2i}
    # (x^2-1)P(x) = \sum_{i=0}^6 p_i x^{2i+2} - \sum_{i=0}^6 p_i x^{2i}
    
    r = np.zeros(8) # indices 0..7
    for j in range(8):
        val = 0.0
        # From P'': term is 2(j+1)(2j+1) p_{j+1} (for j=0..5)
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        # From -2xP': term is -4j p_j (for j=1..6)
        if 1 <= j <= 6:
            val += -4*j*p[j]
        # From x^2 P: term is p_{j-1} (for j=1..7)
        if 1 <= j <= 7:
            val += p[j-1]
        # From -P: term is -p_j (for j=0..6)
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    # Compute \int R(x)^2 e^{-x^2} dx
    s = np.convolve(r, r)
    # s has length 15, indices 0 to 14
    
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    # <xi^2> = (1/4\pi^2) * \int |g''|^2 dx / \int |g|^2 dx
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    # Product
    product = math.sqrt(x2 * xi2)
    
    return product

def _build_eigenvalue_problem():
    """
    Build the matrices for the generalized eigenvalue problem.
    
    We want to minimize sqrt(<x^2> * <xi^2>) subject to the constraint that the 
    coefficients (c0, c4, c8) define a valid function.
    
    Since both <x^2> and <xi^2> are ratios of quadratic forms in the coefficients,
    we can write:
    <x^2> = (c^T A c) / (c^T B c)
    <xi^2> = (c^T C c) / (c^T D c)
    
    where c = [c0, c4, c8, c12]^T and c12 is determined by the constraint g(0)=0.
    
    Actually, since c12 is determined by c0, c4, c8, we can express everything in terms
    of just (c0, c4, c8). Let's define the basis functions and compute the integrals.
    """
    # Basis functions: phi_i(x) = H_{2i}(x) e^{-x^2/2} for i=0,1,2,3
    # But we have the constraint that the coefficient of H12 is determined.
    
    # Let's work with the polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # where c12 = -(c0 + 12*c4 + 168*c8)/1330
    
    # We can write P(x) = c0 * P0(x) + c4 * P4(x) + c8 * P8(x)
    # where:
    # P0(x) = H0(x) - H12(x)/1330
    # P4(x) = H4(x) - 12*H12(x)/1330  
    # P8(x) = H8(x) - 168*H12(x)/1330
    
    # The function is g(x) = P(x) e^{-x^2/2}
    # g(x) = c0 * phi0(x) + c4 * phi4(x) + c8 * phi8(x)
    # where phi_i(x) = P_i(x) e^{-x^2/2}
    
    # We need to compute:
    # N = <g, g> = c^T G c where G_ij = <phi_i, phi_j>
    # X = <x^2 g, g> = c^T Xmat c where Xmat_ij = <x^2 phi_i, phi_j>
    # M = <g'', g''> = c^T Mmat c where Mmat_ij = <phi_i'', phi_j''>
    
    # Then <x^2> = X/N and <xi^2> = M/(4*pi^2*N)
    # Product = sqrt(X*M)/(4*pi*N)
    
    # To minimize this, we can use the fact that it's a ratio of quadratic forms.
    # The minimum occurs at an eigenvector of a certain matrix.
    
    # Let's compute the matrices numerically using Gauss-Hermite quadrature.
    # Gauss-Hermite quadrature: \int f(x) e^{-x^2} dx ≈ sum_i w_i f(x_i)
    # But our weight is e^{-x^2}, and we have g(x) = P(x) e^{-x^2/2}
    # So <g, g> = \int P(x)^2 e^{-x^2} dx
    
    # Use 50-point Gauss-Hermite quadrature
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(50)
    
    # Define the basis functions
    def _hermite(n, x):
        """Compute Hermite polynomial H_n(x)"""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2*x
        elif n == 2:
            return 4*x**2 - 2
        elif n == 4:
            return 16*x**4 - 48*x**2 + 12
        elif n == 8:
            return 256*x**8 - 4480*x**6 + 26880*x**4 - 53760*x**2 + 1680
        elif n == 12:
            return 4096*x**12 - 118272*x**10 + 1330560*x**8 - 7454720*x**6 + 19051520*x**4 - 20901120*x**2 + 133056
        else:
            raise ValueError(f"Hermite polynomial of order {n} not implemented")
    
    # Compute H0, H4, H8, H12 at quadrature nodes
    H0 = _hermite(0, x_nodes)
    H4 = _hermite(4, x_nodes)
    H8 = _hermite(8, x_nodes)
    H12 = _hermite(12, x_nodes)
    
    # Define the basis functions with the constraint g(0)=0
    # phi0(x) = (H0(x) - H12(x)/1330) * e^{-x^2/2}
    # phi4(x) = (H4(x) - 12*H12(x)/1330) * e^{-x^2/2}
    # phi8(x) = (H8(x) - 168*H12(x)/1330) * e^{-x^2/2}
    
    phi0 = (H0 - H12/1330.0)
    phi4 = (H4 - 12*H12/1330.0)
    phi8 = (H8 - 168*H12/1330.0)
    
    # Note: we don't multiply by e^{-x^2/2} because the weight e^{-x^2} in the
    # quadrature already accounts for it. The integral is \int P(x)^2 e^{-x^2} dx
    # = sum_i w_i * P(x_i)^2
    
    # Compute the Gram matrix G where G_ij = <phi_i, phi_j> = sum_k w_k * phi_i(x_k) * phi_j(x_k)
    # This is the norm squared matrix
    G = np.zeros((3, 3))
    G[0, 0] = np.sum(w_nodes * phi0 * phi0)
    G[0, 1] = np.sum(w_nodes * phi0 * phi4)
    G[0, 2] = np.sum(w_nodes * phi0 * phi8)
    G[1, 0] = G[0, 1]
    G[1, 1] = np.sum(w_nodes * phi4 * phi4)
    G[1, 2] = np.sum(w_nodes * phi4 * phi8)
    G[2, 0] = G[0, 2]
    G[2, 1] = G[1, 2]
    G[2, 2] = np.sum(w_nodes * phi8 * phi8)
    
    # Compute Xmat where Xmat_ij = <x^2 phi_i, phi_j>
    Xmat = np.zeros((3, 3))
    x2 = x_nodes**2
    Xmat[0, 0] = np.sum(w_nodes * x2 * phi0 * phi0)
    Xmat[0, 1] = np.sum(w_nodes * x2 * phi0 * phi4)
    Xmat[0, 2] = np.sum(w_nodes * x2 * phi0 * phi8)
    Xmat[1, 0] = Xmat[0, 1]
    Xmat[1, 1] = np.sum(w_nodes * x2 * phi4 * phi4)
    Xmat[1, 2] = np.sum(w_nodes * x2 * phi4 * phi8)
    Xmat[2, 0] = Xmat[0, 2]
    Xmat[2, 1] = Xmat[1, 2]
    Xmat[2, 2] = np.sum(w_nodes * x2 * phi8 * phi8)
    
    # Compute Mmat where Mmat_ij = <phi_i'', phi_j''>
    # phi_i(x) = P_i(x) where P_i are polynomials
    # We need P_i''(x)
    # For P0(x) = H0 - H12/1330: P0'' = H0'' - H12''/1330
    # H0'' = 0
    # H12'' = derivative of H12 twice
    
    # Let's compute the second derivatives analytically
    # H12(x) = 4096*x^12 - 118272*x^10 + 1330560*x^8 - 7454720*x^6 + 19051520*x^4 - 20901120*x^2 + 133056
    # H12'(x) = 49152*x^11 - 1182720*x^9 + 10644480*x^7 - 44728320*x^5 + 76206080*x^3 - 41802240*x
    # H12''(x) = 540672*x^10 - 10644480*x^8 + 74511360*x^6 - 223641600*x^4 + 228618240*x^2 - 41802240
    
    # H4(x) = 16*x^4 - 48*x^2 + 12
    # H4'(x) = 64*x^3 - 96*x
    # H4''(x) = 192*x^2 - 96
    
    # H8(x) = 256*x^8 - 4480*x^6 + 26880*x^4 - 53760*x^2 + 1680
    # H8'(x) = 2048*x^7 - 26880*x^5 + 107520*x^3 - 107520*x
    # H8''(x) = 14336*x^6 - 134400*x^4 + 322560*x^2 - 107520
    
    # P0''(x) = -H12''(x)/1330
    P0_dd = -(540672*x_nodes**10 - 10644480*x_nodes**8 + 74511360*x_nodes**6 - 223641600*x_nodes**4 + 228618240*x_nodes**2 - 41802240) / 1330.0
    
    # P4''(x) = H4''(x) - 12*H12''(x)/1330
    H4_dd = 192*x_nodes**2 - 96
    P4_dd = H4_dd - 12*(540672*x_nodes**10 - 10644480*x_nodes**8 + 74511360*x_nodes**6 - 223641600*x_nodes**4 + 228618240*x_nodes**2 - 41802240) / 1330.0
    
    # P8''(x) = H8''(x) - 168*H12''(x)/1330
    H8_dd = 14336*x_nodes**6 - 134400*x_nodes**4 + 322560*x_nodes**2 - 107520
    P8_dd = H8_dd - 168*(540672*x_nodes**10 - 10644480*x_nodes**8 + 74511360*x_nodes**6 - 223641600*x_nodes**4 + 228618240*x_nodes**2 - 41802240) / 1330.0
    
    # Now compute Mmat_ij = sum_k w_k * P_i_dd(x_k) * P_j_dd(x_k)
    Mmat = np.zeros((3, 3))
    Mmat[0, 0] = np.sum(w_nodes * P0_dd * P0_dd)
    Mmat[0, 1] = np.sum(w_nodes * P0_dd * P4_dd)
    Mmat[0, 2] = np.sum(w_nodes * P0_dd * P8_dd)
    Mmat[1, 0] = Mmat[0, 1]
    Mmat[1, 1] = np.sum(w_nodes * P4_dd * P4_dd)
    Mmat[1, 2] = np.sum(w_nodes * P4_dd * P8_dd)
    Mmat[2, 0] = Mmat[0, 2]
    Mmat[2, 1] = Mmat[1, 2]
    Mmat[2, 2] = np.sum(w_nodes * P8_dd * P8_dd)
    
    return G, Xmat, Mmat

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit per task contract
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    def _objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    # 1. Build and solve the eigenvalue problem
    # We want to minimize sqrt(<x^2> * <xi^2>) = sqrt(X * M) / (4*pi*N)
    # where X = c^T Xmat c, M = c^T Mmat c, N = c^T G c
    # 
    # This is equivalent to minimizing (X * M) / (N^2) since the constant factor doesn't matter
    # for finding the optimal c.
    #
    # We can use the fact that at the minimum, the gradient of (X*M)/(N^2) is zero,
    # which leads to a generalized eigenvalue problem.
    #
    # A simpler approach: use the method of Lagrange multipliers or directly solve
    # the nonlinear eigenvalue problem.
    #
    # Alternatively, we can parameterize c on the unit sphere (c^T G c = 1) and
    # minimize X * M subject to this constraint.
    
    try:
        G, Xmat, Mmat = _build_eigenvalue_problem()
        
        # Normalize: we want to minimize (c^T Xmat c) * (c^T Mmat c) subject to c^T G c = 1
        # This is a constrained optimization problem.
        
        # One approach: use the fact that the minimum of a product of two quadratic
        # forms subject to a constraint can be found by solving a related eigenvalue problem.
        
        # Let's try a different approach: since we have only 3 variables, we can use
        # a change of variables to remove the constraint.
        
        # First, diagonalize G: G = U D U^T where D is diagonal
        eigvals_G, eigvecs_G = np.linalg.eigh(G)
        
        # Check that G is positive definite
        if np.any(eigvals_G <= 1e-10):
            raise ValueError("G is not positive definite")
        
        # Transform to coordinates where G is identity
        # c = U * D^{-1/2} * y where y^T y = 1
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_G))
        
        # Transform the matrices
        Xmat_t = eigvecs_G.T @ Xmat @ eigvecs_G
        Mmat_t = eigvecs_G.T @ Mmat @ eigvecs_G
        
        # Now we want to minimize (y^T Xmat_t y) * (y^T Mmat_t y) subject to y^T y = 1
        # where y = D^{1/2} U^T c
        
        # This is still a non-convex problem, but we can use iterative methods.
        # Let's try a simple approach: use the fact that at the minimum, 
        # the gradient of the product is proportional to the constraint gradient.
        
        # Actually, let's just use scipy.optimize with the constraint.
        from scipy.optimize import minimize
        
        def _constrained_objective(y):
            X_val = y @ Xmat_t @ y
            M_val = y @ Mmat_t @ y
            return X_val * M_val
        
        # Initial guess: transform the seed coefficients
        c_seed = seed_coeffs
        y_seed = D_inv_sqrt @ (eigvecs_G.T @ c_seed)
        y_seed = y_seed / np.linalg.norm(y_seed)
        
        # Constraint: y^T y = 1
        constraints = [{'type': 'eq', 'fun': lambda y: y @ y - 1.0}]
        
        res_eig = minimize(
            _constrained_objective,
            y_seed,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-15}
        )
        
        if res_eig.success:
            y_opt = res_eig.x
            # Transform back to c
            c_opt = (eigvecs_G @ D_inv_sqrt @ y_opt)
            val_opt = _compute_bound(c_opt[0], c_opt[1], c_opt[2])
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
                
    except Exception as e:
        # If eigenvalue approach fails, fall back to iterative optimization
        pass
    
    # 2. Local Optimization from Seed (as backup)
    if time.time() < deadline - 30.0:
        try:
            res_local = scipy.optimize.minimize(
                _objective,
                seed_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_local.success and res_local.fun < best_val:
                best_val = res_local.fun
                best_coeffs = res_local.x.copy()
        except Exception:
            pass
    
    # 3. Multi-start Local Optimization
    if time.time() < deadline - 10.0:
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            # Start near the current best with some noise
            noise = np.random.normal(0, 0.5, 3)
            start_point = best_coeffs + noise
            try:
                res_restart = scipy.optimize.minimize(
                    _objective,
                    start_point,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res_restart.success and res_restart.fun < best_val:
                    best_val = res_restart.fun
                    best_coeffs = res_restart.x.copy()
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        try:
            res_final = scipy.optimize.minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return best_coeffs

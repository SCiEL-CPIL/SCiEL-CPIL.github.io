import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # Factorials needed: 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    fact = {0: 1.0, 4: 24.0, 8: 40320.0, 12: 479001600.0}
    
    def compute_exact_integrals(c0, c4, c8):
        """Compute I0, Ix2, Ideriv exactly using Hermite orthogonality."""
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        c = {0: c0, 4: c4, 8: c8, 12: c12}
        degrees = [0, 4, 8, 12]
        
        # I0 = sum_k c_k^2 * k!
        I0 = 0.0
        for k in degrees:
            I0 += c[k]**2 * fact[k]
        
        # Ix2 = sum over k,l of c_k c_l * <H_k x^2 H_l>
        # x^2 H_k = 0.5*H_{k+2} + k*H_k + 0.5*k*(k-1)*H_{k-2}
        Ix2 = 0.0
        for k in degrees:
            ck = c[k]
            if abs(ck) < 1e-30:
                continue
            # Contribution from 0.5*H_{k+2}
            if (k+2) in fact:
                Ix2 += ck * c[k+2] * 0.5 * fact[k+2]
            # Contribution from k*H_k
            Ix2 += ck * c[k] * k * fact[k]
            # Contribution from 0.5*k*(k-1)*H_{k-2}
            if k >= 2 and (k-2) in fact:
                Ix2 += ck * c[k-2] * 0.5 * k * (k-1) * fact[k-2]
        
        # Ideriv = sum over k,l of c_k c_l * <H_k' H_l'>
        # H_k' = 2k H_{k-1}, so H_k' H_l' = 4kl H_{k-1} H_{l-1}
        # <H_{k-1} H_{l-1}> = (k-1)! delta_{k-1,l-1} = (k-1)! delta_{k,l}
        Ideriv = 0.0
        for k in degrees:
            if k == 0:
                continue  # H_0' = 0
            ck = c[k]
            Ideriv += ck**2 * 4.0 * k * k * fact[k-1]
        
        return I0, Ix2, Ideriv
    
    def objective(coeffs):
        """Compute the uncertainty product A(f)*A(Fhat(f))."""
        c0, c4, c8 = coeffs
        try:
            I0, Ix2, Ideriv = compute_exact_integrals(c0, c4, c8)
            if I0 <= 0 or not np.isfinite(I0):
                return 1e10
            if Ix2 < 0 or not np.isfinite(Ix2):
                return 1e10
            if Ideriv < 0 or not np.isfinite(Ideriv):
                return 1e10
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            if A_f_sq <= 0 or A_gf_sq <= 0:
                return 1e10
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return 1e10
    
    def solve_coordinate(coeffs, axis):
        """
        Analytically solve for optimal value of coeffs[axis] given the other two.
        The objective squared is proportional to (Ix2 * Ideriv) / I0^2.
        Each of I0, Ix2, Ideriv is quadratic in the variable:
          I = a + b*t + c*t^2
        We minimize f(t) = (B(t)*D(t)) / (A(t))^2 where B=Ix2, D=Ideriv, A=I0.
        Taking d/dt ln(f) = 0 gives:
          (B'D + BD')/A^2 - 2 B D A' / A^3 = 0
          => (B'D + BD') * A - 2 B D A' = 0
        This is a polynomial in t of degree up to 4 (since B,D,A are quadratic,
        their products and derivatives give degree up to 4).
        """
        c0, c4, c8 = list(coeffs)
        others = [c0, c4, c8]
        
        # Build quadratic coefficients for I0, Ix2, Ideriv as functions of t = coeffs[axis]
        # We need to express each integral as a + b*t + c*t^2
        # where t is the variable we're optimizing
        
        # Strategy: compute integrals symbolically by substituting t for the axis variable
        # and collecting powers of t.
        
        def get_quad_coeffs(axis_idx):
            """Return (a, b, c) for each of I0, Ix2, Ideriv as a + b*t + c*t^2."""
            # Use finite differences / direct computation:
            # Evaluate at t = -1, 0, 1 to recover quadratic coefficients
            vals = []
            for t in [-1.0, 0.0, 1.0]:
                cc = list(others)
                cc[axis_idx] = t
                I0, Ix2, Ideriv = compute_exact_integrals(cc[0], cc[1], cc[2])
                vals.append((I0, Ix2, Ideriv))
            
            # For each integral, fit quadratic: f(t) = a + b*t + c*t^2
            # Using values at -1, 0, 1:
            # f(0) = a
            # f(1) = a + b + c
            # f(-1) = a - b + c
            # => b = (f(1) - f(-1))/2, c = (f(1) + f(-1) - 2*f(0))/2
            
            quad = []
            for i in range(3):  # I0, Ix2, Ideriv
                f_minus1 = vals[0][i]
                f_0 = vals[1][i]
                f_1 = vals[2][i]
                a = f_0
                b = (f_1 - f_minus1) / 2.0
                c = (f_1 + f_minus1 - 2.0 * f_0) / 2.0
                quad.append((a, b, c))
            
            return quad
        
        A = get_quad_coeffs(axis)  # (a_I0, b_I0, c_I0), (a_Ix2, b_Ix2, c_Ix2), (a_Id, b_Id, c_Id)
        aA, bA, cA = A[0]
        aB, bB, cB = A[1]
        aD, bD, cD = A[2]
        
        # f(t) = (B(t)*D(t)) / (A(t))^2
        # B(t)*D(t) = (aB + bB*t + cB*t^2)(aD + bD*t + cD*t^2)
        #   = aB*aD + (aB*bD + bB*aD)*t + (aB*cD + bB*bD + cB*aD)*t^2 + (bB*cD + cB*bD)*t^3 + cB*cD*t^4
        # A(t)^2 = (aA + bA*t + cA*t^2)^2
        #   = aA^2 + 2*aA*bA*t + (bA^2 + 2*aA*cA)*t^2 + 2*bA*cA*t^3 + cA^2*t^4
        
        # N(t) = B(t)*D(t), denominator = A(t)^2
        # Minimize N(t)/denom(t). Critical point: N'(t)*denom(t) - N(t)*denom'(t) = 0
        
        # N(t) coefficients (degree 4):
        n0 = aB * aD
        n1 = aB * bD + bB * aD
        n2 = aB * cD + bB * bD + cB * aD
        n3 = bB * cD + cB * bD
        n4 = cB * cD
        
        # N'(t) = n1 + 2*n2*t + 3*n3*t^2 + 4*n4*t^3
        
        # denom(t) = A(t)^2 coefficients (degree 4):
        d0 = aA * aA
        d1 = 2.0 * aA * bA
        d2 = bA * bA + 2.0 * aA * cA
        d3 = 2.0 * bA * cA
        d4 = cA * cA
        
        # denom'(t) = d1 + 2*d2*t + 3*d3*t^2 + 4*d4*t^3
        
        # g(t) = N'(t)*denom(t) - N(t)*denom'(t) = 0
        # This is a polynomial of degree up to 7 (3+4=7 from N'*denom, 4+3=7 from N*denom')
        
        # Let's build the polynomial coefficients for g(t) = sum_{k=0}^{7} gk * t^k
        
        # N'(t)*denom(t):
        # N' coefficients: [n1, 2*n2, 3*n3, 4*n4] (degrees 0,1,2,3)
        # denom coefficients: [d0, d1, d2, d3, d4] (degrees 0,1,2,3,4)
        
        np_coeff = [n1, 2.0*n2, 3.0*n3, 4.0*n4]  # N' at degrees 0..3
        nd_coeff = [d0, d1, d2, d3, d4]  # denom at degrees 0..4
        
        # Product N'*denom: degree up to 7
        prod1 = [0.0] * 8
        for i in range(4):
            for j in range(5):
                prod1[i+j] += np_coeff[i] * nd_coeff[j]
        
        # N(t)*denom'(t):
        # N coefficients: [n0, n1, n2, n3, n4] (degrees 0..4)
        # denom' coefficients: [d1, 2*d2, 3*d3, 4*d4] (degrees 0..3)
        
        n_coeff = [n0, n1, n2, n3, n4]
        ndp_coeff = [d1, 2.0*d2, 3.0*d3, 4.0*d4]
        
        prod2 = [0.0] * 8
        for i in range(5):
            for j in range(4):
                prod2[i+j] += n_coeff[i] * ndp_coeff[j]
        
        # g(t) = prod1 - prod2
        g_coeff = [prod1[k] - prod2[k] for k in range(8)]
        
        # Find real roots of g(t) = 0
        # Use numpy roots
        try:
            roots = np.roots(g_coeff[::-1])  # np.roots expects highest degree first
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and np.isfinite(r.real)]
        except Exception:
            real_roots = []
        
        # Evaluate objective at each real root and at the current value
        best_t = coeffs[axis]
        best_obj = objective(coeffs)
        
        for t in real_roots:
            if not np.isfinite(t):
                continue
            # Check that A(t) > 0 (I0 must be positive)
            new_coeffs = list(coeffs)
            new_coeffs[axis] = t
            I0_check, _, _ = compute_exact_integrals(new_coeffs[0], new_coeffs[1], new_coeffs[2])
            if I0_check <= 0:
                continue
            obj_val = objective(new_coeffs)
            if obj_val < best_obj:
                best_obj = obj_val
                best_t = t
        
        # Also check boundary: if no valid root, keep current
        return best_t, best_obj
    
    def coordinate_descent(x0, max_iter=200, tol=1e-14):
        """Cycle through coordinates, solving analytically each time."""
        x = np.array(x0, dtype=float)
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            max_change = 0.0
            for axis in range(3):
                new_val, obj = solve_coordinate(x, axis)
                change = abs(new_val - x[axis])
                max_change = max(max_change, change)
                x[axis] = new_val
            if max_change < tol:
                break
        return x, objective(x)
    
    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Coordinate descent from seed
    if time.time() < deadline - 30:
        try:
            x_cd, obj_cd = coordinate_descent(x0, max_iter=500)
            if np.isfinite(obj_cd) and obj_cd < best_score:
                best_score = obj_cd
                best_coeffs = x_cd.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B polish from coordinate descent result
    if time.time() < deadline - 25:
        try:
            res = opt.minimize(objective, best_coeffs, method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Random restarts with coordinate descent
    num_restarts = 60
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if i % 4 == 0:
            # Perturb around best
            scale = 0.05 * (1.0 + i/30.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        elif i % 4 == 1:
            # Perturb around seed
            scale = 0.1
            x_start = x0 + rng.normal(0, scale, 3)
        elif i % 4 == 2:
            # Wider exploration
            scale = 0.3
            x_start = x0 + rng.normal(0, scale, 3)
        else:
            # Very wide exploration
            scale = 0.5
            x_start = x0 + rng.normal(0, scale, 3)
        
        try:
            # Coordinate descent from this start
            x_cd, obj_cd = coordinate_descent(x_start, max_iter=300)
            if np.isfinite(obj_cd) and obj_cd < best_score:
                best_score = obj_cd
                best_coeffs = x_cd.copy()
            # Quick L-BFGS polish
            if time.time() < deadline - 10:
                res = opt.minimize(objective, best_coeffs, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 30})
                if np.isfinite(res.fun) and res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision coordinate descent + polish
    if time.time() < deadline - 5:
        try:
            x_cd, obj_cd = coordinate_descent(best_coeffs, max_iter=1000, tol=1e-16)
            if np.isfinite(obj_cd) and obj_cd < best_score:
                best_score = obj_cd
                best_coeffs = x_cd.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 3:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 2000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(objective, best_coeffs, method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)

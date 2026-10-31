import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin
    
    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280
    
    # Coefficients at x=0 for the constraint P(0)=0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # c0 + 12c4 + 1680c8 + 665280c12 = 0
    # c12 = -(c0 + 12c4 + 1680c8) / 665280
    
    # Precompute monomial coefficients for P(x) = sum_{j=0,2,..,12} p_j x^j
    # p_j depends on c0, c4, c8, c12
    # H0: p0 += 1
    # H4: p2 += -48, p4 += 16
    # H8: p2 += -48384, p4 += 26880, p6 += -4480, p8 += 256
    # H12: p2 += -1597440, p4 += 16636928, p6 += -7654656, p8 += 1548288, p10 += -129024, p12 += 4096
    
    # Let's define the matrix M such that p = M @ c
    # c = [c0, c4, c8, c12]
    # p indices: 0, 2, 4, 6, 8, 10, 12
    M = np.zeros((7, 4))
    # H0
    M[0, 0] = 1.0
    # H4
    M[1, 1] = -48.0
    M[2, 1] = 16.0
    # H8
    M[1, 2] = -48384.0
    M[2, 2] = 26880.0
    M[3, 2] = -4480.0
    M[4, 2] = 256.0
    # H12
    M[1, 3] = -1597440.0
    M[2, 3] = 16636928.0
    M[3, 3] = -7654656.0
    M[4, 3] = 1548288.0
    M[5, 3] = -129024.0
    M[6, 3] = 4096.0
    
    # Constraint: c12 = -(c0 + 12c4 + 1680c8) / 665280
    # Let's substitute c12 into p.
    # p = M[:3] @ c[:3] + M[:, 3] * (-(c0 + 12c4 + 1680c8) / 665280)
    # p = (M[:3] - M[:, 3] * [1, 12, 1680] / 665280) @ c[:3]
    
    coeff_constraint = np.array([1.0, 12.0, 1680.0])
    M_reduced = M[:, :3] - np.outer(M[:, 3], coeff_constraint) / 665280.0
    
    # Precompute integrals I_k = int_{-inf}^{inf} x^k e^{-x^2} dx
    # I_k = 0 for odd k.
    # I_{2n} = (2n-1)!! * sqrt(pi) / 2^n
    # I_0 = sqrt(pi)
    # I_2 = 0.5 * sqrt(pi)
    # I_4 = 0.75 * 0.5 * sqrt(pi) = 0.375 * sqrt(pi)
    # ...
    # Max degree of P(x)^2 is 24. We need x^k up to 2+24=26.
    max_power = 26
    I = np.zeros(max_power + 1)
    I[0] = np.sqrt(np.pi)
    for k in range(2, max_power + 1, 2):
        I[k] = (k - 1) / 2.0 * I[k - 2]
        
    # Precompute Q coefficients relation
    # Q_j = p_j * (-1)^(j/2)
    # p_j corresponds to x^j.
    # j=0: 1, j=2: -1, j=4: 1, j=6: -1, j=8: 1, j=10: -1, j=12: 1
    signs_q = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    M_q_reduced = M_reduced * signs_q[:, np.newaxis]
    
    def get_p(c):
        return M_reduced @ c
        
    def get_q(c):
        return M_q_reduced @ c

    def compute_moments(p):
        # m_k = sum_{i,j} p_i p_j I_{2i+2j+k} ? No.
        # P(x) = sum p_i x^i (where i is 0,2,..)
        # P(x)^2 = sum_{i,j} p_i p_j x^{i+j}
        # int x^k P(x)^2 e^{-x^2} = sum_{i,j} p_i p_j int x^{k+i+j} e^{-x^2}
        # Let indices i, j refer to the monomial power.
        # p array indices: 0->x^0, 1->x^2, 2->x^4, ...
        # Let p_vec be the coefficients.
        # m_k = sum_{a,b} p_vec[a] p_vec[b] I[k + 2*a + 2*b]
        
        m = np.zeros(3) # m0, m2 (m1 is 0 for even functions)
        for k in range(3):
            s = 0.0
            for a in range(7):
                pa = p[a]
                if pa == 0: continue
                for b in range(7):
                    pb = p[b]
                    if pb == 0: continue
                    idx = k + 2*a + 2*b
                    if idx <= max_power:
                        s += pa * pb * I[idx]
            m[k] = s
        return m

    def compute_q_moments(q):
        n = np.zeros(3)
        for k in range(3):
            s = 0.0
            for a in range(7):
                qa = q[a]
                if qa == 0: continue
                for b in range(7):
                    qb = q[b]
                    if qb == 0: continue
                    idx = k + 2*a + 2*b
                    if idx <= max_power:
                        s += qa * qb * I[idx]
            n[k] = s
        return n

    def objective_and_grad(c):
        p = get_p(c)
        q = get_q(c)
        
        m = compute_moments(p)
        n = compute_q_moments(q)
        
        m0, m2 = m[0], m[2]
        n0, n2 = n[0], n[2]
        
        if m0 <= 1e-15 or n0 <= 1e-15 or m2 <= 0 or n2 <= 0:
            return 1e18, np.zeros(3)
            
        # Variance x = m2/m0
        # Variance xi = n2/n0
        # Product = sqrt( (m2/m0) * (n2/n0) )
        
        var_x = m2 / m0
        var_xi = n2 / n0
        val = np.sqrt(var_x * var_xi)
        
        # Gradient via finite difference is safer for coding simplicity here, 
        # but analytic is faster. Given the polynomial nature, finite difference 
        # with small h is extremely accurate and robust.
        h = 1e-8
        grad = np.zeros(3)
        for i in range(3):
            c_plus = c.copy()
            c_plus[i] += h
            val_plus = objective(c_plus)
            c_minus = c.copy()
            c_minus[i] -= h
            val_minus = objective(c_minus)
            grad[i] = (val_plus - val_minus) / (2*h)
            
        return val, grad

    def objective(c):
        val, _ = objective_and_grad(c)
        return val

    # Initial guess from seed
    c_init = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_c = c_init.copy()
    best_val = objective(c_init)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Local refinement from seed
    try:
        res = minimize(objective, c_init, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
        
    # Phase 2: Global search with restarts
    # The landscape is likely non-convex. We use a mix of random restarts 
    # and differential evolution if time permits.
    
    # Simple random restarts in a reasonable box
    # c0, c4, c8
    # c0 is O(1), c4 is O(0.1), c8 is O(0.001)
    
    for i in range(50):
        if time.time() > deadline - 10:
            break
            
        # Random start
        if i == 0:
            c_rand = best_c + rng.normal(0, 0.1, 3)
        else:
            c_rand = np.array([
                rng.uniform(-3.0, 2.0),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.01, 0.01)
            ])
            
        try:
            res_r = minimize(objective, c_rand, method='L-BFGS-B', jac=True,
                             options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            if res_r.fun < best_val:
                best_val = res_r.fun
                best_c = res_r.x.copy()
        except Exception:
            pass
            
    # Phase 3: Fine-tuning around the best found
    if time.time() < deadline - 5:
        try:
            res_final = minimize(objective, best_c, method='L-BFGS-B', jac=True,
                                 options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_c = res_final.x.copy()
        except Exception:
            pass

    return best_c


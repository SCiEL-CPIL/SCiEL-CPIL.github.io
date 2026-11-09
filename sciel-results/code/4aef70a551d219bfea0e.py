import time
import math
import numpy as np

def construct():
    """Return 12 double-root locations z_1..z_12."""
    # The task requires finding z_i such that B/(2*pi) is minimized.
    # B is the largest real root of q(x) = g(x) / (x * prod (x-z_i)^2) that is a simple root (sign change).
    # g(x) is a linear combination of L_0, L_2, ..., L_50 (generalized Laguerre with alpha=-1/2).
    # g satisfies g(0)=0, g'(0)=1, and g(z_i)=g'(z_i)=0.
    # This means g(x) has double roots at z_i and a simple root at 0.
    # Since deg(g) <= 50, and we have 12 double roots + 1 simple root = 25 roots.
    # The dimension of the space of such g is 26 (coefficients c_0..c_25).
    # The constraints are 1 + 1 + 12*2 = 26.
    # So for any distinct positive z_i, there is a unique g (up to scaling, but fixed by g'(0)=1).
    
    # The key insight is that we want to push the "next" root of g (which becomes a root of q) as close to 0 as possible?
    # No, we want to MINIMIZE B. B is the largest counted root of q.
    # q(x) is a polynomial of degree 50 - (1 + 24) = 25.
    # Wait, deg(g) is at most 50.
    # Roots of g: 0, z_1..z_12 (double). Total multiplicity 1 + 24 = 25.
    # So g(x) = x * prod (x-z_i)^2 * q(x).
    # deg(q) = 50 - 25 = 25.
    # q(x) is a polynomial of degree 25.
    # The roots of q are the "extra" roots of g.
    # We want to minimize the largest real root of q (that is simple).
    # Ideally, we want all roots of q to be negative or complex, so B=0.
    # Or at least, the largest real simple root to be as small as possible.
    
    # Let's try to find z_i such that q(x) has no positive real roots.
    # This is a hard optimization problem.
    
    # Heuristic:
    # The Laguerre polynomials L_n^{-1/2}(x) have positive roots.
    # The combination g is determined by the interpolation conditions.
    
    # Let's try a grid search or local optimization on z_i.
    # Given the time limit, we should be very careful.
    
    # Initial guess:
    # Maybe z_i are related to the roots of some specific polynomial?
    # If we choose z_i to be the roots of L_{12}^{-1/2}(x), does that help?
    # Or maybe we should just return a fixed set that works reasonably well.
    
    # Let's try to compute the score for a few candidate sets and pick the best.
    # But computing the score requires solving a 26x26 linear system and finding roots of a degree 25 polynomial.
    # This is fast in numpy.
    
    def _score(z_list):
        try:
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            alpha = -0.5
            # Get Laguerre polynomials L_{2k}^{-1/2}
            # We need coefficients of L_n^{-1/2}(x)
            # We can use numpy.polynomial.laguerre or compute via recurrence.
            # Let's use the recurrence for coefficients.
            
            def laguerre_coeffs(n, alpha):
                if n == 0:
                    return np.array([1.0])
                if n == 1:
                    return np.array([1.0 + alpha, -1.0])
                
                c_nm2 = np.array([1.0]) # L_0
                c_nm1 = np.array([1.0 + alpha, -1.0]) # L_1
                
                for k in range(2, n+1):
                    # k L_k = (2k-1+alpha) L_{k-1} - (k-1) L_{k-2} - x L_{k-1}
                    # Wait, recurrence: x L_n' = n L_{n-1} - n L_n
                    # (n+1) L_{n+1} = (2n+1+alpha-x) L_n - (n+alpha) L_{n-1}
                    
                    # L_{k} = ((2k-1+alpha) L_{k-1} - x L_{k-1} - (k-1+alpha) L_{k-2}) / k
                    
                    # Coeffs of L_{k-1}: c_nm1
                    # Coeffs of L_{k-2}: c_nm2
                    
                    # Term 1: (2k-1+alpha) * c_nm1
                    t1 = c_nm1 * (2*k - 1 + alpha)
                    
                    # Term 2: -x * c_nm1
                    # Shift c_nm1 by 1
                    t2 = np.zeros(len(c_nm1) + 1)
                    t2[1:] = -c_nm1
                    
                    # Term 3: -(k-1+alpha) * c_nm2
                    t3 = c_nm2 * (-(k - 1 + alpha))
                    
                    # Align lengths
                    max_len = max(len(t1), len(t2), len(t3))
                    t1 = np.pad(t1, (0, max_len - len(t1)))
                    t2 = np.pad(t2, (0, max_len - len(t2)))
                    t3 = np.pad(t3, (0, max_len - len(t3)))
                    
                    c_n = (t1 + t2 + t3) / k
                    c_nm2 = c_nm1
                    c_nm1 = c_n
                return c_nm1
            
            L_coeffs = []
            for k in range(26):
                n = 2*k
                L_coeffs.append(laguerre_coeffs(n, alpha))
            
            # Build matrix A and vector b
            # g(x) = sum c_k L_{2k}(x)
            # Conditions:
            # 1. g(0) = 0
            # 2. g'(0) = 1
            # 3. g(z_i) = 0
            # 4. g'(z_i) = 0
            
            A = np.zeros((26, 26))
            b = np.zeros(26)
            
            # Row 0: g(0) = 0
            for k in range(26):
                A[0, k] = L_coeffs[k][0]
            b[0] = 0
            
            # Row 1: g'(0) = 1
            for k in range(26):
                # Derivative at 0 is coeff of x in L_{2k}
                if len(L_coeffs[k]) > 1:
                    A[1, k] = L_coeffs[k][1]
                else:
                    A[1, k] = 0
            b[1] = 1
            
            for i in range(12):
                zi = z_arr[i]
                # Row 2+2i: g(zi) = 0
                # Row 3+2i: g'(zi) = 0
                for k in range(26):
                    # Evaluate L_{2k}(zi)
                    # L_coeffs[k] is [c_0, c_1, ...]
                    # Use Horner or polyval
                    # polyval expects [c_n, ..., c_0]
                    A[2+2*i, k] = np.polyval(L_coeffs[k][::-1], zi)
                    
                    # Derivative
                    # d/dx L(x) = sum j c_j x^{j-1}
                    # Coeffs of derivative: [1*c_1, 2*c_2, ...]
                    d_coeffs = np.array([j * L_coeffs[k][j] for j in range(1, len(L_coeffs[k]))])
                    if len(d_coeffs) > 0:
                        A[3+2*i, k] = np.polyval(d_coeffs[::-1], zi)
                    else:
                        A[3+2*i, k] = 0
            
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return float('inf')
            
            # Construct g(x)
            # g_coeffs = sum c_k L_coeffs[k]
            max_deg = 50
            g_coeffs = np.zeros(max_deg + 1)
            for k in range(26):
                g_coeffs[:len(L_coeffs[k])] += c[k] * L_coeffs[k]
            
            # Construct D(x) = x * prod (x-z_i)^2
            roots_d = np.concatenate(([0.0], np.repeat(z_arr, 2)))
            d_coeffs = np.poly(roots_d)
            
            # q(x) = g(x) / D(x)
            g_rev = g_coeffs[::-1]
            q_coeffs, remainder = np.polydiv(g_rev, d_coeffs)
            
            if np.max(np.abs(remainder)) > 1e-8:
                # Should be exact, but floating point error
                # If remainder is large, something is wrong
                pass
            
            # Find roots of q
            q_roots = np.roots(q_coeffs)
            
            max_r = 0.0
            for r in q_roots:
                if abs(r.imag) < 1e-9:
                    r_real = r.real
                    # Check sign change
                    # q(r - eps) * q(r + eps) < 0
                    # Use small eps
                    eps = 1e-7
                    q_minus = np.polyval(q_coeffs, r_real - eps)
                    q_plus = np.polyval(q_coeffs, r_real + eps)
                    if q_minus * q_plus < 0:
                        if r_real > max_r:
                            max_r = r_real
            
            B = max(0, max_r)
            return B / (2 * math.pi)
        
        except Exception:
            return float('inf')

    # Candidate generation
    # We want to minimize the score.
    # Let's try a few heuristics.
    
    # 1. Bessel zeros squared
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # 2. Uniform spacing
    z_uniform = [i for i in range(1, 13)]
    
    # 3. Exponential spacing
    z_exp = [math.exp(i) for i in range(1, 13)]
    
    # 4. Roots of L_{12}^{-1/2}
    # We can approximate these or use known values.
    # Let's just use the Bessel ones as a strong baseline.
    
    best_z = z_bessel
    best_score = _score(best_z)
    
    # Local optimization
    # Try to improve by perturbing z_i
    # We have time budget.
    deadline = time.time() + 580.0
    
    current_z = best_z
    current_score = best_score
    
    np.random.seed(42)
    
    # Simple hill climbing
    step_size = 0.1
    for i in range(1000):
        if time.time() > deadline:
            break
        
        # Perturb a few indices
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(12, num_perturb, replace=False)
        
        candidate = list(current_z)
        for idx in indices:
            factor = np.random.uniform(1 - step_size, 1 + step_size)
            candidate[idx] *= factor
        
        if any(z <= 0 for z in candidate):
            continue
            
        c_sorted = sorted(candidate)
        distinct = True
        for j in range(11):
            if abs(c_sorted[j] - c_sorted[j+1]) < 1e-6:
                distinct = False
                break
        if not distinct:
            continue
            
        score = _score(candidate)
        
        if score < current_score:
            current_score = score
            current_z = candidate
            best_z = candidate
            best_score = score
        else:
            # Reduce step size slightly if stuck?
            pass
            
        # Decay step size
        step_size *= 0.98
        
    return best_z

import time
import math
import numpy as np

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    def _score(z_list):
        try:
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            alpha = -0.5
            
            def laguerre_coeffs(n, alpha):
                if n == 0:
                    return np.array([1.0])
                if n == 1:
                    return np.array([1.0 + alpha, -1.0])
                
                c_nm2 = np.array([1.0])
                c_nm1 = np.array([1.0 + alpha, -1.0])
                
                for k in range(2, n+1):
                    t1 = c_nm1 * (2*k - 1 + alpha)
                    t2 = np.zeros(len(c_nm1) + 1)
                    t2[1:] = -c_nm1
                    t3 = c_nm2 * (-(k - 1 + alpha))
                    
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
            
            A = np.zeros((26, 26))
            b = np.zeros(26)
            
            for k in range(26):
                A[0, k] = L_coeffs[k][0]
            b[0] = 0
            
            for k in range(26):
                if len(L_coeffs[k]) > 1:
                    A[1, k] = L_coeffs[k][1]
                else:
                    A[1, k] = 0
            b[1] = 1
            
            for i in range(12):
                zi = z_arr[i]
                for k in range(26):
                    A[2+2*i, k] = np.polyval(L_coeffs[k][::-1], zi)
                    d_coeffs = np.array([j * L_coeffs[k][j] for j in range(1, len(L_coeffs[k]))])
                    if len(d_coeffs) > 0:
                        A[3+2*i, k] = np.polyval(d_coeffs[::-1], zi)
                    else:
                        A[3+2*i, k] = 0
            
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return float('inf')
            
            max_deg = 50
            g_coeffs = np.zeros(max_deg + 1)
            for k in range(26):
                g_coeffs[:len(L_coeffs[k])] += c[k] * L_coeffs[k]
            
            roots_d = np.concatenate(([0.0], np.repeat(z_arr, 2)))
            d_coeffs = np.poly(roots_d)
            
            g_rev = g_coeffs[::-1]
            q_coeffs, remainder = np.polydiv(g_rev, d_coeffs)
            
            if np.max(np.abs(remainder)) > 1e-6:
                return float('inf')
            
            q_roots = np.roots(q_coeffs)
            
            max_r = 0.0
            for r in q_roots:
                if abs(r.imag) < 1e-8:
                    r_real = r.real
                    if r_real <= 0:
                        continue
                    eps = 1e-6
                    q_minus = np.polyval(q_coeffs, r_real - eps)
                    q_plus = np.polyval(q_coeffs, r_real + eps)
                    if q_minus * q_plus < 0:
                        if r_real > max_r:
                            max_r = r_real
            
            B = max(0, max_r)
            return B / (2 * math.pi)
        
        except Exception:
            return float('inf')

    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Bessel zeros squared
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    candidates.append(z_bessel)
    
    # 2. Uniform spacing
    candidates.append([float(i) for i in range(1, 13)])
    
    # 3. Exponential spacing
    candidates.append([math.exp(0.3*i) for i in range(1, 13)])
    
    # 4. Power law spacing
    candidates.append([i**1.5 for i in range(1, 13)])
    
    # 5. Roots of L_12^{-1/2}(x) - approximate
    # Use the fact that for large n, roots of L_n^{alpha} are approximately
    # related to Bessel zeros. For n=12, alpha=-1/2:
    # The roots are approximately at x_k ≈ (k + alpha/2 + 0.25)^2 * pi^2 / 4 for small k
    # But let's just use a reasonable approximation
    z_lag = [((k + 0.25) * math.pi / 2)**2 for k in range(1, 13)]
    candidates.append(z_lag)
    
    # 6. Log spacing
    candidates.append([math.log(i + 1) * 2 for i in range(1, 13)])
    
    # 7. Sine-based spacing
    candidates.append([4 * math.sin((i+1) * math.pi / 26)**2 for i in range(12)])
    
    # 8. Hyperbolic cosine spacing
    candidates.append([math.cosh(0.2*i) for i in range(1, 13)])
    
    # Evaluate all candidates
    best_z = None
    best_score = float('inf')
    
    for z_cand in candidates:
        score = _score(z_cand)
        if score < best_score:
            best_score = score
            best_z = list(z_cand)
    
    if best_z is None:
        return [float(i) for i in range(1, 13)]
    
    current_z = best_z
    current_score = best_score
    
    deadline = time.time() + 570.0  # Safety margin
    
    np.random.seed(42)
    
    step_size = 0.15
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Adaptive step size
        if no_improve_count > max_no_improve:
            # Restart with larger step to escape local minimum
            step_size = max(step_size, 0.3)
            no_improve_count = 0
        
        # Perturb random subset of indices
        num_perturb = np.random.randint(1, 6)  # Perturb 1-5 indices
        indices = np.random.choice(12, num_perturb, replace=False)
        
        candidate = list(current_z)
        valid = True
        for idx in indices:
            factor = np.random.uniform(1 - step_size, 1 + step_size)
            candidate[idx] *= factor
            if candidate[idx] <= 0:
                valid = False
                break
        
        if not valid:
            continue
        
        c_sorted = sorted(candidate)
        distinct = True
        for j in range(11):
            if abs(c_sorted[j] - c_sorted[j+1]) < 1e-8:
                distinct = False
                break
        if not distinct:
            continue
        
        score = _score(candidate)
        
        if score < current_score:
            current_score = score
            current_z = candidate
            no_improve_count = 0
            step_size *= 0.95  # Shrink step on success
        else:
            no_improve_count += 1
            step_size *= 0.98  # Gradual decay
        
        # Prevent step size from getting too small
        step_size = max(step_size, 0.01)
    
    return current_z

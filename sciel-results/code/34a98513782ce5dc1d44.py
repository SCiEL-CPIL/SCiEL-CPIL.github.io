import time
import math
import numpy as np
import sympy as sp

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    deadline = time.time() + 580.0
    
    def _score_exact(z_list):
        try:
            if time.time() > deadline - 60:
                return float('inf')
                
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            # Convert z_i to exact rationals
            z_sympy = [sp.Rational(float(zi)) for zi in z_arr]
            
            alpha = sp.Rational(-1, 2)
            x = sp.symbols('x')
            
            # Precompute Laguerre polynomials L_{2k}^{-1/2}(x) for k=0..25
            Ls = []
            for k in range(26):
                n = 2 * k
                Ls.append(sp.laguerre(n, alpha, x))
            
            # Setup linear system A c = b
            c_syms = sp.symbols('c0:26')
            
            # g(x) = sum c_k L_{2k}(x)
            g_expr = sum(c_syms[k] * Ls[k] for k in range(26))
            
            # Derivative g'(x)
            g_diff = sp.diff(g_expr, x)
            
            A_mat = sp.zeros(26, 26)
            b_vec = sp.zeros(26, 1)
            
            # Row 0: g(0) = 0
            for k in range(26):
                A_mat[0, k] = Ls[k].subs(x, 0)
            b_vec[0, 0] = 0
            
            # Row 1: g'(0) = 1
            for k in range(26):
                A_mat[1, k] = sp.diff(Ls[k], x).subs(x, 0)
            b_vec[1, 0] = 1
            
            for i in range(12):
                zi = z_sympy[i]
                # Row 2+2i: g(zi) = 0
                for k in range(26):
                    A_mat[2+2*i, k] = Ls[k].subs(x, zi)
                b_vec[2+2*i, 0] = 0
                
                # Row 3+2i: g'(zi) = 0
                for k in range(26):
                    A_mat[3+2*i, k] = sp.diff(Ls[k], x).subs(x, zi)
                b_vec[3+2*i, 0] = 0
            
            # Solve A c = b
            try:
                c_sol = sp.LUsolve(A_mat, b_vec)
            except sp.MatrixNonInvertibleError:
                return float('inf')
            except Exception:
                return float('inf')
            
            # Construct g(x) with solved coefficients
            g_sol = sum(c_sol[k] * Ls[k] for k in range(26))
            
            # Construct D(x) = x * prod (x - z_i)^2
            D_expr = x
            for zi in z_sympy:
                D_expr *= (x - zi)**2
            
            # q(x) = g(x) / D(x)
            q_poly, rem = sp.div(g_sol, D_expr, x)
            
            if rem != 0:
                if sp.simplify(rem) != 0:
                    return float('inf')
            
            # Find real roots of q(x)
            try:
                q_roots = sp.real_roots(q_poly, x)
            except Exception:
                return float('inf')
            
            max_r = 0.0
            eps = sp.Rational(1, 10)**198 
            
            for r in q_roots:
                q_minus = q_poly.subs(x, r - eps)
                q_plus = q_poly.subs(x, r + eps)
                
                prod = q_minus * q_plus
                if prod < 0:
                    r_val = float(r)
                    if r_val > max_r:
                        max_r = r_val
            
            B = max(0, max_r)
            return B / (2 * math.pi)
            
        except Exception:
            return float('inf')

    # --- Candidate Generation Strategies ---
    
    # 1. Bessel zeros squared (current baseline heuristic)
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # 2. Laguerre roots of L_12^{-1/2} (Structural Change: Directly related to the basis)
    # The problem involves L_{2k}^{alpha}. The roots of L_N^{alpha} are often good initial guesses
    # for nodes in such interpolation/expansion problems.
    try:
        from scipy.special import roots_genlaguerre
        # Roots of L_12^{-1/2}(x)
        roots_12 = roots_genlaguerre(12, -0.5)
        z_laguerre_12 = sorted([float(r) for r in roots_12.roots if r > 0])
        if len(z_laguerre_12) < 12:
            # Pad if necessary
            max_val = max(z_laguerre_12) if z_laguerre_12 else 1.0
            while len(z_laguerre_12) < 12:
                z_laguerre_12.append(max_val * 1.1)
                max_val *= 1.1
            z_laguerre_12 = z_laguerre_12[:12]
    except Exception:
        z_laguerre_12 = z_bessel # Fallback
    
    # 3. Geometric spacing
    z_geometric = [0.5 * (1.5 ** i) for i in range(12)]
    
    # 4. Uniform spacing
    z_uniform = [float(i) for i in range(1, 13)]
    
    candidates = [
        z_bessel,
        z_laguerre_12,
        z_geometric,
        z_uniform
    ]
    
    best_z = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for z_cand in candidates:
        if time.time() > deadline - 120:
            break
        score = _score_exact(z_cand)
        if score < best_score:
            best_score = score
            best_z = list(z_cand)
    
    if best_z is None:
        return z_bessel
        
    # --- Optimization: Adaptive Perturbation Search ---
    # Instead of fixed random perturbations, we use a greedy local search
    # that tries to move each node slightly to improve the score.
    # This is more robust than random jumps which might miss the narrow valley.
    
    current_z = list(best_z)
    current_score = best_score
    
    # We will iterate through each coordinate and try small steps.
    # Step size starts small and can adapt.
    
    max_iterations = 40 # Limit total evaluations to ensure we finish in time
    iter_count = 0
    step_size = 0.05
    
    while iter_count < max_iterations:
        if time.time() > deadline - 60:
            break
            
        improved = False
        # Try to improve one coordinate at a time
        # To be efficient, we don't shuffle, we just cycle through.
        # But randomizing the order might help escape local minima.
        indices = list(range(12))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 60:
                break
                
            # Try moving z[idx] up and down
            for delta in [step_size, -step_size]:
                candidate = list(current_z)
                new_val = current_z[idx] * (1 + delta)
                
                if new_val <= 0:
                    continue
                
                candidate[idx] = new_val
                candidate.sort()
                
                # Check distinctness
                valid = True
                for j in range(11):
                    if abs(candidate[j] - candidate[j+1]) < 1e-8:
                        valid = False
                        break
                if not valid:
                    continue
                
                score = _score_exact(candidate)
                iter_count += 1
                
                if score < current_score:
                    current_score = score
                    current_z = candidate
                    best_score = score
                    best_z = candidate
                    improved = True
                    # If improved, reset step size or keep it? 
                    # Usually, if we improve, we can try to continue in the same direction 
                    # or shrink the step. Here we just accept and continue loop.
                    break # Break out of delta loop to try next idx or restart
            
            # If we improved, restart the index loop? 
            # Or just continue. Continuing is faster but might miss global improvements.
            # Given the time limit, simple hill climbing with fixed step is okay.
        
        # Reduce step size if no improvement found in this full pass
        if not improved:
            step_size *= 0.5
            if step_size < 1e-4:
                break

    return best_z

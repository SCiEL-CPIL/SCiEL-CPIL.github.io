"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Derivative-free root finder using polynomial interpolation.
    Exploits the fact that f is a smooth polynomial.
    """
    t_start = time.time()
    DEADLINE = 120.0  # seconds, with margin
    
    x = float(x0)
    # Collect points (x_i, f(x_i))
    points = []
    
    def add_point(xv):
        """Evaluate f at xv and add to points list. Returns f(xv)."""
        fx = f(xv)
        if math.isfinite(fx):
            points.append((xv, fx))
        return fx
    
    # Initial point
    fx = add_point(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Set up initial bracketing / exploration points
    # Try a few points to build up the interpolating polynomial
    step = max(1e-3, abs(x) * 0.1)
    
    # Add a few initial points
    for dx in [step, -step, 2*step, -2*step, 4*step, -4*step]:
        if time.time() - t_start > DEADLINE - 5:
            break
        x_trial = x + dx
        fx_trial = add_point(x_trial)
        if abs(fx_trial) <= 1e-8:
            return x_trial
        # Update best
        if abs(fx_trial) < abs(fx):
            fx = fx_trial
            x = x_trial
    
    # Now iteratively refine using polynomial interpolation
    # Newton's divided differences to build interpolating polynomial
    def newton_interpolate(xs, ys):
        """
        Compute Newton's divided differences.
        Returns coefficients for polynomial in Newton form:
        p(x) = c0 + c1*(x-x0) + c2*(x-x0)(x-x1) + ...
        """
        n = len(xs)
        if n == 0:
            return []
        # Compute divided differences table
        dd = [list(ys[:])]  # dd[0][i] = ys[i]
        for level in range(1, n):
            row = [0.0] * (n - level)
            for i in range(n - level):
                denom = xs[i + level] - xs[i]
                if abs(denom) < 1e-15:
                    row[i] = 0.0
                else:
                    row[i] = (dd[level-1][i+1] - dd[level-1][i]) / denom
            dd.append(row)
        return [dd[level][0] for level in range(n)]
    
    def eval_newton_poly(coeffs, xs, x):
        """Evaluate Newton-form polynomial at x."""
        n = len(coeffs)
        if n == 0:
            return 0.0
        result = coeffs[-1]
        for i in range(n - 2, -1, -1):
            result = result * (x - xs[i]) + coeffs[i]
        return result
    
    def find_roots_newton_poly(coeffs, xs):
        """
        Find real roots of a Newton-form polynomial.
        Convert to standard form first, then use numpy roots.
        """
        import numpy as np
        n = len(coeffs)
        if n == 0:
            return []
        if n == 1:
            return []
        
        # Convert Newton form to standard polynomial coefficients
        # p(x) = sum_{k=0}^{n-1} c_k * prod_{j=0}^{k-1} (x - x_j)
        # Build polynomial iteratively
        # Start with c_{n-1} (constant in the highest term)
        # Actually, let's build from highest degree down
        
        # The Newton polynomial: p(x) = c0 + c1*(x-x0) + c2*(x-x0)(x-x1) + ... + c_{n-1}*(x-x0)...(x-x_{n-2})
        # Let's build the standard coefficients
        # Start with the highest-degree term
        poly = [coeffs[-1]]  # coefficient of x^{n-1}
        # Multiply by (x - x_{n-2})
        for k in range(n - 2, -1, -1):
            # poly represents sum of terms from degree n-1-k down to 0 (in reverse)
            # Actually let me redo this more carefully
            pass
        
        # Better approach: build polynomial as list of coefficients [a_n, a_{n-1}, ..., a_0]
        # Start with the leading term
        # p(x) = c_{n-1} * prod_{j=0}^{n-2}(x - x_j) + c_{n-2} * prod_{j=0}^{n-3}(x - x_j) + ... + c_0
        
        # Build each product and accumulate
        # Standard coefficients: index 0 = highest power
        std_coeffs = [0.0] * n  # degree n-1, so n coefficients
        
        for k in range(n):
            # term k: c_k * prod_{j=0}^{k-1} (x - x_j)
            # This is a polynomial of degree k
            c_k = coeffs[k]
            if k == 0:
                # constant term
                std_coeffs[n - 1] += c_k
            else:
                # Build prod_{j=0}^{k-1}(x - x_j)
                # Start with [1] (constant 1)
                prod = [1.0]
                for j in range(k):
                    # Multiply by (x - x_j): new[i] = old[i-1] - x_j * old[i]
                    new_prod = [0.0] * (len(prod) + 1)
                    for i in range(len(prod)):
                        new_prod[i] += prod[i]  # x * prod[i] goes to degree i+1
                        new_prod[i+1] += -xs[j] * prod[i]  # -x_j * prod[i] stays at degree i
                    prod = new_prod
                # prod now has k+1 coefficients, [coeff of x^k, ..., coeff of x^0]
                # Add c_k * prod to std_coeffs
                # std_coeffs is length n, representing degrees n-1 down to 0
                # prod is length k+1, representing degrees k down to 0
                # So prod[0] (x^k) maps to std_coeffs[n-1-k]
                for i in range(len(prod)):
                    std_coeffs[n - 1 - k + i] += c_k * prod[i]
        
        # Use numpy to find roots
        roots = np.roots(std_coeffs)
        # Filter real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_roots.append(float(r.real))
        return real_roots
    
    best_x = x
    best_fx = fx
    max_iter = 50
    
    for iteration in range(max_iter):
        if time.time() - t_start > DEADLINE - 5:
            break
        if len(points) < 2:
            break
        
        # Sort points by x for better interpolation
        points_sorted = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        
        # Limit to last 8 points for numerical stability
        if len(xs) > 8:
            xs = xs[-8:]
            ys = ys[-8:]
        
        # Build Newton interpolating polynomial
        coeffs = newton_interpolate(xs, ys)
        
        # Find roots of the interpolating polynomial
        candidate_roots = find_roots_newton_poly(coeffs, xs)
        
        # Also try secant-method style: use last two points
        if len(xs) >= 2:
            x1, x2 = xs[-2], xs[-1]
            y1, y2 = ys[-2], ys[-1]
            if abs(y2 - y1) > 1e-15:
                secant_root = x2 - y2 * (x2 - x1) / (y2 - y1)
                if math.isfinite(secant_root):
                    candidate_roots.append(secant_root)
        
        if not candidate_roots:
            # Fallback: bisection-like step
            if len(xs) >= 2:
                if ys[-1] * ys[-2] < 0:
                    x_new = (xs[-1] + xs[-2]) / 2
                else:
                    x_new = xs[-1] + 0.5 * (xs[-1] - xs[-2])
            else:
                x_new = xs[0] + 0.5 * step
        else:
            # Pick the candidate root that minimizes |interpolating poly value|
            # and is reasonably close to current best
            best_candidate = None
            best_candidate_score = float('inf')
            for cr in candidate_roots:
                if not math.isfinite(cr):
                    continue
                # Score: prefer roots close to current best and with small interpolation residual
                # Evaluate the interpolating polynomial at cr (should be ~0 by construction)
                # Instead, prefer roots that are "reasonable" - not too far
                dist = abs(cr - best_x)
                # Penalize very distant roots
                if dist > max(10.0, 10.0 * abs(best_x)):
                    continue
                score = dist
                if best_candidate is None or score < best_candidate_score:
                    best_candidate = cr
                    best_candidate_score = score
            
            if best_candidate is not None:
                x_new = best_candidate
            else:
                # Use the secant root if available
                if len(xs) >= 2:
                    x1, x2 = xs[-2], xs[-1]
                    y1, y2 = ys[-2], ys[-1]
                    if abs(y2 - y1) > 1e-15:
                        x_new = x2 - y2 * (x2 - x1) / (y2 - x1)
                        if not math.isfinite(x_new):
                            x_new = xs[-1] + 0.1 * (xs[-1] - xs[-2])
                    else:
                        x_new = xs[-1] + 0.1 * step
                else:
                    x_new = xs[0] + 0.5 * step
        
        # Check if x_new is too close to an existing point
        too_close = False
        for px, _ in points:
            if abs(x_new - px) < 1e-12 * max(1.0, abs(px), abs(x_new)):
                too_close = True
                break
        
        if too_close:
            # Perturb slightly
            x_new = x_new + 1e-6 * max(1.0, abs(x_new))
        
        fx_new = add_point(x_new)
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        if abs(fx_new) < abs(best_fx):
            best_fx = fx_new
            best_x = x_new
        
        # If we have many points and aren't converging, try a different strategy
        if len(points) > 12:
            # Try to use the best few points to build a tighter interpolant
            # Sort by |f(x)| and keep the best 6
            points_sorted_by_f = sorted(points, key=lambda p: abs(p[1]))
            # Keep a mix: best by |f| and some spread
            best_points = points_sorted_by_f[:4]
            # Add a couple for spread
            for p in points:
                if len(best_points) >= 6:
                    break
                if all(abs(p[0] - bp[0]) > 1e-10 for bp in best_points):
                    best_points.append(p)
            points = best_points
    
    # Return the best x found
    return best_x
# EVOLVE-BLOCK-END

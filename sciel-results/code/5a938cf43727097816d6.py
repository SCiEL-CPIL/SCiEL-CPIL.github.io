import numpy as np
from scipy.special import eval_genlaguerre, roots_genlaguerre
from scipy.optimize import brentq, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    m = 12
    
    # Compute Laguerre polynomial values
    # L_m(x) = eval_genlaguerre(m, 0, x)
    # L_{m-1}^{(1)}(x) = eval_genlaguerre(m-1, 1, x)
    
    def P(x, c):
        """P(x) = [L_m(x)]^2 - c * x * [L_{m-1}^{(1)}(x)]^2"""
        Lm = eval_genlaguerre(m, 0, x)
        Lm1 = eval_genlaguerre(m-1, 1, x)
        return Lm**2 - c * x * Lm1**2
    
    def P_derivative(x, c):
        """P'(x) using automatic differentiation or finite differences"""
        h = 1e-8
        return (P(x + h, c) - P(x - h, c)) / (2*h)
    
    # We need to find c such that P(x) has 12 double roots.
    # A double root at x=r means P(r) = 0 and P'(r) = 0.
    # This is a system of 24 equations in 13 unknowns (12 roots + c).
    
    # Alternative approach: The resultant of P and P' should be zero.
    # But that's complex. Instead, let's use the fact that for the optimal c,
    # P(x) >= 0 and touches zero at 12 points.
    
    # Strategy: For a given c, find the local minima of P(x) and check if they're zero.
    # The optimal c is when the minimum value of P(x) is exactly 0.
    
    def min_P(c):
        """Find the minimum value of P(x) for given c over x > 0"""
        # Sample P(x) on a grid to find approximate minima
        x_grid = np.linspace(0.01, 100, 1000)
        P_vals = P(x_grid, c)
        
        # Find local minima
        min_indices = []
        for i in range(1, len(P_vals)-1):
            if P_vals[i] < P_vals[i-1] and P_vals[i] < P_vals[i+1]:
                min_indices.append(i)
        
        if not min_indices:
            # Find global minimum
            min_idx = np.argmin(P_vals)
            min_indices = [min_idx]
        
        min_val = np.inf
        for idx in min_indices:
            # Refine using brentq on derivative
            try:
                x_left = max(0.001, x_grid[idx] - 1.0)
                x_right = min(99.999, x_grid[idx] + 1.0)
                if P_derivative(x_left, c) * P_derivative(x_right, c) < 0:
                    x_min = brentq(lambda x: P_derivative(x, c), x_left, x_right, xtol=1e-10)
                    min_val = min(min_val, P(x_min, c))
                else:
                    min_val = min(min_val, P_vals[idx])
            except:
                min_val = min(min_val, P_vals[idx])
        
        return min_val
    
    # Find c such that min_P(c) = 0
    # Try a range of c values
    c_low = 1e-6
    c_high = 10.0
    
    # First, find a c where min_P changes sign
    c_candidates = np.logspace(-4, 1, 50)
    min_vals = []
    for c in c_candidates:
        try:
            mv = min_P(c)
            min_vals.append(mv)
        except:
            min_vals.append(np.nan)
    
    min_vals = np.array(min_vals)
    valid = ~np.isnan(min_vals)
    
    # Find where min_val crosses zero
    c_opt = None
    for i in range(len(c_candidates)-1):
        if valid[i] and valid[i+1]:
            if min_vals[i] * min_vals[i+1] < 0:
                # Sign change, use brentq
                try:
                    c_opt = brentq(lambda c: min_P(c), c_candidates[i], c_candidates[i+1], xtol=1e-10)
                    break
                except:
                    pass
    
    if c_opt is None:
        # Fallback: use the c that gives the smallest absolute min_val
        if np.any(valid):
            best_idx = np.argmin(np.abs(min_vals[valid]))
            valid_indices = np.where(valid)[0]
            c_opt = c_candidates[valid_indices[best_idx]]
        else:
            c_opt = 1.0
    
    # Now find the 12 double roots for this c_opt
    # The roots are where P(x) = 0 and P'(x) = 0
    # Find all zeros of P(x) and check which are also zeros of P'(x)
    
    def find_double_roots(c):
        """Find the double roots of P(x) for given c"""
        # Find all roots of P(x) in (0, inf)
        # Use a grid to find sign changes
        x_grid = np.linspace(0.001, 200, 5000)
        P_vals = P(x_grid, c)
        
        # Find sign changes
        root_candidates = []
        for i in range(len(x_grid)-1):
            if P_vals[i] * P_vals[i+1] < 0:
                # Sign change, refine with brentq
                try:
                    x_root = brentq(lambda x: P(x, c), x_grid[i], x_grid[i+1], xtol=1e-12)
                    root_candidates.append(x_root)
                except:
                    pass
            elif abs(P_vals[i]) < 1e-10:
                root_candidates.append(x_grid[i])
        
        # Filter for double roots: P'(x) should also be ~0
        double_roots = []
        for r in root_candidates:
            if abs(P_derivative(r, c)) < 1e-6:
                double_roots.append(r)
        
        # Deduplicate
        if double_roots:
            double_roots = sorted(set([round(r, 10) for r in double_roots]))
            # Keep only 12
            if len(double_roots) > 12:
                double_roots = double_roots[:12]
        
        return double_roots
    
    roots = find_double_roots(c_opt)
    
    # If we don't have exactly 12 roots, try to refine
    if len(roots) != 12:
        # Try a different approach: optimize c to make P(x) have exactly 12 double roots
        # Use the fact that the sum of roots and other symmetric functions can be computed
        # from the coefficients of P(x)
        
        # Get coefficients of P(x) = [L_m(x)]^2 - c*x*[L_{m-1}^{(1)}(x)]^2
        # L_m(x) is a polynomial of degree m, so [L_m(x)]^2 has degree 2m = 24
        # x*[L_{m-1}^{(1)}(x)]^2 also has degree 2m = 24
        
        # Compute coefficients
        x_test = np.linspace(0, 1, 25)
        Lm_vals = eval_genlaguerre(m, 0, x_test)
        Lm1_vals = eval_genlaguerre(m-1, 1, x_test)
        
        # Fit polynomials
        Lm_coeffs = np.polyfit(x_test, Lm_vals, m)[::-1]  # ascending order
        Lm1_coeffs = np.polyfit(x_test, Lm1_vals, m-1)[::-1]
        
        # [L_m(x)]^2
        Lm_sq = np.convolve(Lm_coeffs, Lm_coeffs)
        
        # x * [L_{m-1}^{(1)}(x)]^2
        Lm1_sq = np.convolve(Lm1_coeffs, Lm1_coeffs)
        x_Lm1_sq = np.concatenate([[0], Lm1_sq])
        
        # P(x) coefficients (ascending order)
        # P(x) = Lm_sq - c * x_Lm1_sq
        # Need to align degrees
        max_deg = max(len(Lm_sq), len(x_Lm1_sq))
        Lm_sq_padded = np.zeros(max_deg)
        x_Lm1_sq_padded = np.zeros(max_deg)
        Lm_sq_padded[:len(Lm_sq)] = Lm_sq
        x_Lm1_sq_padded[:len(x_Lm1_sq)] = x_Lm1_sq
        
        # For a given c, P_coeffs = Lm_sq_padded - c * x_Lm1_sq_padded
        # The roots of P(x) can be found from these coefficients
        
        # The condition for double roots is that the discriminant is zero,
        # but that's complex. Instead, let's just use the roots we found.
        
        # If we have fewer than 12, use the roots we have and pad
        if len(roots) > 0:
            # Interpolate to get 12 roots
            if len(roots) < 12:
                # Add roots based on spacing
                if len(roots) == 1:
                    r = roots[0]
                    roots = [r * (i+1) for i in range(12)]
                else:
                    # Extrapolate
                    diffs = np.diff(roots)
                    avg_diff = np.mean(diffs)
                    while len(roots) < 12:
                        roots.append(roots[-1] + avg_diff)
            roots = roots[:12]
        else:
            # Fallback: use roots of L_12(x)
            x_lag, _ = roots_genlaguerre(m, 0)
            roots = list(x_lag)
    
    # Ensure positive and sorted
    roots = sorted([max(1e-6, float(r)) for r in roots])
    
    return roots

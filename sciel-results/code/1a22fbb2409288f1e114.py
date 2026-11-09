import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Nelder-Mead simplex minimization on f(x)^2.
    
    Strategy:
    1. Treat root finding as a minimization problem: minimize g(x) = f(x)^2.
    2. Use Nelder-Mead simplex method, which is derivative-free and robust
       for smooth, oscillatory functions.
    3. This avoids the fragility of bracketing (sign changes) which can fail
       if the function touches zero without crossing, or if bracketing requires
       many exploratory calls.
    4. Nelder-Mead typically converges in O(10-20) iterations for 1D problems
       with good initial simplex, significantly reducing call count compared
       to bisection/Illinois which might need 50-100 calls.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Objective function: minimize f(x)^2
    # We cache f(x) evaluations to avoid redundant calls if the simplex
    # revisits points, though in 1D NM, revisits are rare.
    # Actually, for a pure black-box function, we just call f.
    # We will track the best point found.
    
    best_x = x
    best_f = fx
    best_g = fx * fx
    
    # Nelder-Mead Simplex Method for 1D
    # Simplex is a set of 2 points in 1D: [x1, x2]
    # We need to initialize x2.
    # A small step is usually sufficient.
    # Let's try a few small steps to initialize the simplex well.
    # If f(x0) is large, a larger step might be better, but 1.0 is a standard default.
    # Let's use a relative step based on x0 magnitude if x0 is not near 0.
    # However, for oscillatory functions with roots likely O(1), a fixed small step is safer.
    # Let's try h = 1e-1 first.
    
    h = 1e-1
    x1 = x
    x2 = x + h
    
    # Evaluate second point
    fx2 = f(x2)
    if abs(fx2) <= 1e-8:
        return x2
    
    g1 = fx * fx
    g2 = fx2 * fx2
    
    # Update best
    if g2 < best_g:
        best_g = g2
        best_x = x2
        best_f = fx2
    
    # Simplex points: (x1, g1) and (x2, g2)
    # Ensure x1 has lower g? No, NM doesn't strictly require ordering,
    # but it helps to track. Let's just keep the two points.
    
    # NM Parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink
    
    # Max iterations to prevent infinite loops
    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Order points so x_low has lower g
        if g1 <= g2:
            x_low, g_low = x1, g1
            x_high, g_high = x2, g2
        else:
            x_low, g_low = x2, g2
            x_high, g_high = x1, g1
            
        # Check convergence
        if abs(g_low) <= 1e-16: # sqrt(1e-16) = 1e-8
            # We need |f(x)| <= 1e-8, so f(x)^2 <= 1e-16
            # Check the actual x_low
            if abs(f(x_low)) <= 1e-8: # This is an extra call, but we might have cached it?
                # To be safe, we return the one we know is small.
                # We have g_low which is f(x_low)^2.
                # If g_low <= 1e-16, then |f(x_low)| <= 1e-8.
                return x_low
        
        # Also check if the interval is small enough
        if abs(x_high - x_low) < 1e-12:
            # Return the one with smaller g
            return x_low if g_low <= g_high else x_high
            
        # Centroid of the simplex excluding the worst point
        # In 1D, centroid of [x_low] is just x_low.
        # Reflection: x_r = x_low + alpha * (x_low - x_high)
        x_r = x_low + alpha * (x_low - x_high)
        f_r = f(x_r)
        g_r = f_r * f_r
        
        if g_r < best_g:
            best_g = g_r
            best_x = x_r
            best_f = f_r
        
        if g_r <= 1e-16:
            return x_r
            
        if g_r < g_low:
            # Expansion
            x_e = x_low + gamma * (x_low - x_high)
            f_e = f(x_e)
            g_e = f_e * f_e
            
            if g_e < best_g:
                best_g = g_e
                best_x = x_e
                best_f = f_e
                
            if g_e <= 1e-16:
                return x_e
                
            if g_e < g_r:
                # Replace worst (x_high) with expanded point
                x_high, g_high = x_e, g_e
                if g_high < g_low:
                    # Swap roles for next iteration
                    pass # Logic handles ordering at start of loop
                else:
                    pass
                # Update the pair
                if x_low == x1:
                    x2, g2 = x_high, g_high
                else:
                    x1, g1 = x_high, g_high
            else:
                # Replace worst with reflected point
                x_high, g_high = x_r, g_r
                if x_low == x1:
                    x2, g2 = x_high, g_high
                else:
                    x1, g1 = x_high, g_high
                    
        elif g_r < g_high:
            # Accept reflection
            x_high, g_high = x_r, g_r
            if x_low == x1:
                x2, g2 = x_high, g_high
            else:
                x1, g1 = x_high, g_high
                
        else:
            # Contraction
            x_c = x_low + rho * (x_high - x_low)
            f_c = f(x_c)
            g_c = f_c * f_c
            
            if g_c < best_g:
                best_g = g_c
                best_x = x_c
                best_f = f_c
                
            if g_c <= 1e-16:
                return x_c
                
            if g_c < g_high:
                # Replace worst with contracted point
                x_high, g_high = x_c, g_c
                if x_low == x1:
                    x2, g2 = x_high, g_high
                else:
                    x1, g1 = x_high, g_high
            else:
                # Shrink
                # Move both points towards x_low
                x1_new = x_low + sigma * (x1 - x_low)
                x2_new = x_low + sigma * (x2 - x_low)
                
                f1_new = f(x1_new)
                g1_new = f1_new * f1_new
                if g1_new < best_g:
                    best_g = g1_new
                    best_x = x1_new
                    best_f = f1_new
                if g1_new <= 1e-16:
                    return x1_new
                    
                f2_new = f(x2_new)
                g2_new = f2_new * f2_new
                if g2_new < best_g:
                    best_g = g2_new
                    best_x = x2_new
                    best_f = f2_new
                if g2_new <= 1e-16:
                    return x2_new
                    
                x1, g1 = x1_new, g1_new
                x2, g2 = x2_new, g2_new
                
    return best_x
# EVOLVE-BLOCK-END

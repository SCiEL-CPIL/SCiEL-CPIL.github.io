import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Spectral estimation: Sample f around x0 to estimate local frequency and phase.
      2. Predict nearest zero-crossing based on the fitted sinusoid.
      3. Refine with a robust Secant/Bisection hybrid method.
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Helper to check time
    def time_up():
        return time.time() > deadline

    # Step 1: Spectral Sampling
    # We sample at N points over a window [x0 - W, x0 + W]
    # We need to pick a window W that captures at least one period but not too many.
    # Since we don't know the frequency, we start with a small window and expand if needed.
    # However, for oscillatory functions, a small window around x0 usually contains the local oscillation.
    # Let's try a few window sizes to find a good fit.
    
    best_guess = x0
    best_error = float('inf')
    
    # Try different window sizes to estimate frequency
    # The function is oscillatory. Let's assume the period is not extremely small or large relative to x0.
    # We will sample 16 points.
    N = 16
    window_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for W in window_sizes:
        if time_up(): break
        
        # Generate sample points
        xs = [x0 - W + 2*W*i/(N-1) for i in range(N)]
        ys = []
        valid = True
        for x in xs:
            y = f(x)
            if not math.isfinite(y):
                valid = False
                break
            ys.append(y)
        
        if not valid:
            continue
            
        # Fit a sine wave: y = A * sin(omega * t + phi) + C
        # We can use a simple least squares fit for sin and cos terms.
        # y = a * sin(omega * t) + b * cos(omega * t) + C
        # This is linear in a, b, C for a fixed omega.
        # We need to estimate omega.
        # Instead of complex optimization, we can use the fact that for a sine wave,
        # the zero crossings are periodic.
        
        # A simpler heuristic: Find the first sign change in the samples.
        # If we find a sign change, we can interpolate to get a better guess.
        # But the spectral idea is to use the shape.
        
        # Let's try to estimate omega from the samples using autocorrelation or simple zero-crossing detection.
        # Zero crossing detection:
        for i in range(N - 1):
            if ys[i] == 0:
                guess = xs[i]
                if abs(f(guess)) < best_error:
                    best_error = abs(f(guess))
                    best_guess = guess
            elif ys[i] * ys[i+1] < 0:
                # Linear interpolation for zero crossing
                x1, x2 = xs[i], xs[i+1]
                y1, y2 = ys[i], ys[i+1]
                if y1 != y2:
                    t = y1 / (y1 - y2)
                    guess = x1 + t * (x2 - x1)
                    fg = f(guess)
                    if math.isfinite(fg) and abs(fg) < best_error:
                        best_error = abs(fg)
                        best_guess = guess
        if best_error < 1e-8:
            break

    # Step 2: Refinement
    # Use a robust method starting from best_guess
    # We need a bracket [a, b] such that f(a)*f(b) < 0.
    # If we have a good guess, we can expand a small window around it to find a bracket.
    
    x = best_guess
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Find a bracket
    # Start with a small step and expand
    step = 1e-3
    a = x - step
    b = x + step
    fa = f(a)
    fb = f(b)
    
    # Expand bracket if needed
    for _ in range(50):
        if time_up():
            return x # Return best known
        
        if fa == 0:
            return a
        if fb == 0:
            return b
        
        if fa * fb < 0:
            break
        
        # Expand
        step *= 2
        a = x - step
        b = x + step
        fa = f(a)
        fb = f(b)
        
        if not (math.isfinite(fa) and math.isfinite(fb)):
            # If we hit non-finite, try shrinking or falling back to x0
            # For now, break and try secant from x
            break

    # If we found a bracket, use Bisection/Secant hybrid
    if fa * fb < 0:
        # Secant/Bisection hybrid
        # We track (a, fa) and (b, fb)
        # Also keep track of the previous point for secant
        x_prev = x
        f_prev = fx
        
        for _ in range(100):
            if time_up():
                return x if abs(fx) < abs(f_prev) else x_prev
            
            # Check convergence
            if abs(fx) <= 1e-8:
                return x
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Secant step
            if f_prev != fx:
                x_new = x - fx * (x - x_prev) / (fx - f_prev)
            else:
                x_new = (a + b) / 2
            
            # Check if x_new is within bracket
            if x_new <= a or x_new >= b:
                # Bisection step
                x_new = (a + b) / 2
            
            f_new = f(x_new)
            if not math.isfinite(f_new):
                # If new point is bad, bisect
                x_new = (a + b) / 2
                f_new = f(x_new)
                
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
                
            # Update secant history
            x_prev = x
            f_prev = fx
            x = x_new
            fx = f_new
            
            # Convergence check based on interval size
            if abs(b - a) < 1e-12:
                return (a + b) / 2
    else:
        # No bracket found, fall back to Newton/Secant from x0
        x = x0
        fx = f(x)
        x_prev = x0 + 1e-6
        f_prev = f(x_prev)
        
        for _ in range(100):
            if time_up():
                return x
            
            if abs(fx) <= 1e-8:
                return x
                
            if f_prev != fx:
                x_new = x - fx * (x - x_prev) / (fx - f_prev)
            else:
                break # Derivative zero, can't continue secant
                
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            x_prev = x
            f_prev = fx
            x = x_new
            fx = f_new

    return x
# EVOLVE-BLOCK-END

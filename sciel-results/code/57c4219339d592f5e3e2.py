import time

def find_root(f, x0):
    start_time = time.time()
    # 150s kill, 120s eval. Use 115s to be safe.
    deadline = start_time + 115.0
    
    x = float(x0)
    h = 1e-6 * max(1.0, abs(x))
    
    x_prev = x
    x_curr = x + h
    
    try:
        f_prev = f(x_prev)
        f_curr = f(x_curr)
    except Exception:
        return x

    if abs(f_curr) <= 1e-8:
        return x_curr
    if abs(f_prev) <= 1e-8:
        return x_prev

    # If f_prev and f_curr have opposite signs, we have a bracket.
    # We can use Bisection or Secant. Secant is faster.
    # If same sign, we are just extrapolating.
    
    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline:
            # Return the best candidate
            if abs(f_curr) < abs(f_prev):
                return x_curr
            else:
                return x_prev
        
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev

        # Check for convergence of x
        if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr)):
            # If x is stable but f is not small, we might be stuck at a local extremum or flat region.
            # But for polynomials, this is unlikely unless root is multiple.
            # If we are stuck, break.
            break

        if f_curr == f_prev:
            # Flat slope, try a small perturbation
            h_new = 1e-6 * max(1.0, abs(x_curr))
            x_next = x_curr + h_new
            f_next = f(x_next)
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next
            continue

        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # If x_next is not finite, break
        if not (x_next == x_next and x_next != float('inf') and x_next != float('-inf')):
            break

        f_next = f(x_next)
        
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    if abs(f_curr) < abs(f_prev):
        return x_curr
    else:
        return x_prev

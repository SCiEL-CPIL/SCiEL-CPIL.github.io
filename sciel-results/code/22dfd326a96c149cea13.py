import numpy as np

def equation(x, y, params):
    # p0..p5 numerator, q1..q4 denominator (constant term 1)
    p0, p1, p2, p3, p4, p5 = params[0:6]
    q1, q2, q3, q4 = params[6:10]
    
    num = p0 + p1 * x + p2 * y + p3 * x**2 + p4 * y**2 + p5 * x * y
    den = 1.0 + q1 * x + q2 * y + q3 * x**2 + q4 * y**2
    
    # Safety: prevent division by zero
    # If den is close to 0, the value blows up.
    # We can add a small epsilon to the magnitude of den to avoid inf, 
    # but this distorts the function.
    # Better: rely on the optimizer. If it fails, score is 0.
    # To be safer, we can use np.where to avoid 0, but that's non-differentiable.
    # Let's just do direct division. BFGS from ones usually stays in safe region 
    # for simple functions.
    
    return num / den

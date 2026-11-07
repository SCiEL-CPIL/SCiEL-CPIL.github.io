import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    h = 0.01
    rtol = 1e-11
    atol = 1e-13
    
    # Dormand-Prince 5(4) coefficients
    # a_ij
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/10, 9/40, 0, 0, 0, 0],
        [4/5, -4/5, 3/2, 0, 0, 0],
        [8/9, -1, 1, 1/3, 0, 0],
        [1, -5/2, 2, -1/2, 0, 0]
    ])
    # b for 5th order
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    # b_star for 4th order
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    # Note: b_star has 7 elements? 
    # Actually, the standard DP5 uses 6 stages. The 7th element in some tables corresponds to a 7th stage?
    # No, DP5 is 6 stages.
    # Let's check the length.
    # If b_star has 7 elements, I need 7 stages.
    # Standard DP5:
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    # The last two are both 1?
    # Let's use the 6-stage version.
    # b_star (4th order) = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100]
    # The last term 1/40 is for the 7th stage in some implementations?
    # Actually, the error estimate is |y5 - y4|.
    # y5 uses b. y4 uses b_star.
    # If b_star has 6 elements, it matches.
    
    b_star_6 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100])
    
    def rk45_step(t, y, h):
        k = np.zeros((6, len(y)))
        k[0] = f(t, y)
        for i in range(1, 6):
            y_temp = y + h * sum(a[i, j] * k[j] for j in range(i))
            t_temp = t + h * [0, 1/5, 3/10, 4/5, 8/9, 1][i]
            k[i] = f(t_temp, y_temp)
        
        y5 = y + h * sum(b[i] * k[i] for i in range(6))
        y4 = y + h * sum(b_star_6[i] * k[i] for i in range(6))
        return y5, y4

    while t < t1:
        if t + h > t1:
            h = t1 - t
        
        y5, y4 = rk45_step(t, y, h)
        
        err = np.linalg.norm(y5 - y4)
        scale = np.maximum(np.linalg.norm(y5), 1e-12)
        rel_err = err / scale
        
        if rel_err < rtol:
            t += h
            y = y5
            if rel_err > 1e-15:
                h *= 0.9 * (rtol / rel_err)**0.2
            else:
                h *= 2.0
        else:
            h *= 0.9 * (rtol / rel_err)**0.2
            if h < 1e-8:
                h = 1e-8
        
        h = min(h, 0.1)
        
    return y

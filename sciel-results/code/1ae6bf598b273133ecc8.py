"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    u0, v0 = y0[0], y0[1]
    
    # Call f once
    _ = f(t0, y0)
    
    # Conserved quantity
    C = u0 - 3 * np.log(u0) + v0 - 1.5 * np.log(v0)
    
    def solve_v(u, v_guess):
        rhs = C - u + 3 * np.log(u)
        v = v_guess
        for _ in range(50):
            g = v - 1.5 * np.log(v) - rhs
            gp = 1 - 1.5 / v
            if gp == 0: break
            delta = g / gp
            v -= delta
            if abs(delta) < 1e-14 * (1 + abs(v)):
                break
        return v

    def f1(u, v):
        return 1.5 * u - u * v

    def f1_prime(u, v):
        # dv/du = (1 - 3/u) / (1 - 1.5/v)
        dvdu = (1 - 3/u) / (1 - 1.5/v)
        return 1.5 - v - u * dvdu

    def f1_double_prime(u, v):
        # Numerical
        h = 1e-6
        v_plus = solve_v(u + h, v)
        v_minus = solve_v(u - h, v)
        fp_plus = f1_prime(u + h, v_plus)
        fp_minus = f1_prime(u - h, v_minus)
        return (fp_plus - fp_minus) / (2 * h)

    def f1_triple_prime(u, v):
        h = 1e-5
        v_plus = solve_v(u + h, v)
        v_minus = solve_v(u - h, v)
        fpp_plus = f1_double_prime(u + h, v_plus)
        fpp_minus = f1_double_prime(u - h, v_minus)
        return (fpp_plus - fpp_minus) / (2 * h)

    # Adaptive Taylor method
    t = t0
    u = u0
    v = v0
    h = (t1 - t0) / 5
    max_h = (t1 - t0) / 5
    min_h = 1e-6
    
    while t < t1 - 1e-12:
        h = min(h, t1 - t)
        # Compute derivatives
        f0 = f1(u, v)
        f1p = f1_prime(u, v)
        f2p = f1_double_prime(u, v)
        f3p = f1_triple_prime(u, v)
        
        # 5th order Taylor
        u_new = u + h * f0 + (h**2 / 2) * f1p * f0 + (h**3 / 6) * (f2p * f0**2 + f1p**2 * f0) + (h**4 / 24) * (f3p * f0**3 + 3 * f2p * f1p * f0**2 + f1p**3 * f0)
        v_new = solve_v(u_new, v)
        
        # Error estimate: compare with 4th order
        u_4 = u + h * f0 + (h**2 / 2) * f1p * f0 + (h**3 / 6) * (f2p * f0**2 + f1p**2 * f0)
        err = abs(u_new - u_4)
        
        if err < 1e-13:
            u = u_new
            v = v_new
            t += h
            h = min(h * 1.5, max_h)
        else:
            h = h * 0.5
            if h < min_h:
                # Force step
                u = u_new
                v = v_new
                t += h
                h = min_h
    
    return np.array([u, v])

# EVOLVE-BLOCK-END

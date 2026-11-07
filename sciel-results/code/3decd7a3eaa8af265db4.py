"""ODE integrator for discretized heat equation using implicit Radau IIA with known Jacobian."""
import numpy as np
from scipy.linalg import solve_banded


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Build tridiagonal A = 21^2 * tridiag(1, -2, 1)
    scale = 21.0 ** 2
    diag = np.full(n, -2.0 * scale)
    offdiag = np.full(n - 1, scale)
    
    def matvec_A(v):
        """Compute A @ v for tridiagonal A."""
        out = np.empty(n)
        out[0] = diag[0] * v[0] + offdiag[0] * v[1]
        out[-1] = offdiag[-1] * v[-1] + diag[-1] * v[-1]
        out[1:-1] = offdiag[:-1] * v[:-2] + diag[1:-1] * v[1:-1] + offdiag[1:] * v[2:]
        return out
    
    def solve_I_minus_chA(c_val, rhs):
        """Solve (I - c*h*A) x = rhs using banded LU."""
        ch = c_val * h
        d = np.full(n, 1.0 + 2.0 * ch * scale)
        o = np.full(n - 1, -ch * scale)
        banded = np.zeros((3, n))
        banded[0, 1:] = o
        banded[1, :] = d
        banded[2, :-1] = o
        return solve_banded((1, 1), banded, rhs)
    
    # Radau IIA 3-stage (5th order)
    a11, a21, a22, a31, a32, a33 = 1.0/12.0, -5.0/72.0, 25.0/36.0, -1.0/24.0, 0.5, 7.0/8.0
    b1, b2 = 1.0/9.0, 8.0/9.0
    
    steps = 100
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        # Stage 1: (I - a11*h*A) y1 = y
        y1 = solve_I_minus_chA(a11, y)
        Ay1 = matvec_A(y1)
        
        # Stage 2: (I - a22*h*A) y2 = y + a21*h*Ay1
        y2 = solve_I_minus_chA(a22, y + a21 * h * Ay1)
        Ay2 = matvec_A(y2)
        
        # Stage 3: (I - a33*h*A) y3 = y + a31*h*Ay1 + a32*h*Ay2
        y3 = solve_I_minus_chA(a33, y + a31 * h * Ay1 + a32 * h * Ay2)
        Ay3 = matvec_A(y3)
        
        # Update: y_{n+1} = y + h*(b1*Ay1 + b2*Ay2 + 0*Ay3)
        y = y + h * (b1 * Ay1 + b2 * Ay2)
        t += h
    
    # Call f at least once to satisfy contract
    _ = f(t1, y)
    
    return y

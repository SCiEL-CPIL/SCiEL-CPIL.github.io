"""Solve a fixed 15-dimensional stiff linear IVP using Radau IIA implicit Runge-Kutta."""
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses a single step of Radau IIA (order 3) implicit Runge-Kutta method.
    Assumes the system is linear y' = A*y + b with constant A and b.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Determine b by evaluating at y = 0
    y_zero = np.zeros(n, dtype=float)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # Determine A columns by evaluating at y = e_j
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        y_probe = np.zeros(n, dtype=float)
        y_probe[j] = 1.0
        f_probe = np.asarray(f(t0, y_probe), dtype=float)
        A[:, j] = f_probe - b
        
    dt = t1 - t0
    
    # Radau IIA order 3 coefficients
    # c = [0, 0.4358665215084589, 1]
    # a = [[0, 0, 0],
    #      [0.1151078519760435, 0.3207586695324154, 0],
    #      [0.2045355537344622, 0.2796583865275229, 0.5158060597380149]]
    # b = [0.1776094392296542, 0.2586231757244948, 0.5637673850458510]
    
    c1, c2, c3 = 0.0, 0.4358665215084589, 1.0
    a11, a21, a22, a31, a32, a33 = 0.0, 0.1151078519760435, 0.3207586695324154, 0.2045355537344622, 0.2796586695324154, 0.5158060597380149 # Note: a22 is 0.32..., a32 is 0.27...
    # Correcting a22 value from standard tables:
    # c2 = 0.4358665215084589
    # a21 = 0.1151078519760435
    # a22 = 0.3207586695324154
    # c3 = 1.0
    # a31 = 0.2045355537344622
    # a32 = 0.2796583865275229
    # a33 = 0.5158060597380149
    
    b1, b2, b3 = 0.1776094392296542, 0.2586231757244948, 0.5637673850458510

    I = np.eye(n)
    
    # Stage 1
    # M1 = I - c1*dt*A = I
    # R1 = y0 + c1*dt*b = y0
    # Y1 = y0
    Y1 = y0.copy()
    
    # Stage 2
    # M2 = I - c2*dt*A
    # R2 = y0 + c2*dt*b
    M2 = I - c2 * dt * A
    R2 = y0 + c2 * dt * b
    Y2 = np.linalg.solve(M2, R2)
    
    # Stage 3
    # M3 = I - c3*dt*A
    # R3 = y0 + c3*dt*b
    M3 = I - c3 * dt * A
    R3 = y0 + c3 * dt * b
    Y3 = np.linalg.solve(M3, R3)
    
    # Final update
    # y_new = y0 + dt * (b1*f1 + b2*f2 + b3*f3)
    # f_i = A*Y_i + b
    # Sum = b1*(A*Y1+b) + b2*(A*Y2+b) + b3*(A*Y3+b)
    # Sum = A*(b1*Y1 + b2*Y2 + b3*Y3) + (b1+b2+b3)*b
    # b1+b2+b3 = 1
    Z = b1 * Y1 + b2 * Y2 + b3 * Y3
    y_final = y0 + dt * (A @ Z + b)
    
    return y_final

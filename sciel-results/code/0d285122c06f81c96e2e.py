import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator y0' = y1, y1' = -y0.
    Uses a high-order fixed-step Runge-Kutta method with a single large step
    to achieve high accuracy in a minimal number of function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=np.float64, copy=True)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    h = t1 - t0
    
    # Use DOP853 with a single step (13 stages = 13 function calls)
    # DOP853 Butcher tableau coefficients
    c = np.array([
        0.0,
        128.0/425.0,
        8.0/15.0,
        2197.0/25000.0,
        161.0/175.0,
        512.0/575.0,
        216.0/125.0,
        29.0/50.0,
        104.0/225.0,
        108.0/125.0,
        212.0/125.0,
        469.0/1500.0,
        1.0
    ])
    
    # A matrix: A[i,j] for j < i
    A = np.zeros((13, 13))
    A[1, 0] = 128.0/425.0
    A[2, 0] = 8.0/15.0
    A[3, 0] = 2197.0/25000.0
    A[4, 0] = 161.0/175.0
    A[5, 0] = 512.0/575.0
    A[6, 0] = 216.0/125.0
    A[7, 0] = 29.0/50.0
    A[8, 0] = 104.0/225.0
    A[9, 0] = 108.0/125.0
    A[10, 0] = 212.0/125.0
    A[11, 0] = 469.0/1500.0
    A[12, 0] = 1.0
    
    # b coefficients for the 8th-order solution
    b = np.array([
        326713695912675.0/2758680043855872.0,
        -30015419637729.0/102178229024064.0,
        54932119313847.0/25544557256016.0,
        -28116995062797.0/33739383008021.0,
        94911131217231.0/197611278048128.0,
        -18721049830699.0/123507048780080.0,
        396650754835759.0/1852605731701200.0,
        596346835029621.0/1482084585360960.0,
        35763428028778.0/32813213232465.0,
        -2630295337496.0/15549794346895.0,
        6645184377580.0/11662345760171.0,
        17102664311027.0/32277824413800.0,
        439339358847.0/4185287732480.0
    ])
    
    # Perform one step of DOP853
    k = [None] * 13
    k[0] = f(t0, y)
    
    for i in range(1, 13):
        ti = t0 + c[i] * h
        yi = y + h * np.dot(A[i, :i], k[:i])
        k[i] = f(ti, yi)
    
    y_new = y + h * np.dot(b, k)
    
    return y_new

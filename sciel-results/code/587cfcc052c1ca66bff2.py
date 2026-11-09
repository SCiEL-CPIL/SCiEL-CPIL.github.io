import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    t = x / 2.0
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    # We have 10 params.
    # Let's use a rational function: P(x) / Q(x)
    # P(x) = sum_{i=0}^4 a_i T_i(x)
    # Q(x) = sum_{j=0}^4 b_j T_j(x)
    
    a = params[:5]
    b = params[5:]
    
    # Compute P and Q
    # To avoid recomputing T, we can use a loop or unroll
    # Unrolling for speed and clarity
    
    P = a[0]*T0 + a[1]*T1 + a[2]*T2 + a[3]*T3 + a[4]*T4
    Q = b[0]*T0 + b[1]*T1 + b[2]*T2 + b[3]*T3 + b[4]*T4
    
    # Safety: prevent division by zero
    # If Q is very small, add a small epsilon with the sign of Q
    # Or just clip Q
    Q_safe = np.where(np.abs(Q) < 1e-6, 1e-6 * np.sign(Q), Q)
    # If Q is 0, sign is 0, so we need to handle 0 explicitly
    Q_safe = np.where(Q == 0, 1e-6, Q_safe)
    
    return P / Q_safe

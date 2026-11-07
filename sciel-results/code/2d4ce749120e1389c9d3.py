import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # Using recurrence: T_0(t) = 1, T_1(t) = t, T_{n}(t) = 2tT_{n-1}(t) - T_{n-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Store basis in a list or array
    # We need 10 parameters, so T_0 ... T_9
    basis = [T0, T1]
    
    if len(params) > 2:
        T2 = 2.0 * t * T1 - T0
        basis.append(T2)
        
    if len(params) > 3:
        T3 = 2.0 * t * T2 - T1
        basis.append(T3)
        
    if len(params) > 4:
        T4 = 2.0 * t * T3 - T2
        basis.append(T4)
        
    if len(params) > 5:
        T5 = 2.0 * t * T4 - T3
        basis.append(T5)
        
    if len(params) > 6:
        T6 = 2.0 * t * T5 - T4
        basis.append(T6)
        
    if len(params) > 7:
        T7 = 2.0 * t * T6 - T5
        basis.append(T7)
        
    if len(params) > 8:
        T8 = 2.0 * t * T7 - T6
        basis.append(T8)
        
    if len(params) > 9:
        T9 = 2.0 * t * T8 - T7
        basis.append(T9)
        
    # Ensure we have exactly 10 basis functions
    # If params has more, we ignore extras (shouldn't happen, task says length 10)
    # If params has fewer, we pad with 0s (shouldn't happen)
    
    # Compute prediction
    # prediction = sum(params[i] * basis[i])
    
    # Efficient computation
    pred = np.zeros_like(t)
    for i in range(min(10, len(params))):
        pred += params[i] * basis[i]
        
    return pred

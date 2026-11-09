import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a scaled, physically-motivated basis.
    
    The model assumes a form:
    a = p0 + p1*X + p2*V + p3*X^3 + p4*V^3 + p5*tanh(X) + p6*tanh(V) + p7*X*V + p8*X^2*V + p9*V^2*X
    
    Inputs are scaled to [-1, 1] to improve the conditioning of the BFGS optimizer.
    """
    # Scale inputs based on approximate ranges from the task description
    x_scale = 0.6
    v_scale = 0.3
    
    X = x / x_scale
    V = v / v_scale
    
    # Basis functions
    b0 = 1.0
    b1 = X
    b2 = V
    b3 = X**3
    b4 = V**3
    b5 = np.tanh(X)
    b6 = np.tanh(V)
    b7 = X * V
    b8 = X**2 * V
    b9 = V**2 * X
    
    # Linear combination of basis functions
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)

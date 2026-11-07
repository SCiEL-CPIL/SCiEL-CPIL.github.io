import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Nonlinear Physics-Inspired Parametric Model.
    
    Instead of a linear basis (T @ params), we use a parametric model of a 
    nonlinear oscillator (Duffing-Van der Pol hybrid) where the 10 parameters 
    control the physical constants (mass, damping, stiffness, nonlinearity, 
    forcing). 
    
    The model form:
    y = a0 + a1*x + a2*v + a3*x^3 + a4*v^3 + a5*x*v + 
        a6*sin(x) + a7*cos(x) + a8*sin(x*v) + a9*x^2*v
    
    However, to make it "nonlinear in params" and structurally distinct, 
    we can interpret the params as coefficients in a way that BFGS can optimize.
    
    Actually, the previous linear basis failed likely due to scaling or 
    ill-conditioning. A purely linear basis in x, v, x^2, etc. is a 
    linear least squares problem. BFGS is not the best tool for that if 
    the matrix is ill-conditioned. 
    But we cannot change the optimizer.
    
    Let's try a different structural form: 
    y = A * sin(B*x + C*v + D) + E * x + F * v + G * x^3 + H * v^3 + I * x*v + J
    
    This is nonlinear in A, B, C. This forces BFGS to find the frequency/phase 
    and amplitude, which might be more robust if the data is oscillatory.
    
    Parameters:
    p[0]: Amplitude of sine term
    p[1]: Frequency coefficient for x
    p[2]: Frequency coefficient for v
    p[3]: Phase shift
    p[4]: Linear x coeff
    p[5]: Linear v coeff
    p[6]: Cubic x coeff
    p[7]: Cubic v coeff
    p[8]: Cross term x*v coeff
    p[9]: Constant bias
    """
    
    p = params
    
    # Extract parameters
    A = p[0]
    B = p[1]
    C = p[2]
    D = p[3]
    E = p[4]
    F = p[5]
    G = p[6]
    H = p[7]
    I = p[8]
    J = p[9]
    
    # Compute the argument of the sine wave
    # We scale x and v slightly to keep the argument in a reasonable range 
    # for the initial guess of ones.
    # x in [-0.5, 0.5], v in [-0.2, 0.3]
    # If B, C are ~1, argument is small.
    # We want the sine term to be able to capture periodicity if it exists.
    
    arg = B * x + C * v + D
    
    # Main oscillatory component
    osc = A * np.sin(arg)
    
    # Polynomial correction terms (Duffing/Van der Pol style)
    poly = E * x + F * v + G * (x**3) + H * (v**3) + I * (x * v)
    
    # Bias
    bias = J
    
    return osc + poly + bias

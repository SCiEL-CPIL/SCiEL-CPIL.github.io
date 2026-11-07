import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid model:
    Low-order polynomial + Learnable Sinusoidal terms.
    
    Model:
    y(x) = p0 + p1*x + p2*x^2 + p3*x^3 
           + p4*sin(p5*x) 
           + p6*cos(p7*x) 
           + p8*sin(p9*x + x)  # Phase shift added
    
    Wait, p9 is phase. Let's make it cleaner.
    y(x) = p0 + p1*x + p2*x^2 + p3*x^3 
           + p4*sin(p5*x) 
           + p6*cos(p7*x) 
           + p8*sin(x + p9)
    
    Params:
    p0, p1, p2, p3: Polynomial coefficients (deg 3)
    p4, p5: Amplitude, Frequency for Sine 1
    p6, p7: Amplitude, Frequency for Cosine 1
    p8, p9: Amplitude, Phase for Sine 2 (Fixed freq 1? Or learnable?)
    
    If p9 is phase, freq is fixed at 1.
    If we want learnable freq for the 2nd sine, we need another param.
    We have 10 params.
    Poly (4) + Sin1 (2) + Cos1 (2) + Sin2 (2) = 10.
    Let's make Sin2 have learnable frequency too?
    Then we need 11 params.
    
    Let's drop x^3. Use Poly deg 2 (3 params).
    Poly (3) + Sin1 (2) + Cos1 (2) + Sin2 (2) + Const?
    p0: Const
    p1, p2: Poly x, x^2
    p3, p4: Sin1 Amp, Freq
    p5, p6: Cos1 Amp, Freq
    p7, p8: Sin2 Amp, Freq
    p9: Phase for Sin2?
    
    Total 10.
    """
    x_flat = x.flatten()
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    
    p3 = params[3]
    p4 = params[4]
    
    p5 = params[5]
    p6 = params[6]
    
    p7 = params[7]
    p8 = params[8]
    
    p9 = params[9]
    
    # Polynomial part
    y_poly = p0 + p1 * x_flat + p2 * x_flat**2
    
    # Sinusoidal parts
    # Use np.sin and np.cos
    # Ensure arguments are finite
    y_sin1 = p3 * np.sin(p4 * x_flat)
    y_cos1 = p5 * np.cos(p6 * x_flat)
    y_sin2 = p7 * np.sin(p8 * x_flat + p9)
    
    y = y_poly + y_sin1 + y_cos1 + y_sin2
    
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x_flat)
        
    return y

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Fourier series basis on [-2, 2]
    # Fundamental frequency omega_0 = pi/2
    # Basis:
    # p[0]: 1
    # p[1]: cos(pi/2 * x)
    # p[2]: sin(pi/2 * x)
    # p[3]: cos(pi * x)
    # p[4]: sin(pi * x)
    # p[5]: cos(3*pi/2 * x)
    # p[6]: sin(3*pi/2 * x)
    # p[7]: cos(2*pi * x)
    # p[8]: sin(2*pi * x)
    # p[9]: cos(5*pi/2 * x)
    
    # Precompute frequencies
    w1 = 0.5 * np.pi
    w2 = 1.0 * np.pi
    w3 = 1.5 * np.pi
    w4 = 2.0 * np.pi
    w5 = 2.5 * np.pi
    
    y_hat = (p[0] * np.ones_like(x) + 
             p[1] * np.cos(w1 * x) + 
             p[2] * np.sin(w1 * x) +
             p[3] * np.cos(w2 * x) + 
             p[4] * np.sin(w2 * x) +
             p[5] * np.cos(w3 * x) + 
             p[6] * np.sin(w3 * x) +
             p[7] * np.cos(w4 * x) + 
             p[8] * np.sin(w4 * x) +
             p[9] * np.cos(w5 * x))
             
    return y_hat

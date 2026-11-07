import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    e = strain
    t = temp
    p = params
    
    # Basis 1: Constant
    b1 = np.ones_like(e)
    
    # Basis 2-4: Polynomial strain terms to capture elastic and hardening behavior
    b2 = e
    b3 = e**2
    b4 = e**3
    
    # Basis 5-7: Temperature and its interactions with strain
    b5 = t
    b6 = e * t
    b7 = e**2 * t
    
    # Basis 8-9: Saturation terms for yield transition
    # k=0.01 for sharp yield, k=0.2 for broader transition
    k1 = 0.01
    k2 = 0.2
    b8 = e / (e + k1)
    b9 = e / (e + k2)
    
    # Basis 10: Temperature squared
    b10 = t**2
    
    sigma = (p[0] * b1 + 
             p[1] * b2 + 
             p[2] * b3 + 
             p[3] * b4 + 
             p[4] * b5 + 
             p[5] * b6 + 
             p[6] * b7 + 
             p[7] * b8 + 
             p[8] * b9 + 
             p[9] * b10)
             
    return sigma

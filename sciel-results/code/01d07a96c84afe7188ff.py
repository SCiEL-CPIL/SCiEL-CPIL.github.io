import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    e = strain
    t = temp
    p = params
    
    # Basis 1-4: Polynomial/Interaction terms
    b1 = np.ones_like(e)
    b2 = e
    b3 = t
    b4 = e * t
    
    # Basis 5-10: Hyperbolic saturation functions
    # These capture the smooth transition from elastic to plastic behavior
    # Form: e / (e + k)
    # k controls the saturation scale
    k1 = 0.01
    k2 = 0.05
    k3 = 0.1
    k4 = 0.3
    k5 = 0.5
    k6 = 1.0
    
    b5 = e / (e + k1)
    b6 = e / (e + k2)
    b7 = e / (e + k3)
    b8 = e / (e + k4)
    b9 = e / (e + k5)
    b10 = e / (e + k6)
    
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

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    e = strain
    t = temp
    p = params
    
    # Normalize to [-1, 1] for Chebyshev
    # e range approx [-0.002, 0.964] -> mid 0.481, half 0.483
    # t range approx [0.067, 1.0] -> mid 0.533, half 0.467
    
    e_norm = (e - 0.481) / 0.483
    t_norm = (t - 0.533) / 0.467
    
    # Clip to avoid huge values if slightly out of range, though training is in range
    e_norm = np.clip(e_norm, -1.1, 1.1)
    t_norm = np.clip(t_norm, -1.1, 1.1)
    
    # Chebyshev polynomials of first kind
    def T0(x): return np.ones_like(x)
    def T1(x): return x
    def T2(x): return 2 * x**2 - 1
    def T3(x): return 4 * x**3 - 3 * x
    
    # Basis functions
    # We select 10 terms that cover the interaction space
    b1 = T0(e_norm) * T0(t_norm)
    b2 = T1(e_norm) * T0(t_norm)
    b3 = T2(e_norm) * T0(t_norm)
    b4 = T1(e_norm) * T1(t_norm)
    b5 = T0(e_norm) * T1(t_norm)
    b6 = T3(e_norm) * T0(t_norm)
    b7 = T0(e_norm) * T2(t_norm)
    b8 = T2(e_norm) * T1(t_norm)
    b9 = T1(e_norm) * T2(t_norm)
    b10 = T2(e_norm) * T2(t_norm)
    
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

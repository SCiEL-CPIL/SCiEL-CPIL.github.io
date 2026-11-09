import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Polynomial basis: x^0, x^1, x^2, x^3, x^4, x^5
    x0 = np.ones_like(x)
    x1 = x
    x2 = x**2
    x3 = x**3
    x4 = x**4
    x5 = x**5
    
    # 4 RBF terms with fixed centers and widths
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    widths = 0.5
    
    rbf_terms = []
    for c in centers:
        rbf = np.exp(-((x - c)**2) / (2.0 * widths**2))
        rbf_terms.append(rbf)
    
    # Combine: 6 polynomial + 4 RBF
    y_hat = (p[0] * x0 + 
             p[1] * x1 + 
             p[2] * x2 + 
             p[3] * x3 + 
             p[4] * x4 + 
             p[5] * x5 + 
             p[6] * rbf_terms[0] + 
             p[7] * rbf_terms[1] + 
             p[8] * rbf_terms[2] + 
             p[9] * rbf_terms[3])
             
    return y_hat

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p is size 10
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Compute basis
    basis = np.zeros((10, x.size))
    basis[0] = 1.0
    basis[1] = x
    basis[2] = x**2
    basis[3] = x**3
    basis[4] = x**4
    basis[5] = np.sin(x)
    basis[6] = np.cos(x)
    
    sigma = 0.7
    basis[7] = np.exp(-0.5 * ((x - 1.0) / sigma)**2)
    basis[8] = np.exp(-0.5 * ((x - 0.0) / sigma)**2)
    basis[9] = np.exp(-0.5 * ((x + 1.0) / sigma)**2)
    
    return basis.T @ p

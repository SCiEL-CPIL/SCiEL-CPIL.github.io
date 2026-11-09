import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Basis functions
    b0 = np.ones_like(strain)
    b1 = strain
    b2 = temp
    b3 = temp * strain
    
    k1 = 5.0
    k2 = 10.0
    k3 = 30.0
    
    t1 = np.tanh(k1 * strain)
    t2 = np.tanh(k2 * strain)
    t3 = np.tanh(k3 * strain)
    
    b4 = t1
    b5 = temp * t1
    b6 = t2
    b7 = temp * t2
    b8 = t3
    b9 = temp * t3
    
    # Construct design matrix
    # Shape (N, 10)
    X = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Prediction
    return X @ params

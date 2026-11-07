import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
       x = np.asarray(x, dtype=np.float64)
       params = np.asarray(params, dtype=np.float64)
       
       # Basis functions
       f0 = np.ones_like(x)
       f1 = x
       f2 = x**2
       f3 = x**3
       f4 = x**4
       f5 = np.sin(x)
       f6 = np.cos(x)
       f7 = np.exp(-x**2)
       f8 = np.tanh(x)
       f9 = 1.0 / (1.0 + x**2)
       
       features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
       
       # Linear combination
       result = features @ params
       
       # Ensure finite output
       if not np.all(np.isfinite(result)):
           return np.zeros_like(x)
           
       return result

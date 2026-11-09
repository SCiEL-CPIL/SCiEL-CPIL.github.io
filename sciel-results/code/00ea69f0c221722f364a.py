import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
       x = np.asarray(x, dtype=np.float64)
       y = np.asarray(y, dtype=np.float64)
       z = np.asarray(z, dtype=np.float64)
       p = np.asarray(params, dtype=np.float64)
       
       # Full quadratic polynomial in 3 variables (10 coefficients):
       # p0 + p1*x + p2*y + p3*z + p4*x*y + p5*y*z + p6*x*z + p7*x^2 + p8*y^2 + p9*z^2
       
       result = (p[0] + p[1]*x + p[2]*y + p[3]*z +
                 p[4]*x*y + p[5]*y*z + p[6]*x*z +
                 p[7]*x*x + p[8]*y*y + p[9]*z*z)
       
       if not np.all(np.isfinite(result)):
           result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
           
       return result

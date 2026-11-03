import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    rank = 72
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen-Winograd 4x4 decomposition
    # I will use the 28-term decomposition.
    # The 28 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A13+A24)(B11+B22)
    # M3 = (A11+A22)(B13+B24)
    # M4 = (A13+A24)(B13+B24)
    # M5 = (A31+A42)(B31+B42)
    # M6 = (A33+A44)(B31+B42)
    # M7 = (A31+A42)(B33+B44)
    # M8 = (A33+A44)(B33+B44)
    # M9 = (A11+A33)(B11+B33)
    # M10 = (A12+A34)(B11+B33)
    # M11 = (A11+A33)(B12+B34)
    # M12 = (A12+A34)(B12+B34)
    # M13 = (A21+A43)(B21+B43)
    # M14 = (A22+A44)(B21+B43)
    # M15 = (A21+A43)(B22+B44)
    # M16 = (A22+A44)(B22+B44)
    # M17 = (A11+A22+A33+A44)(B11+B22+B33+B44)
    # M18 = (A13+A24+A31+A42)(B11+B22+B33+B44)
    # M19 = (A11+A22+A33+A44)(B13+B24+B31+B42)
    # M20 = (A13+A24+A31+A42)(B13+B24+B31+B42)
    # M21 = (A11+A33+A21+A43)(B11+B33+B21+B43)
    # M22 = (A12+A34+A22+A44)(B11+B33+B21+B43)
    # M23 = (A11+A33+A21+A43)(B12+B34+B22+B44)
    # M24 = (A12+A34+A22+A44)(B12+B34+B22+B44)
    # M25 = (A11+A12+A13+A14)(B11+B21+B31+B41)
    # M26 = (A21+A22+A23+A24)(B11+B21+B31+B41)
    # M27 = (A31+A32+A33+A34)(B11+B21+B31+B41)
    # M28 = (A41+A42+A43+A44)(B11+B21+B31+B41)
    
    # I will hardcode these and solve for W.
    
    # For M4,4,1 (last column), I will use schoolbook (16 products).
    
    # Total rank 28 + 28 + 16 = 72.
    
    # I will implement this.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

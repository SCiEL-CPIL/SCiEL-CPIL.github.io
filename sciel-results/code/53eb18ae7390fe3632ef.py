import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration using a Chebyshev polynomial basis in normalized x and v.
    
    Chebyshev polynomials provide a well-conditioned basis that reduces
    parameter correlation compared to monomials, allowing BFGS to converge
    to better local minima.
    
    We use a 2D Chebyshev expansion: sum_{i,j} c_{ij} T_i(x_n) T_j(v_n)
    with total terms <= 10.
    
    Terms: T0*T0, T1*T0, T0*T1, T2*T0, T0*T2, T1*T1, T3*T0, T0*T3, T2*T1, T1*T2
    (up to degree 3 in one variable, degree 2 in the other, or mixed)
    
    Normalization: x_n = x / 0.6, v_n = v / 0.3 to map to roughly [-1, 1].
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to roughly [-1, 1]
    x_n = x / 0.6
    v_n = v / 0.3
    
    # Clip to [-1, 1] for numerical stability of Chebyshev recurrence
    x_n = np.clip(x_n, -1.0, 1.0)
    v_n = np.clip(v_n, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind via recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    # For x
    T0_x = np.ones_like(x_n)
    T1_x = x_n
    T2_x = 2.0 * x_n * T1_x - T0_x
    T3_x = 2.0 * x_n * T2_x - T1_x
    
    # For v
    T0_v = np.ones_like(v_n)
    T1_v = v_n
    T2_v = 2.0 * v_n * T1_v - T0_v
    T3_v = 2.0 * v_n * T2_v - T1_v
    
    # Build the 2D Chebyshev basis (10 terms)
    # Indexing: (deg_x, deg_v)
    basis = np.stack([
        T0_x * T0_v,  # p0: constant
        T1_x * T0_v,  # p1: x
        T0_x * T1_v,  # p2: v
        T2_x * T0_v,  # p3: x^2
        T0_x * T2_v,  # p4: v^2
        T1_x * T1_v,  # p5: x*v
        T3_x * T0_v,  # p6: x^3
        T0_x * T3_v,  # p7: v^3
        T2_x * T1_v,  # p8: x^2*v
        T1_x * T2_v,  # p9: x*v^2
    ], axis=-1)
    
    # Linear combination
    return basis @ params

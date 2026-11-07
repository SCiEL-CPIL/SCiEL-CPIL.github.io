import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Chebyshev Polynomial Basis.
    
    Instead of a physics-inspired model with sine/cosine terms, we use a pure 
    Chebyshev polynomial basis in the scaled inputs. Chebyshev polynomials have 
    superior approximation properties for bounded domains compared to monomials, 
    reducing oscillation artifacts and improving BFGS convergence stability.
    
    The inputs are scaled to [-1, 1] to match the natural domain of Chebyshev 
    polynomials. We use a tensor product basis of Chebyshev polynomials in x and v.
    
    Parameters:
    p[0]: T_0(x) * T_0(v) = 1
    p[1]: T_1(x) * T_0(v) = x_scaled
    p[2]: T_0(x) * T_1(v) = v_scaled
    p[3]: T_2(x) * T_0(v) = 2*x_scaled^2 - 1
    p[4]: T_0(x) * T_2(v) = 2*v_scaled^2 - 1
    p[5]: T_1(x) * T_1(v) = x_scaled * v_scaled
    p[6]: T_3(x) * T_0(v) = 4*x_scaled^3 - 3*x_scaled
    p[7]: T_0(x) * T_3(v) = 4*v_scaled^3 - 3*v_scaled
    p[8]: T_2(x) * T_1(v) = (2*x_scaled^2 - 1) * v_scaled
    p[9]: T_1(x) * T_2(v) = x_scaled * (2*v_scaled^2 - 1)
    
    This is a linear basis, so BFGS should converge to the global minimum 
    (linear least squares solution) if the basis is well-conditioned.
    """
    
    p = params
    
    # Scale inputs to [-1, 1] for Chebyshev polynomials
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Use symmetric scaling to keep the domain centered
    x_scale = 0.5728  # max abs value
    v_scale = 0.2821  # max abs value
    
    x_s = x / x_scale
    v_s = v / v_scale
    
    # Clip to [-1, 1] to handle any slight out-of-range values
    x_s = np.clip(x_s, -1.0, 1.0)
    v_s = np.clip(v_s, -1.0, 1.0)
    
    # Compute Chebyshev polynomials
    T0_x = np.ones_like(x_s)
    T1_x = x_s
    T2_x = 2 * x_s**2 - 1
    T3_x = 4 * x_s**3 - 3 * x_s
    
    T0_v = np.ones_like(v_s)
    T1_v = v_s
    T2_v = 2 * v_s**2 - 1
    T3_v = 4 * v_s**3 - 3 * v_s
    
    # Tensor product basis
    basis = np.stack([
        T0_x * T0_v,  # p[0]
        T1_x * T0_v,  # p[1]
        T0_x * T1_v,  # p[2]
        T2_x * T0_v,  # p[3]
        T0_x * T2_v,  # p[4]
        T1_x * T1_v,  # p[5]
        T3_x * T0_v,  # p[6]
        T0_x * T3_v,  # p[7]
        T2_x * T1_v,  # p[8]
        T1_x * T2_v   # p[9]
    ], axis=-1)
    
    # Linear combination
    return np.sum(basis * p, axis=-1)

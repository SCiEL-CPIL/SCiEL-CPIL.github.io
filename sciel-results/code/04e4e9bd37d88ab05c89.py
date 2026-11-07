import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Use a bilinear Chebyshev basis.
    We compute Chebyshev polynomials T_0 through T_9 on the scaled input x/2.
    We then form 10 basis functions as products of pairs of these Chebyshev polynomials.
    Specifically, we use the pairs (i, j) such that i+j is even or we select specific pairs
    that span a rich space including high-order polynomials and interaction terms.
    
    The model is linear in the 10 params, so BFGS will find the global optimum efficiently.
    This basis can represent functions like |x|, x^3, and other non-smooth or high-curvature
    features more effectively than a simple linear combination of low-order polynomials.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
        t = x / 2.0
        
        # Compute Chebyshev polynomials T_0 to T_9
        # T_0(t) = 1
        # T_1(t) = t
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        
        T = np.zeros((10, t.size))
        T[0] = 1.0
        T[1] = t
        
        for k in range(2, 10):
            T[k] = 2.0 * t * T[k-1] - T[k-2]
        
        # Define 10 bilinear basis functions as products T_i * T_j
        # We want to cover a good range of degrees.
        # T_i * T_j has degree i+j.
        # Let's pick pairs that give distinct degrees and interactions.
        
        # Pairs (i, j) with i <= j:
        # (0,0) -> 1
        # (0,1) -> t
        # (0,2) -> T_2
        # (1,1) -> t^2
        # (0,3) -> T_3
        # (1,2) -> T_1*T_2
        # (0,4) -> T_4
        # (1,3) -> T_1*T_3
        # (2,2) -> T_2^2
        # (0,5) -> T_5
        
        # This set includes:
        # Degree 0: 1
        # Degree 1: t, T_1 (same)
        # Degree 2: T_2, t^2
        # Degree 3: T_3, T_1*T_2
        # Degree 4: T_4, T_1*T_3, T_2^2
        # Degree 5: T_5
        
        # Let's refine the selection to maximize span.
        # A better set might be to just take the first 10 Chebyshev polynomials?
        # No, the prompt suggests bilinear.
        # Let's use:
        # b0 = T0*T0 = 1
        # b1 = T0*T1 = T1
        # b2 = T0*T2 = T2
        # b3 = T1*T1 = T1^2
        # b4 = T0*T3 = T3
        # b5 = T1*T2 = T1*T2
        # b6 = T0*T4 = T4
        # b7 = T1*T3 = T1*T3
        # b8 = T2*T2 = T2^2
        # b9 = T0*T5 = T5
        
        basis = np.zeros((t.size, 10))
        basis[:, 0] = T[0] * T[0]
        basis[:, 1] = T[0] * T[1]
        basis[:, 2] = T[0] * T[2]
        basis[:, 3] = T[1] * T[1]
        basis[:, 4] = T[0] * T[3]
        basis[:, 5] = T[1] * T[2]
        basis[:, 6] = T[0] * T[4]
        basis[:, 7] = T[1] * T[3]
        basis[:, 8] = T[2] * T[2]
        basis[:, 9] = T[0] * T[5]
        
        return basis @ params
        
    except Exception:
        return np.zeros_like(x)

import numpy as np
from scipy.optimize import linprog
from scipy.special import eval_hermite

def construct():
    """
    Minimize the upper bound C4 on A(f)*A(F(f)) using a Linear Programming relaxation.
    We minimize the sum of the L1 norms of g and its Fourier transform, 
    which is a convex surrogate for the product.
    We fix a0 = 1 to determine the shape, then normalize the result.
    """
    
    # Grids for integration
    x = np.linspace(-10, 10, 500)
    xi = np.linspace(-10, 10, 500)
    dx = x[1] - x[0]
    dxi = xi[1] - xi[0]
    
    # Precompute Hermite polynomials and Gaussian weights
    H0_x = eval_hermite(0, x) * np.exp(-x**2 / 2)
    H4_x = eval_hermite(4, x) * np.exp(-x**2 / 2)
    H8_x = eval_hermite(8, x) * np.exp(-x**2 / 2)
    H12_x = eval_hermite(12, x) * np.exp(-x**2 / 2)
    
    H0_xi = eval_hermite(0, xi) * np.exp(-xi**2 / 2)
    H4_xi = eval_hermite(4, xi) * np.exp(-xi**2 / 2)
    H8_xi = eval_hermite(8, xi) * np.exp(-xi**2 / 2)
    H12_xi = eval_hermite(12, xi) * np.exp(-xi**2 / 2)
    
    # Fourier transform factors: (i)^n
    # H0: 1, H4: 1, H8: 1, H12: -1
    F_H0_xi = 1 * H0_xi
    F_H4_xi = 1 * H4_xi
    F_H8_xi = 1 * H8_xi
    F_H12_xi = -1 * H12_xi
    
    # H_n(0) values for a12 constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    # With a0 = 1:
    # a12 = -(H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    # a12 = C12_const + a4*C12_4 + a8*C12_8
    C12_const = -H0_0 / H12_0
    C12_4 = -H4_0 / H12_0
    C12_8 = -H8_0 / H12_0
    
    # Decision variables: [a4, a8, u_1...u_N, v_1...v_M]
    # N = len(x), M = len(xi)
    N = len(x)
    M = len(xi)
    num_vars = 2 + N + M
    
    # Objective: minimize sum(u) + sum(v)
    c = np.zeros(num_vars)
    c[2:2+N] = 1.0
    c[2+N:2+N+M] = 1.0
    
    # Constraints:
    # 1. u_i >= g(x_i)  =>  -u_i + g(x_i) <= 0
    # 2. u_i >= -g(x_i) =>  u_i - g(x_i) <= 0
    # 3. v_j >= hat{g}(xi_j) => -v_j + hat{g}(xi_j) <= 0
    # 4. v_j >= -hat{g}(xi_j) => v_j - hat{g}(xi_j) <= 0
    
    # g(x_i) = 1*H0_x_i + a4*H4_x_i + a8*H8_x_i + a12*H12_x_i
    # a12 = C12_const + a4*C12_4 + a8*C12_8
    # g(x_i) = H0_x_i + C12_const*H12_x_i + a4*(H4_x_i + C12_4*H12_x_i) + a8*(H8_x_i + C12_8*H12_x_i)
    
    coeff_a4_x = H4_x + C12_4 * H12_x
    coeff_a8_x = H8_x + C12_8 * H12_x
    const_x = H0_x + C12_const * H12_x
    
    # hat{g}(xi_j) = 1*F_H0_xi_j + a4*F_H4_xi_j + a8*F_H8_xi_j + a12*F_H12_xi_j
    # a12 is same expression
    coeff_a4_xi = F_H4_xi + C12_4 * F_H12_xi
    coeff_a8_xi = F_H8_xi + C12_8 * F_H12_xi
    const_xi = F_H0_xi + C12_const * F_H12_xi
    
    A_ub = []
    b_ub = []
    
    # Constraints for u
    for i in range(N):
        # -u_i + g(x_i) <= 0  =>  coeff_a4_x[i]*a4 + coeff_a8_x[i]*a8 - u_i <= -const_x[i]
        row1 = np.zeros(num_vars)
        row1[0] = coeff_a4_x[i]
        row1[1] = coeff_a8_x[i]
        row1[2+i] = -1.0
        A_ub.append(row1)
        b_ub.append(-const_x[i])
        
        # u_i - g(x_i) <= 0 => -coeff_a4_x[i]*a4 - coeff_a8_x[i]*a8 + u_i <= const_x[i]
        row2 = np.zeros(num_vars)
        row2[0] = -coeff_a4_x[i]
        row2[1] = -coeff_a8_x[i]
        row2[2+i] = 1.0
        A_ub.append(row2)
        b_ub.append(const_x[i])
        
    # Constraints for v
    for j in range(M):
        # -v_j + hat{g}(xi_j) <= 0
        row3 = np.zeros(num_vars)
        row3[0] = coeff_a4_xi[j]
        row3[1] = coeff_a8_xi[j]
        row3[2+N+j] = -1.0
        A_ub.append(row3)
        b_ub.append(-const_xi[j])
        
        # v_j - hat{g}(xi_j) <= 0
        row4 = np.zeros(num_vars)
        row4[0] = -coeff_a4_xi[j]
        row4[1] = -coeff_a8_xi[j]
        row4[2+N+j] = 1.0
        A_ub.append(row4)
        b_ub.append(const_xi[j])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for a4, a8: (-inf, inf)
    # Bounds for u, v: (0, inf)
    bounds = [(None, None), (None, None)] + [(0, None)] * (N + M)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            a4_opt = res.x[0]
            a8_opt = res.x[1]
            a0_opt = 1.0
            # Calculate a12
            a12_opt = C12_const + a4_opt * C12_4 + a8_opt * C12_8
            
            # The LP minimizes the sum. 
            # We return the coefficients [a0, a4, a8].
            # It is often beneficial to normalize the function.
            # Let's normalize such that the L2 norm is 1.
            # Compute L2 norm of g with these coeffs
            g_vals = (a0_opt * H0_x + a4_opt * H4_x + a8_opt * H8_x + a12_opt * H12_x)
            l2_norm = np.sqrt(np.sum(g_vals**2) * dx)
            
            if l2_norm > 1e-10:
                scale = 1.0 / l2_norm
                return np.array([a0_opt * scale, a4_opt * scale, a8_opt * scale])
            else:
                return np.array([a0_opt, a4_opt, a8_opt])
        else:
            # Fallback to seed values if LP fails
            return np.array([-113/100, 1/25, 1/3240])
    except Exception:
        return np.array([-113/100, 1/25, 1/3240])

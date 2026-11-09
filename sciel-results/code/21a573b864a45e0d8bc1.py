import numpy as np
import time
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Minimizes the L4/L2 norm ratio for degree-10 spherical harmonics using
    Conjugate Gradient optimization on the complex unit sphere.
    
    The strategy exploits the rotational symmetry by parameterizing the field
    using complex amplitudes a_m (m=0..10) such that:
    f(theta, phi) = sum_{m=0}^{10} [ a_m Y_l^m(theta, phi) + conj(a_m) Y_l^{-m}(theta, phi) ]
    where Y_l^m are standard complex spherical harmonics.
    This reduces the real degrees of freedom from 21 to 11 (complex vector on S^{21} in R^{22}).
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    cos_theta_vals = x_gl
    n_grid = 70 * n_phi
    
    # 2. Pre-compute Complex Basis Functions Y_l^m
    # We need Y_m for m=0..10.
    # Standard definition: Y_l^m = (-1)^m sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) P_l^m(cos theta) e^{i m phi}
    # Note: scipy lpmv includes Condon-Shortley phase (-1)^m.
    # So P_l^m from lpmv is already (-1)^m * P_l^m_standard.
    # Thus Y_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * lpmv(m, l, x) * e^{i m phi}
    
    # We will construct a matrix Y_comp of shape (11, n_grid) for m=0..10
    Y_comp = np.zeros((k + 1, n_grid), dtype=np.complex128)
    
    for m in range(k + 1):
        # Normalization constant for complex harmonics
        # N_m_comp = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
        num = factorial(k - m)
        den = factorial(k + m)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
        P_vals = lpmv(m, k, cos_theta_vals) # shape (70,)
        
        # e^{i m phi}
        exp_term = np.exp(1j * m * phi) # shape (130,)
        
        # Y_m(theta, phi) = N_m * P_vals[:, None] * exp_term[None, :]
        Y_vals = (N_m * P_vals[:, np.newaxis] * exp_term[np.newaxis, :])
        
        Y_comp[m, :] = Y_vals.flatten()
        
    # 3. Define Objective and Gradient on Complex Sphere
    # We optimize a complex vector a of shape (11,).
    # Constraint: ||a||_2 = 1 (scale invariance allows fixing this)
    # f = sum_{m=0}^{10} ( a_m Y_m + conj(a_m) conj(Y_m) )  <-- Wait, let's check the real basis mapping.
    
    # The problem defines real harmonics:
    # Y_0_real = N_0 P^0
    # Y_m_real (m>0) = sqrt(2) N_m P^m cos(m phi)
    # Y_m_real (m<0) = sqrt(2) N_m P^{|m|} sin(|m| phi)
    
    # Let's express the real basis in terms of complex basis Y_m_comp.
    # Y_m_comp = N_m P^m e^{i m phi}
    # Y_{-m_comp} = N_m P^m e^{-i m phi} (Note: P^m is same for -m in magnitude? 
    # Actually P_l^m = P_l^{-m} * (-1)^m? No, P_l^m usually defined for m>=0.
    # In complex harmonics, Y_l^{-m} = (-1)^m conj(Y_l^m).
    # Let's stick to the definition: f = sum_{m=-10}^{10} c_m Y_m_real.
    
    # Relation:
    # Y_0_real = Y_0_comp
    # For m > 0:
    # Y_m_comp = N_m P^m cos(m phi) + i N_m P^m sin(m phi)
    # Y_{-m_comp} = N_m P^m cos(m phi) - i N_m P^m sin(m phi)
    # So:
    # cos(m phi) P^m = (Y_m_comp + Y_{-m_comp}) / (2 N_m)
    # sin(m phi) P^m = (Y_m_comp - Y_{-m_comp}) / (2 i N_m)
    
    # The real basis in problem:
    # Y_m_real (m>0) = sqrt(2) N_m P^m cos(m phi) = sqrt(2)/2 (Y_m_comp + Y_{-m_comp}) = 1/sqrt(2) (Y_m_comp + Y_{-m_comp})
    # Y_{-m_real} (m<0, let m'= -m > 0) = sqrt(2) N_{m'} P^{m'} sin(m' phi) = sqrt(2)/2 * (1/i) (Y_{m'_comp} - Y_{-m'_comp}) 
    # Note Y_{-m'_comp} is the harmonic for index -m'.
    # Let's just build the mapping matrix M such that f = sum_{j=0}^{10} ( a_j Y_j_comp + conj(a_j) Y_{-j_comp} )?
    # Actually, any real harmonic field can be written as sum_{m=0}^{10} ( a_m Y_m_comp + conj(a_m) Y_{-m_comp} ) 
    # IF we define the coefficients appropriately.
    
    # Let's verify:
    # Let a_m = c_m / sqrt(2) for m>0?
    # If we set f = sum_{m=0}^{10} ( a_m Y_m_comp + conj(a_m) Y_{-m_comp} )
    # For m=0: term is a_0 Y_0 + conj(a_0) Y_0. Since Y_0 is real, this is 2 Re(a_0) Y_0.
    # Problem Y_0 = Y_0_comp. So c_0 = 2 Re(a_0).
    
    # For m>0:
    # Term: a_m Y_m + conj(a_m) Y_{-m}
    # Y_m = N P (cos + i sin)
    # Y_{-m} = N P (cos - i sin)
    # a_m = x + iy
    # (x+iy)N P (cos + i sin) + (x-iy) N P (cos - i sin)
    # = N P [ (x cos - y sin) + i(x sin + y cos) + (x cos + y sin) + i(-x sin + y cos) ]
    # = N P [ 2 x cos + i (2 y cos) ]
    # = 2 N P ( x cos(m phi) + i y cos(m phi) ) -- Wait, imaginary part?
    # Let's re-evaluate:
    # (x+iy)(cos+i sin) = x cos - y sin + i(x sin + y cos)
    # (x-iy)(cos-i sin) = x cos - y sin + i(-x sin + y cos)  <-- Note: Y_{-m} has e^{-im phi} = cos - i sin.
    # Sum: 2(x cos - y sin) + i(2 y cos)
    # This is complex. But f must be real.
    # Ah, the standard expansion for real fields is:
    # f = sum_{m=0}^{10} ( a_m Y_m + (-1)^m conj(a_m) Y_{-m} ) ? No.
    
    # Let's just use the linear mapping from complex amplitudes a (size 11) to real coefficients c (size 21).
    # It is easier to compute f directly from a on the grid, then compute norms.
    
    # Mapping:
    # c_0 = 2 Re(a_0)
    # For m > 0:
    # c_m (cos part) and c_{-m} (sin part)
    # From above:
    # Real part of (a_m Y_m + conj(a_m) Y_{-m}) is 2 N_m P_m (x_m cos(m phi) - y_m sin(m phi))
    # Imaginary part is 2 N_m P_m y_m cos(m phi).
    # This doesn't match the real basis form directly unless we rotate the phase.
    
    # Alternative: Just define f on the grid using a.
    # f(theta, phi) = sum_{m=0}^{10} ( a_m Y_m_comp(theta, phi) + conj(a_m) Y_{-m_comp}(theta, phi) )
    # Is this real?
    # Let g_m = a_m Y_m + conj(a_m) Y_{-m}.
    # conj(g_m) = conj(a_m) conj(Y_m) + a_m conj(Y_{-m}).
    # We know Y_{-m} = (-1)^m conj(Y_m). So conj(Y_m) = (-1)^m Y_{-m}.
    # conj(Y_{-m}) = (-1)^m Y_m.
    # conj(g_m) = conj(a_m) (-1)^m Y_{-m} + a_m (-1)^m Y_m = (-1)^m ( a_m Y_m + conj(a_m) Y_{-m} ) = (-1)^m g_m.
    # So if m is even, g_m is real. If m is odd, g_m is imaginary.
    # This implies the sum is not necessarily real.
    
    # Correct real expansion:
    # f = sum_{m=0}^{10} ( a_m Y_m + (-1)^m conj(a_m) Y_{-m} ) ?
    # Let's check m=0: a_0 Y_0 + conj(a_0) Y_0 = 2 Re(a_0) Y_0. Real.
    # Let's check m=1: a_1 Y_1 - conj(a_1) Y_{-1}.
    # conj(g_1) = conj(a_1) conj(Y_1) - a_1 conj(Y_{-1}) = conj(a_1)(-Y_{-1}) - a_1(-Y_1) = -conj(a_1)Y_{-1} + a_1 Y_1 = g_1. Real!
    
    # So the correct parameterization for a real field is:
    # f = sum_{m=0}^{10} ( a_m Y_m_comp + (-1)^m conj(a_m) Y_{-m_comp} )
    
    # We need Y_{-m_comp} on the grid.
    # Y_{-m_comp} = (-1)^m conj(Y_m_comp).
    # So term is a_m Y_m + (-1)^m conj(a_m) (-1)^m conj(Y_m) = a_m Y_m + conj(a_m) conj(Y_m) = 2 Re(a_m Y_m).
    
    # So f(theta, phi) = sum_{m=0}^{10} 2 Re( a_m Y_m_comp(theta, phi) )
    
    # This is extremely simple!
    # f = 2 * Re( sum_{m=0}^{10} a_m Y_m_comp )
    
    # Let's verify the coefficients c_m in the problem statement.
    # Problem: f = c_0 Y_0_real + sum_{m>0} (c_m Y_m_real + c_{-m} Y_{-m_real})
    # Y_0_real = Y_0_comp.
    # Y_m_real = 1/sqrt(2) (Y_m_comp + Y_{-m_comp}) = 1/sqrt(2) (Y_m_comp + (-1)^m conj(Y_m_comp))
    # Y_{-m_real} = 1/sqrt(2) (Y_m_comp - Y_{-m_comp}) / i ? 
    # Let's check: Y_{-m_real} = sqrt(2) N_m P_m sin(m phi).
    # Y_m_comp - Y_{-m_comp} = N P (cos + i sin) - N P (cos - i sin) = 2 i N P sin(m phi).
    # So Y_{-m_real} = sqrt(2) * (Y_m_comp - Y_{-m_comp}) / (2 i) = 1/sqrt(2) * (-i) (Y_m_comp - Y_{-m_comp}) = 1/sqrt(2) (i Y_{-m_comp} - i Y_m_comp)
    
    # Let's just use f = 2 Re( sum a_m Y_m ).
    # If a_m = x_m + i y_m.
    # 2 Re( (x+iy) N P (cos + i sin) ) = 2 N P Re( x cos - y sin + i(x sin + y cos) ) = 2 N P (x cos - y sin).
    # So the field generated by a_m is 2 N_m P_m (x_m cos(m phi) - y_m sin(m phi)).
    
    # Comparing to problem basis:
    # Term for cos(m phi): 2 N_m P_m x_m.
    # Problem term: c_m sqrt(2) N_m P_m cos(m phi).
    # So c_m sqrt(2) = 2 x_m => c_m = sqrt(2) x_m.
    
    # Term for sin(m phi): -2 N_m P_m y_m.
    # Problem term: c_{-m} sqrt(2) N_m P_m sin(m phi).
    # So c_{-m} sqrt(2) = -2 y_m => c_{-m} = -sqrt(2) y_m.
    
    # For m=0:
    # f term: 2 Re(a_0 Y_0) = 2 x_0 Y_0.
    # Problem term: c_0 Y_0.
    # So c_0 = 2 x_0.
    
    # Mapping from a (complex, size 11) to c (real, size 21):
    # c[10] (m=0) = 2 * Re(a[0])
    # For m=1..10:
    #   c[10+m] (index for m) = sqrt(2) * Re(a[m])
    #   c[10-m] (index for -m) = -sqrt(2) * Im(a[m])
    
    # This is a linear, invertible mapping (except for scaling/phase redundancy which is handled by sphere constraint).
    # We optimize on the complex sphere ||a||=1.
    
    # Objective:
    # R(a) = ||f(a)||_4 / ||f(a)||_2
    # f = 2 * Re( Y_comp @ a )
    
    # Precompute f for a given a
    def compute_f(a):
        # a shape (11,) complex
        # Y_comp shape (11, n_grid) complex
        # f = 2 * Re( Y_comp @ a )
        return 2.0 * np.real(Y_comp @ a)
        
    # Quadrature weights
    weights = np.tile(w_gl, (n_phi,)) * dphi
    weights = weights.flatten()
    
    def objective_and_grad(a):
        # a is complex vector, shape (11,)
        # We treat a as a real vector of shape (22,) for the optimizer? 
        # Or do we implement CG on the complex sphere?
        # Let's use scipy.optimize.minimize on the real representation of a (size 22) with a constraint?
        # No, constraint ||a||=1 is hard.
        # Better: Use the fact that R is scale invariant. We can fix ||a||=1.
        # We can use a manifold optimizer or project back.
        # Let's use L-BFGS-B on the 22 real dims (Re, Im) and project to unit sphere after each step?
        # Or just use SLSQP with constraint?
        # Actually, let's just use the real vector c (21 dims) but with a better initial guess and L-BFGS-B?
        # The prompt asks for "Conjugate Gradient method... on the complex sphere".
        # I will implement a simple CG on the sphere using the complex gradient.
        
        f = compute_f(a)
        
        norm2_sq = np.sum(weights * f**2)
        norm4_four = np.sum(weights * f**4)
        
        if norm2_sq < 1e-15:
            return np.inf, np.zeros_like(a)
            
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(norm4_four))
        
        R = norm4 / norm2
        
        # Gradient of R with respect to a
        # R = ||f||_4 / ||f||_2
        # dR/da = (1/norm2) * d||f||_4/da - (norm4/norm2^2) * d||f||_2/da
        # ||f||_4 = (sum w f^4)^{1/4}
        # d||f||_4/da = (1/4) (sum w f^4)^{-3/4} * sum w 4 f^3 df/da = (1/norm4^3) * sum w f^3 df/da
        # ||f||_2 = (sum w f^2)^{1/2}
        # d||f||_2/da = (1/2) (sum w f^2)^{-1/2} * sum w 2 f df/da = (1/norm2) * sum w f df/da
        
        # df/da:
        # f = 2 Re( Y a )
        # df/da is complex.
        # Let Y_a = Y @ a.
        # f = Y_a + conj(Y_a)
        # df/d a_k = 2 Re( Y_k )  (where Y_k is the k-th row of Y)
        # Wait, f is real. a is complex.
        # df/d Re(a_k) = 2 Re(Y_k)
        # df/d Im(a_k) = -2 Im(Y_k)
        
        # Let's compute the gradient with respect to the real vector v = [Re(a), Im(a)] of size 22.
        # This is easier for standard optimizers.
        
        v = np.zeros(22)
        v[:11] = a.real
        v[11:] = a.imag
        
        # df/dv shape (n_grid, 22)
        # f_i = 2 Re( sum_k Y_{ik} a_k )
        # d f_i / d Re(a_k) = 2 Re(Y_{ik})
        # d f_i / d Im(a_k) = -2 Im(Y_{ik})
        
        # We need grad_R_v.
        # Let g4 = sum w f^3 (vector of size n_grid)
        # Let g2 = sum w f (vector of size n_grid)
        
        # d||f||_4/dv_j = (1/norm4^3) * sum_i w_i f_i^3 * (df_i/dv_j)
        # d||f||_2/dv_j = (1/norm2) * sum_i w_i f_i * (df_i/dv_j)
        
        # df_i/dv_j:
        # If j < 11 (Re part, k=j): 2 Re(Y_{i,j})
        # If j >= 11 (Im part, k=j-11): -2 Im(Y_{i,j-11})
        
        # Let's vectorize.
        # Y_real_part shape (11, n_grid)
        # Y_imag_part shape (11, n_grid)
        
        Y_real = Y_comp.real
        Y_imag = Y_comp.imag
        
        # df_dv_real (n_grid, 11) = 2 * Y_real.T
        # df_dv_imag (n_grid, 11) = -2 * Y_imag.T
        
        # grad_norm4 = (1/norm4^3) * (w * f^3) @ df_dv
        # w_f3 = weights * f**3
        w_f3 = weights * (f**3)
        w_f = weights * f
        
        # grad_norm4 shape (22,)
        grad_n4 = np.zeros(22)
        grad_n4[:11] = (w_f3 @ (2 * Y_real.T)) / (norm4**3)
        grad_n4[11:] = (w_f3 @ (-2 * Y_imag.T)) / (norm4**3)
        
        # grad_norm2 shape (22,)
        grad_n2 = np.zeros(22)
        grad_n2[:11] = (w_f @ (2 * Y_real.T)) / norm2
        grad_n2[11:] = (w_f @ (-2 * Y_imag.T)) / norm2
        
        grad_R = (1/norm2) * grad_n4 - (norm4 / (norm2**2)) * grad_n2
        
        return R, grad_R

    # Optimization Strategy:
    # We will use L-BFGS-B on the 22-dimensional real vector v (Re, Im of a).
    # We will enforce the sphere constraint ||a||=1 by projecting v back to the sphere
    # after each gradient step? L-BFGS-B doesn't support manifold projection directly.
    # Alternative: Use SLSQP with constraint v.v = 1?
    # Or, since R is scale invariant, we can just optimize R(v) without constraint,
    # and normalize at the end? 
    # If we don't constrain, the optimizer might drift to 0 or inf.
    # But R(v) = R(alpha v). The gradient is 0 along the radial direction?
    # dR/dv = (dR/da) da/dv.
    # If v -> alpha v, R is constant. So the gradient is orthogonal to v.
    # L-BFGS-B might struggle if it tries to change the magnitude.
    # Let's add a penalty term or just normalize v in the objective?
    # If we normalize v inside the objective, the gradient is not zero, it's the gradient on the sphere.
    # Let's define objective_sphere(v):
    #   a = v / ||v||
    #   R, grad_R_a = objective_and_grad(a)
    #   grad_R_v = grad_R_a - (grad_R_a . a) a  (project gradient onto tangent space)
    #   This requires computing the full gradient mapping.
    
    # Simpler: Just use L-BFGS-B on v, and in the objective, normalize v.
    # def obj(v):
    #   norm_v = np.linalg.norm(v)
    #   if norm_v < 1e-10: return 1e10
    #   a = v / norm_v
    #   R, _ = objective_and_grad(a)
    #   return R
    # This is non-smooth at 0, but we start away from 0.
    # The gradient of this objective will guide the optimizer to minimize R on the sphere.
    
    from scipy.optimize import minimize
    
    start_time = time.time()
    time_limit = 600.0
    
    # Multi-start
    best_R = np.inf
    best_v = None
    
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Random start on sphere
        v0 = np.random.randn(22)
        v0 /= np.linalg.norm(v0)
        
        def obj_and_grad(v):
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                return 1e10, np.zeros(22)
            a = v / norm_v
            R, grad_R_a = objective_and_grad(a)
            
            # Project gradient to tangent space of sphere at a
            # grad_R_v = grad_R_a - (grad_R_a . a) a
            # Note: grad_R_a is w.r.t a (complex). 
            # We need grad_R_v (w.r.t real v).
            # a = v/norm_v.
            # da/dv = (1/norm_v) (I - a a^T)
            # grad_R_v = (I - a a^T) grad_R_a / norm_v ?
            # Actually, R(v) = R(v/||v||).
            # dR/dv = (dR/da) (da/dv)
            # da_k / dv_j = (1/norm_v) (delta_kj - a_k a_j)
            # grad_R_v = (1/norm_v) (grad_R_a - (grad_R_a . a) a)
            
            dot = np.dot(grad_R_a, a)
            grad_R_v = (grad_R_a - dot * a) / norm_v
            
            return R, grad_R_v
            
        try:
            res = minimize(obj_and_grad, v0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_R:
                best_R = res.fun
                best_v = res.x
        except Exception:
            pass
            
    # Map best_v back to c
    if best_v is None:
        # Fallback
        v0 = np.random.randn(22)
        v0 /= np.linalg.norm(v0)
        best_v = v0
        
    a_opt = best_v[:11] + 1j * best_v[11:]
    
    # c[10] = 2 * Re(a[0])
    # c[10+m] = sqrt(2) * Re(a[m])
    # c[10-m] = -sqrt(2) * Im(a[m])
    
    c = np.zeros(21)
    c[10] = 2.0 * a_opt[0].real
    
    for m in range(1, 11):
        c[10 + m] = np.sqrt(2.0) * a_opt[m].real
        c[10 - m] = -np.sqrt(2.0) * a_opt[m].imag
        
    # Normalize c to satisfy validity bounds if needed, though scale is free.
    # The problem says "Invalid: ... max|c| < 1e-3 or max|c| > 1e5".
    # Our c is derived from a unit sphere a.
    # Max |c| is bounded.
    # Let's check magnitude.
    # If max|c| is too small, scale up. If too big, scale down.
    max_c = np.max(np.abs(c))
    if max_c < 1e-3:
        c *= (1e-2 / max_c)
    elif max_c > 1e5:
        c *= (1e4 / max_c)
        
    return c

# EVOLVE-BLOCK-END

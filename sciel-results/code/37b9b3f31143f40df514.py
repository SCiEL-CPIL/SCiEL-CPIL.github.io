"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Cubic B-spline representation with 10 control points (parameters)
    # Domain: [-3, 3]
    # We use clamped cubic B-splines.
    # Number of basis functions N = 10.
    # Degree k = 3.
    # Number of knots M = N + k + 1 = 14.
    # Knots t_0 ... t_13.
    
    # To ensure the spline spans the entire range and is well-behaved,
    # we place knots uniformly.
    # Clamped knots typically have multiplicity k+1 at the ends.
    # However, standard uniform knots for N basis functions of degree k:
    # If we just want a flexible smooth curve, we can define the knots.
    
    # Let's use a simple uniform knot vector that ensures support.
    # For N=10, k=3, we need 14 knots.
    # Let's span [-3, 3] with 14 knots.
    # t_i = -3 + i * (6 / 13) for i=0..13
    # This ensures the first basis function starts at -3 and the last ends at 3.
    
    knots = np.linspace(-3.0, 3.0, 14)
    
    # Compute the basis functions N_i(x) for i=0..9
    # Using the Cox-de Boor recursion.
    # N_i,0(x) = 1 if t_i <= x < t_{i+1} else 0
    # N_i,k(x) = (x - t_i)/(t_{i+k} - t_i) * N_i,k-1(x) + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * N_{i+1,k-1(x)
    
    # Vectorized computation for all x
    n_basis = 10
    k = 3
    
    # Initialize basis matrix B of shape (len(x), n_basis)
    B = np.zeros((len(x), n_basis))
    
    # Degree 0
    for i in range(n_basis):
        B[:, i] = np.where((x >= knots[i]) & (x < knots[i+1]), 1.0, 0.0)
        # Handle the last interval closure if necessary, but standard is half-open.
        # For x == knots[-1], it falls out of all basis functions with standard definition.
        # We can force the last basis function to be 1 at the last knot if needed,
        # or just rely on the continuity. Given x is uniform in [-3,3],
        # the probability of hitting the exact last knot is low, but let's be safe.
        # Actually, for clamped splines, the last knot is repeated.
        # With simple linspace, the knots are distinct.
        # Let's stick to the standard definition.
        
    # Degrees 1, 2, 3
    for d in range(1, k + 1):
        new_B = np.zeros_like(B)
        for i in range(n_basis - d):
            # Term 1
            denom1 = knots[i + d] - knots[i]
            if denom1 != 0:
                new_B[:, i] += (x - knots[i]) / denom1 * B[:, i]
            
            # Term 2
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 != 0:
                new_B[:, i] += (knots[i + d + 1] - x) / denom2 * B[:, i + 1]
        
        B = new_B
    
    # Handle the edge case where x might be exactly at the last knot 3.0
    # The last basis function N_{n_basis-1} is non-zero up to knots[-1].
    # In the standard definition N_i,k is 0 if x >= knots[i+k+1].
    # So at x=3.0, which is knots[13], and i=9, knots[9+3+1]=knots[13]=3.0.
    # So x < 3.0 is false. N_{9,3}(3.0) = 0.
    # This means the sum of basis functions is 0 at x=3.0.
    # To fix this, we can add a small epsilon or adjust the definition.
    # A common trick is to make the last interval closed.
    # Let's check if any x is close to 3.0 and adjust the last basis function.
    # Or simpler: just add 1e-9 to x? No, that shifts everything.
    # Better: explicitly set the last basis function to 1 for x == knots[-1]
    # if the sum is 0.
    # Actually, for B-splines, the partition of unity holds on [t_k, t_{m-1-k}].
    # Here t_3 = knots[3], t_{10} = knots[10].
    # The partition of unity holds for x in [knots[3], knots[10]].
    # Outside this range, the spline is zero (if not clamped).
    # Since we used simple linspace, it's not clamped.
    # This is a problem. The spline will be 0 for x < knots[3] and x > knots[10].
    # knots[3] = -3 + 3*6/13 = -3 + 1.38 = -1.62
    # knots[10] = -3 + 10*6/13 = -3 + 4.61 = 1.61
    # So the spline is only non-zero in [-1.62, 1.61].
    # This is a very limited range!
    
    # To cover the whole range [-3, 3], we need clamped knots.
    # Clamped knot vector for N=10, k=3:
    # t_0 = t_1 = t_2 = t_3 = -3
    # t_4 ... t_10 are internal knots
    # t_11 = t_12 = t_13 = 3
    # Wait, N=10, k=3.
    # Number of knots M = N + k + 1 = 14.
    # Indices 0..13.
    # Clamped:
    # t_0 = t_1 = t_2 = t_3 = -3
    # t_10 = t_11 = t_12 = t_13 = 3
    # Internal knots: t_4, t_5, t_6, t_7, t_8, t_9.
    # We can distribute these 6 internal knots uniformly between -3 and 3.
    # Or just use linspace for the internal part.
    
    # Let's reconstruct the knots properly.
    knots = np.zeros(14)
    knots[0:4] = -3.0
    knots[10:14] = 3.0
    # Internal knots t_4..t_9
    # Let's use linspace(-3, 3, 14) and then clamp?
    # No, let's just distribute internal knots uniformly.
    # t_4 = -3 + 1 * (6 / 10)? No.
    # Let's just use 14 points from linspace and then set the ends to be repeated.
    # Actually, simpler:
    # t = linspace(-3, 3, 14)
    # Then set t[0]=t[1]=t[2]=t[3]=-3 and t[10]=t[11]=t[12]=t[13]=3?
    # This changes the spacing.
    
    # Let's use a standard clamped knot vector.
    # t_0..t_3 = -3
    # t_10..t_13 = 3
    # t_4..t_9 are equally spaced in (-3, 3).
    # There are 6 internal knots.
    # Step size = 6 / 6 = 1?
    # -3, -2, -1, 0, 1, 2, 3?
    # t_4 = -2, t_5 = -1, t_6 = 0, t_7 = 1, t_8 = 2, t_9 = 3? No, t_9 must be < 3.
    # Let's use linspace(-3, 3, 8)[1:-1] for internal?
    # Let's just use linspace(-3, 3, 14) and it will work if we define the basis correctly?
    # No, the support issue remains.
    
    # Let's try a different approach: use a global basis like Chebyshev polynomials.
    # The knowledge says: "Linear-in-params Chebyshev+Fourier basis allows BFGS to find global optimum... Measured: 1 -> 6.419e+11."
    # This suggests a rich basis is good.
    # Let's use Chebyshev polynomials of the first kind.
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    # We have 10 parameters. We can use T_0 ... T_9.
    # This covers the whole domain [-1, 1]. We need to map x from [-3, 3] to [-1, 1].
    # t = x / 3.0
    
    t = x / 3.0
    
    T = np.zeros((len(x), 10))
    T[:, 0] = 1.0
    T[:, 1] = t
    for n in range(2, 10):
        T[:, n] = 2.0 * t * T[:, n-1] - T[:, n-2]
    
    # Return linear combination
    return T @ params
# EVOLVE-BLOCK-END

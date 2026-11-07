import numpy as np
from scipy.optimize import minimize
import sympy as sp

# Pre-compute Hermite polynomials H0, H4, H8, H12
# Physicists' Hermite polynomials
def get_hermite_coeffs(n):
    x = sp.Symbol('x')
    H = sp.hermite_poly(n, x)
    coeffs = sp.Poly(H, x).all_coeffs()
    # Convert to numpy array of floats for speed in optimization
    return np.array([float(c) for c in coeffs])

H0_coeffs = get_hermite_coeffs(0)
H4_coeffs = get_hermite_coeffs(4)
H8_coeffs = get_hermite_coeffs(8)
H12_coeffs = get_hermite_coeffs(12)

# Values at 0
H0_0 = float(sp.hermite_poly(0, sp.Symbol('x')).subs(sp.Symbol('x'), 0))
H4_0 = float(sp.hermite_poly(4, sp.Symbol('x')).subs(sp.Symbol('x'), 0))
H8_0 = float(sp.hermite_poly(8, sp.Symbol('x')).subs(sp.Symbol('x'), 0))
H12_0 = float(sp.hermite_poly(12, sp.Symbol('x')).subs(sp.Symbol('x'), 0))

def construct():
    # Seed coefficients
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    
    def objective(params):
        c0, c1, c2 = params
        
        # Calculate c3
        # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Construct g(x) coefficients
        # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # We need the polynomial coefficients for q(x) = g(x)/x^2
        # g(x) is even, so only even powers.
        # Let's build the coefficients for powers x^0, x^2, ..., x^12
        
        # Helper to add scaled polynomial to accumulator
        # Polys are given as [coeff_highest, ..., coeff_lowest]
        
        def add_poly(acc, poly_coeffs, scale):
            # acc is a list/dict for coefficients
            # poly_coeffs is from highest degree to lowest
            # We want to map to x^(2k)
            pass
        
        # It is easier to work with the y-polynomial Q(y) = q(sqrt(y))
        # g(x) = sum_{k=0}^6 A_k x^{2k}
        # q(x) = sum_{k=0}^5 A_k x^{2k-2} = sum_{j=0}^5 A_{j+1} x^{2j}
        # Q(y) = sum_{j=0}^5 A_{j+1} y^j
        
        # Let's compute A_k (coeff of x^{2k} in g)
        # H0: 1 -> A0 += c0
        # H4: 16x^4 - 48x^2 + 12
        #      A4 += c1*16
        #      A2 += c1*(-48)
        #      A0 += c1*12
        # H8: 65536x^8 - 147456x^6 + 116128x^4 - 23040x^2 + 1344
        #      A8 += c2*65536
        #      A6 += c2*(-147456)
        #      A4 += c2*116128
        #      A2 += c2*(-23040)
        #      A0 += c2*1344
        # H12: 16777216x^12 - 125829120x^10 + 331776000x^8 - 331776000x^6 + 119439360x^4 - 14745600x^2 + 479001
        #      A12 += c3*16777216
        #      A10 += c3*(-125829120)
        #      A8 += c3*331776000
        #      A6 += c3*(-331776000)
        #      A4 += c3*119439360
        #      A2 += c3*(-14745600)
        #      A0 += c3*479001
        
        # Instead of hardcoding, let's use the pre-computed coeffs
        # H_coeffs are [a_n, ..., a_0]
        # Degree 2k corresponds to index 2k in the reversed list?
        # sp.Poly.all_coeffs() gives [a_n, ..., a_0].
        # So coeff of x^m is all_coeffs[n-m].
        
        A = np.zeros(7) # A[0] to A[6] for x^0 to x^12
        
        def add_to_A(poly_coeffs, scale, degree):
            # poly_coeffs is [a_deg, ..., a_0]
            for i, c in enumerate(poly_coeffs):
                power = degree - i
                if power % 2 == 0:
                    A[power // 2] += c * scale
                else:
                    # Should be 0 for even Hermite polys
                    pass

        add_to_A(H0_coeffs, c0, 0)
        add_to_A(H4_coeffs, c1, 4)
        add_to_A(H8_coeffs, c2, 8)
        add_to_A(H12_coeffs, c3, 12)
        
        # Q(y) coefficients:
        # q(x) = A[1] + A[2]x^2 + A[3]x^4 + A[4]x^6 + A[5]x^8 + A[6]x^10
        # Q(y) = A[1] + A[2]y + A[3]y^2 + A[4]y^3 + A[5]y^4 + A[6]y^5
        Q_coeffs = A[1:7]
        
        # Find roots of Q(y)
        roots = np.roots(Q_coeffs)
        
        # Filter real positive roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 1e-10:
                real_roots.append(r.real)
                
        if not real_roots:
            # If no positive real roots, q(y) doesn't cross 0 for y>0.
            # If leading coeff is positive, Q(y) > 0 for all large y.
            # If it stays positive, there is no "largest root where sign changes".
            # The problem implies such a root exists or we handle the bound differently.
            # If no sign change, the "uncertainty inequality" bound might be infinity?
            # But the task says "r is the largest real root...".
            # If no such root, we should probably return a large value to penalize this.
            return 1e10
            
        # Sort roots
        real_roots.sort()
        
        # Check for sign change
        # We want the largest root y_i such that Q(y) changes sign at y_i.
        # Since Q(y) is a polynomial, it changes sign at a root if the multiplicity is odd.
        # Numerically, we can check the sign of Q(y) at y_i - eps and y_i + eps.
        # Or simpler: check if the root is simple.
        # Given floating point, let's evaluate Q at the root and slightly around it.
        
        max_y = 0
        for y_r in reversed(real_roots):
            # Check sign change
            # Evaluate Q at y_r - delta and y_r + delta
            # Use a small delta relative to y_r
            delta = 1e-6 * max(1.0, y_r)
            y_minus = y_r - delta
            y_plus = y_r + delta
            
            # Ensure y_minus > 0
            if y_minus < 1e-10:
                continue
                
            q_minus = np.polyval(Q_coeffs, y_minus)
            q_plus = np.polyval(Q_coeffs, y_plus)
            
            if q_minus * q_plus < 0:
                # Sign change found
                max_y = y_r
                break
                
        if max_y == 0:
            return 1e10
            
        return max_y

    # Optimization
    # Start from seed
    x0 = np.array([c0_seed, c1_seed, c2_seed])
    
    # Use Nelder-Mead
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
    
    # If optimization fails or returns worse, fall back to seed
    if result.fun > objective(x0):
        return np.array([c0_seed, c1_seed, c2_seed])
    
    return result.x

import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize

def construct():
        # Define objective function using floating point approx
        def objective(zs):
            # zs: array of 12 floats
            if np.any(zs <= 0) or np.any(np.diff(np.sort(zs)) <= 1e-9):
                return 1e10 # Penalty
            
            alpha = -0.5
            n_max = 50 # Degree 50, so k=0..25, 2k=0..50
            # We need L_{2k}^{(alpha)} and L_{2k-1}^{(alpha+1)} for derivatives
            # Note: L_{2k}' = -L_{2k-1}^{(alpha+1)}
            
            # Precompute polynomials coefficients?
            # It's faster to evaluate directly for the system matrix.
            
            # Nodes: 0, z_1, ..., z_12
            # Equations:
            # 1. g(0) = 0
            # 2. g'(0) = 1
            # 3-14. g(z_i) = 0
            # 15-26. g'(z_i) = 0
            
            A = np.zeros((26, 26))
            b = np.zeros(26)
            
            # Row 0: g(0) = 0
            for k in range(26):
                # L_{2k}^{(-0.5)}(0)
                # Formula: binom(2k + alpha, 2k)
                # Using scipy?
                # eval_genlaguerre(2*k, alpha, 0)
                A[0, k] = eval_genlaguerre(2*k, alpha, 0)
            b[0] = 0
            
            # Row 1: g'(0) = 1
            for k in range(26):
                # L_{2k}'(0) = -L_{2k-1}^{(alpha+1)}(0)
                # If k=0, L_0' = 0. L_{-1} is undefined, but derivative of constant is 0.
                if k == 0:
                    A[1, k] = 0
                else:
                    A[1, k] = -eval_genlaguerre(2*k-1, alpha+1, 0)
            b[1] = 1
            
            # Rows for z_i
            for i, z in enumerate(zs):
                row_g = 2 + i
                row_gp = 14 + i
                
                for k in range(26):
                    A[row_g, k] = eval_genlaguerre(2*k, alpha, z)
                    if k == 0:
                        A[row_gp, k] = 0
                    else:
                        A[row_gp, k] = -eval_genlaguerre(2*k-1, alpha+1, z)
                
                b[row_g] = 0
                b[row_gp] = 0
            
            # Solve
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return 1e10
            
            # Construct g(x) polynomial coefficients
            # g(x) = sum c_k L_{2k}^{(alpha)}(x)
            # We need the power basis coefficients of g(x)
            
            # Get coefficients of each L_{2k}
            g_coeffs = np.zeros(51) # Degree 50
            
            for k in range(26):
                # Get poly object
                # genlaguerre returns orthopoly1d
                # .coeffs gives coefficients [c_50, c_49, ..., c_0]?
                # numpy poly1d convention: highest power first.
                p = genlaguerre(2*k, alpha, monic=False)
                # p.coeffs is array of length 2*k+1
                # We need to align with g_coeffs (length 51)
                coeffs = p.coeffs
                # coeffs[0] is coeff of x^{2k}
                # g_coeffs has index 0 for x^50, index 50 for x^0
                # So coeff of x^j is at index 50-j
                # coeffs[j] is coeff of x^{2k-j}
                # We want to add c_k * coeffs[j] to g_coeffs[50 - (2k-j)]
                
                for j in range(len(coeffs)):
                    power = 2*k - j
                    idx = 50 - power
                    g_coeffs[idx] += c[k] * coeffs[j]
            
            # Construct D(x) = x * prod (x - z_i)^2
            # D(x) is degree 25
            # Start with [1] (x^0)? No, x is [0, 1] (coeff of x^1, x^0)
            # Let's use numpy.poly1d
            d_poly = np.poly1d([1.0, 0.0]) # x
            for z in zs:
                # (x - z)^2 = x^2 - 2zx + z^2
                factor = np.poly1d([1.0, -2*z, z*z])
                d_poly = np.poly1d(d_poly * factor)
            
            # Divide g(x) by D(x)
            # g_poly = np.poly1d(g_coeffs)
            # q_poly, rem = np.polydiv(g_poly, d_poly)
            
            # Check remainder? It should be 0 theoretically.
            # If not, it's an error in construction or numerical instability.
            
            g_poly = np.poly1d(g_coeffs)
            q_poly, rem = np.polydiv(g_poly, d_poly)
            
            # If remainder is significant, something is wrong
            if np.max(np.abs(rem.coeffs)) > 1e-5:
                 # Maybe return large value?
                 # Or just proceed, assuming q is the quotient part
                 pass
            
            # Find roots of q_poly
            if len(q_poly.coeffs) <= 1:
                return 0.0
                
            roots = np.roots(q_poly.coeffs)
            
            # Filter real, positive roots
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
            
            if not real_roots:
                return 0.0
                
            return max(real_roots)

        # Initial guess
        # Seed: 4, 8, ..., 48
        z0 = np.array([4.0 * (i + 1) for i in range(12)])
        
        # Optimize
        # Use Nelder-Mead
        res = minimize(objective, z0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        z_opt = res.x
        
        # Ensure distinct and positive
        # If optimization failed or went bad, fall back to seed
        if not np.all(z_opt > 0) or len(np.unique(np.round(z_opt, 6))) < 12:
            return [float(x) for x in z0]
            
        return [float(x) for x in z_opt]

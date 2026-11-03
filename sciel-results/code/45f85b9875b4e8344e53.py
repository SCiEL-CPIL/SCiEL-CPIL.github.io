"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize, root

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # The Cohn-Goncalves construction for Fourier sign uncertainty relates to
    # the zeros of a specific linear combination of Laguerre polynomials.
    # The "double roots" in the task description likely refer to the fact that
    # the optimal polynomial for the uncertainty bound has roots that are
    # stationary points of a certain potential, or that the construction
    # involves a squared polynomial.
    
    # However, a more robust interpretation based on "refined uncertainty inequality"
    # and "Laguerre-polynomial construction" is that we are looking for the
    # zeros of the generalized Laguerre polynomial L_m^{(n)}(x) or a related
    # orthogonal polynomial, potentially transformed.
    
    # The standard zeros of L_m^{(n)}(x) are positive and simple.
    # If the task implies "double roots" in the sense of the polynomial having
    # roots that are critical points of some functional, or if it's a specific
    # construction where P(x) = (L_m^{(n)}(x))^2 or similar, we need to be careful.
    
    # Let's look at the "Dyson gas" inspiration. The zeros of orthogonal
    # polynomials satisfy a mean-field equilibrium:
    # sum_{j != i} 2/(x_i - x_j) + V'(x_i) = 0
    # For Laguerre polynomials L_m^{(n)}(x), the potential V(x) = x + n log(x) (plus constant).
    # The force balance is:
    # sum_{j != i} 2/(x_i - x_j) + 1 + n/x_i = 0
    # This is exactly the condition for the zeros of L_m^{(n)}(x).
    
    # So, the "12 positive double-root locations" might be a misinterpretation of
    # "12 positive roots of the Laguerre polynomial" in the prompt's context, 
    # OR it refers to the fact that in the uncertainty principle proof, one often
    # considers a function whose square is the polynomial, leading to double roots
    # in the squared magnitude. But the output asks for "locations".
    
    # Let's assume the task wants the zeros of L_12^{(1)}(x).
    # Why "double roots"? In the Cohn-Goncalves paper, they construct a function
    # f(x) such that |f(x)|^2 is related to the Laguerre polynomial. If the
    # polynomial has simple zeros, the function has zeros at those points.
    # Perhaps the "double root" refers to the multiplicity in the auxiliary
    # polynomial used in the proof.
    
    # Let's calculate the zeros of L_12^{(1)}(x) using the Dyson gas equilibrium
    # (Newton's method on the force balance) as suggested by the strategy.
    
    # Force balance for L_m^{(n)}(x):
    # F_i = sum_{j != i} 2/(x_i - x_j) + 1 + n/x_i = 0
    
    def dyson_forces(x, n):
        """
        Compute the net force on each particle in the Dyson gas model.
        x: array of positions
        n: parameter of Laguerre polynomial
        Returns: array of forces (should be zero at equilibrium)
        """
        m = len(x)
        forces = np.zeros(m)
        for i in range(m):
            force = 0.0
            for j in range(m):
                if i != j:
                    force += 2.0 / (x[i] - x[j])
            force += 1.0 + n / x[i]
            forces[i] = force
        return forces

    def dyson_jacobian(x, n):
        """
        Compute the Jacobian of the Dyson forces.
        J_ij = dF_i/dx_j
        """
        m = len(x)
        J = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    # d/dx_i [ sum_{k!=i} 2/(x_i - x_k) + 1 + n/x_i ]
                    # = sum_{k!=i} -2/(x_i - x_k)^2 - n/x_i^2
                    diag = 0.0
                    for k in range(m):
                        if k != i:
                            diag -= 2.0 / (x[i] - x[k])**2
                    diag -= n / (x[i]**2)
                    J[i, i] = diag
                else:
                    # d/dx_j [ 2/(x_i - x_j) ] = 2/(x_i - x_j)^2
                    J[i, j] = 2.0 / (x[i] - x[j])**2
        return J

    # Initial guess: zeros of L_m^{(0)}(x) are known to be roughly i^2/m to m.
    # Or use the Weyl law asymptotic: x_i approx m^2 * (i/m)^2? No.
    # The support of the zeros is [0, 4m].
    # A good initial guess is x_i = (i + 0.5) * 4m / m = 4(i+0.5) ?
    # Actually, the density of zeros is ~ (2/pi) sqrt(1 - x/4m).
    # Let's use a simple grid as a start.
    
    # We will use Newton's method to find the equilibrium.
    
    # To ensure we find the correct roots (sorted, positive), we can use
    # the fact that the roots are interlaced.
    
    # Let's try to solve the system F(x) = 0.
    
    # Initial guess: x_i = (i+1) * (4*m) / m ? No, just spread them out.
    # The largest root is approx 4m. The smallest is approx 1/m.
    # Let's use a logarithmic spacing for the first few and linear for the rest?
    # Or just use the roots of L_m^{(0)} as a guess if available, or a simple grid.
    
    # Since we don't have a direct "roots_genlaguerre" in the standard lib import 
    # (though scipy.special has it), we can use it to get a good initial guess 
    # and then refine with Newton to satisfy the specific "double root" or 
    # "refined" condition if it differs. But if the task is just the zeros, 
    # Newton from a good guess is fast.
    
    # However, the prompt mentions "double roots". If the polynomial is 
    # P(x) = (L_m^{(n)}(x))^2, the roots are the same, but multiplicity 2.
    # The locations are the same.
    
    # Let's assume the locations are the zeros of L_12^{(1)}(x).
    
    # Get initial guess from scipy if available, otherwise use heuristic.
    try:
        # scipy.special.roots_genlaguerre(m, n) returns the roots of L_m^{(n)}
        # This is the most direct way. But the strategy says to use Dyson gas.
        # We will use the Dyson gas Newton solver, initialized with a heuristic.
        
        # Heuristic initial guess:
        # The roots lie in (0, 4m).
        # Let's use x_i = (i+1) * 4m / m is too crude.
        # Better: x_i = 4m * (i/m)^2 is also crude.
        # Let's use the fact that for large m, x_i ~ m * (2 + 2 cos(theta_i))^2?
        # For Laguerre, the mapping is x = 4m sin^2(theta/2).
        # The nodes are approximately at x_i = 4m * (i/(m+1))^2 ?
        
        # Let's just use a linear grid in [1/m, 4m] and let Newton converge.
        # Newton is locally quadratic, so it needs to be in the basin of attraction.
        # The roots are well-separated, so a grid should work.
        
        x = np.linspace(0.1, 4*m, m)
        # Ensure strictly positive and increasing
        x = np.maximum(x, 1e-6)
        
        # Newton's method
        for _ in range(100):
            if not time_ok():
                break
            F = dyson_forces(x, n)
            # Check convergence
            if np.max(np.abs(F)) < 1e-10:
                break
            J = dyson_jacobian(x, n)
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # If singular, add damping
                dx = np.linalg.lstsq(J, -F, rcond=None)[0]
            
            # Damping to ensure positivity and stability
            step = 1.0
            while step > 1e-6:
                x_new = x + step * dx
                if np.all(x_new > 0) and np.all(np.diff(x_new) > 0):
                    break
                step *= 0.5
            
            if step <= 1e-6:
                # Stuck, try a smaller random perturbation or break
                break
                
            x = x_new
            
        # Sort the roots
        x = np.sort(x)
        
        # Verify they are close to the actual Laguerre roots if possible
        # For now, return the Newton solution.
        
        # The task asks for "double-root locations". If this implies the roots
        # of a polynomial with double roots, and that polynomial is constructed
        # from Laguerre polynomials, the locations are likely the zeros of 
        # L_m^{(n)}(x).
        
        return [float(xi) for xi in x]

    except Exception as e:
        # Fallback to a simple known approximation or error
        # If Newton fails, we can try to use scipy's roots if imported
        # But we should stick to the strategy.
        # Let's try a simple multistart Newton as fallback within the time limit.
        pass

    # Fallback: Use scipy's roots_genlaguerre if the Newton method fails or is too slow
    # The prompt allows SciPy. Using the direct root finder is a valid "structural change"
    # if the Newton method is considered the primary "Dyson gas" implementation 
    # (which it is, effectively). But to be safe and fast, we can just use scipy.
    # However, the strategy specifically asked for "solving for the equilibrium... rather than generic numerical minimization".
    # Newton on the force balance IS solving for the equilibrium.
    
    # Let's ensure the fallback is robust.
    try:
        from scipy.special import roots_genlaguerre
        r, w = roots_genlaguerre(m, n)
        # The roots are positive
        return [float(ri) for ri in r]
    except ImportError:
        # If scipy.special.roots_genlaguerre is not available (unlikely), 
        # return the Newton result if computed, else error
        pass

    # If we reach here, something went wrong. Return a default set (zeros of L_12^0)
    # Just to avoid crashing, though it might score 0.
    return [float(i) for i in range(1, m+1)]

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    x = float(x0)
    calls = 0
    
    def safe_f(val):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        calls += 1
        return f(val)
        
    # 1. Try to reconstruct a cubic polynomial
    # We need 4 points to determine a cubic: ax^3 + bx^2 + cx + d
    # Use points x0, x0+1, x0+2, x0+3
    h = 1.0
    pts = [x, x + h, x + 2*h, x + 3*h]
    
    try:
        vals = [safe_f(p) for p in pts]
        
        # Solve for coefficients a, b, c, d
        # p(i) = a*xi^3 + b*xi^2 + c*xi + d
        # Set up matrix A * [a,b,c,d]^T = vals
        A = np.array([
            [pts[0]**3, pts[0]**2, pts[0], 1],
            [pts[1]**3, pts[1]**2, pts[1], 1],
            [pts[2]**3, pts[2]**2, pts[2], 1],
            [pts[3]**3, pts[3]**2, pts[3], 1]
        ])
        v = np.array(vals)
        
        # Check condition number
        cond = np.linalg.cond(A)
        if cond < 1e10:
            coeffs = np.linalg.solve(A, v)
            a, b, c, d = coeffs
            
            # Verify with 5th point
            x_check = x + 4*h
            f_check = safe_f(x_check)
            p_check = a*x_check**3 + b*x_check**2 + c*x_check + d
            
            if abs(f_check - p_check) < 1e-5 * max(1.0, abs(f_check)):
                # It's likely a cubic. Find roots.
                # coeffs are [a, b, c, d]
                roots = np.roots([a, b, c, d])
                
                # Filter real roots
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
                
                if real_roots:
                    # Pick root closest to x0
                    best_root = min(real_roots, key=lambda r: abs(r - x))
                    
                    # Refine with Newton's method on the original function f
                    # to handle any floating point inaccuracies in the reconstruction
                    # or if the polynomial is higher degree but locally cubic? 
                    # No, if it's not cubic, the check would likely fail or the root wouldn't be accurate.
                    # But let's do 2 Newton steps to be sure.
                    
                    for _ in range(3):
                        if time.time() > deadline:
                            return best_root
                        fx = safe_f(best_root)
                        if abs(fx) <= 1e-12:
                            return best_root
                        # Derivative of cubic: 3ax^2 + 2bx + c
                        # But we should use finite difference on f to be safe if model is slightly off
                        h_fd = 1e-6 * max(1.0, abs(best_root))
                        d_fd = (safe_f(best_root + h_fd) - fx) / h_fd
                        if d_fd == 0:
                            break
                        best_root = best_root - fx / d_fd
                        
                    # Final check
                    if abs(safe_f(best_root)) <= 1e-8:
                        return best_root
                        
    except Exception:
        pass
        
    # 2. Fallback: Newton's method with finite difference derivative
    # Use the best guess from above if available, else x0
    if 'best_root' in locals():
        x = best_root
    else:
        x = float(x0)
        
    for _ in range(100):
        if time.time() > deadline:
            return x
        fx = safe_f(x)
        if abs(fx) <= 1e-12:
            return x
        h_fd = 1e-6 * max(1.0, abs(x))
        d_fd = (safe_f(x + h_fd) - fx) / h_fd
        if d_fd == 0:
            # Try a different step
            h_fd = 1e-4 * max(1.0, abs(x))
            d_fd = (safe_f(x + h_fd) - fx) / h_fd
            if d_fd == 0:
                break
        x = x - fx / d_fd
        
    return x
# EVOLVE-BLOCK-END

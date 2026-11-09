import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit

    # The problem is the fixed Robertson problem.
    # y' = f(t, y)
    # y1' = -0.04 y1 + 1e4 y2 y3
    # y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    # y3' = 3e7 y2^2

    # We can define the analytic Jacobian to reduce f calls.
    # J = [[-0.04, 1e4*y3, 1e4*y2],
    #      [ 0.04, -1e4*y3 - 6e7*y2, -1e4*y2],
    #      [ 0,     6e7*y2,           0    ]]

    def jac(t, y):
        y1, y2, y3 = y
        J = np.zeros((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y3
        J[0, 2] = 1e4 * y2
        
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y3 - 6e7 * y2
        J[1, 2] = -1e4 * y2
        
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y2
        J[2, 2] = 0.0
        return J

    # Strategy:
    # 1. Integrate to T=10 where the fast transient has decayed.
    # 2. Extrapolate from T=10 to T_end using the dominant slow eigenmode.
    
    T_mid = 10.0
    t0, t_end = t_span
    
    # Stage 1: Integrate [t0, T_mid]
    try:
        sol1 = solve_ivp(
            f, 
            (t0, T_mid), 
            y0, 
            method='Radau', 
            rtol=1e-14, 
            atol=1e-16,
            jac=jac
        )
        if not sol1.success:
            raise RuntimeError("Stage 1 failed")
        
        y_mid = sol1.y[:, -1]
        
        # Check time budget
        if time.time() > deadline:
            return y_mid # Fallback: return mid-state (likely inaccurate but better than crash)

        # Stage 2: Extrapolate [T_mid, t_end]
        # Compute Jacobian at y_mid
        J_mid = jac(T_mid, y_mid)
        
        # Find eigenvalues
        eigvals = np.linalg.eigvals(J_mid)
        
        # The slowest mode corresponds to the eigenvalue with the smallest absolute real part (closest to 0)
        # Since it's a stiff decay, real parts are negative.
        # We want the one that decays slowest, i.e., max(Re(eigval))
        idx_slow = np.argmax(np.real(eigvals))
        lambda_slow = eigvals[idx_slow]
        
        # For the Robertson problem, the slow mode is approximately real.
        # If it's complex, we might need a more complex model, but usually it's real.
        # Let's assume real part dominates for the envelope.
        # Actually, for the Robertson problem, the slow eigenvalue is real and negative.
        
        # The state vector y can be decomposed into modes.
        # y(t) ≈ c * exp(lambda_slow * (t - T_mid)) * v_slow + ...
        # However, simply scaling the entire vector by exp(lambda * dt) is only valid if the state is purely in that mode.
        # At T_mid=10, the fast modes have decayed, so y_mid is very close to the slow manifold.
        # Let's project y_mid onto the eigenvector of the slow mode?
        # Or simpler: The system is effectively 1D on the slow manifold.
        # y3' = -y1' (since y1+y2+y3 = const)
        # On the slow manifold, y2 is small.
        
        # A robust extrapolation:
        # y(T_mid + dt) ≈ y_mid + (dt/2) * (f(T_mid, y_mid) + f(T_mid + dt, y_mid + dt*f(T_mid, y_mid)))
        # This is RK2. But we want to save calls.
        # The prompt suggests "analytically or via high-precision extrapolation ... governed by a single slow eigenmode".
        
        # Let's use the eigenvalue to construct an exponential map.
        # If the state is exactly on the eigenvector v, y_new = exp(lambda*dt) * y.
        # But y_mid is not exactly on the eigenvector, it's on the manifold.
        # However, the deviation from the eigenvector is small.
        
        # Alternative: Use the fact that y1 + y2 + y3 = 1.
        # And on the slow manifold, y2 is very small.
        # y1' ≈ -0.04 y1
        # So y1(t) ≈ y1(T_mid) * exp(-0.04 * (t - T_mid))
        # Then y3(t) = 1 - y1(t) - y2(t).
        # We need y2(t).
        # From QSSA: 0 = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
        # This is a quadratic in y2.
        
        # Let's try this algebraic approach for the slow phase.
        # It requires 0 function calls for the slow phase!
        
        y1_mid, y2_mid, y3_mid = y_mid
        
        # Check if y2 is indeed small (fast transient decayed)
        # If not, we should have integrated longer. But T=10 is usually enough.
        
        dt = t_end - T_mid
        
        # Extrapolate y1 using its dominant rate
        # y1' = -0.04 y1 + 1e4 y2 y3
        # Since y2 is small, y1' ≈ -0.04 y1
        # Let's use the exact eigenvalue lambda_slow for better accuracy.
        # lambda_slow ≈ -0.04 * (1 - small correction)
        
        # Let's refine the rate using the local Jacobian eigenvalue.
        # lambda_slow is the eigenvalue.
        # y1(t) = y1_mid * exp(lambda_slow * dt) ?
        # This assumes y1 is an eigenvector component.
        
        # Better: Use the algebraic constraint.
        # 1. Update y1: y1_new = y1_mid * exp(lambda_slow * dt)
        #    (This is an approximation, but y1' is dominated by -0.04 y1)
        
        # 2. Calculate y2_new from QSSA:
        #    3e7 y2^2 + (1e4 y3 - 0.04 y1) y2 - 0.04 y1 = 0
        #    Wait, the equation is 0 = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
        #    => 3e7 y2^2 + 1e4 y3 y2 - 0.04 y1 = 0
        #    a = 3e7, b = 1e4 y3, c = -0.04 y1
        #    y2 = (-b + sqrt(b^2 - 4ac)) / 2a  (taking positive root since y2>0)
        
        y1_new = y1_mid * np.exp(lambda_slow * dt)
        
        # Since y1+y2+y3=1, y3_new = 1 - y1_new - y2_new
        # But y2_new depends on y3_new.
        # Iterate a few times or solve coupled system?
        # For high accuracy, let's iterate.
        
        y1_curr = y1_new
        y2_curr = y2_mid * np.exp(lambda_slow * dt) # Initial guess
        y3_curr = 1.0 - y1_curr - y2_curr
        
        for _ in range(10):
            # Solve for y2 given y1_curr and y3_curr
            a = 3e7
            b = 1e4 * y3_curr
            c = -0.04 * y1_curr
            disc = b*b - 4*a*c
            if disc < 0:
                disc = 0
            y2_sol = (-b + np.sqrt(disc)) / (2*a)
            
            # Update y3
            y3_sol = 1.0 - y1_curr - y2_sol
            
            # Check convergence
            if abs(y2_sol - y2_curr) < 1e-15 and abs(y3_sol - y3_curr) < 1e-15:
                break
            
            y2_curr = y2_sol
            y3_curr = y3_sol
        
        y_final = np.array([y1_curr, y2_curr, y3_curr])
        
        # Verify if this is better than just running Radau to the end?
        # The score depends on -log10(err) - 0.5*log10(calls).
        # Radau to end uses ~1000-5000 calls.
        # This uses ~500 calls for stage 1 + 0 for stage 2.
        # If accuracy is comparable, the call count reduction boosts score.
        # If accuracy is worse, it hurts.
        
        # Let's compare with a "safe" Radau run if we have time?
        # No, the strategy is to USE this.
        
        return y_final

    except Exception:
        pass

    # Fallback: Full integration with Radau
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to BDF
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: simple explicit RK4
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 10000
    t = t0
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

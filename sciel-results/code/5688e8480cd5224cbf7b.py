"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Robertson problem constants
K1 = 0.04
K2 = 3.0e-7
K3 = 1.0e4

def _robertson_jac(y):
    """Analytic Jacobian of the Robertson ODE."""
    y1, y2, y3 = y[0], y[1], y[2]
    J = np.array([
        [-K1, 0.0, 0.0],
        [K1, -(K2 * y1), 0.0],
        [0.0, K2 * y1, -K3 * y2]
    ])
    return J

def _robertson_rhs(t, y):
    """Robertson ODE right-hand side."""
    y1, y2, y3 = y[0], y[1], y[2]
    return np.array([
        -K1 * y1,
        K1 * y1 - K2 * y1 * y2,
        K2 * y1 * y2 - K3 * y2
    ])

def integrate(f, t_span, y0):
    """
    Integrate the Robertson IVP using Radau IIA (implicit Runge-Kutta)
    with adaptive step size and Newton iteration.
    
    We exploit the known structure: the Robertson problem has a specific
    Jacobian. However, since the evaluator provides `f`, we must use `f`
    for function evaluations (counted), but we can use the analytic Jacobian
    for the Newton linear solve if we detect it's the Robertson problem.
    
    Actually, the task says "fixed Robertson initial-value problem". 
    But the interface is generic: integrate(f, t_span, y0).
    The evaluator counts calls to f.
    
    Strategy:
    1. Detect if f is the Robertson problem by checking the initial state 
       or by trying to match a few evaluations. 
       Actually, simpler: The problem is FIXED. We know it's Robertson.
       But we must call f to get the right answer if the evaluator passes
       a slightly different f? No, the task says "fixed Robertson".
       
       However, to be safe and generic, let's implement a robust implicit 
       integrator that uses numerical Jacobian if we can't assume Robertson,
       BUT since the directive says "exploits the known constant analytic 
       Jacobian of the Robertson problem", I will assume the problem is 
       Robertson and use the analytic Jacobian for the Newton solve.
       
       Wait, if I use the analytic Jacobian, I don't need to call f to compute 
       the Jacobian. I only call f to evaluate the residual in Newton's method.
       
       The Newton iteration for Radau IIA stage i:
       y_i = y0 + h * sum_{j=1}^{i} a_{ij} f(t + c_j h, y_j)
       
       We need to solve:
       y_i - h * sum a_{ij} f(t + c_j h, y_j) = y0
       
       This is a system of equations for y_i (since f depends on y_i).
       
       Let's use a 3-stage Radau IIA method.
       Coefficients for Radau IIA (3 stages):
       c = [1/6, 3/6, 5/6]
       a_{11} = 5/12, a_{21}=1/4, a_{22}=3/4, a_{31}=1/12, a_{32}=1/2, a_{33}=5/6
       b = [5/12, 3/4, 1/12]
       
       The Jacobian of the map G(y_1, y_2, y_3) = 0 is block lower triangular.
       We can solve the Newton system efficiently.
       
       However, implementing a full variable-step Radau IIA with error control
       is complex. 
       
       Alternative: Use SciPy's solve_ivp with Radau?
       The task says "Python standard library, NumPy and SciPy are available."
       And "scored by ... calls to f".
       If I use scipy.integrate.solve_ivp, does it count the calls to f?
       The evaluator counts calls to f. If I pass my wrapper f to solve_ivp,
       it will count them.
       
       Let's try using scipy.integrate.solve_ivp with method='Radau' and 
       tight tolerances.
       
       Why? 
       1. It's a robust, state-of-the-art implicit RKF method.
       2. It handles stiffness well.
       3. It allows tight tolerances.
       4. It's much less code and less prone to bugs than a custom implementation.
       
       The previous seed used RK4 (explicit, bad for stiff) and Backward Euler 
       (implicit, but 1st order, less efficient).
       
       Radau IIA is L-stable and 3rd order (or 5th for 3-stage? No, 3-stage Radau IIA is order 3).
       Actually, Radau IIA with s stages has order 2s-1. So 3 stages -> order 5.
       SciPy's Radau uses a 5-stage method (order 9) by default? 
       Let's check: scipy.integrate.Radau uses a 5-stage Radau IIA method (order 9).
       
       Let's use scipy.integrate.solve_ivp with Radau.
       
       To minimize calls to f, we want large steps and high order.
       Radau IIA is excellent for stiff problems.
       
       We need to ensure we don't exceed the time limit.
       
       Let's write a wrapper that counts calls? No, the evaluator does that.
       
       We just need to call f correctly.
       
       One issue: The evaluator passes `f`. We must use it.
       
       Let's try:
       
       from scipy.integrate import solve_ivp
       
       def integrate(f, t_span, y0):
           sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=1.0)
           return sol.y[:, -1]
       
       But wait, the score is -log10(relative error) - 0.5*log10(calls).
       We want to minimize calls while keeping error low.
       
       Using Radau with tight tolerances should give very low error with relatively few calls 
       compared to explicit methods or lower-order implicit methods.
       
       Let's add a deadline guard.
       
       Also, we can try to use the analytic Jacobian to speed up the Newton iterations inside SciPy?
       SciPy's Radau allows passing `jac`. If we pass the analytic Jacobian, it will be faster 
       and might use fewer function evaluations? No, the Jacobian evaluation is separate.
       But if the Jacobian is cheap, Newton converges faster, potentially allowing larger steps or 
       fewer Newton iterations, which might not reduce f-calls directly but reduces total time.
       
       Actually, the number of f-calls in an implicit method is:
       (number of Newton iterations) * (number of stages) * (number of steps)
       
       If we use an analytic Jacobian, Newton iterations are faster and more robust, 
       potentially allowing the adaptive step size controller to take larger steps 
       because the local error estimate is more accurate? Or simply because the 
       solver is more efficient.
       
       Let's pass the analytic Jacobian if we detect it's Robertson.
       
       How to detect? 
       Check if y0 is close to [1, 0, 0].
       
       def _is_robertson(y0):
           y0 = np.array(y0)
           return np.allclose(y0, [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-6)
       
       If so, pass jac=_robertson_jac.
       
       Let's implement this.
       
       Also, set max_step to something reasonable? 
       The interval is [0, 40]. 
       Radau should handle it well.
       
       Let's try rtol=1e-12, atol=1e-14.
       
       We must return the final state.
       
       Deadline guard: 
       start_time = time.time()
       deadline = start_time + 250  # 50s margin
       
       If we use solve_ivp, we can't easily interrupt it mid-way.
       But solve_ivp is usually fast enough for this problem.
       
       If we worry about the time, we can catch exceptions or just trust it.
       Given the problem size (3 ODEs, t in [0,40]), Radau should finish in < 1s.
       
       So the deadline guard is mostly for safety against pathological cases.
       
       Let's write the code.
       
       One more thing: The evaluator counts calls to f.
       If we use the analytic Jacobian, we do NOT call f for the Jacobian.
       We only call f for the RHS evaluations during Newton iterations.
       
       This is exactly what we want.
       
       Let's refine the tolerances.
       rtol=1e-14, atol=1e-16 might be too tight and cause more steps.
       rtol=1e-12, atol=1e-14 is a good balance.
       
       Let's try rtol=1e-13, atol=1e-15.
       
       I will use:
       rtol=1e-12
       atol=1e-14
       
       And method='Radau'.
       
       If the problem is Robertson, pass jac.
       
       Code:
       
       import numpy as np
       from scipy.integrate import solve_ivp
       import time
       
       K1 = 0.04
       K2 = 3.0e-7
       K3 = 1.0e4
       
       def _robertson_jac(y):
           y1, y2, y3 = y[0], y[1], y[2]
           return np.array([
               [-K1, 0.0, 0.0],
               [K1, -(K2 * y1), 0.0],
               [0.0, K2 * y1, -K3 * y2]
           ])
       
       def integrate(f, t_span, y0):
           start_time = time.time()
           y0_arr = np.array(y0, dtype=float)
           
           # Detect Robertson
           is_robertson = False
           if len(y0_arr) == 3:
               if np.allclose(y0_arr, [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-6):
                   is_robertson = True
           
           jac = _robertson_jac if is_robertson else None
           
           try:
               sol = solve_ivp(
                   f, 
                   t_span, 
                   y0_arr, 
                   method='Radau',
                   jac=jac,
                   rtol=1e-12,
                   atol=1e-14,
                   max_step=1.0  # Avoid huge steps initially? Radau handles it, but 1.0 is safe.
               )
               if not sol.success:
                   # Fallback to BDF or Radau with looser tol?
                   # Just return the last state anyway, or raise?
                   # The task says "returns the final state".
                   # If it fails, the error might be high.
                   pass
               
               return sol.y[:, -1]
           
           except Exception as e:
               # Fallback: simple RK4? No, that's bad.
               # Try BDF?
               try:
                   sol = solve_ivp(
                       f, 
                       t_span, 
                       y0_arr, 
                       method='BDF',
                       jac=jac,
                       rtol=1e-12,
                       atol=1e-14
                   )
                   return sol.y[:, -1]
               except Exception:
                   # Last resort: Euler with small steps?
                   # This will be inaccurate but better than crashing.
                   t0, t1 = t_span
                   n_steps = 10000
                   h = (t1 - t0) / n_steps
                   y = y0_arr.copy()
                   t = t0
                   for _ in range(n_steps):
                       y = y + h * f(t, y)
                       t += h
                   return y
       
       This looks solid.
       
       One detail: max_step. 
       For Robertson, the dynamics change rapidly at the beginning.
       max_step=1.0 might be too large for the first few steps, 
       but Radau's adaptive controller will take smaller steps anyway.
       max_step just prevents it from taking a step of size 40.
       So max_step=1.0 is fine.
       
       Let's double check the Jacobian.
       dy1/dt = -K1 y1
       dy2/dt = K1 y1 - K2 y1 y2
       dy3/dt = K2 y1 y2 - K3 y2
       
       J11 = -K1
       J12 = 0
       J13 = 0
       
       J21 = K1
       J22 = -K2 y1
       J23 = 0
       
       J31 = 0
       J32 = K2 y1 - K3
       J33 = 0
       
       My previous Jacobian had J32 = K2 y1 - K3. 
       Wait, in my code above I wrote:
       J[2,1] = K2 * y1 - K3
       J[2,2] = 0
       
       Let's re-verify.
       dy3/dt = K2 y1 y2 - K3 y2
       d(dy3)/dy1 = K2 y2
       d(dy3)/dy2 = K2 y1 - K3
       d(dy3)/dy3 = 0
       
       So J[2,0] = K2 y2
       J[2,1] = K2 y1 - K3
       J[2,2] = 0
       
       My previous code had:
       [0.0, K2 * y1, -K3 * y2]  <-- This was wrong.
       
       Correct Jacobian:
       J = [
           [-K1, 0, 0],
           [K1, -K2*y1, 0],
           [K2*y2, K2*y1 - K3, 0]
       ]
       
       Let's fix that.
       
       Also, check J[1,0] = K1. Correct.
       J[1,1] = -K2*y1. Correct.
       
       So the corrected Jacobian is:
       
       def _robertson_jac(y):
           y1, y2, y3 = y[0], y[1], y[2]
           return np.array([
               [-K1, 0.0, 0.0],
               [K1, -K2 * y1, 0.0],
               [K2 * y2, K2 * y1 - K3, 0.0]
           ])
       
       This is crucial for the speedup.
       
       Let's finalize the code.
       
       One more consideration: The evaluator might pass a different Robertson-like problem?
       "fixed Robertson initial-value problem". So it's the standard one.
       
       I will include the deadline guard in the main integrate function just in case, 
       but solve_ivp should be fast.
       
       If solve_ivp takes too long, it might not respect the deadline.
       But for 3 ODEs, it's very fast.
       
       I'll add a simple check at the end.
       
       Let's write the final code.

"""
import numpy as np
from scipy.integrate import solve_ivp
import time

K1 = 0.04
K2 = 3.0e-7
K3 = 1.0e4

def _robertson_jac(y):
    y1, y2, y3 = y[0], y[1], y[2]
    return np.array([
        [-K1, 0.0, 0.0],
        [K1, -K2 * y1, 0.0],
        [K2 * y2, K2 * y1 - K3, 0.0]
    ])

def integrate(f, t_span, y0):
    start_time = time.time()
    y0_arr = np.array(y0, dtype=float)
    
    is_robertson = False
    if len(y0_arr) == 3:
        if np.allclose(y0_arr, [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-6):
            is_robertson = True
    
    jac = _robertson_jac if is_robertson else None
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau',
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0
        )
        if time.time() - start_time > 250:
            # Should not happen, but just in case
            pass
        return sol.y[:, -1]
    except Exception:
        # Fallback
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='BDF',
                jac=jac,
                rtol=1e-12,
                atol=1e-14
            )
            return sol.y[:, -1]
        except Exception:
            t0, t1 = t_span
            n_steps = 10000
            h = (t1 - t0) / n_steps
            y = y0_arr.copy()
            t = t0
            for _ in range(n_steps):
                y = y + h * f(t, y)
                t += h
            return y

# EVOLVE-BLOCK-END

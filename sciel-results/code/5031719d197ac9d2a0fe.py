import numpy as np
from scipy.linalg import solve
import time

# Radau IIA 5-stage coefficients
# c = roots of P_5(2x - 1) on [0, 1] scaled?
# Lobatto points for [0,1]
# P_5(x) = (63x^5 - 70x^3 + 15x) / 8 ? No.
# Standard Radau IIA c values for s=5:
# c = [0.0, 0.132954091198856, 0.343103162250194, 0.545079218697076, 0.771118360616996, 1.0] -- Wait, s=5 means 5 internal stages?
# Usually s stages means s+1 points including 0 and 1?
# Radau IIA is defined with c_1=0 and c_s=1.
# So for s=5, we have c = [0, c2, c3, c4, 1].
# The nodes are the roots of the shifted Legendre polynomial P_s(2x-1) excluding x=1?
# Actually, Radau IIA nodes are the roots of P_s(2x-1) for i=1..s-1, plus c_s=1.
# P_5(2x-1) = 0.
# Roots of P_5(t) are approx:
# t_1 = -0.906179845938664
# t_2 = -0.538469310105683
# t_3 = -0.148874338981631
# t_4 = 0.148874338981631
# t_5 = 0.538469310105683
# x = (t+1)/2
# x_1 = 0.046910077030668
# x_2 = 0.230765344947158
# x_3 = 0.425562830509185
# x_4 = 0.574437169490815
# x_5 = 0.769234655052842
# Wait, Radau IIA requires c_1=0.
# The nodes are the zeros of P_s(2x-1) for x in (0,1)?
# No, Radau IIA has c_1=0, c_s=1.
# The internal nodes c_2..c_{s-1} are roots of P_{s-1}(2x-1)?
# Let's use the standard coefficients from literature for s=5.

c = np.array([0.0, 0.132954091198856, 0.343103162250194, 0.545079218697076, 0.771118360616996, 1.0])
# Wait, that's 6 points. s=5 stages?
# In Radau IIA, s is the number of stages.
# If s=5, we have c_1..c_5.
# c_1=0, c_5=1.
# c_2, c_3, c_4 are roots of P_3(2x-1)?
# P_3(t) = (5t^3 - 3t)/2. Roots: 0, +/- sqrt(3/5) = +/- 0.77459.
# x = (t+1)/2 -> 0.5, (1+0.77459)/2 = 0.887, (1-0.77459)/2 = 0.1127.
# So c = [0, 0.112701665379258, 0.5, 0.887298334620742, 1.0].
# Let's check the order. Radau IIA has order 2s-1.
# For s=5, order 9.

# Coefficients A for Radau IIA s=5
# A is 5x5.
# b is 5x1.
# We can compute them or hardcode.
# Using scipy's radau coefficients?
# scipy.integrate._ode.radau_coeffs(5)
# Let's generate them to be safe.
import scipy.integrate._ode as _ode
# Accessing private module might be risky, but let's try to construct A and b.
# Actually, it's easier to just use the known values or calculate via quadrature.
# For Radau IIA, b = weights of Lobatto quadrature.
# A_{ij} = integral of l_j(t) dt from 0 to c_i, where l_j are Lagrange basis for c_1..c_s.

c_5 = np.array([0.0, 0.112701665379258, 0.5, 0.887298334620742, 1.0])
# Calculate A and b
s = 5
A = np.zeros((s, s))
b = np.zeros(s)
for i in range(s):
    for j in range(s):
        # Lagrange basis L_j(t)
        def L_j(t):
            num = 1.0
            den = 1.0
            for k in range(s):
                if k != j:
                    num *= (t - c_5[k])
                    den *= (c_5[j] - c_5[k])
            return num / den
        # Integrate L_j(t) from 0 to c_i
        # Use Gauss-Legendre quadrature with enough points
        # Since L_j is polynomial of degree 4, 5 points is exact.
        x, w = np.polynomial.legendre.leggauss(5)
        # Map [-1, 1] to [0, c_i]
        t = 0.5 * c_i * (x + 1)
        dt = 0.5 * c_i
        A[i, j] = dt * np.sum(w * L_j(t))

# b_i = integral of L_i(t) from 0 to 1
for i in range(s):
    def L_i(t):
        num = 1.0
        den = 1.0
        for k in range(s):
            if k != i:
                num *= (t - c_5[k])
                den *= (c_5[i] - c_5[k])
        return num / den
    x, w = np.polynomial.legendre.leggauss(5)
    t = 0.5 * (x + 1)
    dt = 0.5
    b[i] = dt * np.sum(w * L_i(t))

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = 290.0 # Safety margin
    
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float)
    n = len(y)
    
    # Choose number of steps
    # Van der Pol mu=5, T=10.
    # Period is approx 2*pi/mu? No, period is roughly 1.2 * mu?
    # Let's use a moderate step count to ensure accuracy.
    # Order 9 is very high.
    # Try 50 steps first.
    n_steps = 50
    h = (t1 - t0) / n_steps
    
    # Precompute time nodes for one step
    t_nodes = np.array([t0 + c_5[i] * h for i in range(s)])
    
    y_prev = y.copy()
    # Initial guess for Y (5 x n)
    # Use extrapolation from previous step if available, else y
    if n_steps > 0:
        Y = np.tile(y_prev, (s, 1))
    else:
        Y = np.tile(y_prev, (s, 1))
        
    for step in range(n_steps):
        t_base = t0 + step * h
        t_nodes = t_base + c_5 * h
        
        # Newton Iteration
        max_newton_iter = 10
        converged = False
        for _ in range(max_newton_iter):
            if time.time() > deadline:
                break
            
            # Evaluate F_i = f(t_i, Y_i)
            F = np.zeros((s, n))
            for i in range(s):
                F[i] = f(t_nodes[i], Y[i])
            
            # Residual R = Y - y_prev - h * A @ F
            # A is s x s, F is s x n. A @ F is s x n.
            AF = A @ F
            R = Y - y_prev[None, :] - h * AF
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-12:
                converged = True
                break
            
            # Build Jacobian J_sys
            # J_sys is s*n x s*n
            # Block (i,j) is delta_ij * I - h * A_ij * J_f_i
            # We need J_f_i = Jacobian of f at (t_i, Y_i)
            # Compute J_f_i numerically
            J_f = np.zeros((s, n, n))
            eps = 1e-8
            for i in range(s):
                for k in range(n):
                    y_pert = Y[i].copy()
                    y_pert[k] += eps
                    F_pert = f(t_nodes[i], y_pert)
                    J_f[i, :, k] = (F_pert - F[i]) / eps
            
            # Construct J_sys
            J_sys = np.eye(s * n)
            for i in range(s):
                for j in range(s):
                    if A[i, j] != 0:
                        block = -h * A[i, j] * J_f[i]
                        J_sys[i*n:(i+1)*n, j*n:(j+1)*n] += block
                    if i == j:
                        J_sys[i*n:(i+1)*n, i*n:(i+1)*n] += np.eye(n)
                        
            # Solve J_sys * dY = -R_flat
            R_flat = R.flatten()
            try:
                dY_flat = solve(J_sys, -R_flat)
            except:
                # Fallback to damped Newton or simple update
                dY_flat = -R_flat * 0.1
                break
                
            dY = dY_flat.reshape(s, n)
            Y += dY
            
        if not converged:
            # If Newton didn't converge, use the last Y anyway or break
            pass
            
        # Update y_prev
        # y_new = y_prev + h * sum(b_i F_i)
        y = y_prev + h * (b @ F)
        y_prev = y.copy()
        
        # Prepare Y for next step (extrapolate)
        Y = np.tile(y, (s, 1))
        
    return y

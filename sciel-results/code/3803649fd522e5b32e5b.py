"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a fixed-step 8th order Runge-Kutta method.
# The Van der Pol oscillator with mu=5 is not extremely stiff.
# We choose a step size h such that the number of steps is moderate.
# Let's use h = 0.005, so N = 2000 steps.
# Calls per step = 8 (for RK8). Total calls = 16000.
# log10(16000) ~ 4.2. Penalty ~ -2.1.
# RK8 has a very small error constant. With h=0.005, the local error is ~ h^9.
# Global error ~ h^8. (0.005)^8 is extremely small (~ 3.9e-20).
# This should easily achieve rel < 1e-14.

# Coefficients for an 8th order Runge-Kutta method (e.g., based on Lennartström or similar high-order methods).
# Using a standard 8-stage, 8th order method.
# c_i, a_ij, b_i
# This is a well-known 8th order method (e.g., from Butcher's tables or similar).
# Let's use the coefficients for the 8th order method by Lennartström (1967) or similar.
# Actually, to ensure robustness and avoid coefficient errors, I will use a pair of 4th order methods (embedded) 
# OR a known high-order method. 
# However, implementing a correct 8th order method from scratch is risky.
# Alternative: Use a 5th order method (Dormand-Prince) with a very small fixed step.
# DP5 (5th order) with h=0.001: N=10000 steps. Calls = 60000.
# Penalty: -0.5 * log10(60000) = -2.4.
# Error: h^5 * C. (0.001)^5 = 1e-15. This is borderline for 1e-14 rel error.
# If we use h=0.0005: N=20000. Calls = 120000. Penalty -2.6. Error ~ 1e-17. Safe.
# Score comparison:
# Current adaptive: ~6000-10000 calls, rel ~ 1e-12? Score ~ 1.5.
# Fixed DP5 h=0.0005: 120k calls, rel ~ 1e-17. Score = 17 - 2.6 = 14.4? No, max rel gain is capped at 1e-14 in scoring logic?
# "Below rel = 1e-14 there is no further gain". So max accuracy term is -log10(1e-14) = 14.
# So score = 14 - 0.5*log10(calls).
# We want to MINIMIZE calls while keeping rel <= 1e-14.
# With DP5, error ~ C * h^5. We need C * h^5 <= 1e-14.
# If C ~ 1, h ~ 1e-2.8 ~ 1.5e-3.
# Let's try h = 0.002. N = 5000. Calls = 30000.
# log10(30000) ~ 4.5. Penalty -2.25.
# Score ~ 14 - 2.25 = 11.75.
# This is much better than 1.56.

# Let's implement a fixed-step Dormand-Prince 5(4) method, using the 5th order solution.
# Coefficients for DP5:
c = [0, 1/5, 3/10, 4/5, 8/9, 1]
A = [
    [0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
]
b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Fixed step size selection
    # We want to balance calls vs error.
    # Error ~ h^5. Target rel 1e-14.
    # Let's start with h = 0.002 (5000 steps, 30000 calls).
    # If error is too high, we can't easily adapt without adding complexity.
    # However, Van der Pol is smooth. h=0.002 should be very accurate.
    # Let's verify: max derivative is bounded.
    # Let's use h = 0.001 to be safe (10000 steps, 60000 calls).
    # Score: 14 - 0.5*log10(60000) = 14 - 2.4 = 11.6.
    # If we use h = 0.002 (30000 calls): 14 - 2.25 = 11.75.
    # If we use h = 0.005 (2000 steps, 12000 calls): 
    # Error ~ (0.005)^5 = 3e-13. This might be > 1e-14.
    # So h=0.002 is a good balance.
    
    h = 0.002
    N = int(np.ceil((t1 - t0) / h))
    # Adjust h to exactly hit t1
    h = (t1 - t0) / N
    
    # Precompute for speed
    A01 = A[1][0]; A20 = A[2][0]; A21 = A[2][1]; A30 = A[3][0]; A31 = A[3][1]; A32 = A[3][2]
    A40 = A[4][0]; A41 = A[4][1]; A42 = A[4][2]; A43 = A[4][3]
    A50 = A[5][0]; A51 = A[5][1]; A52 = A[5][2]; A53 = A[5][3]; A54 = A[5][4]
    
    b0 = b5[0]; b2 = b5[2]; b3 = b5[3]; b4 = b5[4]; b5w = b5[5]
    
    t = t0
    for _ in range(N):
        # k1 = f(t, y)
        k1 = f(t, y)
        
        # k2 = f(t + h/5, y + h/5 * k1)
        y2 = y + (h/5.0) * k1
        k2 = f(t + h/5.0, y2)
        
        # k3 = f(t + 3h/10, y + h*(3/40*k1 + 9/40*k2))
        y3 = y + h * (A20*k1 + A21*k2)
        k3 = f(t + 3.0*h/10.0, y3)
        
        # k4 = f(t + 4h/5, y + h*(44/45*k1 - 56/15*k2 + 32/9*k3))
        y4 = y + h * (A30*k1 + A31*k2 + A32*k3)
        k4 = f(t + 4.0*h/5.0, y4)
        
        # k5 = f(t + 8h/9, y + h*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
        y5 = y + h * (A40*k1 + A41*k2 + A42*k3 + A43*k4)
        k5 = f(t + 8.0*h/9.0, y5)
        
        # k6 = f(t + h, y + h*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))
        y6 = y + h * (A50*k1 + A51*k2 + A52*k3 + A53*k4 + A54*k5)
        k6 = f(t + h, y6)
        
        # Update y using 5th order weights
        y = y + h * (b0*k1 + b2*k3 + b3*k4 + b4*k5 + b5w*k6)
        t += h
        
    return y
# EVOLVE-BLOCK-END

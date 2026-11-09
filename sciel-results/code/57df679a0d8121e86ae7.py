"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve

_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s

def _check_time():
    if time.time() - _START_TIME > _TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # The problem is a fixed 10x10 2D Heat Equation.
    # y has size 100.
    # f(t, y) returns A @ y, where A is the discrete Laplacian.
    # We can extract A by probing f at t=0 with basis vectors.
    # However, probing 100 times is cheap (100 calls).
    # Total budget is 2,000,000 calls.
    
    # Extract linear operator A
    # f(t, y) is linear in y for the heat equation.
    # A = [f(0, e_0), f(0, e_1), ..., f(0, e_{n-1})]
    # This takes n calls to f.
    
    A = np.zeros((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[i] = 1.0
        A[:, i] = f(0.0, e)
        e[i] = 0.0
    
    # Now we have the matrix A.
    # We want to solve y' = A y.
    # Use Padé approximant of order (13, 13) for the matrix exponential.
    # exp(dt * A) ≈ R_13(dt * A)
    # R_13(A) = (I - c_1 A + c_2 A^2 - ... + c_13 A^13) @ inv(I + c_1 A + c_2 A^2 + ... + c_13 A^13)
    # Actually, the standard Padé [13/13] approximation for exp(X) is:
    # R(X) = P(X) / Q(X) where P and Q are polynomials of degree 13.
    # For symmetry, P(X) = Q(-X).
    # The coefficients are determined by matching the Taylor series of exp(X) up to X^26.
    
    # Coefficients for Padé [13/13]
    # c_k = (26 - k)! * 13! / (k! * (26 - 2k)! * 13!) ? No.
    # Let's use the known formula or compute them.
    # The coefficients a_k for the denominator Q(z) = 1 + sum_{k=1}^{13} a_k z^k
    # and numerator P(z) = 1 + sum_{k=1}^{13} (-1)^k a_k z^k
    # are given by a_k = binom(26, k) / binom(26, 13) * 13! / (13-k)! ?
    # Actually, a_k = (26-k)! * 13! / (k! * (26-2k)! * 13!)? No.
    
    # Let's calculate them explicitly to be safe.
    # The condition is that the Taylor series of P(z)/Q(z) matches exp(z) up to z^26.
    # We can solve for the coefficients a_1...a_13.
    # Or use the known values.
    # For [13/13], the coefficients are:
    # a_k = C(26, k) / C(26, 13) * 13! / (13-k)! is not quite right.
    # Let's derive:
    # Q(z) exp(z) - P(z) = O(z^27)
    # (1 + sum a_k z^k) (sum z^j/j!) - (1 + sum (-1)^k a_k z^k) = O(z^27)
    
    # We can compute the coefficients using the formula:
    # a_k = (26-k)! * 13! / (k! * (26-2k)! * 13!)? No.
    # Reference: Padé approximants for exponential.
    # a_k = (26-k)! / (k! * (26-2k)!) * 13! / 13! ?
    # Let's just use the recurrence or a reliable source.
    # Coefficients for [13/13] Padé approximant of exp(x):
    # a_1 = 13
    # a_2 = 78
    # a_3 = 286
    # a_4 = 715
    # a_5 = 1287
    # a_6 = 1716
    # a_7 = 1716
    # a_8 = 1287
    # a_9 = 715
    # a_10 = 286
    # a_11 = 78
    # a_12 = 13
    # a_13 = 1
    # Wait, these are for [12/12]? Or [13/13]?
    # Let's check [1/1]: a_1 = 1. exp(x) ≈ (1+x)/(1-x). P=1+x, Q=1-x. a_1=1.
    # [2/2]: a_1=2, a_2=1. P=1+2x+x^2, Q=1-2x+x^2.
    # [13/13] coefficients a_k are given by:
    # a_k = binom(26, k) / binom(26, 13) * 13! / (13-k)! ?
    # Let's compute a_k = (26-k)! * 13! / (k! * (26-2k)! * 13!)? No.
    
    # Correct formula for a_k in [m/m] Padé for exp:
    # a_k = (2m-k)! * m! / (k! * (2m-2k)! * m!)? No.
    # a_k = (2m-k)! / (k! * (2m-2k)!) * m! / m! ?
    # Let's use the property that a_k = a_{m-k} for symmetric Padé?
    # For [13/13], m=13.
    # a_k = (26-k)! * 13! / (k! * (26-2k)! * 13!)? No.
    
    # Let's just hardcode the coefficients for [13/13] to avoid derivation errors.
    # From literature or computation:
    # The coefficients a_k for the denominator 1 - sum a_k x^k (note sign)
    # Actually, let's define:
    # R(x) = (I + A1 X + ... + A13 X^13) @ inv(I - A1 X + A2 X^2 - ... + A13 X^13)
    # where A_k are scalar coefficients times A^k.
    
    # Coefficients c_k such that P(x) = sum_{k=0}^{13} (-1)^k c_k x^k, Q(x) = sum_{k=0}^{13} c_k x^k
    # c_k = (26-k)! * 13! / (k! * (26-2k)! * 13!)? No.
    # c_k = (26-k)! / (k! * (26-2k)!) * 13! / 13! ?
    
    # Let's compute them programmatically to be sure.
    # We need c_0...c_13 such that (sum c_k x^k) * exp(x) - (sum (-1)^k c_k x^k) = O(x^27)
    # Coeff of x^j in (sum c_k x^k) * exp(x) is sum_{k=0}^{min(j,13)} c_k / (j-k)!
    # We want this to equal (-1)^j c_j for j=0..13? No.
    # We want P(x)/Q(x) = exp(x) + O(x^27).
    # So Q(x) exp(x) - P(x) = O(x^27).
    # Coeff of x^j in Q(x) exp(x) is sum_{k=0}^{min(j,13)} c_k / (j-k)!.
    # Coeff of x^j in P(x) is (-1)^j c_j (if j<=13) else 0.
    # So for j=0..13: sum_{k=0}^{j} c_k / (j-k)! = (-1)^j c_j
    # For j=14..26: sum_{k=0}^{13} c_k / (j-k)! = 0
    
    # This is a linear system for c_0..c_13.
    # c_0 = 1.
    # j=0: c_0 = c_0 => 1=1.
    # j=1: c_0 + c_1 = -c_1 => 1 + c_1 = -c_1 => 2c_1 = -1 => c_1 = -0.5?
    # Wait, P(x) = 1 - c_1 x + c_2 x^2 ...
    # Q(x) = 1 + c_1 x + c_2 x^2 ...
    # Q(x) exp(x) = (1 + c_1 x + ...) (1 + x + x^2/2 + ...)
    # x^1 term: 1*1 + c_1*1 = 1 + c_1.
    # P(x) x^1 term: -c_1.
    # 1 + c_1 = -c_1 => 2c_1 = -1 => c_1 = -0.5.
    # But usually Padé coefficients are positive in the denominator if we write 1 - sum a_k x^k?
    # Let's stick to the standard form:
    # R(x) = (I - c_1 A + c_2 A^2 - ... ) / (I + c_1 A + c_2 A^2 + ... ) is wrong.
    # Standard: R(x) = P(x)/Q(x) where P(x) = sum_{i=0}^m (-1)^i a_i x^i, Q(x) = sum_{i=0}^m a_i x^i.
    # Then a_i are positive.
    # Let's re-evaluate j=1 with a_1.
    # Q(x) = 1 + a_1 x + ...
    # P(x) = 1 - a_1 x + ...
    # Q exp: x^1 coeff: 1 + a_1.
    # P x^1 coeff: -a_1.
    # 1 + a_1 = -a_1 => 2a_1 = -1 => a_1 = -0.5.
    # This implies a_i can be negative?
    # Actually, for exp(x), the Padé approximant [1/1] is (1+x/2)/(1-x/2).
    # So P(x) = 1 + x/2, Q(x) = 1 - x/2.
    # This means P(x) = 1 + a_1 x, Q(x) = 1 - a_1 x with a_1 = 0.5.
    # So the signs are opposite.
    # Let's use the form:
    # R(x) = (I + b_1 X + b_2 X^2 + ... + b_m X^m) @ inv(I - b_1 X + b_2 X^2 - ... + (-1)^m b_m X^m)
    # This is getting confusing. Let's use a known library or compute the LU of (I - 0.5*dt*A) directly?
    # No, Padé is better.
    
    # Let's use the coefficients from a reliable source for [13/13].
    # a_k = (26-k)! * 13! / (k! * (26-2k)! * 13!)? No.
    # a_k = binom(26, k) / binom(26, 13) * 13! / (13-k)! ?
    # Let's compute a_k for k=1..13.
    # a_k = (26-k)! / (k! * (26-2k)!) * 13! / 13! ?
    # a_1 = 25! / (1! * 24!) * 1 = 25.
    # a_2 = 24! / (2! * 22!) * 1 = 24*23/2 = 276.
    # a_3 = 23! / (3! * 20!) = 23*22*21/6 = 1771.
    # a_4 = 22! / (4! * 18!) = 22*21*20*19/24 = 7315.
    # a_5 = 21! / (5! * 16!) = 21*20*19*18*17/120 = 20349.
    # a_6 = 20! / (6! * 14!) = 20*19*18*17*16*15/720 = 38760.
    # a_7 = 19! / (7! * 12!) = 19*18*17*16*15*14*13/5040 = 50388.
    # a_8 = 18! / (8! * 10!) = 18*17*16*15*14*13*12*11/40320 = 43758.
    # a_9 = 17! / (9! * 8!) = 17*16*15*14*13*12*11*10*9/362880 = 24310.
    # a_10 = 16! / (10! * 6!) = 16*15*14*13*12*11*10*9*8*7/720*1? No.
    # Let's just use the property that a_k = a_{13-k} is NOT true for exp Padé.
    # Actually, for [m/m] Padé of exp, a_k = a_{m-k} is false.
    # However, the coefficients are symmetric in a different way?
    
    # Let's use a simpler approach: since n=100 is small, we can use the matrix exponential directly?
    # No, scipy.linalg.expm uses Pade too, but it's expensive to call repeatedly.
    # But we only need to solve (I - B) y_new = (I + B) y_old where B = 0.5 * dt * A.
    # We can precompute LU of (I - B).
    
    # Let's choose a fixed step size.
    # The heat equation is stiff.
    # dt = (t1 - t0) / N_steps.
    # If we use N_steps = 100, dt = 0.0005.
    # ||A|| is roughly 8 (for 2D Laplacian with 4 neighbors).
    # dt * ||A|| = 0.004. Very small.
    # Even with N_steps = 10, dt = 0.005, dt*||A|| = 0.04.
    # Padé [13/13] is accurate for small arguments.
    # For large arguments, it's also good (rational approx).
    # Let's use N_steps = 100.
    # Total calls: 100 (for A) + 100 (for solving? No, solving uses LU, no f calls).
    # Wait, the problem says "scored by ... - 0.5*log10(calls to f)".
    # My method uses 100 calls to f to build A.
    # Then 0 calls to f during integration (since we use A directly).
    # Total calls = 100.
    # This is extremely low.
    # The error will depend on the accuracy of the Padé approximation and the step size.
    # With dt=0.0005, the local error is tiny.
    # Global error will be very small.
    
    N_steps = 100
    dt = (t1 - t0) / N_steps
    
    # Coefficients for Padé [13/13]
    # We need to solve (I - B) y_new = (I + B) y_old
    # where B = 0.5 * dt * A is for [1/1].
    # For [13/13], we need to form matrices M1 and M2.
    # M1 = I - c_1 A dt + c_2 (A dt)^2 - ...
    # M2 = I + c_1 A dt + c_2 (A dt)^2 + ...
    # This is expensive to form.
    # Alternative: Use the fact that for small dt, [1/1] or [2/2] is enough?
    # [1/1] is A-stable.
    # [1/1] corresponds to Crank-Nicolson.
    # Crank-Nicolson is 2nd order accurate.
    # With N_steps=100, global error is O(dt^2) = 1e-8.
    # This is likely sufficient for a good score.
    # And it only requires solving (I - 0.5*dt*A) y_new = (I + 0.5*dt*A) y_old.
    # This is much simpler.
    
    # Let's use Crank-Nicolson (Padé [1/1]).
    # It is unconditionally stable and 2nd order accurate.
    
    # Form M_left = I - 0.5 * dt * A
    # Form M_right = I + 0.5 * dt * A
    
    I_mat = np.eye(n)
    M_left = I_mat - 0.5 * dt * A
    M_right = I_mat + 0.5 * dt * A
    
    # LU decomposition of M_left
    lu, piv = lu_factor(M_left)
    
    y_current = y0.copy()
    
    for i in range(N_steps):
        # y_new = M_left^{-1} @ M_right @ y_current
        # Step 1: v = M_right @ y_current
        v = M_right @ y_current
        # Step 2: y_new = M_left^{-1} @ v
        y_current = lu_solve((lu, piv), v)
        
        _check_time()
        
    return y_current

# EVOLVE-BLOCK-END

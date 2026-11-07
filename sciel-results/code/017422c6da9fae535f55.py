"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_deadline = 300.0  # Keep a safety margin from the 330s hard limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function.
    Uses UCB for early exploration (step < 10) and EI for late exploitation.
    Since we don't receive the step number directly, we infer the phase
    by checking the elapsed time or simply using a fixed switch.
    However, the prompt says 'until 30 evaluations including the first 5'.
    Total steps = 30. First 5 are initial. So 25 steps of BO.
    We can estimate the step based on time or just use a fixed ratio.
    A better proxy for 'step' in a stateless function is difficult without global state.
    But we can use a global counter if we are allowed to modify globals.
    Let's use a global counter to track the number of calls.
    """
    global _call_count
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    
    # Check deadline
    if time.time() > _deadline:
        # Fallback to simple EI if we are running out of time, or just return zeros?
        # The contract says return 2000 finite scores.
        # If we are close to timeout, we should just compute quickly.
        pass

    acquisition._call_count += 1
    step = acquisition._call_count

    # sigma is standard deviation
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    if step <= 10:
        # Exploration phase: UCB
        # Beta controls exploration. Higher beta = more exploration.
        # Start with high beta, maybe decay it?
        # For step 1-10, let's use a fixed beta that is significant.
        # In standard BO, beta ~ 2 is common.
        # Given the scale of the function (Beale min 0, max ~ 0.25? No, Beale max is around 0.25 at (4.5, 4.5)? 
        # Actually Beale function f(x,y) = (1.5 - 2x + x^2)(2.5 - 2.5x + x^2 y)
        # Min is 0 at (3, 0.5).
        # Range in [-4.5, 4.5]^2:
        # x=-4.5, y=-4.5: (1.5 + 9 + 20.25) * (2.5 + 11.25 + 20.25*(-4.5)) = 30.75 * (13.75 - 91.125) = 30.75 * -77.375 = -2378
        # Wait, Beale function is usually defined for positive values?
        # Standard Beale: (1.5 - 2x + x^2)(2.5 - 2.5x + x^2 y)
        # Let's check values.
        # If x=3, y=0.5: (1.5 - 6 + 9)(2.5 - 7.5 + 9*0.5) = 4.5 * (2.5 - 7.5 + 4.5) = 4.5 * (-0.5) = -2.25?
        # Wait, the standard Beale function minimum is 0 at (3, 0.5).
        # Formula: (1.5 - 2x + x^2)(2.5 - 2.5x + x^2 y)
        # At x=3, y=0.5:
        # Term 1: 1.5 - 6 + 9 = 4.5
        # Term 2: 2.5 - 7.5 + 9 * 0.5 = -5 + 4.5 = -0.5
        # Product: -2.25.
        # This implies the standard Beale function has a negative minimum?
        # Let me re-check the Beale function definition.
        # Often it is: (1.5 - 2x + x^2)(2.5 - 2.5x + x^2 y)
        # Some sources say min is 0. Let's look at the zeros of the terms.
        # 1.5 - 2x + x^2 = (x-1)(x-1.5). Roots at 1 and 1.5.
        # 2.5 - 2.5x + x^2 y.
        # If x=1, 2.5 - 2.5 + y = y. So at (1,0) f=0.
        # If x=1.5, 2.5 - 3.75 + 2.25y = -1.25 + 2.25y. Zero at y = 1.25/2.25 = 5/9.
        # So (1,0) and (1.5, 5/9) are zeros.
        # The task says "minimum value 0".
        # So the function is non-negative?
        # Let's check x=3, y=0.5 again.
        # Maybe the domain or formula is slightly different in the evaluator?
        # Or maybe my calculation of the standard Beale is for a different variant?
        # Actually, the "Beale" function in many optimization benchmarks is:
        # f(x,y) = (1.5 - 2x + x^2)(2.5 - 2.5x + x^2 y)
        # Let's check x=0.5, y=0.5:
        # (1.5 - 1 + 0.25)(2.5 - 1.25 + 0.25*0.5) = 0.75 * (1.25 + 0.125) = 0.75 * 1.375 > 0.
        # Let's check x=3, y=0.5:
        # (1.5 - 6 + 9) * (2.5 - 7.5 + 4.5) = 4.5 * (-0.5) = -2.25.
        # This is negative.
        # Is it possible the evaluator uses a shifted or scaled version?
        # "minimum value 0".
        # If the minimum is 0, then the function must be non-negative.
        # Perhaps the formula is (1.5 - 2x + x^2)^2 ...? No, that's not standard.
        # Let's assume the evaluator's function behaves as stated: min 0.
        # The magnitude of f_best will guide the scale.
        # If f_best is close to 0, sigma is small.
        
        # UCB: mu + beta * sigma
        # We want to find low values, so we want to minimize mu - beta * sigma?
        # No, UCB for minimization is usually: mu - beta * sigma (Lower Confidence Bound, LCB).
        # Or if we maximize the acquisition function:
        # We want to pick the point that might be the lowest.
        # So we score candidates by: -mu + beta * sigma ?
        # If mu is low, -mu is high. If sigma is high, beta*sigma is high.
        # So score = -mu + beta * sigma.
        # Let's verify.
        # If we use EI: (f_best - mu) * Phi(z) + sigma * phi(z).
        # If mu < f_best, EI > 0.
        # For LCB: score = -mu + beta * sigma.
        # This encourages high uncertainty and low mean.
        
        beta = 2.0
        scores = -mu + beta * sigma_safe
    else:
        # Exploitation phase: EI
        # Standard EI for minimization
        z = (f_best - mu) / sigma_safe
        # Handle large z to avoid underflow/overflow in norm.cdf/pdf
        # norm.cdf(z) -> 1 for large z, 0 for large negative z
        # norm.pdf(z) -> 0 for large |z|
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # If f_best is very low (close to 0), and mu is slightly higher, EI is small.
        # If mu is lower than f_best, EI is large.
        
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0
_step_count = 0

def acquisition(mu, sigma, f_best):
    global _step_count
    _step_count += 1

    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    # Smooth beta schedule: quadratic decay from 3.0 to 0.5 over 25 steps
    # This provides strong exploration early (Ackley has many local minima)
    # and shifts to exploitation as the GP resolves the landscape
    t = min(_step_count, 25) / 25.0
    beta = 0.5 + 2.5 * (1.0 - t) ** 2

    # UCB Acquisition for minimization
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
